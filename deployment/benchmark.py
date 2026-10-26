#!/usr/bin/env python3
"""
benchmark.py — Unified benchmarking for YOLOv8, YOLO11, and RF-DETR on Jetson Orin Nano.

Modes:
  (a) Live webcam with overlay:
      python3 benchmark.py --mode live --model yolov8 --engine engines/best_fp16.engine
  (b) Headless benchmark on video/images:
      python3 benchmark.py --mode headless --all-engines engines/ --source val_images/ --num-frames 500

Full run across all models:
    python3 benchmark.py --mode headless \
        --yolov8-engine engines/best_fp16.engine \
        --yolov8-pt Models/Yolov8/best.pt \
        --yolo11-engine engines/yolo26n_fp16.engine \
        --yolo11-pt Models/Yolov11/yolo26n.pt \
        --rfdetr-engine engines/rfdetr_fp16.engine \
        --source val_images/ --num-frames 500 \
        --val-dir val_images/ --val-ann val_annotations.json
"""

import argparse
import csv
import json
import os
import sys
import time
import threading
import queue
import glob
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (graceful fallback if not on Jetson)
# ─────────────────────────────────────────────────────────────────────────────
def _import_trt():
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
    return trt, cuda

def _import_ultralytics():
    from ultralytics import YOLO
    return YOLO

def _import_torch():
    import torch
    return torch


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0: Utility helpers
# ═════════════════════════════════════════════════════════════════════════════
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def file_size_mb(p):
    return os.path.getsize(p) / (1024**2) if p and os.path.exists(p) else 0.0

def compute_stats(arr):
    """Compute latency statistics for an array of ms values."""
    if len(arr) == 0:
        return {k: 0.0 for k in ["mean","median","std","min","max","p90","p95","p99"]}
    a = np.array(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with letterbox padding (used for YOLO)."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: TensorRT YOLO wrapper
# ═════════════════════════════════════════════════════════════════════════════
class TRTYOLOEngine:
    """TensorRT inference for YOLO (v8/v11/v26) detection models."""

    def __init__(self, engine_path, imgsz=640, conf=0.25, iou=0.45):
        trt_mod, cuda_mod = _import_trt()
        self.trt = trt_mod
        self.cuda = cuda_mod
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.logger = trt_mod.Logger(trt_mod.Logger.WARNING)
        t0 = time.time()
        with open(engine_path, "rb") as f:
            runtime = trt_mod.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.load_time = time.time() - t0

        self.context = self.engine.create_execution_context()
        self.stream = cuda_mod.Stream()

        # Discover I/O
        self.bindings = {}
        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt_mod.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            host = cuda_mod.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda_mod.mem_alloc(nbytes)
            mode = self.engine.get_tensor_mode(name)
            self.bindings[name] = {"shape": shape, "dtype": dtype, "host": host,
                                    "device": device, "nbytes": nbytes}
            if mode == trt_mod.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        in_shape = self.bindings[self.input_name]["shape"]
        self.input_h, self.input_w = in_shape[2], in_shape[3]
        print(f"[YOLO TRT] {engine_path}  input={in_shape}  load={self.load_time*1000:.0f}ms")

    def preprocess(self, img_bgr):
        padded, ratio, (dw, dh) = letterbox(img_bgr, (self.input_h, self.input_w))
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])
        return blob, ratio, dw, dh

    def infer(self, blob):
        cuda = self.cuda
        timings = {}
        inp = self.bindings[self.input_name]
        np.copyto(inp["host"], blob.ravel())

        t0 = time.time()
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.stream.synchronize()
        timings["h2d_ms"] = (time.time() - t0) * 1000

        self.context.set_tensor_address(self.input_name, int(inp["device"]))
        for on in self.output_names:
            self.context.set_tensor_address(on, int(self.bindings[on]["device"]))

        start_ev = cuda.Event()
        end_ev = cuda.Event()
        start_ev.record(self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        end_ev.record(self.stream)
        end_ev.synchronize()
        timings["infer_ms"] = start_ev.time_till(end_ev)

        t0 = time.time()
        outputs = {}
        for on in self.output_names:
            o = self.bindings[on]
            cuda.memcpy_dtoh_async(o["host"], o["device"], self.stream)
        self.stream.synchronize()
        timings["d2h_ms"] = (time.time() - t0) * 1000

        for on in self.output_names:
            o = self.bindings[on]
            outputs[on] = o["host"].reshape(o["shape"]).copy()
        return outputs, timings

    def postprocess(self, outputs, ratio, dw, dh, orig_shape):
        """NMS-based postprocessing for YOLO detection output."""
        # YOLO output: (1, 4+nc, num_preds) or (1, num_preds, 4+nc)
        out = list(outputs.values())[0]
        if out.ndim == 3:
            out = out[0]
        # Determine layout
        if out.shape[0] < out.shape[1]:
            out = out.T  # now (num_preds, 4+nc)

        boxes_xywh = out[:, :4]
        scores_all = out[:, 4:]
        num_classes = scores_all.shape[1]

        max_scores = scores_all.max(axis=1)
        mask = max_scores >= self.conf
        if mask.sum() == 0:
            return np.empty((0,4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32)

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = scores_all[mask].argmax(axis=1)

        # xywh → xyxy
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Undo letterbox
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Clip
        oh, ow = orig_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

        # NMS
        keep = _nms(boxes, max_scores, self.iou)
        return boxes[keep], max_scores[keep].astype(np.float32), class_ids[keep].astype(np.int32)

    def warmup(self, n=10):
        dummy = np.random.randn(1, 3, self.input_h, self.input_w).astype(np.float32)
        for _ in range(n):
            self.infer(dummy)


def _nms(boxes, scores, iou_thresh):
    """Simple greedy NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Threaded camera capture
# ═════════════════════════════════════════════════════════════════════════════
class CameraCapture:
    """Threaded camera reader so capture doesn't block inference."""

    def __init__(self, source=0, width=1920, height=1080):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        self.frames_read = 0
        self.frames_dropped = 0
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            self.frames_read += 1
            if self.q.full():
                try:
                    self.q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if self.stopped and self.q.empty():
            return None
        try:
            return self.q.get(timeout=5.0)
        except queue.Empty:
            return None

    def release(self):
        self.stopped = True
        self.cap.release()


class ImageFolderSource:
    """Iterator over images in a folder (headless benchmark)."""

    def __init__(self, path, loop=False):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.files = []
        for e in exts:
            self.files.extend(glob.glob(os.path.join(path, e)))
            self.files.extend(glob.glob(os.path.join(path, "**", e), recursive=True))
        self.files = sorted(set(self.files))
        if not self.files:
            raise FileNotFoundError(f"No images in {path}")
        self.idx = 0
        self.loop = loop
        self.frames_read = 0
        self.frames_dropped = 0
        self.actual_fps = 0
        print(f"[ImageFolder] {len(self.files)} images from {path}")

    def read(self):
        if self.idx >= len(self.files):
            if self.loop:
                self.idx = 0
            else:
                return None
        img = cv2.imread(self.files[self.idx])
        self.idx += 1
        self.frames_read += 1
        return img

    def release(self):
        pass

    def filepath(self, idx=None):
        i = (idx if idx is not None else self.idx - 1) % len(self.files)
        return self.files[i]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Resource monitor (jtop / tegrastats)
# ═════════════════════════════════════════════════════════════════════════════
class ResourceMonitor:
    """Sample GPU/CPU/power/temp at ~10 Hz via jtop or tegrastats parsing."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _sample_loop(self):
        try:
            from jtop import jtop
            with jtop(interval=int(self.interval * 1000)) as jetson:
                while jetson.ok() and not self._stop.is_set():
                    s = {
                        "ts": time.time(),
                        "gpu_util": jetson.stats.get("GPU", 0),
                        "gpu_freq_mhz": jetson.stats.get("GPU freq", 0),
                        "cpu_util": [],
                        "ram_used_mb": jetson.stats.get("RAM", 0),
                        "swap_used_mb": jetson.stats.get("SWAP", 0),
                        "power_total_mw": jetson.stats.get("Power TOT", 0),
                        "power_gpu_mw": jetson.stats.get("Power GPU", 0),
                        "power_cpu_mw": jetson.stats.get("Power CPU", 0),
                        "temp_cpu": jetson.stats.get("Temp CPU", 0),
                        "temp_gpu": jetson.stats.get("Temp GPU", 0),
                    }
                    # Per-core CPU
                    for k, v in jetson.stats.items():
                        if k.startswith("CPU") and k[3:].isdigit():
                            s["cpu_util"].append(v)
                    self.samples.append(s)
        except Exception:
            # Fallback: parse tegrastats
            self._sample_tegrastats()

    def _sample_tegrastats(self):
        """Fallback tegrastats parser."""
        import psutil
        while not self._stop.is_set():
            t0 = time.time()
            cpu_pcts = psutil.cpu_percent(percpu=True)
            mem = psutil.virtual_memory()
            # Try reading sysfs for GPU/power/temp
            gpu_util = _read_sysfs("/sys/devices/gpu.0/load", div=10, default=0)
            power_mw = _read_sysfs("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_input",
                                    default=0)
            temp_cpu = _read_sysfs("/sys/devices/virtual/thermal/thermal_zone*/temp",
                                    div=1000, default=0, pick_max=True)
            self.samples.append({
                "ts": t0,
                "gpu_util": gpu_util,
                "gpu_freq_mhz": 0,
                "cpu_util": cpu_pcts,
                "ram_used_mb": mem.used / (1024**2),
                "swap_used_mb": 0,
                "power_total_mw": power_mw,
                "power_gpu_mw": 0,
                "power_cpu_mw": 0,
                "temp_cpu": temp_cpu,
                "temp_gpu": temp_cpu,
            })
            elapsed = time.time() - t0
            time.sleep(max(0, self.interval - elapsed))

    def summarize(self):
        if not self.samples:
            return {}
        gpu_utils = [s["gpu_util"] for s in self.samples]
        cpu_avgs = [np.mean(s["cpu_util"]) if s["cpu_util"] else 0 for s in self.samples]
        powers = [s["power_total_mw"] for s in self.samples]
        temps_cpu = [s["temp_cpu"] for s in self.samples]
        temps_gpu = [s["temp_gpu"] for s in self.samples]
        rams = [s["ram_used_mb"] for s in self.samples]
        return {
            "gpu_util_avg": float(np.mean(gpu_utils)),
            "gpu_util_max": float(np.max(gpu_utils)),
            "cpu_util_avg": float(np.mean(cpu_avgs)),
            "cpu_util_max": float(np.max(cpu_avgs)),
            "ram_peak_mb": float(np.max(rams)),
            "power_avg_w": float(np.mean(powers)) / 1000,
            "power_peak_w": float(np.max(powers)) / 1000,
            "temp_cpu_avg": float(np.mean(temps_cpu)),
            "temp_cpu_max": float(np.max(temps_cpu)),
            "temp_gpu_avg": float(np.mean(temps_gpu)),
            "temp_gpu_max": float(np.max(temps_gpu)),
            "thermal_throttle_count": sum(1 for t in temps_gpu if t > 95),
            "num_samples": len(self.samples),
        }


def _read_sysfs(pattern, div=1, default=0, pick_max=False):
    """Read an integer from a sysfs file (glob-safe)."""
    import glob as g
    files = g.glob(pattern)
    if not files:
        return default
    vals = []
    for f in files:
        try:
            with open(f) as fh:
                vals.append(int(fh.read().strip()) / div)
        except Exception:
            pass
    if not vals:
        return default
    return max(vals) if pick_max else vals[0]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Accuracy evaluation (COCO mAP via pycocotools)
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_accuracy(engine_wrapper, source_dir, ann_json=None, class_names=None,
                       conf=0.25, iou=0.45, imgsz=640, is_rfdetr=False, out_dir="results"):
    """
    Run model on all images and compute COCO-style mAP.

    If ann_json is provided (COCO format), use it for GT.
    Otherwise, look for YOLO-format .txt labels alongside images.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ensure_dir(out_dir)
    img_files = sorted(glob.glob(os.path.join(source_dir, "**", "*.jpg"), recursive=True) +
                        glob.glob(os.path.join(source_dir, "**", "*.png"), recursive=True))
    if not img_files:
        print(f"[WARN] No images found in {source_dir} for accuracy eval")
        return {}

    print(f"\n  Evaluating accuracy on {len(img_files)} images...")

    # Build COCO GT dataset
    if ann_json and os.path.exists(ann_json):
        coco_gt = COCO(ann_json)
        img_id_map = {os.path.basename(coco_gt.imgs[i]["file_name"]): i for i in coco_gt.imgs}
    else:
        # Build from YOLO .txt labels
        coco_gt_dict, img_id_map = _build_coco_from_yolo_labels(img_files, class_names)
        if coco_gt_dict is None:
            print("  [WARN] No labels found — accuracy eval skipped.")
            return {}
        tmp_gt_path = os.path.join(out_dir, "_tmp_gt.json")
        with open(tmp_gt_path, "w") as f:
            json.dump(coco_gt_dict, f)
        coco_gt = COCO(tmp_gt_path)

    # Run inference and collect detections
    detections = []
    det_counts = []
    for img_path in img_files:
        fname = os.path.basename(img_path)
        if fname not in img_id_map:
            continue
        img_id = img_id_map[fname]
        img = cv2.imread(img_path)
        if img is None:
            continue

        if is_rfdetr:
            blob, sx, sy = engine_wrapper.preprocess(img)
            outputs, _ = engine_wrapper.infer(blob)
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, sx, sy, conf_thresh=conf)
        else:
            blob, ratio, dw, dh = engine_wrapper.preprocess(img)
            outputs, _ = engine_wrapper.infer(blob)
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, ratio, dw, dh, img.shape)

        det_counts.append(len(boxes))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                "image_id": img_id,
                "category_id": int(class_ids[i]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # xywh
                "score": float(scores[i]),
            })

    if not detections:
        print("  No detections — mAP=0")
        return {"mAP50": 0, "mAP50_95": 0, "avg_dets_per_img": 0}

    # Save detections
    det_path = os.path.join(out_dir, "_tmp_det.json")
    with open(det_path, "w") as f:
        json.dump(detections, f)

    # COCO eval
    coco_dt = coco_gt.loadRes(det_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    results = {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "avg_dets_per_img": float(np.mean(det_counts)) if det_counts else 0,
        "total_images": len(img_files),
        "total_dets": len(detections),
    }

    # Per-class AP
    if class_names:
        cat_ids = coco_gt.getCatIds()
        per_class = {}
        for ci in cat_ids:
            coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval_cls.params.catIds = [ci]
            coco_eval_cls.evaluate()
            coco_eval_cls.accumulate()
            coco_eval_cls.summarize()
            cname = class_names.get(ci, str(ci)) if isinstance(class_names, dict) else str(ci)
            per_class[cname] = {
                "AP50": float(coco_eval_cls.stats[1]),
                "AP50_95": float(coco_eval_cls.stats[0]),
            }
        results["per_class"] = per_class

    # Precision / Recall / F1 at fixed conf
    tp = sum(1 for d in detections if d["score"] >= conf)
    total_gt = len(coco_gt.getAnnIds())
    precision = tp / len(detections) if detections else 0
    recall = tp / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    results.update({"precision_at_conf": precision, "recall_at_conf": recall, "f1_at_conf": f1,
                     "conf_thresh_used": conf})

    return results


def _build_coco_from_yolo_labels(img_files, class_names):
    """Convert YOLO-format .txt labels to COCO GT dict."""
    images = []
    annotations = []
    img_id_map = {}
    ann_id = 1

    label_dirs = set()
    for f in img_files:
        # Try: same dir, or ../labels/
        txt = os.path.splitext(f)[0] + ".txt"
        if not os.path.exists(txt):
            parent = os.path.dirname(f)
            lbl_dir = os.path.join(os.path.dirname(parent), "labels", os.path.basename(parent))
            txt = os.path.join(lbl_dir, os.path.splitext(os.path.basename(f))[0] + ".txt")
        if os.path.exists(txt):
            label_dirs.add(os.path.dirname(txt))

    if not label_dirs:
        return None, None

    for idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = idx + 1
        fname = os.path.basename(img_path)
        img_id_map[fname] = img_id
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        txt = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt):
            parent = os.path.dirname(img_path)
            lbl_dir = os.path.join(os.path.dirname(parent), "labels", os.path.basename(parent))
            txt = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(txt):
            continue

        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                bw_px = bw * w
                bh_px = bh * h
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls_id,
                    "bbox": [x1, y1, bw_px, bh_px], "area": bw_px * bh_px, "iscrowd": 0
                })
                ann_id += 1

    # Categories
    all_cats = set(a["category_id"] for a in annotations)
    categories = []
    for c in sorted(all_cats):
        name = class_names.get(c, str(c)) if isinstance(class_names, dict) else str(c)
        categories.append({"id": c, "name": name})

    return {"images": images, "annotations": annotations, "categories": categories}, img_id_map


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PyTorch baseline (for accuracy-drop and speedup comparison)
# ═════════════════════════════════════════════════════════════════════════════
def pytorch_latency(pt_path, imgsz, num_frames=100, warmup=10):
    """Measure PyTorch (.pt) inference latency for comparison."""
    YOLO = _import_ultralytics()
    torch = _import_torch()

    model = YOLO(pt_path)
    # Warmup
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(num_frames):
        t0 = time.time()
        model.predict(dummy, imgsz=imgsz, verbose=False)
        latencies.append((time.time() - t0) * 1000)

    stats = compute_stats(latencies)
    fps = 1000.0 / stats["mean"] if stats["mean"] > 0 else 0
    return {"pt_latency_ms": stats, "pt_fps": fps}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main benchmark loop
# ═════════════════════════════════════════════════════════════════════════════
def run_benchmark(engine_wrapper, source, num_frames, warmup, is_rfdetr=False, is_live=False):
    """
    Run the main timing benchmark and return per-stage latency arrays.
    """
    print(f"\n  Warmup: {warmup} frames...")
    engine_wrapper.warmup(warmup)

    # First-frame latency
    frame = source.read()
    if frame is None:
        raise RuntimeError("Could not read first frame")

    t_cold = time.time()
    if is_rfdetr:
        blob, sx, sy = engine_wrapper.preprocess(frame)
        outputs, _ = engine_wrapper.infer(blob)
        _ = engine_wrapper.postprocess(outputs, sx, sy)
    else:
        blob, ratio, dw, dh = engine_wrapper.preprocess(frame)
        outputs, _ = engine_wrapper.infer(blob)
        _ = engine_wrapper.postprocess(outputs, ratio, dw, dh, frame.shape)
    first_frame_ms = (time.time() - t_cold) * 1000
    print(f"  First-frame latency: {first_frame_ms:.1f} ms")

    # Main loop
    timings = defaultdict(list)
    wall_start = time.time()
    frames_done = 0

    print(f"  Running {num_frames} frames...")
    for i in range(num_frames):
        t_total_start = time.time()

        # Capture
        t0 = time.time()
        frame = source.read()
        if frame is None:
            break
        timings["capture_ms"].append((time.time() - t0) * 1000)

        # Preprocess
        t0 = time.time()
        if is_rfdetr:
            blob, sx, sy = engine_wrapper.preprocess(frame)
        else:
            blob, ratio, dw, dh = engine_wrapper.preprocess(frame)
        timings["preprocess_ms"].append((time.time() - t0) * 1000)

        # Inference (with H2D / infer / D2H breakdown)
        outputs, infer_timings = engine_wrapper.infer(blob)
        timings["h2d_ms"].append(infer_timings["h2d_ms"])
        timings["infer_ms"].append(infer_timings["infer_ms"])
        timings["d2h_ms"].append(infer_timings["d2h_ms"])

        # Postprocess
        t0 = time.time()
        if is_rfdetr:
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, sx, sy)
        else:
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, ratio, dw, dh, frame.shape)
        timings["postprocess_ms"].append((time.time() - t0) * 1000)

        # Draw (measure even in headless to count the cost)
        t0 = time.time()
        if is_live:
            _draw_boxes(frame, boxes, scores, class_ids)
        timings["draw_ms"].append((time.time() - t0) * 1000)

        # Total
        timings["total_ms"].append((time.time() - t_total_start) * 1000)
        frames_done += 1

    wall_elapsed = time.time() - wall_start
    wall_fps = frames_done / wall_elapsed if wall_elapsed > 0 else 0

    # Compute all stats
    result = {"frames": frames_done, "wall_time_s": wall_elapsed, "wall_fps": wall_fps,
              "first_frame_ms": first_frame_ms}

    for stage, vals in timings.items():
        result[stage] = compute_stats(vals)

    # Inference-only FPS
    infer_mean = result.get("infer_ms", {}).get("mean", 0)
    result["infer_fps"] = 1000.0 / infer_mean if infer_mean > 0 else 0
    total_mean = result.get("total_ms", {}).get("mean", 0)
    result["e2e_fps"] = 1000.0 / total_mean if total_mean > 0 else 0

    # Jitter
    totals = np.array(timings["total_ms"])
    med = np.median(totals)
    result["jitter_std_ms"] = float(np.std(totals))
    result["jitter_range_ms"] = float(np.max(totals) - np.min(totals))
    result["stutter_count"] = int(np.sum(totals > 2 * med))

    # Dropped frames
    result["camera_frames_read"] = getattr(source, "frames_read", 0)
    result["camera_frames_dropped"] = getattr(source, "frames_dropped", 0)
    result["camera_fps"] = getattr(source, "actual_fps", 0)

    return result


def _draw_boxes(frame, boxes, scores, class_ids, class_names=None):
    """Draw detection boxes on frame."""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        color = _color_for_class(int(class_ids[i]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_ids[i]}:{scores[i]:.2f}"
        if class_names and int(class_ids[i]) in class_names:
            label = f"{class_names[int(class_ids[i])]}:{scores[i]:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _color_for_class(cls_id):
    np.random.seed(cls_id + 42)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Sustained-load test
# ═════════════════════════════════════════════════════════════════════════════
def sustained_load_test(engine_wrapper, source_dir, duration_s=600, imgsz=640,
                         is_rfdetr=False, out_dir="results"):
    """Run for `duration_s` seconds, logging FPS/latency/temp/power over time."""
    ensure_dir(out_dir)
    source = ImageFolderSource(source_dir, loop=True)
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    log = []
    t_start = time.time()
    frame_count = 0
    window_size = 30  # rolling window for FPS

    print(f"\n  Sustained-load test: {duration_s}s ...")
    recent_times = []

    while (time.time() - t_start) < duration_s:
        frame = source.read()
        if frame is None:
            source.idx = 0
            continue

        t0 = time.time()
        if is_rfdetr:
            blob, sx, sy = engine_wrapper.preprocess(frame)
            outputs, _ = engine_wrapper.infer(blob)
            _ = engine_wrapper.postprocess(outputs, sx, sy)
        else:
            blob, ratio, dw, dh = engine_wrapper.preprocess(frame)
            outputs, _ = engine_wrapper.infer(blob)
            _ = engine_wrapper.postprocess(outputs, ratio, dw, dh, frame.shape)
        elapsed_ms = (time.time() - t0) * 1000
        recent_times.append(elapsed_ms)

        frame_count += 1

        # Log every second
        if frame_count % window_size == 0:
            now = time.time() - t_start
            rolling_fps = 1000.0 / np.mean(recent_times[-window_size:])
            latest_resource = monitor.samples[-1] if monitor.samples else {}
            log.append({
                "time_s": now,
                "fps": rolling_fps,
                "latency_ms": np.mean(recent_times[-window_size:]),
                "temp_gpu": latest_resource.get("temp_gpu", 0),
                "power_w": latest_resource.get("power_total_mw", 0) / 1000,
            })

    monitor.stop()

    # Save log
    log_path = os.path.join(out_dir, "sustained_load_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Sustained test done: {frame_count} frames in {duration_s}s")
    print(f"  Log saved: {log_path}")

    return log, monitor.summarize()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Plotting
# ═════════════════════════════════════════════════════════════════════════════
def generate_plots(all_results, sustained_logs, out_dir="results"):
    """Generate matplotlib comparison charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    models = list(all_results.keys())
    if not models:
        return

    # ── Latency bar (stacked by stage) ──
    stages = ["preprocess_ms", "h2d_ms", "infer_ms", "d2h_ms", "postprocess_ms"]
    stage_labels = ["Preprocess", "H2D", "Inference", "D2H", "Postprocess"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for stage, label in zip(stages, stage_labels):
        vals = [all_results[m].get("latency", {}).get(stage, {}).get("mean", 0) for m in models]
        ax.bar(x, vals, bottom=bottoms, label=label, width=0.5)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Breakdown (mean)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_stacked.png"), dpi=150)
    plt.close()

    # ── FPS comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    infer_fps = [all_results[m].get("latency", {}).get("infer_fps", 0) for m in models]
    e2e_fps = [all_results[m].get("latency", {}).get("e2e_fps", 0) for m in models]
    w = 0.35
    ax.bar(x - w/2, infer_fps, w, label="Inference-only FPS")
    ax.bar(x + w/2, e2e_fps, w, label="End-to-end FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("FPS")
    ax.set_title("Throughput Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps_comparison.png"), dpi=150)
    plt.close()

    # ── FPS per watt ──
    fig, ax = plt.subplots(figsize=(8, 5))
    fps_per_w = []
    for m in models:
        fps = all_results[m].get("latency", {}).get("e2e_fps", 0)
        power = all_results[m].get("resources", {}).get("power_avg_w", 1)
        fps_per_w.append(fps / max(power, 0.1))
    ax.bar(models, fps_per_w, color="teal")
    ax.set_ylabel("FPS / Watt")
    ax.set_title("Energy Efficiency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps_per_watt.png"), dpi=150)
    plt.close()

    # ── mAP vs Latency scatter ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        lat = all_results[m].get("latency", {}).get("infer_ms", {}).get("mean", 0)
        mAP = all_results[m].get("accuracy", {}).get("mAP50", 0)
        ax.scatter(lat, mAP, s=100, label=m)
        ax.annotate(m, (lat, mAP), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Accuracy vs Speed")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_vs_latency.png"), dpi=150)
    plt.close()

    # ── Latency box plot ──
    # (we'd need raw arrays; store them in all_results if desired)

    # ── Sustained-load time series ──
    for model_name, log in sustained_logs.items():
        if not log:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ts = [e["time_s"] for e in log]
        axes[0].plot(ts, [e["fps"] for e in log], color="blue")
        axes[0].set_ylabel("FPS")
        axes[0].set_title(f"Sustained Load — {model_name}")
        axes[1].plot(ts, [e["latency_ms"] for e in log], color="orange")
        axes[1].set_ylabel("Latency (ms)")
        axes[2].plot(ts, [e["temp_gpu"] for e in log], color="red", label="GPU Temp")
        ax2 = axes[2].twinx()
        ax2.plot(ts, [e["power_w"] for e in log], color="green", label="Power (W)")
        ax2.set_ylabel("Power (W)")
        axes[2].set_ylabel("Temp (°C)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sustained_{model_name}.png"), dpi=150)
        plt.close()

    print(f"  Plots saved to {out_dir}/")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: Output (tables, CSV, JSON, summary.md)
# ═════════════════════════════════════════════════════════════════════════════
def print_comparison_table(all_results):
    """Print a clean terminal comparison table."""
    from tabulate import tabulate

    models = list(all_results.keys())
    rows = []

    metrics = [
        ("Infer Latency (ms)", lambda r: r.get("latency",{}).get("infer_ms",{}).get("mean",0), ".2f"),
        ("E2E Latency (ms)", lambda r: r.get("latency",{}).get("total_ms",{}).get("mean",0), ".2f"),
        ("Infer FPS", lambda r: r.get("latency",{}).get("infer_fps",0), ".1f"),
        ("E2E FPS", lambda r: r.get("latency",{}).get("e2e_fps",0), ".1f"),
        ("Wall FPS", lambda r: r.get("latency",{}).get("wall_fps",0), ".1f"),
        ("mAP@0.5", lambda r: r.get("accuracy",{}).get("mAP50",0), ".3f"),
        ("mAP@0.5:0.95", lambda r: r.get("accuracy",{}).get("mAP50_95",0), ".3f"),
        ("First Frame (ms)", lambda r: r.get("latency",{}).get("first_frame_ms",0), ".1f"),
        ("Jitter Std (ms)", lambda r: r.get("latency",{}).get("jitter_std_ms",0), ".2f"),
        ("Stutter Count", lambda r: r.get("latency",{}).get("stutter_count",0), "d"),
        ("GPU Util %", lambda r: r.get("resources",{}).get("gpu_util_avg",0), ".1f"),
        ("Power Avg (W)", lambda r: r.get("resources",{}).get("power_avg_w",0), ".2f"),
        ("FPS/W", lambda r: _safe_div(r.get("latency",{}).get("e2e_fps",0),
                                        r.get("resources",{}).get("power_avg_w",1)), ".2f"),
        ("Temp GPU Max (°C)", lambda r: r.get("resources",{}).get("temp_gpu_max",0), ".1f"),
        ("Engine Size (MB)", lambda r: r.get("model_info",{}).get("engine_mb",0), ".1f"),
        ("Engine Build (s)", lambda r: r.get("model_info",{}).get("build_time",0), ".0f"),
    ]

    for label, fn, fmt in metrics:
        row = [label]
        for m in models:
            val = fn(all_results[m])
            row.append(f"{val:{fmt}}")
        rows.append(row)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPARISON")
    print("=" * 80)
    print(tabulate(rows, headers=["Metric"] + models, tablefmt="grid"))
    print()


def _safe_div(a, b):
    return a / b if b > 0 else 0


def save_results(all_results, out_dir="results"):
    """Save results as CSV, JSON, and summary.md."""
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV (flat)
    csv_path = os.path.join(out_dir, "results.csv")
    models = list(all_results.keys())
    flat_rows = []
    for m in models:
        r = all_results[m]
        row = {"model": m, "timestamp": ts}
        # Flatten latency
        for stage in ["capture_ms","preprocess_ms","h2d_ms","infer_ms","d2h_ms","postprocess_ms","total_ms"]:
            for stat_name, stat_val in r.get("latency",{}).get(stage,{}).items():
                row[f"{stage}_{stat_name}"] = stat_val
        # Scalar latency fields
        for k in ["infer_fps","e2e_fps","wall_fps","first_frame_ms","jitter_std_ms","stutter_count"]:
            row[k] = r.get("latency",{}).get(k, "")
        # Accuracy
        for k, v in r.get("accuracy",{}).items():
            if not isinstance(v, dict):
                row[f"acc_{k}"] = v
        # Resources
        for k, v in r.get("resources",{}).items():
            row[f"res_{k}"] = v
        # Model info
        for k, v in r.get("model_info",{}).items():
            row[f"model_{k}"] = v
        flat_rows.append(row)

    if flat_rows:
        keys = list(flat_rows[0].keys())
        for r in flat_rows[1:]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat_rows)

    # summary.md
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"Generated: {ts}\n\n")

        # Winners
        f.write("## Winners by Metric\n\n")
        winner_metrics = [
            ("Lowest Inference Latency",
             lambda r: r.get("latency",{}).get("infer_ms",{}).get("mean",9999), False),
            ("Highest FPS",
             lambda r: r.get("latency",{}).get("e2e_fps",0), True),
            ("Best mAP@0.5",
             lambda r: r.get("accuracy",{}).get("mAP50",0), True),
            ("Best FPS/Watt",
             lambda r: _safe_div(r.get("latency",{}).get("e2e_fps",0),
                                  r.get("resources",{}).get("power_avg_w",1)), True),
            ("Smallest Engine",
             lambda r: r.get("model_info",{}).get("engine_mb",9999), False),
        ]
        for label, fn, higher_better in winner_metrics:
            vals = {m: fn(all_results[m]) for m in models}
            if higher_better:
                winner = max(vals, key=vals.get)
            else:
                winner = min(vals, key=vals.get)
            f.write(f"| {label} | **{winner}** ({vals[winner]:.3f}) |\n")

        f.write("\n## Detailed Results\n\n")
        f.write("See `results.json` and `results.csv` for full data.\n")
        f.write("See `latency_stacked.png`, `fps_comparison.png`, etc. for plots.\n")

    print(f"\n  Results saved: {json_path}, {csv_path}, {md_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: Live mode
# ═════════════════════════════════════════════════════════════════════════════
def run_live(engine_wrapper, camera_index, imgsz, is_rfdetr, class_names=None):
    """Live webcam mode with overlay."""
    cap = CameraCapture(camera_index, 1920, 1080)
    monitor = ResourceMonitor()
    monitor.start()

    print("\n  Live mode — press 'q' to quit")
    fps_smooth = 0
    while True:
        frame = cap.read()
        if frame is None:
            break

        t0 = time.time()
        if is_rfdetr:
            blob, sx, sy = engine_wrapper.preprocess(frame)
            outputs, timings = engine_wrapper.infer(blob)
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, sx, sy)
        else:
            blob, ratio, dw, dh = engine_wrapper.preprocess(frame)
            outputs, timings = engine_wrapper.infer(blob)
            boxes, scores, class_ids = engine_wrapper.postprocess(outputs, ratio, dw, dh, frame.shape)
        total_ms = (time.time() - t0) * 1000
        fps = 1000.0 / total_ms if total_ms > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps

        _draw_boxes(frame, boxes, scores, class_ids, class_names)

        # Overlay
        latest = monitor.samples[-1] if monitor.samples else {}
        overlay_lines = [
            f"FPS: {fps_smooth:.1f}",
            f"Infer: {timings['infer_ms']:.1f}ms  Total: {total_ms:.1f}ms",
            f"GPU: {latest.get('gpu_util',0):.0f}%  Power: {latest.get('power_total_mw',0)/1000:.1f}W",
            f"Temp: {latest.get('temp_gpu',0):.0f}C  Dets: {len(boxes)}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Benchmark Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    monitor.stop()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11: Power-mode comparison
# ═════════════════════════════════════════════════════════════════════════════
def get_current_power_mode():
    """Read current nvpmodel mode."""
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "NV Power Mode" in line:
                return line.strip()
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown"


def set_power_mode(mode_id, jetson_clocks=False):
    """Set nvpmodel mode and optionally enable jetson_clocks."""
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", str(mode_id)], check=True,
                        capture_output=True, text=True)
        if jetson_clocks:
            subprocess.run(["sudo", "jetson_clocks"], check=True,
                            capture_output=True, text=True)
        time.sleep(5)  # Allow mode switch to settle
        return get_current_power_mode()
    except Exception as e:
        print(f"  [WARN] Could not set power mode {mode_id}: {e}")
        return "failed"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12: Model info
# ═════════════════════════════════════════════════════════════════════════════
def get_model_info(pt_path=None, onnx_path=None, engine_path=None):
    """Gather parameter count, GFLOPs, file sizes."""
    info = {}
    if pt_path and os.path.exists(pt_path):
        info["pt_mb"] = file_size_mb(pt_path)
        try:
            YOLO = _import_ultralytics()
            m = YOLO(pt_path)
            info["parameters"] = sum(p.numel() for p in m.model.parameters())
            # GFLOPs via ultralytics
            info["gflops"] = getattr(m.model, "flops", 0) / 1e9 if hasattr(m.model, "flops") else "N/A"
        except Exception:
            info["parameters"] = "N/A"
            info["gflops"] = "N/A"
    if onnx_path and os.path.exists(onnx_path):
        info["onnx_mb"] = file_size_mb(onnx_path)
    if engine_path and os.path.exists(engine_path):
        info["engine_mb"] = file_size_mb(engine_path)
    return info


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Jetson Orin Nano Object Detection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode
    parser.add_argument("--mode", choices=["live", "headless"], default="headless")

    # Engine paths
    parser.add_argument("--yolov8-engine", type=str, help="YOLOv8 TensorRT engine")
    parser.add_argument("--yolo11-engine", type=str, help="YOLO11 TensorRT engine")
    parser.add_argument("--rfdetr-engine", type=str, help="RF-DETR TensorRT engine")

    # PT paths (for accuracy drop comparison)
    parser.add_argument("--yolov8-pt", type=str, help="YOLOv8 .pt for PT baseline")
    parser.add_argument("--yolo11-pt", type=str, help="YOLO11 .pt for PT baseline")

    # Single model mode
    parser.add_argument("--model", choices=["yolov8", "yolo11", "rfdetr"], help="Single model for live mode")
    parser.add_argument("--engine", type=str, help="Engine path for single model mode")

    # Source
    parser.add_argument("--source", type=str, default="0", help="Camera index, video path, or image dir")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live mode")

    # Parameters
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rfdetr-imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=500)

    # Accuracy
    parser.add_argument("--val-dir", type=str, help="Validation images dir")
    parser.add_argument("--val-ann", type=str, help="COCO annotation JSON for val set")

    # Sustained load
    parser.add_argument("--sustained", action="store_true", help="Run sustained-load test")
    parser.add_argument("--sustained-duration", type=int, default=600, help="Seconds (default 600)")

    # Power modes
    parser.add_argument("--power-modes", type=int, nargs="+", help="nvpmodel modes to test (e.g. 0 1 2)")
    parser.add_argument("--jetson-clocks", action="store_true", help="Enable jetson_clocks in each mode")

    # Output
    parser.add_argument("--out-dir", type=str, default="results")

    # RF-DETR classes
    parser.add_argument("--rfdetr-num-classes", type=int, default=None)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    # ── Live mode ──
    if args.mode == "live":
        if not args.engine:
            parser.error("--engine required for live mode")
        model_type = args.model or "yolov8"
        is_rf = model_type == "rfdetr"
        if is_rf:
            from trt_rfdetr import TRTRFDETREngine
            eng = TRTRFDETREngine(args.engine, conf_thresh=args.conf,
                                   num_classes=args.rfdetr_num_classes)
        else:
            eng = TRTYOLOEngine(args.engine, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
        eng.warmup(args.warmup)
        run_live(eng, args.camera, args.imgsz, is_rf)
        return

    # ── Headless mode ──
    print("=" * 70)
    print(" Jetson Orin Nano — Headless Benchmark")
    print("=" * 70)
    print(f"  Power mode: {get_current_power_mode()}")

    # Collect engines to benchmark
    engines_to_run = {}
    if args.yolov8_engine:
        engines_to_run["yolov8"] = {"engine": args.yolov8_engine, "type": "yolo",
                                     "pt": args.yolov8_pt, "imgsz": args.imgsz}
    if args.yolo11_engine:
        engines_to_run["yolo11"] = {"engine": args.yolo11_engine, "type": "yolo",
                                     "pt": args.yolo11_pt, "imgsz": args.imgsz}
    if args.rfdetr_engine:
        engines_to_run["rfdetr"] = {"engine": args.rfdetr_engine, "type": "rfdetr",
                                     "pt": None, "imgsz": args.rfdetr_imgsz}

    if not engines_to_run:
        print("[ERROR] No engines specified. Use --yolov8-engine, --yolo11-engine, --rfdetr-engine")
        sys.exit(1)

    # Power mode sweep
    power_modes_to_test = args.power_modes if args.power_modes else [None]

    all_results_all_modes = {}

    for pm in power_modes_to_test:
        if pm is not None:
            mode_name = set_power_mode(pm, args.jetson_clocks)
            print(f"\n  ════ Power mode: {pm} ({mode_name}) ════")
        else:
            mode_name = get_current_power_mode()

        all_results = {}
        sustained_logs = {}

        for name, cfg in engines_to_run.items():
            print(f"\n{'─'*60}")
            print(f"  Model: {name}  Engine: {cfg['engine']}")
            print(f"{'─'*60}")

            is_rf = cfg["type"] == "rfdetr"
            if is_rf:
                from trt_rfdetr import TRTRFDETREngine
                eng = TRTRFDETREngine(cfg["engine"], conf_thresh=args.conf,
                                       num_classes=args.rfdetr_num_classes)
            else:
                eng = TRTYOLOEngine(cfg["engine"], imgsz=cfg["imgsz"],
                                     conf=args.conf, iou=args.iou)

            # Source
            src_path = args.source
            if os.path.isdir(src_path):
                source = ImageFolderSource(src_path, loop=True)
            elif os.path.isfile(src_path):
                source = CameraCapture(src_path)
            else:
                source = CameraCapture(int(src_path), 1920, 1080)

            # Resource monitor
            monitor = ResourceMonitor()
            monitor.start()

            # Latency benchmark
            lat_result = run_benchmark(eng, source, args.num_frames, args.warmup, is_rfdetr=is_rf)
            monitor.stop()
            source.release()

            # Model info
            model_info = get_model_info(pt_path=cfg.get("pt"), engine_path=cfg["engine"])
            model_info["load_time_ms"] = getattr(eng, "load_time", 0) * 1000

            # Resources
            res_summary = monitor.summarize()

            # Energy per frame
            power_w = res_summary.get("power_avg_w", 0)
            e2e_fps = lat_result.get("e2e_fps", 0)
            if e2e_fps > 0 and power_w > 0:
                res_summary["energy_per_frame_mj"] = (power_w / e2e_fps) * 1000
                res_summary["fps_per_watt"] = e2e_fps / power_w

            # Accuracy
            acc_result = {}
            if args.val_dir:
                acc_result = evaluate_accuracy(eng, args.val_dir, args.val_ann,
                                                conf=args.conf, iou=args.iou,
                                                imgsz=cfg["imgsz"], is_rfdetr=is_rf,
                                                out_dir=args.out_dir)

            # PT baseline
            pt_info = {}
            if cfg.get("pt"):
                print(f"  Measuring PyTorch baseline for {name}...")
                pt_info = pytorch_latency(cfg["pt"], cfg["imgsz"], num_frames=100)
                pt_mean = pt_info.get("pt_latency_ms", {}).get("mean", 0)
                trt_mean = lat_result.get("infer_ms", {}).get("mean", 0)
                if trt_mean > 0 and pt_mean > 0:
                    pt_info["speedup_factor"] = pt_mean / trt_mean

            all_results[name] = {
                "latency": lat_result,
                "accuracy": acc_result,
                "resources": res_summary,
                "model_info": model_info,
                "pytorch_baseline": pt_info,
                "power_mode": mode_name,
                "imgsz": cfg["imgsz"],
                "conf": args.conf,
                "iou": args.iou,
            }

            # Sustained load
            if args.sustained and args.val_dir:
                slog, sres = sustained_load_test(eng, args.val_dir,
                                                   duration_s=args.sustained_duration,
                                                   imgsz=cfg["imgsz"], is_rfdetr=is_rf,
                                                   out_dir=args.out_dir)
                sustained_logs[name] = slog
                all_results[name]["sustained_resources"] = sres

        # Output
        print_comparison_table(all_results)
        save_results(all_results, args.out_dir)
        generate_plots(all_results, sustained_logs, args.out_dir)

        suffix = f"_mode{pm}" if pm is not None else ""
        all_results_all_modes[f"mode{pm}" if pm is not None else "default"] = all_results

    # Save combined
    if len(power_modes_to_test) > 1:
        combined_path = os.path.join(args.out_dir, "results_all_modes.json")
        with open(combined_path, "w") as f:
            json.dump(all_results_all_modes, f, indent=2, default=str)
        print(f"\n  All-modes results: {combined_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
