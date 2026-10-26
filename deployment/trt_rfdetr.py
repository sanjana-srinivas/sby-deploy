#!/usr/bin/env python3
"""
trt_rfdetr.py — Custom TensorRT inference wrapper for RF-DETR.

RF-DETR outputs:
  - dets:   [1, 300, 4]  → (cx, cy, w, h) normalized [0,1]
  - labels: [1, 300, C]  → class logits (NOT softmaxed)

This wrapper handles:
  - Engine loading with deserialization
  - Pinned host memory + CUDA streams for async H2D / inference / D2H
  - Preprocessing: resize, mean/std normalization (ImageNet defaults)
  - Postprocessing: sigmoid on logits, confidence filter, cx/cy/w/h → x1/y1/x2/y2
  - No NMS needed (DETR uses set prediction with Hungarian matching)
"""

import os
import time
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa — initializes CUDA context


class TRTRFDETREngine:
    """TensorRT inference engine for RF-DETR."""

    # ImageNet normalization (RF-DETR default)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, engine_path, conf_thresh=0.5, num_classes=None):
        """
        Args:
            engine_path: Path to .engine file
            conf_thresh: Confidence threshold for detections
            num_classes: Override class count (auto-detected from engine output shape)
        """
        self.conf_thresh = conf_thresh
        self.logger = trt.Logger(trt.Logger.WARNING)

        # ── Load engine ──
        t0 = time.time()
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.load_time = time.time() - t0

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # ── Discover bindings ──
        self.input_name = None
        self.output_names = []
        self.bindings = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            # Allocate pinned host memory and device memory
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(size)

            self.bindings[name] = {
                "shape": tuple(shape),
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
                "size": size,
            }

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # Extract dimensions
        in_shape = self.bindings[self.input_name]["shape"]  # (1, 3, H, W)
        self.batch_size = in_shape[0]
        self.input_h = in_shape[2]
        self.input_w = in_shape[3]

        # Detect num_classes from the labels output
        for oname in self.output_names:
            oshape = self.bindings[oname]["shape"]
            if len(oshape) == 3 and oshape[2] != 4:  # not the box output
                self.num_classes = num_classes or oshape[2]
                self.num_queries = oshape[1]
                break

        # Create CUDA stream
        self.stream = cuda.Stream()

        print(f"[RF-DETR TRT] Loaded {engine_path}")
        print(f"  Input:  {self.input_name} {in_shape}")
        print(f"  Queries: {self.num_queries}, Classes: {self.num_classes}")
        print(f"  Load time: {self.load_time*1000:.1f} ms")

    def preprocess(self, image_bgr):
        """
        Resize + normalize a BGR image for RF-DETR.

        Args:
            image_bgr: numpy HWC BGR uint8 image

        Returns:
            blob: float32 NCHW array, normalized with ImageNet mean/std
            scale_x, scale_y: scale factors to map detections back to original coords
        """
        orig_h, orig_w = image_bgr.shape[:2]
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        # Resize (no letterboxing for RF-DETR — it expects exact input size)
        resized = _resize_cv2(image_bgr, self.input_w, self.input_h)

        # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
        img = resized[:, :, ::-1].astype(np.float32) / 255.0

        # ImageNet normalization
        img = (img - self.MEAN) / self.STD

        # HWC → NCHW
        blob = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])
        return blob, scale_x, scale_y

    def infer(self, blob):
        """
        Run TensorRT inference with async memory transfers.

        Args:
            blob: preprocessed NCHW float32 numpy array

        Returns:
            dict of output_name → numpy array
            timing dict with h2d_ms, infer_ms, d2h_ms
        """
        timings = {}

        # ── Host → Device (async) ──
        inp = self.bindings[self.input_name]
        np.copyto(inp["host"], blob.ravel())

        t0 = time.time()
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.stream.synchronize()
        timings["h2d_ms"] = (time.time() - t0) * 1000

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(inp["device"]))
        for oname in self.output_names:
            self.context.set_tensor_address(oname, int(self.bindings[oname]["device"]))

        # ── Inference (CUDA events for accurate timing) ──
        start_event = cuda.Event()
        end_event = cuda.Event()

        start_event.record(self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        end_event.record(self.stream)
        end_event.synchronize()
        timings["infer_ms"] = start_event.time_till(end_event)

        # ── Device → Host (async) ──
        t0 = time.time()
        outputs = {}
        for oname in self.output_names:
            out = self.bindings[oname]
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        timings["d2h_ms"] = (time.time() - t0) * 1000

        for oname in self.output_names:
            out = self.bindings[oname]
            outputs[oname] = out["host"].reshape(out["shape"]).copy()

        return outputs, timings

    def postprocess(self, outputs, scale_x, scale_y, conf_thresh=None):
        """
        Decode RF-DETR outputs to bounding boxes.

        Args:
            outputs: dict from infer() — expects 'dets' [1,300,4] and 'labels' [1,300,C]
            scale_x, scale_y: from preprocess()
            conf_thresh: override instance threshold

        Returns:
            boxes: Nx4 float32 (x1, y1, x2, y2) in original image coords
            scores: N float32
            class_ids: N int
        """
        thresh = conf_thresh if conf_thresh is not None else self.conf_thresh

        # Find the two outputs — boxes and logits
        dets_key = None
        labels_key = None
        for k, v in outputs.items():
            if v.shape[-1] == 4:
                dets_key = k
            else:
                labels_key = k

        raw_boxes = outputs[dets_key][0]    # (300, 4) — cx, cy, w, h normalized
        raw_logits = outputs[labels_key][0]  # (300, C)

        # Sigmoid on logits to get confidence scores
        scores_all = _sigmoid(raw_logits)  # (300, C)

        # Max score per query
        max_scores = scores_all.max(axis=1)  # (300,)
        max_classes = scores_all.argmax(axis=1)  # (300,)

        # Confidence filter
        mask = max_scores >= thresh
        if mask.sum() == 0:
            return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32)

        boxes_cxcywh = raw_boxes[mask]
        scores = max_scores[mask]
        class_ids = max_classes[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2 and scale to original image
        boxes_xyxy = np.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * self.input_w * scale_x
        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * self.input_h * scale_y
        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * self.input_w * scale_x
        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * self.input_h * scale_y

        return boxes_xyxy.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)

    def warmup(self, n=10):
        """Run N dummy inferences to warm up the engine."""
        dummy = np.random.randn(1, 3, self.input_h, self.input_w).astype(np.float32)
        for _ in range(n):
            self.infer(dummy)

    def __del__(self):
        """Clean up CUDA allocations."""
        for b in self.bindings.values():
            try:
                b["device"].free()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _resize_cv2(img, w, h):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="RF-DETR .engine file")
    ap.add_argument("--image", required=True, help="Test image")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--classes", type=int, default=None)
    args = ap.parse_args()

    engine = TRTRFDETREngine(args.engine, conf_thresh=args.conf, num_classes=args.classes)
    engine.warmup(5)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    blob, sx, sy = engine.preprocess(img)
    outputs, timings = engine.infer(blob)
    boxes, scores, class_ids = engine.postprocess(outputs, sx, sy)

    print(f"\nTimings: H2D={timings['h2d_ms']:.2f}ms  Infer={timings['infer_ms']:.2f}ms  D2H={timings['d2h_ms']:.2f}ms")
    print(f"Detections: {len(boxes)}")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        print(f"  class={class_ids[i]}  conf={scores[i]:.3f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    # Draw
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_ids[i]}:{scores[i]:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("rfdetr_test_output.jpg", img)
    print("Saved rfdetr_test_output.jpg")
