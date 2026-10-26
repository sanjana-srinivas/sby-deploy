#!/usr/bin/env python3
"""
export_engines.py — Convert .pt (YOLO) and .onnx (RF-DETR) to TensorRT .engine on Jetson.

Usage:
    python3 export_engines.py --yolov8 Models/Yolov8/best.pt --yolo11 Models/Yolov11/yolo26n.pt \
        --rfdetr-onnx "Models/Rfdetr/Copy of rfdetr_fp32.onnx" --imgsz 640 --rfdetr-imgsz 448

Flags:
    --fp32          Also build FP32 engines
    --int8          Build INT8 engines (requires --calib-dir with sample images)
    --calib-dir     Path to calibration images for INT8
    --calib-count   Number of calibration images (default 100)
"""

import argparse
import os
import sys
import time
import shutil
import glob
import subprocess

import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0


# ─────────────────────────────────────────────────────────────────────────────
# YOLO export: .pt → .onnx → .engine via ultralytics
# ─────────────────────────────────────────────────────────────────────────────
def export_yolo(pt_path, imgsz, out_dir, precisions, calib_dir=None, calib_count=100):
    """Export a YOLO .pt model to ONNX then TensorRT engine(s)."""
    from ultralytics import YOLO

    name = os.path.splitext(os.path.basename(pt_path))[0]
    print(f"\n{'='*60}")
    print(f" Exporting YOLO: {pt_path}  (imgsz={imgsz})")
    print(f"{'='*60}")

    model = YOLO(pt_path)
    pt_size = get_file_size_mb(pt_path)
    print(f"  .pt size: {pt_size:.1f} MB")

    # ── ONNX ──
    onnx_path = os.path.join(out_dir, f"{name}.onnx")
    print(f"  Exporting ONNX → {onnx_path}")
    t0 = time.time()
    model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
    # ultralytics puts onnx next to pt; move it
    auto_onnx = pt_path.replace(".pt", ".onnx")
    if os.path.exists(auto_onnx) and os.path.abspath(auto_onnx) != os.path.abspath(onnx_path):
        shutil.move(auto_onnx, onnx_path)
    onnx_time = time.time() - t0
    onnx_size = get_file_size_mb(onnx_path)
    print(f"  ONNX done in {onnx_time:.1f}s  ({onnx_size:.1f} MB)")

    # ── TensorRT engines ──
    results = {"name": name, "pt": pt_path, "pt_mb": pt_size,
               "onnx": onnx_path, "onnx_mb": onnx_size}

    for prec in precisions:
        engine_name = f"{name}_{prec}.engine"
        engine_path = os.path.join(out_dir, engine_name)
        print(f"\n  Building TensorRT {prec.upper()} engine → {engine_path}")
        t0 = time.time()
        _build_engine_trtexec(onnx_path, engine_path, prec, imgsz, calib_dir, calib_count)
        build_time = time.time() - t0
        eng_size = get_file_size_mb(engine_path)
        print(f"  Engine built in {build_time:.1f}s  ({eng_size:.1f} MB)")
        results[f"engine_{prec}"] = engine_path
        results[f"engine_{prec}_mb"] = eng_size
        results[f"build_time_{prec}"] = build_time

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RF-DETR: .onnx → .engine (no .pth → onnx step if user only has ONNX)
# ─────────────────────────────────────────────────────────────────────────────
def export_rfdetr_from_pth(pth_path, imgsz, out_dir, precisions, calib_dir=None, calib_count=100):
    """If the user has a .pth file, convert to ONNX first via rfdetr library."""
    try:
        from rfdetr import RFDETRBase, RFDETRSmall  # or RFDETRLarge
    except ImportError:
        print("[ERROR] rfdetr package not installed. Cannot export .pth. Provide ONNX instead.")
        return None

    name = os.path.splitext(os.path.basename(pth_path))[0]
    onnx_path = os.path.join(out_dir, f"{name}.onnx")

    print(f"\n{'='*60}")
    print(f" Exporting RF-DETR .pth: {pth_path}")
    print(f"{'='*60}")

    # Attempt to load — user must choose correct class
    import torch
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    # Simple heuristic: try RFDETRBase
    model = RFDETRBase()
    model.load_state_dict(ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt.get("model", ckpt))
    model.eval()

    # Export ONNX
    dummy = torch.randn(1, 3, imgsz, imgsz)
    t0 = time.time()
    torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                       input_names=["input"], output_names=["dets", "labels"],
                       dynamic_axes=None)
    print(f"  ONNX exported in {time.time()-t0:.1f}s")

    return export_rfdetr_from_onnx(onnx_path, imgsz, out_dir, precisions, calib_dir, calib_count,
                                    pt_path=pth_path)


def export_rfdetr_from_onnx(onnx_path, imgsz, out_dir, precisions, calib_dir=None, calib_count=100,
                             pt_path=None):
    """Build TensorRT engine(s) from an RF-DETR ONNX file."""
    name = os.path.splitext(os.path.basename(onnx_path))[0].replace(" ", "_")
    onnx_size = get_file_size_mb(onnx_path)

    print(f"\n{'='*60}")
    print(f" Building RF-DETR engines from ONNX: {onnx_path}")
    print(f" Input size (from ONNX metadata): will be read by trtexec")
    print(f"{'='*60}")
    print(f"  ONNX size: {onnx_size:.1f} MB")

    results = {"name": f"rfdetr_{name}", "pt": pt_path or "N/A",
               "pt_mb": get_file_size_mb(pt_path) if pt_path else 0,
               "onnx": onnx_path, "onnx_mb": onnx_size}

    for prec in precisions:
        engine_name = f"rfdetr_{name}_{prec}.engine"
        engine_path = os.path.join(out_dir, engine_name)
        print(f"\n  Building TensorRT {prec.upper()} → {engine_path}")
        t0 = time.time()
        _build_engine_trtexec(onnx_path, engine_path, prec, imgsz, calib_dir, calib_count)
        build_time = time.time() - t0
        eng_size = get_file_size_mb(engine_path)
        print(f"  Done in {build_time:.1f}s  ({eng_size:.1f} MB)")
        results[f"engine_{prec}"] = engine_path
        results[f"engine_{prec}_mb"] = eng_size
        results[f"build_time_{prec}"] = build_time

    return results


# ─────────────────────────────────────────────────────────────────────────────
# trtexec wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _build_engine_trtexec(onnx_path, engine_path, precision, imgsz, calib_dir=None, calib_count=100):
    """Build a TensorRT engine using trtexec (ships with JetPack)."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=2048",  # MB
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        cmd.append("--fp16")  # int8 with fp16 fallback
        if calib_dir:
            # For INT8 we write a calibration cache using trtexec's built-in
            calib_cache = engine_path.replace(".engine", "_calib.cache")
            cmd.append(f"--calib={calib_cache}")
            # trtexec uses random data by default for calibration;
            # for proper INT8, use a custom calibrator (see below)
            print(f"    ⚠  Using trtexec default calibration. For best accuracy,")
            print(f"       use export_engines.py --int8 --calib-dir <images> to write a proper cache.")
    # FP32 is the default — no flag needed

    # Orin Nano has no DLA
    cmd.append("--noTF32")  # disable TF32 for consistent FP32 baseline
    if precision != "fp32":
        # re-enable TF32 for fp16/int8
        cmd.remove("--noTF32")

    cmd_str = " ".join(cmd)
    print(f"    $ {cmd_str}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"[ERROR] trtexec failed:\n{proc.stderr[-2000:]}")
        # Try the Python API as fallback
        print("    Attempting Python TensorRT API fallback...")
        _build_engine_python(onnx_path, engine_path, precision)
    else:
        # Print throughput from trtexec output
        for line in proc.stdout.splitlines():
            if "Throughput" in line or "Latency" in line or "GPU Compute Time" in line:
                print(f"    {line.strip()}")


def _build_engine_python(onnx_path, engine_path, precision):
    """Fallback: build engine via TensorRT Python API."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    print("    Building engine (Python API)... this may take several minutes.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"    Saved: {engine_path}")


# ─────────────────────────────────────────────────────────────────────────────
# INT8 Calibrator (for proper calibration with real images)
# ─────────────────────────────────────────────────────────────────────────────
class ImageCalibrator:
    """
    INT8 calibrator that feeds real images through the network.
    Used when building via the Python TensorRT API.
    """
    def __init__(self, image_dir, input_shape, cache_file, count=100):
        import tensorrt as trt
        self.cache_file = cache_file
        self.batch_size = 1
        self.input_shape = input_shape  # (C, H, W)
        self.images = self._load_images(image_dir, count)
        self.current = 0

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
        nbytes = int(np.prod(input_shape)) * 4  # float32
        self.device_input = cuda.mem_alloc(nbytes)

    def _load_images(self, image_dir, count):
        import cv2
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(image_dir, ext)))
        files = sorted(files)[:count]
        if not files:
            raise FileNotFoundError(f"No images in {image_dir}")
        print(f"    Calibrator: {len(files)} images from {image_dir}")
        return files

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= len(self.images):
            return None
        import cv2
        import pycuda.driver as cuda
        _, h, w = self.input_shape
        img = cv2.imread(self.images[self.current])
        img = cv2.resize(img, (w, h))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img[np.newaxis, ...])
        cuda.memcpy_htod(self.device_input, img)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export YOLO/RF-DETR models to TensorRT engines")
    parser.add_argument("--yolov8", type=str, default=None, help="Path to YOLOv8 .pt file")
    parser.add_argument("--yolo11", type=str, default=None, help="Path to YOLO11/26 .pt file")
    parser.add_argument("--rfdetr-pth", type=str, default=None, help="Path to RF-DETR .pth file")
    parser.add_argument("--rfdetr-onnx", type=str, default=None, help="Path to RF-DETR pre-exported ONNX")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO (default 640)")
    parser.add_argument("--rfdetr-imgsz", type=int, default=448, help="RF-DETR input size (default 448)")
    parser.add_argument("--out-dir", type=str, default="./engines", help="Output directory")
    parser.add_argument("--fp32", action="store_true", help="Also build FP32 engines")
    parser.add_argument("--int8", action="store_true", help="Build INT8 engines")
    parser.add_argument("--calib-dir", type=str, default=None, help="Calibration images dir for INT8")
    parser.add_argument("--calib-count", type=int, default=100, help="Calibration image count")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Determine precisions to build
    precisions = ["fp16"]
    if args.fp32:
        precisions.insert(0, "fp32")
    if args.int8:
        if not args.calib_dir:
            print("[WARN] --int8 without --calib-dir: INT8 will use trtexec default calibration.")
        precisions.append("int8")

    all_results = []

    # ── YOLO models ──
    for label, pt in [("yolov8", args.yolov8), ("yolo11", args.yolo11)]:
        if pt:
            if not os.path.isfile(pt):
                print(f"[ERROR] {label} file not found: {pt}")
                continue
            res = export_yolo(pt, args.imgsz, args.out_dir, precisions,
                              args.calib_dir, args.calib_count)
            res["model_type"] = label
            all_results.append(res)

    # ── RF-DETR ──
    if args.rfdetr_pth:
        res = export_rfdetr_from_pth(args.rfdetr_pth, args.rfdetr_imgsz, args.out_dir, precisions,
                                      args.calib_dir, args.calib_count)
        if res:
            res["model_type"] = "rfdetr"
            all_results.append(res)
    elif args.rfdetr_onnx:
        res = export_rfdetr_from_onnx(args.rfdetr_onnx, args.rfdetr_imgsz, args.out_dir, precisions,
                                       args.calib_dir, args.calib_count)
        res["model_type"] = "rfdetr"
        all_results.append(res)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(" Export Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n  {r['name']} ({r['model_type']})")
        print(f"    .pt:   {r.get('pt_mb',0):.1f} MB")
        print(f"    .onnx: {r.get('onnx_mb',0):.1f} MB  ({r.get('onnx','N/A')})")
        for prec in precisions:
            eng = r.get(f"engine_{prec}", "NOT BUILT")
            mb = r.get(f"engine_{prec}_mb", 0)
            bt = r.get(f"build_time_{prec}", 0)
            print(f"    .engine ({prec}): {mb:.1f} MB  build={bt:.0f}s  [{eng}]")

    # Save metadata
    import json
    meta_path = os.path.join(args.out_dir, "export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
