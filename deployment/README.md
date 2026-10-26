# Jetson Orin Nano — Object Detection Benchmark

Benchmarks YOLOv8, YOLO11 (v26n), and RF-DETR on Jetson Orin Nano using TensorRT FP16.

## Files in the Zip

```
Models/
  Yolov8/
    best.pt          ← Custom 8-class YOLOv8 (aluminium_foil, bottle, cement_dust, floor, obstacle, paper, wooden_dust, wrapper)
    yolov8n.pt       ← Pretrained COCO 80-class YOLOv8n
    last.pt          ← Same custom training, last checkpoint
    yolo26n.pt       ← YOLO26n (duplicate of Yolov11/ copy)
    best_fp32.onnx, best_fp16.onnx, best_int8.onnx  ← Pre-exported ONNX (from best.pt)
  Yolov11/
    yolo26n.pt       ← YOLO26n pretrained on COCO (80-class, 640px)
    yolo11n-seg.pt   ← YOLOv11n segmentation (not used for detection benchmark)
  Rfdetr/
    Copy of rfdetr_fp32.onnx  ← RF-DETR ONNX, 10-class, input 448×448, output dets[1,300,4] + labels[1,300,10]
    Copy of rfdetr_fp16.onnx
    Copy of rfdetr_int8.onnx
```

### Assumptions

- **YOLOv8** → `best.pt` (your custom 8-class model)
- **YOLO11** → `yolo26n.pt` from `Yolov11/` (COCO 80-class; file is actually YOLO26n)
- **RF-DETR** → only ONNX available (no `.pth`); 10 classes, 448×448 input
- **No validation set or `data.yaml` in the zip.** You must supply `--val-dir` pointing to labeled images for accuracy evaluation. Labels can be COCO JSON (`--val-ann`) or YOLO `.txt` files alongside images.
- **Different class counts** (8 vs 80 vs 10) mean accuracy comparison is only valid within the same dataset. Latency/throughput/resource comparisons are always valid.

## Run Order

### 1. Install dependencies

```bash
chmod +x setup.sh
sudo ./setup.sh
```

If PyTorch is not installed, the script prints instructions for downloading NVIDIA's Jetson wheel. Install that first, then rerun.

### 2. Set power mode (optional but recommended)

```bash
# MAXN (15W on Orin Nano)
sudo nvpmodel -m 0
sudo jetson_clocks

# Or 7W low-power
sudo nvpmodel -m 1
```

### 3. Export TensorRT engines

Engines must be built **on the Jetson itself** — never reuse engines from another machine.

```bash
# Unzip models
unzip Document_from_Sanjana_Srinivas.zip

# Export all three
python3 export_engines.py \
    --yolov8 Models/Yolov8/best.pt \
    --yolo11 Models/Yolov11/yolo26n.pt \
    --rfdetr-onnx "Models/Rfdetr/Copy of rfdetr_fp32.onnx" \
    --imgsz 640 \
    --rfdetr-imgsz 448 \
    --out-dir engines/

# Optional: also build FP32 and INT8
python3 export_engines.py \
    --yolov8 Models/Yolov8/best.pt \
    --yolo11 Models/Yolov11/yolo26n.pt \
    --rfdetr-onnx "Models/Rfdetr/Copy of rfdetr_fp32.onnx" \
    --imgsz 640 --rfdetr-imgsz 448 \
    --fp32 --int8 --calib-dir path/to/calibration_images/ \
    --out-dir engines/
```

### 4. Run headless benchmark

```bash
python3 benchmark.py --mode headless \
    --yolov8-engine engines/best_fp16.engine \
    --yolov8-pt Models/Yolov8/best.pt \
    --yolo11-engine engines/yolo26n_fp16.engine \
    --yolo11-pt Models/Yolov11/yolo26n.pt \
    --rfdetr-engine engines/rfdetr_Copy_of_rfdetr_fp32_fp16.engine \
    --source path/to/test_images_or_video/ \
    --val-dir path/to/val_images/ \
    --val-ann path/to/annotations.json \
    --num-frames 500 \
    --warmup 50 \
    --conf 0.25 --iou 0.45 \
    --rfdetr-num-classes 10 \
    --out-dir results/
```

### 5. Run live webcam mode

```bash
python3 benchmark.py --mode live \
    --model yolov8 \
    --engine engines/best_fp16.engine \
    --camera 0 --imgsz 640 --conf 0.25
```

Press `q` to quit.

### 6. Sustained-load test (thermal stability)

```bash
python3 benchmark.py --mode headless \
    --yolov8-engine engines/best_fp16.engine \
    --source path/to/images/ \
    --val-dir path/to/images/ \
    --sustained --sustained-duration 600 \
    --out-dir results/
```

### 7. Power-mode sweep

```bash
python3 benchmark.py --mode headless \
    --yolov8-engine engines/best_fp16.engine \
    --yolo11-engine engines/yolo26n_fp16.engine \
    --rfdetr-engine engines/rfdetr_fp16.engine \
    --source path/to/images/ \
    --power-modes 0 1 \
    --jetson-clocks \
    --num-frames 300 \
    --out-dir results/
```

### 8. Test RF-DETR standalone

```bash
python3 trt_rfdetr.py \
    --engine engines/rfdetr_fp16.engine \
    --image test.jpg \
    --conf 0.5
```

## Output

```
results/
├── results.json          # Full structured results
├── results.csv           # Flat CSV for spreadsheets
├── summary.md            # Ranked winners per metric
├── latency_stacked.png   # Per-stage latency breakdown
├── fps_comparison.png    # Throughput bar chart
├── fps_per_watt.png      # Energy efficiency
├── map_vs_latency.png    # Accuracy vs speed scatter
├── sustained_*.png       # Time-series plots
└── sustained_load_log.json
```

## Key Notes

- **Orin Nano has no DLA** — all inference runs on GPU only.
- Webcam is captured at 1080p; inference runs at model `imgsz` via letterbox resize (YOLO) or direct resize (RF-DETR).
- Camera capture is threaded so reads don't block inference.
- CUDA events provide accurate GPU-side inference timing; `time.time()` for CPU-side stages.
- The `--rfdetr-num-classes` flag should match the RF-DETR ONNX output (10 for your model).
