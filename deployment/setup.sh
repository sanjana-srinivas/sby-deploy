#!/usr/bin/env bash
# ============================================================================
# setup.sh — Install all dependencies on Jetson Orin Nano (JetPack 5.x / 6.x)
# Run: chmod +x setup.sh && sudo ./setup.sh
# ============================================================================
set -euo pipefail

echo "============================================================"
echo " Jetson Orin Nano Benchmark — Environment Setup"
echo "============================================================"

# ---------- Detect JetPack / L4T ----------
if [ -f /etc/nv_tegra_release ]; then
    echo "[INFO] L4T: $(cat /etc/nv_tegra_release | head -1)"
else
    echo "[WARN] /etc/nv_tegra_release not found — are you on a Jetson?"
fi

JETPACK=""
if dpkg -l nvidia-jetpack &>/dev/null; then
    JETPACK=$(dpkg -l nvidia-jetpack | awk '/nvidia-jetpack/{print $3}')
    echo "[INFO] JetPack version: $JETPACK"
fi

PYTHON_VER=$(python3 --version 2>&1)
echo "[INFO] Python: $PYTHON_VER"

ARCH=$(uname -m)
echo "[INFO] Architecture: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    echo "[WARN] Not aarch64 — this script is designed for Jetson (ARM64)."
    echo "       Continuing anyway (useful for dry-run), but PyTorch wheels won't match."
fi

# ---------- System packages ----------
echo ""
echo "[1/7] System packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv \
    build-essential cmake pkg-config \
    libopencv-dev python3-opencv \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools v4l-utils \
    libjpeg-dev libpng-dev libtiff-dev \
    libhdf5-dev libatlas-base-dev \
    curl wget git unzip nano \
    2>/dev/null || true

# ---------- Upgrade pip ----------
echo ""
echo "[2/7] Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel 2>/dev/null || \
    pip3 install --upgrade pip setuptools wheel

# ---------- PyTorch + torchvision ----------
# ⚠ VERSION PITFALL: You MUST use NVIDIA's prebuilt wheels for Jetson.
# pip install torch from PyPI gives x86 wheels that will NOT work.
# Official list: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
echo ""
echo "[3/7] PyTorch & torchvision..."
echo "       ⚠  If torch is already installed from NVIDIA's Jetson wheel, skipping."
echo "       ⚠  Do NOT install torch from PyPI — it ships x86-only wheels."

if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "       torch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "       CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
else
    echo ""
    echo "  ╔════════════════════════════════════════════════════════════╗"
    echo "  ║  torch NOT found. Install NVIDIA's Jetson PyTorch wheel: ║"
    echo "  ║                                                          ║"
    echo "  ║  JetPack 5.x (L4T R35):                                 ║"
    echo "  ║    torch 2.1 + torchvision 0.16                         ║"
    echo "  ║    https://developer.download.nvidia.com/compute/        ║"
    echo "  ║      redist/jp/v512/pytorch/                             ║"
    echo "  ║                                                          ║"
    echo "  ║  JetPack 6.x (L4T R36):                                 ║"
    echo "  ║    torch 2.3 / 2.4 + torchvision 0.18 / 0.19            ║"
    echo "  ║    https://developer.download.nvidia.com/compute/        ║"
    echo "  ║      redist/jp/v60/pytorch/                              ║"
    echo "  ║                                                          ║"
    echo "  ║  Download the .whl for your JP version + Python, then:   ║"
    echo "  ║    pip3 install torch-<ver>-cp<X>-aarch64.whl            ║"
    echo "  ║    pip3 install torchvision (or build from source)       ║"
    echo "  ╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  After installing, rerun this script."
    exit 1
fi

# ---------- TensorRT Python bindings ----------
echo ""
echo "[4/7] TensorRT bindings..."
# TensorRT C++ libs ship with JetPack; Python bindings may need explicit install.
if python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)" 2>/dev/null; then
    echo "       TensorRT already available."
else
    echo "       Installing tensorrt Python bindings..."
    # JetPack 5: tensorrt is in /usr/lib/python3/dist-packages
    # JetPack 6: may need pip install tensorrt or nvidia-tensorrt
    pip3 install tensorrt 2>/dev/null || \
        echo "[WARN] Could not pip-install tensorrt. Ensure JetPack installed it system-wide."
fi

# ---------- Core Python packages ----------
echo ""
echo "[5/7] Python packages..."
# ⚠ PITFALL: onnxruntime-gpu on Jetson must come from NVIDIA's jetson-containers
#    or be built from source. PyPI onnxruntime-gpu is x86 CUDA.
#    We only strictly need it for optional ONNX validation; TensorRT is the target runtime.
pip3 install --no-cache-dir \
    ultralytics \
    onnx \
    onnxslim \
    pycuda \
    pycocotools \
    opencv-python-headless \
    matplotlib \
    pandas \
    tabulate \
    tqdm \
    psutil \
    2>/dev/null || true

# rfdetr — install if available (used only for PT→ONNX export if user has .pth weights)
pip3 install rfdetr 2>/dev/null || echo "[WARN] rfdetr package not found; not required if you already have ONNX."

# jetson-stats (jtop)
pip3 install jetson-stats 2>/dev/null || echo "[WARN] jetson-stats install failed. Try: sudo pip3 install jetson-stats"

# onnxruntime-gpu: attempt install, warn if it fails
pip3 install onnxruntime-gpu 2>/dev/null || \
    echo "[WARN] onnxruntime-gpu from PyPI is x86. For Jetson, get the wheel from:"
    echo "       https://elinux.org/Jetson_Zoo#ONNX_Runtime"

# ---------- cuda-python (alternative to pycuda) ----------
echo ""
echo "[6/7] cuda-python..."
pip3 install cuda-python 2>/dev/null || echo "[INFO] cuda-python not installed (pycuda is sufficient)."

# ---------- Verify ----------
echo ""
echo "[7/7] Verification..."
python3 -c "
import sys
print(f'Python {sys.version}')
pkgs = {
    'torch': lambda: __import__('torch').__version__,
    'torchvision': lambda: __import__('torchvision').__version__,
    'tensorrt': lambda: __import__('tensorrt').__version__,
    'ultralytics': lambda: __import__('ultralytics').__version__,
    'onnx': lambda: __import__('onnx').__version__,
    'cv2': lambda: __import__('cv2').__version__,
    'pycuda': lambda: __import__('pycuda').__version__,
    'pycocotools': lambda: 'OK',
    'matplotlib': lambda: __import__('matplotlib').__version__,
}
for name, fn in pkgs.items():
    try:
        v = fn()
        print(f'  ✓ {name:20s} {v}')
    except Exception as e:
        print(f'  ✗ {name:20s} MISSING ({e})')

import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================================"
echo " Setup complete. Next: python3 export_engines.py --help"
echo "============================================================"
