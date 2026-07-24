"""
CropCare — Convert PyTorch Model to ONNX
==========================================
Run this script on Google Colab (or wherever you trained the model).

Prerequisites
─────────────
  • crop_model_best.pth must be in the current directory
  • pip install torch torchvision onnx onnxruntime

Usage
─────
  python convert_to_onnx.py

Output
──────
  crop_model.onnx  — ready for deployment with ONNX Runtime
"""

import os
import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (must match crop_detection.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 5) -> nn.Module:
    """Recreate the same architecture used during training."""
    model = models.resnet50(weights=None)          # no pretrained weights needed
    in_features = model.fc.in_features             # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PTH_PATH  = "crop_model_best.pth"      # trained PyTorch weights
ONNX_PATH = "crop_model.onnx"          # output ONNX file
NUM_CLASSES = 5                         # jute, maize, rice, sugarcane, wheat


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD TRAINED WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(PTH_PATH):
    raise FileNotFoundError(
        f"'{PTH_PATH}' not found. Make sure you run this script in the same "
        "directory as your trained model file."
    )

print(f"[1/4] Loading trained weights from  {PTH_PATH} …")
model = build_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
model.eval()
print("  ✓  Weights loaded successfully\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPORT TO ONNX
# ─────────────────────────────────────────────────────────────────────────────

print(f"[2/4] Exporting to ONNX format …")
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f"  ✓  Exported to  {ONNX_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATE THE ONNX MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("[3/4] Validating ONNX model …")

import onnx                       # noqa: E402
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("  ✓  ONNX model is valid\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEST INFERENCE WITH ONNX RUNTIME
# ─────────────────────────────────────────────────────────────────────────────

print("[4/4] Running a test inference with ONNX Runtime …")

import numpy as np                # noqa: E402
import onnxruntime as ort         # noqa: E402

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {"input": test_input})
print(f"  ✓  Inference OK — output shape: {output[0].shape}\n")

size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)

print("=" * 62)
print(f"  ✅  SUCCESS!  Model saved to: {ONNX_PATH}  ({size_mb:.1f} MB)")
print("=" * 62)
print()
print("  NEXT STEPS:")
print()
print("  1.  Upload  crop_model.onnx  to Hugging Face Hub (free):")
print("        → https://huggingface.co  →  New Model  →  Upload file")
print()
print("  2.  Copy the direct download link:")
print("        https://huggingface.co/YOUR_USERNAME/CropCare/resolve/main/crop_model.onnx")
print()
print("  3.  In Streamlit Cloud → Settings → Secrets, add:")
print('        MODEL_URL = "https://huggingface.co/YOUR_USERNAME/CropCare/resolve/main/crop_model.onnx"')
print()
print("  4.  Push the updated code to GitHub and deploy!")
print()
