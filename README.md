# 🌾 CropCare

AI-powered crop detection using image classification. Upload a photo of a crop and get instant identification with confidence scores.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)
![ONNX](https://img.shields.io/badge/Runtime-ONNX-005CED)

## 🌱 Supported Crops

| Crop       | Emoji |
| ---------- | ----- |
| Jute       | 🌿    |
| Maize      | 🌽    |
| Rice       | 🌾    |
| Sugarcane  | 🎋    |
| Wheat      | 🌾    |

## 🏗️ Architecture

- **Training**: ResNet-50 (pretrained on ImageNet), fine-tuned on crop images using PyTorch
- **Deployment**: ONNX Runtime (~40 MB vs ~2 GB for PyTorch)
- **Frontend**: Streamlit with custom CSS
- **Input**: 224 × 224 RGB images

## 🚀 Deployment Guide

### Step 1 — Train the Model (Google Colab)

```bash
# Upload crop_detection.py and your dataset to Colab
python crop_detection.py
```

This trains the ResNet-50 model and saves `crop_model_best.pth`.

Your dataset should be structured as:

```
crop_dataset/
├── jute/
│   ├── img001.jpg
│   └── ...
├── maize/
├── rice/
├── sugarcane/
└── wheat/
```

### Step 2 — Convert to ONNX (Google Colab)

```bash
pip install onnx onnxruntime
python convert_to_onnx.py
```

This converts `crop_model_best.pth` → `crop_model.onnx` and verifies the export.

### Step 3 — Host the Model (Hugging Face Hub)

The ONNX model is too large for GitHub (~100 MB). Host it for free on Hugging Face:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New** → **Model** → name it `CropCare`
3. Upload `crop_model.onnx` to the repository
4. Your direct download URL will be:
   ```
   https://huggingface.co/YOUR_USERNAME/CropCare/resolve/main/crop_model.onnx
   ```

### Step 4 — Deploy on Streamlit Cloud

1. Push this code to your GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your `CropCare` repo
4. In **Settings → Secrets**, add:
   ```toml
   MODEL_URL = "https://huggingface.co/YOUR_USERNAME/CropCare/resolve/main/crop_model.onnx"
   ```
5. Click **Deploy** 🎉

### Step 5 — Run Locally (optional)

```bash
pip install -r requirements.txt
# Place crop_model.onnx in this directory
streamlit run app.py
```

## 📁 Project Structure

```
CropCare/
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── app.py                    # Streamlit web app (ONNX Runtime)
├── crop_detection.py         # Model training script (PyTorch)
├── convert_to_onnx.py        # PyTorch → ONNX conversion script
├── requirements.txt          # Deployment dependencies (lightweight)
├── runtime.txt               # Python version for Streamlit Cloud
└── README.md
```

## ⚙️ Tech Stack

| Component         | Technology                       |
| ----------------- | -------------------------------- |
| Training          | PyTorch, torchvision, ResNet-50  |
| Inference         | ONNX Runtime                     |
| Web App           | Streamlit                        |
| Image Processing  | Pillow, NumPy                    |
| Model Hosting     | Hugging Face Hub                 |
| Deployment        | Streamlit Community Cloud        |

## 📝 Why ONNX Instead of PyTorch?

| Metric              | PyTorch        | ONNX Runtime  |
| ------------------- | -------------- | ------------- |
| Package Size        | ~2 GB          | ~40 MB        |
| Cold Start          | ~30s           | ~3s           |
| RAM Usage           | ~1.5 GB        | ~200 MB       |
| Streamlit Cloud     | ❌ Crashes     | ✅ Works      |

## 📄 License

MIT
