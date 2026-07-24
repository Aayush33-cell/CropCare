"""
CropCare — Crop Detection Web App
==================================
Lightweight Streamlit app using ONNX Runtime for crop image classification.
Supports: Jute, Maize, Rice, Sugarcane, Wheat

Deploy on Streamlit Community Cloud with minimal resources.
Run locally:  streamlit run app.py
"""

import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Jute", "Maize", "Rice", "Sugarcane", "Wheat"]
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH = "crop_model.onnx"

# ── Model URL: set via Streamlit secrets or environment variable ─────────
# In Streamlit Cloud → Settings → Secrets, add:
#   MODEL_URL = "https://huggingface.co/YOUR_USER/CropCare/resolve/main/crop_model.onnx"
MODEL_URL = ""
try:
    MODEL_URL = st.secrets.get("MODEL_URL", "")
except Exception:
    MODEL_URL = os.environ.get("MODEL_URL", "")

CROP_EMOJI = {
    "Jute": "🌿",
    "Maize": "🌽",
    "Rice": "🌾",
    "Sugarcane": "🎋",
    "Wheat": "🌾",
}

CROP_INFO = {
    "Jute": "A natural fibre crop grown primarily in warm, humid climates. India and Bangladesh are the largest producers.",
    "Maize": "Also known as corn, it's one of the most widely grown crops globally, used for food, feed, and fuel.",
    "Rice": "A staple food for more than half the world's population, primarily cultivated in flooded paddy fields.",
    "Sugarcane": "A tropical grass cultivated for its juice, from which sugar is processed. Also used for ethanol production.",
    "Wheat": "One of the world's most important cereal crops, grown in temperate regions for bread, pasta, and pastries.",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CropCare — AI Crop Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header ─────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.75rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2D5016 0%, #4A8B2C 50%, #6BBF3B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6B7280;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Result card ────────────────────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(34, 197, 94, 0.12);
        animation: fadeIn 0.5s ease;
    }
    .result-emoji { font-size: 3rem; }
    .result-crop  {
        font-size: 2rem;
        font-weight: 700;
        color: #15803D;
        margin: 0.4rem 0;
    }
    .result-confidence {
        font-size: 1rem;
        color: #16A34A;
        font-weight: 500;
    }

    /* ── Confidence bars ────────────────────────────────────────── */
    .bar-container {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin: 0.35rem 0;
        border: 1px solid #F3F4F6;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #374151;
    }
    .bar-track {
        background: #E5E7EB;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Info box ────────────────────────────────────────────────── */
    .crop-info-box {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.5;
        margin-top: 0.8rem;
    }

    /* ── Warning ─────────────────────────────────────────────────── */
    .low-conf {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #92400E;
        font-size: 0.88rem;
        margin-top: 0.8rem;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #15803D;
    }
    .sidebar-sub {
        font-size: 0.82rem;
        color: #6B7280;
        margin-bottom: 1rem;
    }
    .crop-chip {
        display: inline-block;
        background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
        border: 1px solid #BBF7D0;
        border-radius: 20px;
        padding: 4px 13px;
        margin: 3px 2px;
        font-size: 0.82rem;
        color: #15803D;
        font-weight: 500;
    }
    .model-card {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 0.8rem;
        color: #0C4A6E;
        line-height: 1.6;
    }

    /* ── Empty state ─────────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: #9CA3AF;
    }
    .empty-state .icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
    .empty-state p { margin: 0.2rem 0; }

    /* ── Animation ───────────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Hide Streamlit chrome ───────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING  (replaces torchvision transforms — zero PyTorch needed)
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, normalise with ImageNet stats, reshape for ONNX."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0       # [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD           # normalise
    arr = arr.transpose(2, 0, 1)                         # HWC → CHW
    return arr[np.newaxis, ...].astype(np.float32)       # (1, 3, 224, 224)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────


def _download_model(url: str, dest: str) -> None:
    """Download model file from URL."""
    with st.spinner("📥 Downloading model (first run only — this may take a minute)…"):
        urllib.request.urlretrieve(url, dest)
    st.toast("✅ Model downloaded!", icon="✅")


@st.cache_resource(show_spinner="🔄 Loading model…")
def load_model() -> ort.InferenceSession:
    """Load ONNX model; download from MODEL_URL if the local file is absent."""
    if not os.path.exists(MODEL_PATH):
        if MODEL_URL:
            _download_model(MODEL_URL, MODEL_PATH)
        else:
            st.error(
                "**Model file not found!**\n\n"
                f"Place `{MODEL_PATH}` in the app directory, "
                "or set the **MODEL_URL** secret in Streamlit Cloud settings.\n\n"
                "See the project README for step-by-step instructions."
            )
            st.stop()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        MODEL_PATH, opts, providers=["CPUExecutionProvider"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────


def predict(session: ort.InferenceSession, image: Image.Image) -> dict:
    """Run inference and return class probabilities."""
    tensor = preprocess(image)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits)
    top_idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {
            name: float(p) for name, p in zip(CLASS_NAMES, probs)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🌾 CropCare</div>'
        '<div class="sidebar-sub">AI-powered crop identification</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 📋 How to Use")
    st.markdown(
        "1. Upload a photo of a crop field\n"
        "2. The model predicts the crop type\n"
        "3. Review confidence scores below"
    )

    st.divider()

    st.markdown("#### 🌱 Supported Crops")
    chips = "".join(
        f'<span class="crop-chip">{CROP_EMOJI.get(c, "🌿")} {c}</span>'
        for c in CLASS_NAMES
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="model-card">'
        "<strong>Model</strong> — ResNet-50 (fine-tuned)<br>"
        "<strong>Runtime</strong> — ONNX Runtime<br>"
        "<strong>Input</strong> — 224 × 224 RGB<br>"
        "<strong>Classes</strong> — 5 crop types"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Built with ❤️ using Streamlit & ONNX Runtime")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    "<h1>🌾 CropCare</h1>"
    "<p>Upload a crop image for instant AI-powered identification</p>"
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a crop image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WebP  •  Max 10 MB",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_gap, col_res = st.columns([5, 1, 5])

    # ── Left column: uploaded image ──────────────────────────────
    with col_img:
        st.markdown("#### 📷 Uploaded Image")
        st.image(image, use_container_width=True)

    # ── Right column: results ────────────────────────────────────
    with col_res:
        session = load_model()
        result = predict(session, image)

        crop = result["class"]
        conf = result["confidence"]
        emoji = CROP_EMOJI.get(crop, "🌿")

        # Prediction card
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-emoji">{emoji}</div>'
            f'<div class="result-crop">{crop}</div>'
            f'<div class="result-confidence">{conf * 100:.1f}% confidence</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Low-confidence warning
        if conf < 0.50:
            st.markdown(
                '<div class="low-conf">'
                "⚠️ <strong>Low confidence.</strong> The image might not show "
                "a recognisable crop, or it may be outside the trained categories."
                "</div>",
                unsafe_allow_html=True,
            )

        # Crop info
        info = CROP_INFO.get(crop, "")
        if info:
            st.markdown(
                f'<div class="crop-info-box">'
                f"<strong>ℹ️ About {crop}:</strong> {info}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📊 All Probabilities")

        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )
        bar_colors = ["#16A34A", "#22C55E", "#4ADE80", "#86EFAC", "#BBF7D0"]

        for i, (name, prob) in enumerate(sorted_probs):
            color = bar_colors[min(i, len(bar_colors) - 1)]
            e = CROP_EMOJI.get(name, "🌿")
            st.markdown(
                f'<div class="bar-container">'
                f'<div class="bar-label">'
                f"<span>{e} {name}</span><span>{prob * 100:.1f}%</span>"
                f"</div>"
                f'<div class="bar-track">'
                f'<div class="bar-fill" style="width:{prob * 100:.1f}%;background:{color};"></div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

else:
    # ── Empty / landing state ────────────────────────────────────
    st.markdown("---")
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📸</div>'
            "<p><strong>No image uploaded yet</strong></p>"
            "<p style='font-size:0.85rem;'>Drag & drop or click the button above to begin</p>"
            "</div>",
            unsafe_allow_html=True,
        )
