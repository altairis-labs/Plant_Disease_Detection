import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p { color: #8b949e; font-size: 1.05rem; font-weight: 300; }

.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    margin-bottom: 0.2rem;
    color: #4ade80;
}
.confidence-bar-bg { background: #21262d; border-radius: 999px; height: 10px; margin-top: 0.6rem; }
.confidence-bar-fill { background: linear-gradient(90deg, #4ade80, #22d3ee); border-radius: 999px; height: 10px; }

.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; margin-bottom: 1rem; }
.badge-healthy { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade8044; }
.badge-disease { background: #3a1a1a; color: #f87171; border: 1px solid #f8717144; }

.upload-hint { color: #8b949e; font-size: 0.88rem; text-align: center; margin-top: 0.5rem; }

div[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 12px;
    padding: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    color: #0d1117;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
}

.section-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
hr { border-color: #21262d; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 128

CLASS_NAMES = {
    0: 'Apple - Apple Scab',
    1: 'Apple - Black Rot',
    2: 'Apple - Cedar Apple Rust',
    3: 'Apple - Healthy',
    4: 'Blueberry - Healthy',
    5: 'Cherry - Powdery Mildew',
    6: 'Cherry - Healthy',
    7: 'Corn - Cercospora Leaf Spot',
    8: 'Corn - Common Rust',
    9: 'Corn - Northern Leaf Blight',
    10: 'Corn - Healthy',
    11: 'Grape - Black Rot',
    12: 'Grape - Esca (Black Measles)',
    13: 'Grape - Leaf Blight',
    14: 'Grape - Healthy',
    15: 'Orange - Haunglongbing (Citrus Greening)',
    16: 'Peach - Bacterial Spot',
    17: 'Peach - Healthy',
    18: 'Pepper Bell - Bacterial Spot',
    19: 'Pepper Bell - Healthy',
    20: 'Potato - Early Blight',
    21: 'Potato - Late Blight',
    22: 'Potato - Healthy',
    23: 'Raspberry - Healthy',
    24: 'Soybean - Healthy',
    25: 'Squash - Powdery Mildew',
    26: 'Strawberry - Leaf Scorch',
    27: 'Strawberry - Healthy',
    28: 'Tomato - Bacterial Spot',
    29: 'Tomato - Early Blight',
    30: 'Tomato - Late Blight',
    31: 'Tomato - Leaf Mold',
    32: 'Tomato - Septoria Leaf Spot',
    33: 'Tomato - Spider Mites',
    34: 'Tomato - Target Spot',
    35: 'Tomato - Yellow Leaf Curl Virus',
    36: 'Tomato - Mosaic Virus',
    37: 'Tomato - Healthy',
}

HEALTHY_CLASSES = {3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37}

# ─────────────────────────────────────────────
# Load TFLite model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="plant_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

# ─────────────────────────────────────────────
# TFLite inference
# ─────────────────────────────────────────────
def predict(interpreter, img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    class_idx  = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])
    return class_idx, confidence

# ─────────────────────────────────────────────
# Grad-CAM (uses TFLite for prediction but
# needs the keras model — skipped for TFLite
# deployment; replaced with a clean overlay)
# ─────────────────────────────────────────────
def make_attention_overlay(pil_img):
    """
    Simple saliency-style overlay using image gradients
    as a lightweight alternative to Grad-CAM when only
    the TFLite model is available.
    """
    img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Sobel edge map as a proxy for regions of interest
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx**2 + sy**2)
    magnitude = cv2.GaussianBlur(magnitude, (15, 15), 0)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() + 1e-8)
    heatmap   = np.uint8(255 * magnitude)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = heatmap_color * 0.35 + img_np.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image to identify diseases using deep learning</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    try:
        interpreter = load_tflite_model()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}\n\nMake sure `plant_model_quant.tflite` is in the repo.")
        st.stop()

# Upload
uploaded_file = st.file_uploader(
    "Drop a leaf image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Supports JPG, JPEG, PNG</p>', unsafe_allow_html=True)

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">Original Image</p>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)

    # Preprocess
    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.expand_dims(
        np.array(img_resized).astype(np.float32) / 255.0, axis=0
    )

    with st.spinner("Analysing..."):
        class_idx, confidence = predict(interpreter, img_array)
        overlay_img = make_attention_overlay(pil_img)

    with col2:
        st.markdown('<p class="section-label">Region Highlight</p>', unsafe_allow_html=True)
        st.image(overlay_img, use_container_width=True)

    # Result card
    class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    is_healthy  = class_idx in HEALTHY_CLASSES
    badge_class = "badge-healthy" if is_healthy else "badge-disease"
    badge_label = "✅ Healthy" if is_healthy else "⚠️ Disease Detected"
    conf_pct    = int(confidence * 100)

    st.markdown(f"""
    <div class="result-card">
        <span class="badge {badge_class}">{badge_label}</span>
        <h3>{class_name}</h3>
        <div class="section-label" style="margin-top:1rem;">Confidence</div>
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <div class="confidence-bar-bg" style="flex:1;">
                <div class="confidence-bar-fill" style="width:{conf_pct}%;"></div>
            </div>
            <span style="font-family:'Syne',sans-serif; font-weight:700; color:#4ade80;">{conf_pct}%</span>
        </div>
        <div class="section-label" style="margin-top:1.2rem;">Class Index</div>
        <span style="color:#8b949e;">#{class_idx}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">About the Highlight</p>', unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#8b949e; font-size:0.9rem;'>"
        "The region highlight uses edge detection to emphasise visually significant areas of the leaf. "
        "<b style='color:#f87171;'>Warm areas (red/yellow)</b> indicate high-contrast regions such as lesions, spots, or discolouration. "
        "<b style='color:#60a5fa;'>Cool areas (blue/purple)</b> are smoother, lower-contrast regions."
        "</span>",
        unsafe_allow_html=True
    )
