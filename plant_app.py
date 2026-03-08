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
.confidence-bar-low  { background: linear-gradient(90deg, #f59e0b, #f87171); border-radius: 999px; height: 10px; }

.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; margin-bottom: 1rem; }
.badge-healthy { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade8044; }
.badge-disease { background: #3a1a1a; color: #f87171; border: 1px solid #f8717144; }
.badge-unsure  { background: #3a2e1a; color: #f59e0b; border: 1px solid #f59e0b44; }

.warning-box {
    background: #2a1f0a;
    border: 1px solid #f59e0b44;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    color: #f59e0b;
    font-size: 0.92rem;
}

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
IMG_SIZE             = 128
CONFIDENCE_THRESHOLD = 0.75  # Below this → show low confidence warning

CLASS_NAMES = {
    0: 'Apple - Apple Scab',       1: 'Apple - Black Rot',
    2: 'Apple - Cedar Apple Rust', 3: 'Apple - Healthy',
    4: 'Blueberry - Healthy',      5: 'Cherry - Powdery Mildew',
    6: 'Cherry - Healthy',         7: 'Corn - Cercospora Leaf Spot',
    8: 'Corn - Common Rust',       9: 'Corn - Northern Leaf Blight',
    10: 'Corn - Healthy',          11: 'Grape - Black Rot',
    12: 'Grape - Esca (Black Measles)', 13: 'Grape - Leaf Blight',
    14: 'Grape - Healthy',         15: 'Orange - Haunglongbing (Citrus Greening)',
    16: 'Peach - Bacterial Spot',  17: 'Peach - Healthy',
    18: 'Pepper Bell - Bacterial Spot', 19: 'Pepper Bell - Healthy',
    20: 'Potato - Early Blight',   21: 'Potato - Late Blight',
    22: 'Potato - Healthy',        23: 'Raspberry - Healthy',
    24: 'Soybean - Healthy',       25: 'Squash - Powdery Mildew',
    26: 'Strawberry - Leaf Scorch',27: 'Strawberry - Healthy',
    28: 'Tomato - Bacterial Spot', 29: 'Tomato - Early Blight',
    30: 'Tomato - Late Blight',    31: 'Tomato - Leaf Mold',
    32: 'Tomato - Septoria Leaf Spot', 33: 'Tomato - Spider Mites',
    34: 'Tomato - Target Spot',    35: 'Tomato - Yellow Leaf Curl Virus',
    36: 'Tomato - Mosaic Virus',   37: 'Tomato - Healthy',
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
# Background removal
# ─────────────────────────────────────────────
def remove_background(pil_img):
    """
    Remove white/light backgrounds from leaf images.
    Replaces background with neutral gray similar to PlantVillage training images.
    Uses GrabCut + color-based detection combined.
    """
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    # ── GrabCut ──────────────────────────────────────────────────────────────
    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin    = max(5, min(h, w) // 10)
    rect      = (margin, margin, w - 2 * margin, h - 2 * margin)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grab_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    except Exception:
        grab_mask = np.ones((h, w), dtype='uint8')

    # ── Color-based: detect white/light/gray background ───────────────────────
    img_hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(img_hsv, (0, 0, 180), (180, 50, 255))
    gray_mask  = cv2.inRange(img_hsv, (0, 0, 100), (180, 30, 200))
    bg_mask    = cv2.bitwise_or(white_mask, gray_mask)
    color_fg   = (cv2.bitwise_not(bg_mask) // 255).astype('uint8')

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = cv2.bitwise_and(grab_mask, color_fg)
    kernel   = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Skip if mask removed too much (non-white bg image)
    if combined.sum() < (h * w * 0.10):
        return pil_img

    # Replace background with neutral gray
    result              = img_np.copy()
    result[combined == 0] = [180, 180, 180]
    return Image.fromarray(result)

# ─────────────────────────────────────────────
# TFLite inference
# ─────────────────────────────────────────────
def predict(interpreter, img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    class_idx   = int(np.argmax(predictions[0]))
    confidence  = float(predictions[0][class_idx])
    return class_idx, confidence, predictions[0]

# ─────────────────────────────────────────────
# Region highlight overlay
# ─────────────────────────────────────────────
def make_attention_overlay(pil_img):
    img_np    = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    gray      = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sx        = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy        = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx**2 + sy**2)
    magnitude = cv2.GaussianBlur(magnitude, (15, 15), 0)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() + 1e-8)
    heatmap   = np.uint8(255 * magnitude)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay   = heatmap_color * 0.35 + img_np.astype(np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image to identify diseases using deep learning</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    try:
        interpreter = load_tflite_model()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

uploaded_file = st.file_uploader(
    "Drop a leaf image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Supports JPG, JPEG, PNG — works best on clear, close-up leaf photos</p>', unsafe_allow_html=True)

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">Original Image</p>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)

    with st.spinner("Preprocessing & analysing..."):
        clean_img   = remove_background(pil_img)
        img_resized = clean_img.resize((IMG_SIZE, IMG_SIZE))
        img_array   = np.expand_dims(
            np.array(img_resized).astype(np.float32) / 255.0, axis=0
        )
        class_idx, confidence, all_preds = predict(interpreter, img_array)
        overlay_img = make_attention_overlay(clean_img)

    with col2:
        st.markdown('<p class="section-label">Region Highlight</p>', unsafe_allow_html=True)
        st.image(overlay_img, use_container_width=True)

    # ── Result ────────────────────────────────────────────────────────────────
    class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    is_healthy = class_idx in HEALTHY_CLASSES
    low_conf   = confidence < CONFIDENCE_THRESHOLD
    conf_pct   = int(confidence * 100)

    if low_conf:
        badge_class, badge_label = "badge-unsure", "⚠️ Uncertain"
    elif is_healthy:
        badge_class, badge_label = "badge-healthy", "✅ Healthy"
    else:
        badge_class, badge_label = "badge-disease", "⚠️ Disease Detected"

    bar_class  = "confidence-bar-low" if low_conf else "confidence-bar-fill"
    conf_color = "#f59e0b" if low_conf else "#4ade80"

    st.markdown(f"""
    <div class="result-card">
        <span class="badge {badge_class}">{badge_label}</span>
        <h3>{class_name}</h3>
        <div class="section-label" style="margin-top:1rem;">Confidence</div>
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <div class="confidence-bar-bg" style="flex:1;">
                <div class="{bar_class}" style="width:{conf_pct}%;"></div>
            </div>
            <span style="font-family:'Syne',sans-serif; font-weight:700; color:{conf_color};">{conf_pct}%</span>
        </div>
        <div class="section-label" style="margin-top:1.2rem;">Class Index</div>
        <span style="color:#8b949e;">#{class_idx}</span>
    </div>
    """, unsafe_allow_html=True)

    if low_conf:
        top3_idx  = np.argsort(all_preds)[::-1][:3]
        top3_text = " &nbsp;|&nbsp; ".join(
            [f"{CLASS_NAMES.get(i, f'Class {i}')} ({int(all_preds[i]*100)}%)" for i in top3_idx]
        )
        st.markdown(f"""
        <div class="warning-box">
            <b>⚠️ Low confidence ({conf_pct}%)</b> — the model is uncertain about this image.<br>
            Try a clearer, closer photo of the leaf on a dark or natural background.<br><br>
            <b>Top possibilities:</b><br>{top3_text}
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
    st.markdown(
        "<br><span style='color:#8b949e; font-size:0.85rem;'>"
        "💡 <b>Tip:</b> For best results, photograph the leaf close-up on a dark or natural background in good lighting."
        "</span>",
        unsafe_allow_html=True
    )
