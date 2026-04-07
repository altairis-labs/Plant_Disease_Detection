import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import io
import base64
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from disease_info import get_disease_info

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 128
TFLITE_MODEL   = "plant_model_quant.tflite"
KERAS_MODEL    = "best_model.keras"
CONFIDENCE_THR = 0.75          # minimum confidence to accept prediction
GRADCAM_THR    = 0.82          # minimum confidence to show Grad-CAM (above this = reliable heatmap)
GRADCAM_LAYER  = None          # resolved at runtime — last Conv2D layer of best_model.keras

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

SEVERITY_COLORS = {
    "none":    ("#22c55e", "#052e16"),   # green
    "moderate":("#f59e0b", "#1c1003"),   # amber
    "severe":  ("#ef4444", "#1c0202"),   # red
    "unknown": ("#6b7280", "#111827"),   # gray
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0a0f0a;
      color: #e8f5e9;
  }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }

  .stButton>button {
      background: linear-gradient(135deg, #166534, #15803d);
      color: white;
      border: none;
      border-radius: 8px;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      padding: 0.6rem 1.6rem;
      transition: all .2s ease;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #15803d, #22c55e);
      transform: translateY(-1px);
  }

  /* care card */
  .care-card {
      background: #0f1a0f;
      border: 1px solid #1a3a1a;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 0.8rem;
  }
  .care-card h4 {
      font-family: 'Syne', sans-serif;
      font-size: 0.9rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin: 0 0 0.5rem 0;
  }
  .care-card p {
      font-size: 0.92rem;
      line-height: 1.65;
      margin: 0;
      color: #bbf7d0;
  }

  /* severity badge */
  .severity-badge {
      display: inline-block;
      padding: .25rem .75rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      font-family: 'Syne', sans-serif;
      letter-spacing: .06em;
      text-transform: uppercase;
  }

  .stTabs [data-baseweb="tab"] {
      font-family: 'Syne', sans-serif;
      font-weight: 700;
  }

  .gradcam-warning {
      background: #1c1003;
      border: 1px solid #f59e0b;
      border-radius: 8px;
      padding: .8rem 1rem;
      color: #fde68a;
      font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_tflite():
    interp = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interp.allocate_tensors()
    return interp

@st.cache_resource
def load_keras():
    """Load full Keras model for Grad-CAM. Returns None if file not present."""
    try:
        model = tf.keras.models.load_model(KERAS_MODEL)
        # dummy forward pass to initialise all layers (required for Keras 3)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        model(dummy, training=False)
        return model
    except Exception:
        return None

def get_last_conv_layer(model):
    """Find the last Conv2D layer in the model for highest-resolution Grad-CAM."""
    last = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last = layer.name
    return last

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def grabcut_segment(img_rgb: np.ndarray) -> np.ndarray:
    """GrabCut background removal. Returns RGB image with background blacked out."""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    result = img_rgb * mask2[:, :, np.newaxis]
    return result

def preprocess_for_tflite(img_pil: Image.Image, use_grabcut: bool = True):
    img_rgb = np.array(img_pil.convert("RGB"))
    if use_grabcut:
        img_rgb = grabcut_segment(img_rgb)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)

def preprocess_for_gradcam(img_pil: Image.Image, use_grabcut: bool = False):
    """
    For Grad-CAM we deliberately skip GrabCut by default —
    background removal can strip disease regions near leaf edges,
    leading to misleading / uncertain heatmaps.
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    if use_grabcut:
        img_rgb = grabcut_segment(img_rgb)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — TFLite
# ─────────────────────────────────────────────────────────────────────────────
def predict_tflite(interp, img_tensor):
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    interp.set_tensor(in_det[0]["index"], img_tensor)
    interp.invoke()
    probs = interp.get_tensor(out_det[0]["index"])[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx]), probs

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM — using last Conv2D layer for sharper, more accurate activations
# ─────────────────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_tensor, keras_model, last_conv_name, pred_index=None):
    """
    Generate Grad-CAM heatmap.
    Uses the LAST Conv2D layer for highest spatial resolution activations.
    Returns a (H, W) float32 heatmap normalised to [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=keras_model.input,
        outputs=[keras_model.get_layer(last_conv_name).output, keras_model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)                    # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_gradcam(orig_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Overlay jet-colormap heatmap on original image. Returns PIL Image."""
    orig_np = np.array(orig_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))

    # Use jet colormap
    colormap = cm.get_cmap("jet")
    heatmap_color = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr    = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)
    orig_bgr        = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_rgb)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def severity_badge(severity: str) -> str:
    labels = {"none": "Healthy", "moderate": "Moderate", "severe": "Severe", "unknown": "Unknown"}
    fg, bg = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
    return (
        f'<span class="severity-badge" style="background:{bg};color:{fg};border:1px solid {fg};">'
        f'{labels.get(severity, severity)}</span>'
    )

def format_label(label: str) -> str:
    """Apple___Apple_scab → Apple — Apple Scab"""
    parts = label.split("___")
    plant   = parts[0].replace("_", " ").replace(",", "").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{plant} — {disease}" if disease else plant

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# WORD CLOUD
# ─────────────────────────────────────────────────────────────────────────────
def show_wordcloud(text: str, title: str):
    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=600, height=300,
            background_color="#0a0f0a",
            colormap="Greens",
            max_words=60,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0a0f0a")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, color="#bbf7d0", fontsize=11)
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.info("Install `wordcloud` (`pip install wordcloud`) to see the word cloud.")

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT & CARE TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_care_tab(label: str, confidence: float):
    info = get_disease_info(label)
    severity = info.get("severity", "unknown")
    fg_color, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])

    st.markdown(
        f"**Severity:** {severity_badge(severity)} &nbsp;&nbsp; "
        f"**Confidence:** `{confidence:.1%}`",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Cure / Treatment
    icon = "🚨" if severity == "severe" else ("⚠️" if severity == "moderate" else "✅")
    st.markdown(f"""
    <div class="care-card">
      <h4 style="color:{fg_color};">{icon} Treatment & Cure</h4>
      <p>{info['cure']}</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="care-card">
          <h4 style="color:#fde047;">☀️ Sunlight</h4>
          <p>{info['sun']}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="care-card">
          <h4 style="color:#67e8f9;">💧 Water</h4>
          <p>{info['water']}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="care-card">
          <h4 style="color:#a78bfa;">🌱 Nutrients</h4>
          <p>{info['nutrients']}</p>
        </div>""", unsafe_allow_html=True)

    # Word cloud of care text
    st.markdown("#### 🔤 Key Care Terms")
    all_text = " ".join([info["cure"], info["sun"], info["water"], info["nutrients"]])
    show_wordcloud(all_text, f"Care keywords for {format_label(label)}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne;font-size:2.4rem;color:#4ade80;margin-bottom:0;'>🌿 Plant Disease Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#6ee7b7;font-size:1rem;margin-top:.2rem;'>"
    "Upload a leaf image · Get diagnosis · See treatment & care advice</p>",
    unsafe_allow_html=True,
)

# Load models
tflite_interp = load_tflite()
keras_model   = load_keras()

# Resolve last conv layer name once
if keras_model is not None:
    last_conv_layer = get_last_conv_layer(keras_model)
else:
    last_conv_layer = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_grabcut_predict = st.toggle("Background removal (prediction)", value=True)
    gradcam_alpha       = st.slider("Grad-CAM overlay intensity", 0.2, 0.8, 0.45, 0.05)
    show_top3           = st.toggle("Show top-3 predictions", value=True)
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280;'>Built by altairis-labs<br>"
        "14 crops · 38 classes · TFLite inference</small>",
        unsafe_allow_html=True,
    )

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a leaf image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="border:2px dashed #1a3a1a;border-radius:16px;padding:3rem;text-align:center;color:#4b6b4b;margin-top:1rem;">
      <div style="font-size:3rem;">🍃</div>
      <p style="font-family:Syne;font-size:1.1rem;margin:.5rem 0 0;">Drop a leaf image to get started</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Process ──────────────────────────────────────────────────────────────────
orig_img = Image.open(uploaded)

with st.spinner("Analysing leaf..."):
    # TFLite prediction
    tensor_pred  = preprocess_for_tflite(orig_img, use_grabcut=use_grabcut_predict)
    top_idx, top_conf, all_probs = predict_tflite(tflite_interp, tensor_pred)
    predicted_label = CLASS_NAMES[top_idx]

    # Top-3
    top3_idx  = np.argsort(all_probs)[::-1][:3]
    top3      = [(CLASS_NAMES[i], float(all_probs[i])) for i in top3_idx]

    # Grad-CAM (only if confidence is high enough and Keras model is available)
    gradcam_img    = None
    gradcam_reason = None

    if keras_model is None:
        gradcam_reason = f"`{KERAS_MODEL}` not found in the working directory — Grad-CAM unavailable."
    elif top_conf < GRADCAM_THR:
        gradcam_reason = (
            f"Confidence is {top_conf:.1%}, below the {GRADCAM_THR:.0%} threshold for reliable Grad-CAM. "
            f"The model is uncertain — try a clearer, better-lit photo of the leaf."
        )
    else:
        try:
            # Use raw image (no GrabCut) for Grad-CAM — avoids stripping disease regions at edges
            tensor_gc  = preprocess_for_gradcam(orig_img, use_grabcut=False)
            heatmap    = make_gradcam_heatmap(tensor_gc, keras_model, last_conv_layer, pred_index=top_idx)
            gradcam_img = overlay_gradcam(orig_img, heatmap, alpha=gradcam_alpha)
        except Exception as e:
            gradcam_reason = f"Grad-CAM generation failed: {e}"

# ── Results header ───────────────────────────────────────────────────────────
info = get_disease_info(predicted_label)
severity = info.get("severity", "unknown")
fg_color, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])

st.markdown("---")
st.markdown(
    f"<h2 style='font-family:Syne;color:{fg_color};'>"
    f"{format_label(predicted_label)}</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"Confidence: **`{top_conf:.1%}`** &nbsp; {severity_badge(severity)}",
    unsafe_allow_html=True,
)

if top_conf < CONFIDENCE_THR:
    st.warning(
        f"⚠️ Low confidence ({top_conf:.1%}). The top-3 predictions are shown below. "
        "Try uploading a clearer, well-lit photo focusing on the leaf."
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_results, tab_care, tab_gradcam, tab_wordcloud = st.tabs(
    ["📊 Results", "💊 Treatment & Care", "🔬 Grad-CAM", "🔤 Word Cloud"]
)

with tab_results:
    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(orig_img, use_container_width=True)

    with col_info:
        st.markdown("**Prediction**")
        st.markdown(
            f"<div style='background:#0f1a0f;border:1px solid {fg_color};"
            f"border-radius:10px;padding:1rem;'>"
            f"<div style='font-family:Syne;font-size:1.05rem;color:{fg_color};'>"
            f"{format_label(predicted_label)}</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{fg_color};'>{top_conf:.1%}</div>"
            f"<div style='color:#6b7280;font-size:0.85rem;'>confidence</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if show_top3:
            st.markdown("**Top-3 Predictions**")
            for lbl, prob in top3:
                bar_w = int(prob * 100)
                lbl_fg, _ = SEVERITY_COLORS.get(
                    get_disease_info(lbl).get("severity", "unknown"),
                    SEVERITY_COLORS["unknown"],
                )
                st.markdown(
                    f"<div style='margin:.3rem 0;'>"
                    f"<div style='font-size:.82rem;color:#9ca3af;margin-bottom:2px;'>"
                    f"{format_label(lbl)}</div>"
                    f"<div style='display:flex;align-items:center;gap:.5rem;'>"
                    f"<div style='flex:1;background:#1a2a1a;border-radius:999px;height:8px;'>"
                    f"<div style='width:{bar_w}%;background:{lbl_fg};height:8px;border-radius:999px;'></div>"
                    f"</div><span style='color:{lbl_fg};font-weight:700;font-size:.85rem;'>{prob:.1%}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

with tab_care:
    render_care_tab(predicted_label, top_conf)

with tab_gradcam:
    st.markdown("#### 🔬 Grad-CAM Activation Map")
    st.markdown(
        "<small style='color:#6b7280;'>Highlights the regions the model focused on. "
        "Red/yellow = high activation (disease region). Only shown when confidence ≥ "
        f"{GRADCAM_THR:.0%} to avoid misleading heatmaps.</small>",
        unsafe_allow_html=True,
    )

    if gradcam_img is not None:
        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(orig_img, use_container_width=True)
        with col_cam:
            st.markdown("**Grad-CAM Overlay**")
            st.image(gradcam_img, use_container_width=True)
        st.caption(
            f"Layer used: `{last_conv_layer}` (last Conv2D — highest spatial resolution). "
            "GrabCut is disabled for Grad-CAM to preserve disease regions at leaf edges."
        )
    else:
        st.markdown(
            f'<div class="gradcam-warning">⚠️ {gradcam_reason}</div>',
            unsafe_allow_html=True,
        )

with tab_wordcloud:
    st.markdown("#### 🔤 Disease & Care Word Cloud")
    info = get_disease_info(predicted_label)
    all_text = " ".join([
        predicted_label.replace("_", " ").replace("___", " "),
        info["cure"], info["sun"], info["water"], info["nutrients"]
    ])
    show_wordcloud(all_text, f"{format_label(predicted_label)} — Key Terms")
