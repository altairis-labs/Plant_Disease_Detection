import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.cm as cm

from disease_info import get_disease_info

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224          # EfficientNetB3 native size (change to 128 if keeping old model)
TFLITE_MODEL    = "plant_model_quant.tflite"
CONFIDENCE_THR  = 0.75         # below this → low-confidence warning shown
OOD_THR         = 0.55         # below this → reject as unrecognised image
GRADCAM_THR     = 0.75         # below this → skip heatmap
HEATMAP_ALPHA   = 0.50         # overlay blend strength

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
    "none":     ("#22c55e", "#052e16"),
    "moderate": ("#f59e0b", "#1c1003"),
    "severe":   ("#ef4444", "#1c0202"),
    "unknown":  ("#6b7280", "#111827"),
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP  — no sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

  /* hide sidebar toggle and sidebar completely */
  [data-testid="collapsedControl"] { display: none !important; }
  section[data-testid="stSidebar"]  { display: none !important; }

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #080d08;
      color: #e8f5e9;
  }
  h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

  .stButton>button {
      background: linear-gradient(135deg, #166534, #15803d);
      color: white; border: none; border-radius: 8px;
      font-family: 'Syne', sans-serif; font-weight: 700;
      padding: 0.6rem 1.6rem; transition: all .2s ease;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #15803d, #22c55e);
      transform: translateY(-1px);
  }

  .care-card {
      background: #0f1a0f; border: 1px solid #1a3a1a;
      border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
  }
  .care-card h4 {
      font-family: 'Syne', sans-serif; font-size: 0.9rem;
      letter-spacing: .08em; text-transform: uppercase; margin: 0 0 0.5rem 0;
  }
  .care-card p { font-size: 0.92rem; line-height: 1.65; margin: 0; color: #bbf7d0; }

  .severity-badge {
      display: inline-block; padding: .25rem .75rem; border-radius: 999px;
      font-size: 0.78rem; font-weight: 700; font-family: 'Syne', sans-serif;
      letter-spacing: .06em; text-transform: uppercase;
  }

  .stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-weight: 700; }

  .ood-card {
      background: #1a0a0a; border: 1px solid #ef4444; border-radius: 12px;
      padding: 1.8rem; text-align: center; margin-top: 1rem;
  }
  .ood-card .icon { font-size: 2.8rem; }
  .ood-card h3 { color: #f87171; font-family: 'Syne', sans-serif; margin: .6rem 0 .3rem; }
  .ood-card p  { color: #fca5a5; font-size: 0.9rem; margin: 0; }
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

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def grabcut_segment(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    return img_rgb * mask2[:, :, np.newaxis]

def preprocess(img_pil: Image.Image, use_grabcut: bool = True) -> np.ndarray:
    img_rgb = np.array(img_pil.convert("RGB"))
    if use_grabcut:
        try:
            img_rgb = grabcut_segment(img_rgb)
        except Exception:
            pass
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_tflite(interp, img_tensor):
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    interp.set_tensor(in_det[0]["index"], img_tensor)
    interp.invoke()
    probs   = interp.get_tensor(out_det[0]["index"])[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx]), probs

# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY-BASED OOD CHECK
# High entropy = model is uncertain across all classes = likely not a leaf
# ─────────────────────────────────────────────────────────────────────────────
def is_out_of_distribution(probs: np.ndarray, top_conf: float) -> bool:
    if top_conf < OOD_THR:
        return True
    eps      = 1e-9
    entropy  = float(-np.sum(probs * np.log(probs + eps)))
    max_ent  = np.log(len(probs))   # ~3.64 for 38 classes
    norm_ent = entropy / max_ent    # 0=confident, 1=uniform
    return norm_ent > 0.65

# ─────────────────────────────────────────────────────────────────────────────
# OCCLUSION SALIENCY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def occlusion_heatmap(interp, img_tensor: np.ndarray, pred_idx: int, grid: int = 14) -> np.ndarray:
    H, W    = IMG_SIZE, IMG_SIZE
    patch_h = H // grid
    patch_w = W // grid
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    interp.set_tensor(in_idx, img_tensor)
    interp.invoke()
    base_score = float(interp.get_tensor(out_idx)[0][pred_idx])

    mean_val = float(img_tensor.mean())
    heatmap  = np.zeros((grid, grid), dtype=np.float32)

    for i in range(grid):
        for j in range(grid):
            occluded = img_tensor.copy()
            r0, r1 = i * patch_h, min((i + 1) * patch_h, H)
            c0, c1 = j * patch_w, min((j + 1) * patch_w, W)
            occluded[0, r0:r1, c0:c1, :] = mean_val
            interp.set_tensor(in_idx, occluded)
            interp.invoke()
            occ_score     = float(interp.get_tensor(out_idx)[0][pred_idx])
            heatmap[i, j] = max(base_score - occ_score, 0.0)

    heatmap_up = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    if heatmap_up.max() > 0:
        heatmap_up /= heatmap_up.max()
    return heatmap_up


def overlay_heatmap(orig_pil: Image.Image, heatmap: np.ndarray, alpha: float = HEATMAP_ALPHA) -> Image.Image:
    """Smooth jet heatmap blended over original — no boxes, no contours."""
    orig_np      = np.array(orig_pil.convert("RGB"))
    hmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    # smooth to look organic
    hmap_resized = cv2.GaussianBlur(hmap_resized, (31, 31), 0)
    if hmap_resized.max() > 0:
        hmap_resized /= hmap_resized.max()
    colormap      = cm.get_cmap("RdYlGn_r")  # green→yellow→red
    heatmap_color = (colormap(hmap_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr   = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)
    orig_bgr      = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    blended       = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

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
    parts   = label.split("___")
    plant   = parts[0].replace("_", " ").replace(",", "").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{plant} — {disease}" if disease else plant

def render_care_tab(label: str, confidence: float):
    info     = get_disease_info(label)
    severity = info.get("severity", "unknown")
    fg_color, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
    st.markdown(
        f"**Severity:** {severity_badge(severity)} &nbsp;&nbsp; "
        f"**Confidence:** `{confidence:.1%}`",
        unsafe_allow_html=True,
    )
    st.markdown("---")
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

# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne;font-size:2.4rem;color:#4ade80;margin-bottom:0;'>"
    "🌿 Plant Disease Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#6ee7b7;font-size:1rem;margin-top:.2rem;margin-bottom:1.4rem;'>"
    "Upload a leaf image · Get instant diagnosis · See treatment & care advice</p>",
    unsafe_allow_html=True,
)

tflite_interp = load_tflite()

# ─────────────────────────────────────────────────────────────────────────────
# ── UPLOAD ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ── PROCESS ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
orig_img = Image.open(uploaded)

with st.spinner("Analysing leaf…"):
    tensor                       = preprocess(orig_img, use_grabcut=True)
    top_idx, top_conf, all_probs = predict_tflite(tflite_interp, tensor)
    predicted_label              = CLASS_NAMES[top_idx]
    top3_idx                     = np.argsort(all_probs)[::-1][:3]
    top3                         = [(CLASS_NAMES[i], float(all_probs[i])) for i in top3_idx]

# ─────────────────────────────────────────────────────────────────────────────
# ── OOD GATE ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if is_out_of_distribution(all_probs, top_conf):
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown("**Uploaded Image**")
        st.image(orig_img, use_container_width=True)
    with col_r:
        st.markdown("""
        <div class="ood-card">
          <div class="icon">🚫</div>
          <h3>Not a recognised plant leaf</h3>
          <p>The model couldn't confidently match this image to any supported class.
          This usually means the image is not a leaf, is too blurry or dark,
          or is a crop not in the dataset.</p>
          <br>
          <p style="color:#fde68a;font-size:0.82rem;">
          Supported crops: Apple · Blueberry · Cherry · Corn · Grape ·
          Orange · Peach · Bell Pepper · Potato · Raspberry · Soybean ·
          Squash · Strawberry · Tomato</p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ── RESULTS HEADER ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
info        = get_disease_info(predicted_label)
severity    = info.get("severity", "unknown")
fg_color, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])

st.markdown("---")
st.markdown(
    f"<h2 style='font-family:Syne;color:{fg_color};margin-bottom:.2rem;'>"
    f"{format_label(predicted_label)}</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"Confidence: **`{top_conf:.1%}`** &nbsp; {severity_badge(severity)}",
    unsafe_allow_html=True,
)
if top_conf < CONFIDENCE_THR:
    st.warning(
        f"⚠️ Low confidence ({top_conf:.1%}). Try a clearer, well-lit close-up of the leaf."
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── TABS ──────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
tab_results, tab_care, tab_detection = st.tabs(
    ["📊 Results", "💊 Treatment & Care", "🔬 Detection Map"]
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
            f"border-radius:10px;padding:1rem;margin-bottom:1rem;'>"
            f"<div style='font-family:Syne;font-size:1.05rem;color:{fg_color};'>"
            f"{format_label(predicted_label)}</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{fg_color};'>{top_conf:.1%}</div>"
            f"<div style='color:#6b7280;font-size:0.85rem;'>confidence</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Top-3 Predictions**")
        for lbl, prob in top3:
            bar_w     = int(prob * 100)
            lbl_fg, _ = SEVERITY_COLORS.get(
                get_disease_info(lbl).get("severity", "unknown"),
                SEVERITY_COLORS["unknown"],
            )
            st.markdown(
                f"<div style='margin:.3rem 0;'>"
                f"<div style='font-size:.82rem;color:#9ca3af;margin-bottom:2px;'>{format_label(lbl)}</div>"
                f"<div style='display:flex;align-items:center;gap:.5rem;'>"
                f"<div style='flex:1;background:#1a2a1a;border-radius:999px;height:8px;'>"
                f"<div style='width:{bar_w}%;background:{lbl_fg};height:8px;border-radius:999px;'></div>"
                f"</div><span style='color:{lbl_fg};font-weight:700;font-size:.85rem;'>{prob:.1%}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

with tab_care:
    render_care_tab(predicted_label, top_conf)

with tab_detection:
    st.markdown("#### 🔬 Disease Detection Map")
    st.markdown(
        "<small style='color:#6b7280;'>"
        "Areas shaded red/yellow are where the model focused most when making its prediction. "
        f"Only shown for diseased leaves with confidence ≥ {GRADCAM_THR:.0%}."
        "</small>",
        unsafe_allow_html=True,
    )

    is_healthy = "healthy" in predicted_label.lower()

    if is_healthy:
        st.markdown("""
        <div style="background:#052e16;border:1px solid #22c55e;border-radius:10px;
                    padding:1.5rem;text-align:center;margin-top:.5rem;">
          <div style="font-size:2.5rem;">✅</div>
          <div style="font-family:'Syne',sans-serif;color:#4ade80;font-size:1.1rem;
                      font-weight:700;margin:.5rem 0 .3rem;">Plant looks healthy!</div>
          <div style="color:#86efac;font-size:0.88rem;">No disease regions to highlight.</div>
        </div>""", unsafe_allow_html=True)

    elif top_conf < GRADCAM_THR:
        st.markdown(
            f'<div style="background:#1c1003;border:1px solid #f59e0b;border-radius:8px;'
            f'padding:.8rem 1rem;color:#fde68a;font-size:0.88rem;">'
            f'⚠️ Confidence {top_conf:.1%} is below the {GRADCAM_THR:.0%} threshold — '
            f'heatmap not shown. Try a clearer, well-lit close-up.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Generating heatmap…"):
            try:
                tensor_raw  = preprocess(orig_img, use_grabcut=False)
                heatmap     = occlusion_heatmap(tflite_interp, tensor_raw, top_idx, grid=14)
                heatmap_img = overlay_heatmap(orig_img, heatmap)

                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("**Original**")
                    st.image(orig_img, use_container_width=True)
                with col2:
                    st.markdown("**Attention Heatmap**")
                    st.image(heatmap_img, use_container_width=True)
                st.caption("Occlusion saliency · 14×14 grid · Red = high attention · Green = low attention")
            except Exception as e:
                st.error(f"Heatmap generation failed: {e}")
