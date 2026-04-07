import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from disease_info import get_disease_info

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 128
TFLITE_MODEL   = "plant_model_quant.tflite"
CONFIDENCE_THR = 0.75
GRADCAM_THR    = 0.75

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
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

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
  .gradcam-warning {
      background: #1c1003; border: 1px solid #f59e0b; border-radius: 8px;
      padding: .8rem 1rem; color: #fde68a; font-size: 0.88rem;
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
        img_rgb = grabcut_segment(img_rgb)
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
# OCCLUSION SALIENCY MAP
# Works with any TFLite model — no Keras / best_model.keras needed.
# Occludes patches one by one, measures how much the predicted score drops.
# Score drop = importance of that region → heatmap.
# ─────────────────────────────────────────────────────────────────────────────
def occlusion_heatmap(interp, img_tensor: np.ndarray, pred_idx: int, grid: int = 8) -> np.ndarray:
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
            occ_score         = float(interp.get_tensor(out_idx)[0][pred_idx])
            heatmap[i, j]     = max(base_score - occ_score, 0.0)

    heatmap_up = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    if heatmap_up.max() > 0:
        heatmap_up /= heatmap_up.max()
    return heatmap_up


def overlay_heatmap(orig_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    orig_np       = np.array(orig_pil.convert("RGB"))
    hmap_resized  = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    colormap      = cm.get_cmap("jet")
    heatmap_color = (colormap(hmap_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr   = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)
    orig_bgr      = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    blended       = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def draw_detection_boxes(orig_pil: Image.Image, heatmap: np.ndarray, threshold: float = 0.5) -> Image.Image:
    orig_np      = np.array(orig_pil.convert("RGB")).copy()
    hmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))

    # noise floor: only draw boxes on regions that are meaningfully above baseline
    # prevents shadow/lighting artifacts from generating false boxes
    noise_floor  = np.percentile(hmap_resized, 75)   # top 25% of activations only
    effective_thr = max(threshold, noise_floor)

    binary       = (hmap_resized >= effective_thr).astype(np.uint8) * 255
    contours, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_bgr   = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    img_area     = orig_np.shape[0] * orig_np.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # skip tiny noise regions AND boxes that cover most of the image (background noise)
        if area < 100 or area > img_area * 0.4:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_bgr, (x, y), (x + w, y + h), (0, 255, 80), 2)
        cv2.putText(
            result_bgr, "Disease Region",
            (x, max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA,
        )

    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

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

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT & CARE TAB
# ─────────────────────────────────────────────────────────────────────────────
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
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne;font-size:2.4rem;color:#4ade80;margin-bottom:0;'>"
    "🌿 Plant Disease Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#6ee7b7;font-size:1rem;margin-top:.2rem;'>"
    "Upload a leaf image · Get diagnosis · See treatment & care advice</p>",
    unsafe_allow_html=True,
)

tflite_interp = load_tflite()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_grabcut   = st.toggle("Background removal", value=True)
    gradcam_alpha = st.slider("Heatmap intensity", 0.2, 0.8, 0.45, 0.05)
    gradcam_grid  = st.select_slider(
        "Detection detail",
        options=[8, 12, 16],
        value=8,
        help="8 = fast · 16 = finer boxes but slower",
    )
    show_top3     = st.toggle("Show top-3 predictions", value=True)
    box_thresh    = st.slider("Detection box sensitivity", 0.3, 0.8, 0.5, 0.05)
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280;'>Built by altairis-labs<br>"
        "14 crops · 38 classes · TFLite</small>",
        unsafe_allow_html=True,
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a leaf image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="border:2px dashed #1a3a1a;border-radius:16px;padding:3rem;
                text-align:center;color:#4b6b4b;margin-top:1rem;">
      <div style="font-size:3rem;">🍃</div>
      <p style="font-family:Syne;font-size:1.1rem;margin:.5rem 0 0;">
        Drop a leaf image to get started</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
orig_img = Image.open(uploaded)

with st.spinner("Analysing leaf..."):
    tensor           = preprocess(orig_img, use_grabcut=use_grabcut)
    top_idx, top_conf, all_probs = predict_tflite(tflite_interp, tensor)
    predicted_label  = CLASS_NAMES[top_idx]
    top3_idx         = np.argsort(all_probs)[::-1][:3]
    top3             = [(CLASS_NAMES[i], float(all_probs[i])) for i in top3_idx]

    gradcam_img    = None
    detected_img   = None
    gradcam_reason = None
    is_healthy     = "healthy" in predicted_label.lower()

    if is_healthy:
        gradcam_reason = "healthy"
    elif top_conf < GRADCAM_THR:
        gradcam_reason = (
            f"Confidence {top_conf:.1%} is below the {GRADCAM_THR:.0%} threshold. "
            "Try a clearer, well-lit, close-up photo of the leaf."
        )
    else:
        try:
            # No GrabCut for detection — avoids stripping disease regions at edges
            tensor_gc    = preprocess(orig_img, use_grabcut=False)
            heatmap      = occlusion_heatmap(tflite_interp, tensor_gc, top_idx, grid=gradcam_grid)
            gradcam_img  = overlay_heatmap(orig_img, heatmap, alpha=gradcam_alpha)
            detected_img = draw_detection_boxes(orig_img, heatmap, threshold=box_thresh)
        except Exception as e:
            gradcam_reason = f"Detection map failed: {e}"

# ── Results header ────────────────────────────────────────────────────────────
info        = get_disease_info(predicted_label)
severity    = info.get("severity", "unknown")
fg_color, _ = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])

st.markdown("---")
st.markdown(
    f"<h2 style='font-family:Syne;color:{fg_color};'>{format_label(predicted_label)}</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"Confidence: **`{top_conf:.1%}`** &nbsp; {severity_badge(severity)}",
    unsafe_allow_html=True,
)
if top_conf < CONFIDENCE_THR:
    st.warning(
        f"⚠️ Low confidence ({top_conf:.1%}). Top-3 shown. "
        "Try a clearer, well-lit close-up of the leaf."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_results, tab_care, tab_detection = st.tabs(
    ["📊 Results", "💊 Treatment & Care", "🔬 Detection Map"]
)

# ── Tab 1: Results ─────────────────────────────────────────────────────────────
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
        if show_top3:
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

# ── Tab 2: Treatment & Care ────────────────────────────────────────────────────
with tab_care:
    render_care_tab(predicted_label, top_conf)

# ── Tab 3: Detection Map ───────────────────────────────────────────────────────
with tab_detection:
    st.markdown("#### 🔬 Disease Detection Map")
    st.markdown(
        "<small style='color:#6b7280;'>"
        "Heatmap: red/yellow = regions the model focused on most. "
        "Detected Regions: green boxes mark likely disease areas on the original image. "
        f"Shown when confidence ≥ {GRADCAM_THR:.0%}."
        "</small>",
        unsafe_allow_html=True,
    )

    if gradcam_reason == "healthy":
        st.markdown("""
        <div style="background:#052e16;border:1px solid #22c55e;border-radius:10px;
                    padding:1.5rem;text-align:center;">
          <div style="font-size:2.5rem;">✅</div>
          <div style="font-family:'Syne',sans-serif;color:#4ade80;font-size:1.1rem;
                      font-weight:700;margin:.5rem 0 .3rem;">Plant looks healthy!</div>
          <div style="color:#86efac;font-size:0.88rem;">
            No disease regions to highlight — detection map is only shown for diseased predictions.
          </div>
        </div>""", unsafe_allow_html=True)
    elif gradcam_img is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original**")
            st.image(orig_img, use_container_width=True)
        with col2:
            st.markdown("**Heatmap Overlay**")
            st.image(gradcam_img, use_container_width=True)
        with col3:
            st.markdown("**Detected Regions**")
            st.image(detected_img, use_container_width=True)
        st.caption(
            f"Occlusion saliency · Grid {gradcam_grid}×{gradcam_grid} · "
            f"Box sensitivity {box_thresh:.0%} · GrabCut OFF for detection accuracy"
        )
    else:
        st.markdown(
            f'<div class="gradcam-warning">⚠️ {gradcam_reason}</div>',
            unsafe_allow_html=True,
        )
