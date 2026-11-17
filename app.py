from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------- CONFIG --------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "potato_cnn_model.keras"
CLASS_PATH = APP_DIR / "model" / "classification_report.txt"
IMG_SIZE = (128, 128)  # Same input size used while training

st.set_page_config(
    page_title="LeafLens | Potato Leaf Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- THEME / CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --cream: #F6F3E9;
    --cream2: #FFFEF9;
    --green: #234D35;
    --green2: #3F7652;
    --sage: #DCE8D8;
    --ink: #20352A;
    --muted: #718074;
}
.stApp { background: var(--cream); color: var(--ink); font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.brand { display:flex; align-items:center; gap:10px; color:var(--green); font-weight:800; font-size:1.1rem; letter-spacing:-.3px; }
.brand-icon { background:var(--green); color:#fff; border-radius:13px; width:38px; height:38px; display:flex; align-items:center; justify-content:center; font-size:20px; }
.hero { background:var(--green); border-radius:25px; padding:36px 40px; color:#fff; margin-top:20px; position:relative; overflow:hidden; }
.hero:after { content:'✳'; position:absolute; right:48px; top:-42px; font-size:230px; color:rgba(255,255,255,.06); }
.eyebrow { color:#C9DCCB; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-family:'Playfair Display',serif; font-size: clamp(2rem,4vw,3.25rem); line-height:1.12; margin:12px 0; color:#fff; }
.hero p { color:#E1EADF; font-size:15px; max-width:610px; margin-bottom:0; line-height:1.7; }
.section-title { font-family:'Playfair Display',serif; color:var(--green); font-size:1.65rem; margin:28px 0 5px; }
.subtle { color:var(--muted); font-size:14px; }
.panel { background:var(--cream2); border:1px solid #E7E8DC; border-radius:20px; padding:23px; box-shadow:0 8px 28px rgba(35,77,53,.045); }
div[data-testid="stFileUploader"] { background:#F8FAF4; border:1.5px dashed #A8BCA8; border-radius:16px; padding:14px; }
div[data-testid="stFileUploader"] section { border:0; }
.stButton > button { background:var(--green); color:#fff; border:0; border-radius:12px; padding: .65rem 1.3rem; font-weight:700; width:100%; }
.stButton > button:hover { background:var(--green2); color:#fff; border:0; }
.metric-card { background:#F0F5EC; border-radius:16px; padding:17px 19px; }
.metric-label { color:#718074; font-size:12px; text-transform:uppercase; letter-spacing:1px; font-weight:700; }
.metric-value { color:var(--green); font-size:1.75rem; font-weight:800; margin-top:5px; }
.result-name { font-family:'Playfair Display',serif; color:var(--green); font-size:1.8rem; margin:4px 0; }
.pill { display:inline-block; background:#E3EEDF; color:#315B3E; border-radius:99px; padding:5px 11px; font-size:12px; font-weight:700; }
.tip { background:#EEF3E9; border-left:4px solid #6D956F; border-radius:9px; padding:13px 15px; color:#425B48; font-size:13px; line-height:1.6; }
.footer { text-align:center; color:#879287; font-size:12px; padding-top:30px; }
hr { border-color:#E3E6DA; }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_class_names():
    # In the supplied notebook, classification_report.txt contains one class name per line.
    names = [line.strip() for line in CLASS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names

def pretty_label(label):
    return label.replace("potato__", "").replace("_", " ").strip().title()

# -------------------- HEADER --------------------
st.markdown("""
<div class="brand"><div class="brand-icon">🌿</div> LeafLens <span style="font-weight:500;color:#879287;font-size:13px;margin-left:2px;">POTATO LEAF CLASSIFIER</span></div>
<div class="hero">
  <div class="eyebrow">AI-powered plant health assistant</div>
  <h1>Understand your leaf.<br>Grow with confidence.</h1>
  <p>Upload a clear photograph of a potato leaf and let the trained CNN model identify the predicted class in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Check required files early, with a helpful message.
missing = []
if not MODEL_PATH.exists():
    missing.append("model/potato_cnn_model.keras")
if not CLASS_PATH.exists():
    missing.append("model/classification_report.txt")

if missing:
    st.error("Model files are missing. Add the following files to the project folder:")
    for item in missing:
        st.code(item)
    st.info("See README.md for the exact setup instructions. The notebook's classification_report.txt should contain the class names, one per line.")
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(f"Could not load the model or class names: {exc}")
    st.stop()

if not class_names:
    st.error("No class names found in model/classification_report.txt.")
    st.stop()

# -------------------- MAIN SINGLE-PAGE UI --------------------
st.markdown('<div class="section-title">Leaf disease detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Choose an image to analyse. For best results, use a well-lit, focused photo with one leaf clearly visible.</div>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.02, 0.98], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 01 · Upload leaf image")
    uploaded_file = st.file_uploader(
        "Drop an image here or browse files",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please upload a valid JPG or PNG file.")
            image = None
    else:
        image = None
        st.markdown("""
        <div style="text-align:center;padding:35px 12px 24px;color:#829082;">
          <div style="font-size:48px;">🍃</div>
          <div style="font-weight:700;color:#42634A;margin-top:8px;">Your leaf image goes here</div>
          <div style="font-size:12px;margin-top:5px;">Clear photo · Good lighting · Minimal background</div>
        </div>
        """, unsafe_allow_html=True)
    predict_clicked = st.button("Analyse leaf  →", disabled=(image is None), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 02 · Prediction")
    if predict_clicked and image is not None:
        try:
            # Training notebook uses RGB images resized to 128x128.
            # The model itself contains Rescaling(1/255), so do NOT divide by 255 here.
            resized = image.resize(IMG_SIZE)
            input_array = np.asarray(resized, dtype=np.float32)
            input_array = np.expand_dims(input_array, axis=0)

            scores = model.predict(input_array, verbose=0)[0]
            if len(scores) != len(class_names):
                st.error(
                    f"Output mismatch: model returned {len(scores)} scores, "
                    f"but {len(class_names)} class names were found. "
                    "Check the order and contents of classification_report.txt."
                )
            else:
                best_index = int(np.argmax(scores))
                confidence = float(scores[best_index]) * 100
                predicted = pretty_label(class_names[best_index])

                st.markdown('<span class="pill">PREDICTION COMPLETE</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-name">{predicted}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="subtle">Model confidence for the top predicted class</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown("**Class-wise probability**")
                for label, score in sorted(zip(class_names, scores), key=lambda x: x[1], reverse=True):
                    st.progress(float(score), text=f"{pretty_label(label)} · {float(score)*100:.1f}%")
                st.markdown("""
                <div class="tip"><b>Note:</b> This is an AI model prediction, not a confirmed agricultural diagnosis. If the plant appears unhealthy, compare with trusted local agricultural guidance.</div>
                """, unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:46px 12px;color:#829082;">
          <div style="font-size:45px;">✦</div>
          <div style="font-weight:700;color:#42634A;margin-top:10px;">Your result will appear here</div>
          <div style="font-size:12px;margin-top:6px;">Upload a leaf image and select Analyse leaf.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
for col, number, title, desc in [
    (c1, "01", "Upload", "Select a clear potato leaf image in JPG or PNG format."),
    (c2, "02", "Analyse", "The CNN processes the image at 128 × 128 pixels."),
    (c3, "03", "View result", "See the predicted class and class-wise confidence scores."),
]:
    with col:
        st.markdown(f"""
        <div class="panel" style="height:150px;">
          <div style="color:#7C9B7B;font-weight:800;font-size:12px;letter-spacing:1px;">STEP {number}</div>
          <div style="font-family:'Playfair Display',serif;color:#234D35;font-size:21px;margin:8px 0;">{title}</div>
          <div class="subtle">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">LeafLens · CNN-based Potato Leaf Classification · For educational use</div>', unsafe_allow_html=True)
