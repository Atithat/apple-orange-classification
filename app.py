"""FruitLens: apple vs orange CNN image classification demo."""
from pathlib import Path
import json
import streamlit as st
from PIL import Image, UnidentifiedImageError
from inference import predict_fruit

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "apple_orange_cnn.keras"
EVAL_PATH = BASE / "outputs" / "evaluation.json"
FIG = BASE / "outputs" / "figures"
EXAMPLES = BASE / "examples"

st.set_page_config(page_title="FruitLens · Apple or Orange", page_icon="🍎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
[data-testid="stAppViewContainer"] {background:#fbfcf7;}
[data-testid="stSidebar"] {background:#f1f5eb;}
.block-container {max-width:1140px;padding-top:1.8rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Outfit',sans-serif !important;color:#263328;}
p,span,label,div {font-family:'DM Sans',sans-serif;}
.hero {background:linear-gradient(120deg,#fff0d9,#fff9ed 47%,#fbece4);border:1px solid #f3dfce;
border-radius:24px;padding:32px 39px;margin-bottom:22px;}
.eyebrow {font-size:.79rem;letter-spacing:.16em;color:#b0672c;font-weight:800;}
.hero h1 {font-size:3.2rem;line-height:1.12;margin:12px 0;color:#253528;}
.hero p {max-width:730px;color:#69756a;font-size:1.06rem;}
.pill {display:inline-block;background:#fff;border:1px solid #f0dfca;border-radius:100px;
padding:7px 12px;font-size:.78rem;color:#976226;font-weight:700;margin-right:8px;margin-top:14px;}
.result {border:1px solid #e6eade;background:#fff;border-radius:18px;padding:28px 30px;margin-top:12px;}
.result-name {font-size:2.9rem;font-weight:800;color:#263328;}
.small-cap {font-size:.78rem;letter-spacing:.08em;color:#8c937c;font-weight:700;}
[data-testid="stFileUploader"] section {border:2px dashed #bfd2b5;border-radius:16px;background:white;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained CNN...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def read_metrics():
    if EVAL_PATH.exists():
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    return None

st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ COMPUTER VISION / IMAGE CLASSIFICATION</div>
  <h1>One photo. Two fruits.<br><span style="color:#d4802e">Smart classification.</span></h1>
  <p>Upload an image or take a photo to see whether a trained convolutional neural
  network identifies an apple or an orange.</p>
  <span class="pill">🍎 APPLE</span><span class="pill">🍊 ORANGE</span>
  <span class="pill">🧠 CNN + TENSORFLOW</span>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.warning("The trained model has not been generated or uploaded yet. No predictions are fabricated.")
    st.markdown("### Train the classifier before demonstrating the site")
    st.markdown("Download the repository, install its requirements and run:")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.info("Training downloads images from Fruits-360 once, then produces a Keras model and measured results. "
            "The app becomes interactive when models/apple_orange_cnn.keras is present.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("Cannot open the trained model: " + str(error))
    st.stop()

metrics = read_metrics()
tab_predict, tab_results, tab_method = st.tabs([
    "🍎 Classify an image", "📊 Training results", "🧠 How the model works"
])

with tab_predict:
    st.subheader("Try the classifier")
    left, right = st.columns([1, 1], gap="large")
    image = None
    with left:
        source = st.radio("Image source", ["Upload photo", "Camera", "Held-out example"],
                          horizontal=True)
        if source == "Upload photo":
            item = st.file_uploader("Select a JPG or PNG", type=["jpg", "jpeg", "png"])
        elif source == "Camera":
            item = st.camera_input("Take an apple or orange photo")
        else:
            samples = sorted(EXAMPLES.glob("test_*.jpg"))
            item = st.selectbox("Choose a sample", samples,
                                format_func=lambda path: path.stem.replace("_", " ").title()) if samples else None
            if not samples:
                st.info("Samples are created after training.")
        if item is not None:
            try:
                image = Image.open(item).convert("RGB")
                st.image(image, caption="Input image", use_container_width=True)
            except (UnidentifiedImageError, OSError) as error:
                st.error("Unable to read image: " + str(error))
    with right:
        if image is not None:
            with st.spinner("Running inference..."):
                result = predict_fruit(image, model)
            st.markdown(
                "<div class='result'><div class='small-cap'>MODEL PREDICTION</div>"
                "<div class='result-name'>" + ("🍎 " if result["label"] == "Apple" else "🍊 ") +
                result["label"] + "</div><p>Estimated confidence: <b>" +
                ("%.1f%%" % (100 * result["confidence"])) + "</b></p></div>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Predicted class probabilities")
            for label, score in result["scores"].items():
                st.write(("🍎" if label == "Apple" else "🍊") + " " + label)
                st.progress(float(score), text="%.1f%%" % (100 * score))
            if result["confidence"] < 0.7:
                st.warning("Lower confidence. Try a single fruit with good lighting.")
            else:
                st.success("Classification complete.")
            st.caption("This is a binary classifier, not a general object detector. "
                       "Non-fruit images will still receive a label.")
        else:
            st.markdown("<div class='result'><div class='small-cap'>READY TO CLASSIFY</div>"
                        "<h3>Your prediction will appear here</h3>"
                        "<p>Upload, capture, or select a sample image to begin.</p></div>",
                        unsafe_allow_html=True)

with tab_results:
    st.subheader("Training and held-out evaluation")
    if metrics is None:
        st.info("The training run will save real metrics to outputs/evaluation.json.")
    else:
        cols = st.columns(4)
        cols[0].metric("Training images", metrics["train_images"])
        cols[1].metric("Validation images", metrics["validation_images"])
        cols[2].metric("Test images", metrics["test_images"])
        cols[3].metric("Test accuracy", "%.1f%%" % (100 * metrics["test_accuracy"]))
        st.caption("Adam optimizer · %d completed epochs · 128×128 RGB · %s parameters" %
                   (metrics["completed_epochs"], format(metrics["model_parameters"], ",")))
        for name, title in [
            ("training_history.png", "Training and validation accuracy / loss"),
            ("confusion_matrix.png", "Held-out test confusion matrix"),
            ("four_test_predictions.png", "Four different held-out test images"),
        ]:
            path = FIG / name
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)
        st.caption("Validation ratio: %.1f%% of training image count." %
                   (100 * metrics["validation_images"] / metrics["train_images"]))

with tab_method:
    st.subheader("How FruitLens works")
    st.markdown("""
    1. **Data:** Fruits-360 provides apple and orange images. The official Test folders
       are held out entirely from the training/validation folders.
    2. **Preprocessing:** Resize to 128×128 RGB; the trained CNN normalizes pixels to 0–1.
    3. **Architecture:** 3 Conv2D layers with 24, 48 and 96 filters and pooling,
       then GlobalAveragePooling, 64 hidden neurons and a sigmoid output.
    4. **Training:** Adam optimizer, binary cross-entropy, up to 12 epochs,
       early stopping and mild data augmentation.
    5. **Inference:** The sigmoid is the estimated probability of Orange.
       Apple probability equals 1 minus that output.
    """)
    st.info("Fruits-360 photographs typically have plain backgrounds. Results may be less "
            "reliable for complex real-world photos, occluded fruit or different lighting.")
    st.caption("Data credit: Fruits-360 (Mihai Oltean et al.), CC BY-SA 4.0. "
               "https://github.com/fruits-360/fruits-360-100x100")

st.divider()
st.caption("FruitLens · Educational classification project · Streamlit + TensorFlow/Keras")
