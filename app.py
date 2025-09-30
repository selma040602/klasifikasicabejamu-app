import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.layers import Dropout
import time
import os
import gdown

# ------------ Definisi FixedDropout ------------
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# ------------ Mapping Model ------------

MODEL_LINKS = {
    "adam_fold2 (1).h5": "https://drive.google.com/uc?id=1Ig-2S3q0fJsP996UFRhvC3TG4i8P1Wrv",
    "model_cnn_aug.h5": "https://drive.google.com/uc?id=17kJ-clysgvI5bKbwEyGGD41_c8VzHt0P",
    "sgd_fold5.h5": "https://drive.google.com/uc?id=1kKMBnrw1dmMwHHZCjwtHkxXT4kIESJTi",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(path):
        st.info(f"📥 Downloading {model_name} dari Google Drive...")
        gdown.download(MODEL_LINKS[model_name], path, quiet=False)
    return path

# ------------ Fungsi Load Model ------------
@st.cache_resource
def load_model(model_name):
    model_path = download_model(model_name)
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "swish": tf.keras.activations.swish,
            "FixedDropout": FixedDropout
        }
    )

# ------------ UI Streamlit ------------
st.title("🌶 Klasifikasi Cabe Jamu: Segar atau Busuk")

# Dropdown pilih model
model_choice = st.selectbox(
    "Pilih model yang akan digunakan:",
    list(MODEL_LINKS.keys())
)

# Load model
model = load_model(model_choice)

# ------------ Fungsi Prediksi ------------
def predict_image(img, model):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    img_size = model.input_shape[1:3]
    img = img.resize(img_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    start_time = time.time()
    pred = model.predict(img_array)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000
    st.write(f"⏱️ Waktu inferensi: {inference_time_ms:.2f} ms")

    if pred.ndim == 1 or pred.shape[1] == 1:  # sigmoid
        prob = float(pred[0] if pred.ndim == 1 else pred[0][0])
        if prob > 0.5:
            st.success(f"Hasil: *Segar* 🌶 (Probabilitas: {prob*100:.2f}%)")
        else:
            st.error(f"Hasil: *Busuk* 🤢 (Probabilitas: {(1-prob)*100:.2f}%)")
    else:  # softmax
        label = ["Busuk", "Segar"]
        kelas = np.argmax(pred, axis=1)[0]
        st.success(f"Hasil: *{label[kelas]}* (Probabilitas: {pred[0][kelas]*100:.2f}%)")

# ------------ Upload Gambar ------------
st.subheader("📂 Upload Gambar Cabai")
uploaded_file = st.file_uploader("Upload gambar cabe...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar dari upload", use_container_width=True)
    predict_image(img, model)

# ------------ Kamera HP ------------
st.subheader("📷 Ambil Gambar dari Kamera HP")
camera_image = st.camera_input("Gunakan kamera HP untuk foto cabai")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")
    st.image(img, caption="Gambar dari kamera HP", use_container_width=True)
    predict_image(img, model)

