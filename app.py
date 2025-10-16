import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.layers import Dropout
import time
import os
import gdown
import cv2

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

# ------------ FUNGSI PENINGKATAN KUALITAS GAMBAR ------------
def preprocess_image_advanced(img):
    """
    Preprocessing gambar dengan teknik maksimal untuk meningkatkan akurasi
    """
    # Convert PIL to numpy array
    img_array = np.array(img)
    
    # 1. Denoising - Mengurangi noise
    img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL
    img = Image.fromarray(img_array)
    
    # 3. Enhance Sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # 4. Enhance Color
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.3)
    
    # 5. Enhance Contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # 6. Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img


def preprocess_image_moderate(img):
    """
    Preprocessing sedang - lebih seimbang
    """
    # Convert to array for OpenCV processing
    img_array = np.array(img)
    
    # 1. Bilateral Filter
    img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # 2. CLAHE dengan parameter lebih soft
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    img = Image.fromarray(img_array)
    
    # 3. Enhancement PIL
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    return img

# ------------ Fungsi Prediksi ------------
def predict_image(img, model, preprocessing_level):
    # Pilih level preprocessing
    if preprocessing_level == "Maksimal":
        img_processed = preprocess_image_advanced(img)
    elif preprocessing_level == "Sedang":
        img_processed = preprocess_image_moderate(img)
    else:  # Original
        img_processed = img.copy()
        enhancer = ImageEnhance.Contrast(img_processed)
        img_processed = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(img_processed)
        img_processed = enhancer.enhance(1.5)
    
    # Tampilkan gambar original saja
    st.image(img, caption="Gambar Input", use_container_width=True)
    
    # Preprocessing untuk model
    img_size = model.input_shape[1:3]
    img_resized = img_processed.resize(img_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi + Hitung Waktu
    start_time = time.time()
    pred = model.predict(img_array)
    end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    st.write(f"⏱️ Waktu inferensi: {inference_time_ms:.2f} ms")
    
    # Hasil Prediksi
    if pred.ndim == 1 or pred.shape[1] == 1:  # sigmoid
        prob = float(pred[0] if pred.ndim == 1 else pred[0][0])
        if prob > 0.5:
            st.success(f"✅ Hasil: **Segar** 🌶 (Probabilitas: {prob*100:.2f}%)")
        else:
            st.error(f"❌ Hasil: **Busuk** 🤢 (Probabilitas: {(1-prob)*100:.2f}%)")
    else:  # softmax
        label = ["Busuk", "Segar"]
        kelas = np.argmax(pred, axis=1)[0]
        prob_busuk = pred[0][0] * 100
        prob_segar = pred[0][1] * 100
        
        if kelas == 1:
            st.success(f"✅ Hasil: **{label[kelas]}** 🌶")
        else:
            st.error(f"❌ Hasil: **{label[kelas]}** 🤢")
        
        # Tampilkan detail probabilitas
        st.write(f"📊 Detail Probabilitas:")
        st.write(f"- Busuk: {prob_busuk:.2f}%")
        st.write(f"- Segar: {prob_segar:.2f}%")

# ------------ UI Streamlit ------------
st.title("🌶 Klasifikasi Cabe Jamu: Segar atau Busuk")
st.write("**Enhanced Version** dengan preprocessing maksimal untuk akurasi lebih tinggi")

# Dropdown pilih model
model_choice = st.selectbox(
    "Pilih model yang akan digunakan:",
    list(MODEL_LINKS.keys())
)

# Pilih level preprocessing
preprocessing_level = st.radio(
    "Pilih Level Preprocessing:",
    ["Maksimal", "Sedang", "Original"]
)

# Load model
model = load_model(model_choice)

st.write("---")

# ------------ Upload Gambar ------------
st.subheader("📂 Upload Gambar Cabai")
uploaded_file = st.file_uploader("Upload gambar cabe...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    predict_image(img, model, preprocessing_level)

st.write("---")

# ------------ Kamera HP ------------
st.subheader("📷 Ambil Gambar dari Kamera HP")
camera_image = st.camera_input("Gunakan kamera HP untuk foto cabai")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")
    predict_image(img, model, preprocessing_level)

st.write("---")
st.info("""
💡 **Tips untuk hasil terbaik:**
1. Gunakan pencahayaan yang baik dan merata
2. Fokus pada objek cabai, hindari background yang ramai
3. Ambil foto dari jarak yang cukup dekat
4. Pastikan cabai terlihat jelas, tidak blur
5. Coba berbagai level preprocessing untuk gambar Anda
""")
