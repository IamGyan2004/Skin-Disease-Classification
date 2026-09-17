"""Streamlit interface for educational HAM10000 predictions."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/skin_disease_model.keras")
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="Skin Disease Classification", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #020d1a;
            color: #edf3ff;
        }
        .main .block-container {
            max-width: 980px;
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }
        h1 {
            color: #f7fafc !important;
            font-size: 2.75rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: -0.05em;
        }
        .caption {
            color: #aab8c8 !important;
            font-size: 0.96rem !important;
            margin-bottom: 1rem !important;
        }
        [data-testid="stFileUploader"] {
            width: 100%;
            max-width: 980px;
            margin: 0 auto;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 1px solid rgba(126, 151, 180, 0.9);
            border-radius: 12px 12px 0 0;
            background: rgba(12, 23, 35, 0.9);
            min-height: 90px;
            padding: 0.4rem 0.7rem;
            border-bottom: 1px dashed rgba(142, 167, 193, 0.75);
        }
        [data-testid="stFileUploaderDropzone"] > div {
            border-radius: 10px;
            background: rgba(8, 18, 30, 0.85);
            min-height: 70px;
        }
        [data-testid="stFileUploaderDropzone"] label,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] button {
            display: none !important;
        }
        .uploaded-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            width: 100%;
            max-width: 980px;
            margin: 0 auto;
            background: rgba(12, 23, 35, 0.9);
            border: 1px solid rgba(126, 151, 180, 0.9);
            border-top: none;
            border-radius: 0 0 12px 12px;
            min-height: 82px;
            padding: 0.5rem 0.8rem;
            box-sizing: border-box;
        }
        .thumb-box {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: rgba(255,255,255,0.02);
            border-radius: 10px;
            padding: 0.2rem 0.4rem;
            min-width: 185px;
        }
        .thumb-box img {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            object-fit: cover;
            display: block;
        }
        .thumb-name {
            font-size: 0.88rem;
            font-weight: 600;
            color: #edf3ff;
        }
        .thumb-size {
            font-size: 0.7rem;
            color: #a6b7cf;
        }
        .close-mini {
            margin-left: auto;
            color: #dfe9f7;
            font-size: 1.2rem;
            line-height: 1;
        }
        .plus-btn {
            margin-left: auto;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            border: 1px solid rgba(173, 189, 207, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #edf3ff;
            font-size: 1.8rem;
            line-height: 1;
        }
        .preview-wrap {
            width: 100%;
            max-width: 980px;
            margin: 1rem auto 0 auto;
        }
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .preview-grid > div {
            border-radius: 12px;
            overflow: hidden;
            background: rgba(255,255,255,0.02);
        }
        .preview-grid img {
            width: 100% !important;
            height: 290px !important;
            object-fit: cover;
            border-radius: 12px;
            display: block;
        }
        .uploaded-label {
            font-size: 0.9rem;
            color: #b8c8d9;
            text-align: center;
            width: 100%;
            max-width: 980px;
            margin: 0.5rem auto 0 auto;
        }
        .result-card {
            width: 100%;
            max-width: 980px;
            margin: 1.1rem auto 0 auto;
            background: rgba(17, 34, 54, 0.9);
            border: 1px solid rgba(126, 145, 178, 0.35);
            border-radius: 12px;
            padding: 1rem 1.5rem 1.1rem 1.5rem;
            box-sizing: border-box;
        }
        .result-title {
            color: #eef4ff;
            font-size: clamp(2.2rem, 3vw, 3.1rem) !important;
            font-weight: 700 !important;
            margin: 0 !important;
            line-height: 1.05 !important;
        }
        .result-subtitle {
            color: #dfeaf6;
            font-size: 1.05rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0 !important;
        }
        .warning-box {
            width: 100%;
            max-width: 980px;
            margin: 1rem auto 0 auto;
            border-radius: 10px;
            background: rgba(140, 164, 100, 0.18);
            border: 1px solid rgba(154, 183, 116, 0.8);
            color: #def0d6;
            font-size: 1.05rem;
            padding: 0.9rem 1rem;
            box-sizing: border-box;
        }
        .stFileUploader > section > div { display: block !important; }
        .stFileUploader > section { padding: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Skin Disease Classification")
st.caption("Educational/research classification model. Not a medical diagnosis tool.")

uploaded_files = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    thumb_cols = st.columns(min(len(uploaded_files), 4))
    for idx, uploaded_file in enumerate(uploaded_files[:4]):
        with thumb_cols[idx]:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=52)
            st.caption(f"{uploaded_file.name}")
            st.caption(f"{max(1, uploaded_file.size // 1024)}KB")
    st.markdown('<div class="plus-btn" style="margin-top: 0.5rem;">+</div>', unsafe_allow_html=True)

    images = [Image.open(uploaded_file).convert("RGB") for uploaded_file in uploaded_files[:4]]
    st.markdown('<div class="preview-wrap"><div class="preview-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, image in enumerate(images):
        with cols[idx % 2]:
            st.image(image, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="uploaded-label">Uploaded image</div>', unsafe_allow_html=True)

    selected_image = images[0]
    tensor = tf.image.resize(np.asarray(selected_image), IMAGE_SIZE)[None, ...]
    probabilities = load_model().predict(tensor, verbose=0)[0]
    index = int(np.argmax(probabilities))

    st.markdown(
        """
        <div class='result-card'>
            <div class='result-title'>Predicted class: {label}</div>
            <div class='result-subtitle'>Model confidence: {confidence}</div>
        </div>
        """.format(label=CLASS_NAMES[index], confidence=f"{probabilities[index]:.2%}"),
        unsafe_allow_html=True,
    )
    st.markdown("<div class='warning-box'>This result is for research and education only and must not guide medical decisions.</div>", unsafe_allow_html=True)
