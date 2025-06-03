
import streamlit as st
from predict import predict_emotion
import tempfile
import os

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognition App")
st.write("Upload a `.wav` file and the app will predict your emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/wav')
    st.write("🎧 Analyzing your audio...")

    emotion = predict_emotion(tmp_path)
    st.success(f"🧠 Predicted Emotion: **{emotion.upper()}**")

    os.remove(tmp_path)
