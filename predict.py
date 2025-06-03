
import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model('best_model_cnn_lstm_0_001.h5')
labels = ['angry', 'happy', 'neutral', 'sad']

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1, 1)
    prediction = model.predict(features)
    return labels[np.argmax(prediction)]
