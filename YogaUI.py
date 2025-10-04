import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('yoga_model.h5')

LABELS = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

IMGSIZE = (160 ,160)

def prepare_image(img):
    img = img.convert("RGB").resize(IMGSIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_pose(img):
    arr = prepare_image(img)
    pred = model.predict(arr)[0]
    pred_idx = np.argmax(pred)
    return LABELS[pred_idx], float(pred[pred_idx])

st.title("Yoga Pose Prediction App")
st.write("Upload a yoga pose image to get the predicted pose and confidence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    pose, confidence = predict_pose(image)
    st.success(f"**Predicted Pose:** {pose}")
    st.info(f"**Confidence:** {confidence:.2f}")
