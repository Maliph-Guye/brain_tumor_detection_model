import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((150, 150))
    img_array = np.array(image).reshape(1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f"### 🎯 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")
