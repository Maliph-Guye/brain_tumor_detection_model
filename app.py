import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for sleek UI
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px dashed #667eea;
        text-align: center;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-text {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .info-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Tumor information
tumor_info = {
    'glioma': 'A tumor that begins in the glial cells of the brain or spinal cord. Most common type of brain tumor.',
    'meningioma': 'A tumor that forms in the meninges (protective membranes covering the brain and spinal cord). Usually benign.',
    'notumor': 'No tumor detected in the MRI scan. Brain appears normal.',
    'pituitary': 'A tumor in the pituitary gland. Often affects hormone production and can cause various symptoms.'
}

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🧠 Brain Tumor Detection System</h1>
        <p class="header-subtitle">Advanced AI-powered MRI Analysis for Tumor Classification</p>
    </div>
""", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Image")
    st.markdown("""
        <div class="info-card">
            <div class="info-title">Supported Formats</div>
            JPG, JPEG, PNG
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if uploaded_file is not None:
        with st.spinner('🔍 Analyzing MRI scan...'):
            # Preprocess the image
            image_resized = image.resize((150, 150))
            img_array = np.array(image_resized).reshape(1, 150, 150, 3)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)
            predicted_class = labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display results
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">🎯 Diagnosis Result</div>
                    <div class="prediction-text">{predicted_class}</div>
                    <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(confidence / 100)
            
            # Tumor information
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-title">ℹ️ About {predicted_class.title()}</div>
                    <p style="margin: 0; color: #555;">{tumor_info[predicted_class]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # All predictions
            with st.expander("📈 View All Predictions"):
                for i, label in enumerate(labels):
                    conf = prediction[0][i] * 100
                    st.write(f"**{label.title()}:** {conf:.2f}%")
                    st.progress(conf / 100)
    else:
        st.info("👆 Please upload an MRI image to begin analysis")

# Footer information
st.markdown("---")
st.markdown("""
    <div class="info-card">
        <div class="info-title">⚠️ Medical Disclaimer</div>
        <p style="margin: 0; color: #555;">
            This tool is for educational and research purposes only. Always consult with qualified healthcare 
            professionals for medical diagnosis and treatment decisions. AI predictions should not replace 
            professional medical advice.
        </p>
    </div>
""", unsafe_allow_html=True)