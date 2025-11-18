import streamlit as st
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    import tensorflow as tf
    import urllib.request
    
    model_path = 'alzheimer_model.h5'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from Hugging Face... (one-time, ~284MB)'):
            try:
                url = 'https://huggingface.co/muditbilala/alzheimer-detection-model/resolve/main/alzheimer_model.h5'
                urllib.request.urlretrieve(url, model_path)
                st.success('Model downloaded!')
            except Exception as e:
                st.error(f'Download failed: {e}')
                return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f'Load failed: {e}')
        return None

st.set_page_config(page_title="NeuroLens", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
.stButton>button {background-color: #667eea; color: white; border-radius: 10px; padding: 10px 24px;}
</style>
""", unsafe_allow_html=True)

classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
model = load_model()

st.title("🧠 NeuroLens - Alzheimer's Detection")
st.markdown("### AI-Powered Brain MRI Analysis")
st.markdown("---")

with st.sidebar:
    st.header("📊 System Info")
    if model:
        st.success("✅ Model Active")
    else:
        st.warning("⚠️ Loading...")
    st.info("**Accuracy:** 99.1%\n\n**Classes:** 4\n\n**Model:** CNN")
    st.markdown("---")
    st.write("**Mudit Bhargava**\nBennett University")
    st.warning("⚠️ Educational use only")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload MRI")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

with col2:
    st.subheader("🔍 Results")
    if uploaded and model:
        img = image.convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        with st.spinner("Analyzing..."):
            pred = model.predict(img_array, verbose=0)
            idx = np.argmax(pred[0])
            conf = pred[0][idx] * 100
            result = classes[idx]
        
        if result == 'Non Demented':
            st.success(f"✅ {result}")
        else:
            st.error(f"🔴 {result}")
        st.metric("Confidence", f"{conf:.2f}%")
        st.progress(conf/100)
        
        st.subheader("📊 Probabilities")
        chart = {classes[i]: float(pred[0][i]*100) for i in range(4)}
        st.bar_chart(chart)
    elif uploaded:
        st.error("Model not loaded")
    else:
        st.info("Upload MRI to begin")

st.markdown("---")
st.markdown("**NeuroLens** | Mudit Bhargava | Bennett University")
