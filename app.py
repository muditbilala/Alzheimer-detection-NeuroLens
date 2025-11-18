import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="NeuroLens", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
.stButton>button {
    background-color: #667eea;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('alzheimer_model.h5')
    except:
        st.warning("Model file not found. Add alzheimer_model.h5 to folder.")
        return None

model = load_model()
classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

st.title("�� NeuroLens - Alzheimer's Detection")
st.markdown("### AI-Powered Brain MRI Analysis")
st.markdown("---")

with st.sidebar:
    st.header("📊 System Info")
    st.info("**Accuracy:** 99.1%\n\n**Classes:** 4\n\n**Model:** CNN")
    st.markdown("---")
    st.write("**Mudit Bhargava**")
    st.write("Bennett University")
    st.warning("Educational use only")

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
        st.metric("Confidence", f"{conf:.1f}%")
        
        st.subheader("📊 Probabilities")
        chart = {classes[i]: pred[0][i]*100 for i in range(4)}
        st.bar_chart(chart)
    elif uploaded:
        st.error("Model not loaded")
    else:
        st.info("Upload MRI to begin")

st.markdown("---")
st.markdown("**NeuroLens** | Mudit Bhargava | Bennett University")
