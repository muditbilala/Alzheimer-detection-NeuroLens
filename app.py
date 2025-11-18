import streamlit as st
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    import tensorflow as tf
    import gdown
    
    model_path = 'alzheimer_model.h5'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from Google Drive... (one-time, ~284MB)'):
            try:
                file_id = '1hfO3Q-R0rK7hpRHhubrw4JPbv2D_7I-Q'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, model_path, quiet=False)
                st.success('✅ Model downloaded successfully!')
            except Exception as e:
                st.error(f'❌ Download failed: {e}')
                return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f'❌ Load failed: {e}')
        return None

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

classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
model = load_model()

st.title("🧠 NeuroLens - Alzheimer's Detection")
st.markdown("### AI-Powered Brain MRI Analysis")
st.markdown("---")

with st.sidebar:
    st.header("�� System Info")
    if model:
        st.success("✅ **Model Status:** Active")
    else:
        st.warning("⚠️ **Model Status:** Loading...")
    
    st.info("""
    **Accuracy:** 99.1%
    
    **Classes:**
    - Non Demented
    - Very Mild Demented
    - Mild Demented
    - Moderate Demented
    
    **Model:** CNN
    """)
    st.markdown("---")
    st.write("**Developer**")
    st.write("Mudit Bhargava")
    st.write("Bennett University")
    st.warning("⚠️ Educational use only")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload MRI")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

with col2:
    st.subheader("🔍 Results")
    if uploaded and model:
        img = image.convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        with st.spinner("🧠 Analyzing..."):
            pred = model.predict(img_array, verbose=0)
            idx = np.argmax(pred[0])
            conf = pred[0][idx] * 100
            result = classes[idx]
        
        if result == 'Non Demented':
            st.success(f"✅ **{result}**")
        elif result == 'Very Mild Demented':
            st.warning(f"⚠️ **{result}**")
        else:
            st.error(f"🔴 **{result}**")
        
        st.metric("Confidence", f"{conf:.2f}%")
        st.progress(conf/100)
        
        st.markdown("---")
        st.subheader("📊 Class Probabilities")
        chart = {classes[i]: float(pred[0][i]*100) for i in range(4)}
        st.bar_chart(chart)
        
        with st.expander("📈 Detailed Probabilities"):
            for i, cls in enumerate(classes):
                st.write(f"**{cls}:** {pred[0][i]*100:.2f}%")
                
    elif uploaded:
        st.error("🔴 Model not loaded")
    else:
        st.info("👆 Upload MRI scan to begin")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p><strong>NeuroLens</strong> | Mudit Bhargava | Bennett University 2025</p>
</div>
""", unsafe_allow_html=True)
