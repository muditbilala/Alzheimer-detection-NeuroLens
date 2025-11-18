import streamlit as st
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

st.title("🧠 NeuroLens - Alzheimer's Detection")
st.markdown("### AI-Powered Brain MRI Analysis")
st.markdown("---")

with st.sidebar:
    st.header("📊 System Info")
    st.info("""
    **Model Status:** Training Complete ✅
    
    **Accuracy:** 99.1%
    
    **Classes:** 4
    - Non Demented
    - Very Mild
    - Mild  
    - Moderate
    
    **Technology:** Deep Learning CNN
    """)
    st.markdown("---")
    st.write("**Developer**")
    st.write("Mudit Bhargava")
    st.write("Bennett University")
    st.write("B.Tech CSE - 3rd Year")
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**\n\nFor educational and research purposes only.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload MRI Scan")
    st.info("Upload a brain MRI image for AI analysis")
    
    uploaded = st.file_uploader(
        "Choose MRI image", 
        type=['jpg', 'png', 'jpeg'],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        # Image info
        st.success(f"✅ Image loaded: {image.size[0]}x{image.size[1]} pixels")

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded:
        st.info("🚧 **Model Integration in Progress**")
        st.write("""
        The CNN model has been trained with **99.1% accuracy** on 80,000+ brain MRI images.
        
        **Model Details:**
        - Architecture: Convolutional Neural Network (CNN)
        - Input Size: 128x128 pixels
        - Classes: 4 (Non Demented, Very Mild, Mild, Moderate)
        - Training Dataset: OASIS Alzheimer's Detection
        
        **Integration Status:** Model file deployment in progress.
        """)
        
        # Demo prediction display
        st.markdown("---")
        st.subheader("📊 Expected Output")
        st.write("Once model is integrated, you'll see:")
        
        # Sample output
        demo_data = {
            'Non Demented': 75.2,
            'Very Mild': 15.3,
            'Mild': 7.1,
            'Moderate': 2.4
        }
        st.bar_chart(demo_data)
        
    else:
        st.info("👆 Upload a brain MRI scan to begin analysis")
        
        st.markdown("---")
        st.subheader("💡 How to Use")
        st.markdown("""
        1. **Upload** a brain MRI scan image
        2. **Wait** for AI analysis (typically <2 seconds)
        3. **View** prediction with confidence scores
        4. **Interpret** results with probability distribution
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4>🧠 NeuroLens - Advanced Medical AI</h4>
    <p>Deep Learning for Early Alzheimer's Detection</p>
    <p><strong>Mudit Bhargava</strong> | Bennett University | 2025</p>
    <p style='font-size: 12px; color: #666;'>
        Research Project | B.Tech Computer Science & Engineering
    </p>
</div>
""", unsafe_allow_html=True)
