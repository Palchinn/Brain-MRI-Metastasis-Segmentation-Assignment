import streamlit as st
import requests
import base64
from PIL import Image
import io

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    if st.button("Predict Metastasis"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            mask_base64 = result["mask"]
            mask_image = Image.open(io.BytesIO(base64.b64decode(mask_base64)))
            
            st.image(mask_image, caption="Predicted Metastasis Mask", use_column_width=True)
        else:
            st.error("Error occurred during prediction. Please try again.")

st.markdown("""
## About this app

This application uses a deep learning model (Nested U-Net or Attention U-Net) to segment brain metastases in MRI images. 
The model was trained on a dataset of brain MRI scans and their corresponding metastasis masks.

To use the app:
1. Upload a brain MRI image
2. Click the "Predict Metastasis" button
3. View the segmentation result

Please note that this is a demonstration and should not be used for clinical diagnosis.
""")