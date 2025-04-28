import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="YOLOv8n Object Detection", layout="wide")

# Title
st.title("YOLOv8n Object Detection App")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# Function to load model (with caching)
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

model = load_model()

# Function to perform detection
def detect_objects(image):
    if model is None:
        st.error("Model not loaded. Please check the error above.")
        return image
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Perform detection
    results = model.predict(
        source=img_array,
        conf=confidence_threshold,
        save=False,
        save_txt=False
    )
    
    # Plot results on the image
    detected_img = results[0].plot()
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    
    return detected_img

# Main content
option = st.radio("Select Input Type:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Perform detection
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                detected_image = detect_objects(image)
                st.image(detected_image, caption="Detected Objects", use_column_width=True)

else:  # Webcam option
    picture = st.camera_input("Take a picture")
    
    if picture:
        # Display original image
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Perform detection
        with st.spinner("Detecting objects..."):
            detected_image = detect_objects(image)
            st.image(detected_image, caption="Detected Objects", use_column_width=True)

# Add some info
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses YOLOv8n (nano version) for object detection. "
    "Adjust the confidence threshold to filter detections."
)

# Add requirements for Streamlit Cloud
st.sidebar.markdown("### Requirements for Deployment")
st.sidebar.code("""
ultralytics==8.0.0
streamlit==1.22.0
opencv-python==4.7.0.72
pillow==9.5.0
numpy==1.24.3
""")
