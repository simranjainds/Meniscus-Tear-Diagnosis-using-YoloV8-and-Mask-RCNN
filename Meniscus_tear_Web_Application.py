import streamlit as st
import os
import cv2
from mrcnn import visualize
import mrcnn.model as mrcnn_model
from mrcnn.config import Config
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import PIL
import base64

# Custom Config for Mask R-CNN
class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + Hard_hat, Safety_vest
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.2

# Load the Mask R-CNN model
inference_config = InferenceConfig()
mrcnn_model = mrcnn_model.MaskRCNN(mode="inference", config=inference_config, model_dir="logs")
model_path_mrcnn = 'mrcnn.h5'
print("Loading weights from ", model_path_mrcnn)
mrcnn_model.keras_model.load_weights(model_path_mrcnn, by_name=True)

# Replace the relative path to your YOLO weight file
model_path_yolo = 'best.pt'

# Background images
main_bg_path = 'Untitled design.jpg'  # Replace with your main content background image path

# Convert images to base64
main_bg_base64 = base64.b64encode(open(main_bg_path, "rb").read()).decode()

# Custom CSS to set background image and text color
def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    # Convert the image to a base64-encoded string
    image_base64 = base64.b64encode(open(side_bg, "rb").read()).decode()

    # Set the sidebar background and text color using custom CSS
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{image_base64});
        }}
        [data-testid="stSidebar"] > div:first-child * {{
            color: white;  /* Set text color to white */
        }}
        /* Set transparent background with white border for buttons and selectbox */
        [data-testid="stSidebar"] button, [data-testid="stSidebar"] .stSelectbox {{
            background-color: transparent !important;
            border: 1px solid white !important;
            color: black !important;
            font-weight: bold !important;
        }}
        /* Set transparent background with black border for buttons and selectbox */
        [data-testid="stSelectbox"] button, [data-testid="stSidebar"] .stSelectbox {{
            background-color: transparent !important;
            border: 1px solid white !important;
            color: black !important;
            font-weight: bold !important;
        }}
        /* Set transparent background for the file uploader */
        [data-testid="stFileUploader"] .stFileDropzone {{
            background-color: transparent !important;
            border: none !important;
            color: black !important;
            box-shadow: none !important;
        }}
        [data-testid="stFileUploader"] .stFileDropzone small {{
            background-color: transparent !important;
            border: 1px solid black !important;
            color: black !important;
            font-weight: bold !important;
        }}
        [data-testid="stFileUploader"] .stFileDropzone div[role="button"] {{
            color: black !important;
            box-shadow: 2px !important;
        }}
        /* Set transparent background for the file uploader container */
        [data-testid="stFileUploader"] .st-bx {{
            background-color: transparent !important;
            border: black !important;
            color: black !important;
            box-shadow: 2px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set the background image and text color for the sidebar
side_bg = '/Users/simranjain/Desktop/Data_split/mrcnn/Untitled design.png'
sidebar_bg(side_bg)

# Add background image and set text color using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{main_bg_base64}) fixed center;
        background-size: cover;
    }}
    .model-selection {{
        font-family: sans-serif;
        color: white;
        font-size: 18px;
        margin-bottom: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Creating sidebar
with st.sidebar:
    st.header("Object Detection Config")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))

    # Model selection
    model_choice = st.selectbox("**Select Model**", ["Mask R-CNN", "YOLOv8"])

    # Confidence slider
    confidence = st.slider("Select Confidence Level", 0.0, 1.0, 0.7, 0.05)

    # Predict button
    predict_button = st.button("Predict")

# Creating main page heading
new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Meniscus Tear Detection</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Display selected model under the title
if model_choice:
    st.markdown(f'<p class="model-selection">Selected Model: {model_choice}</p>', unsafe_allow_html=True)

# Main app logic
if predict_button and uploaded_file is not None:
    try:
        # Read the image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection based on the selected model
        if model_choice == "Mask R-CNN":
            results = mrcnn_model.detect([img], verbose=1)
            r = results[0]

            # Convert BGR image to RGB for displaying in Streamlit
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the image and the detected instances
            st.image(rgb_image, caption="Uploaded Image", use_column_width=True)

            # Create a Matplotlib figure
            fig, ax = plt.subplots()
            visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'],
                                        ["BG", "Healthy Meniscus", "Meniscus tear"], r['scores'], ax=ax)
            # Pass the Matplotlib figure to st.pyplot()
            st.pyplot(fig)

        elif model_choice == "YOLOv8":
            yolo_model = YOLO(model_path_yolo)
            res = yolo_model.predict(img, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            # Display the detected image
            st.image(res_plotted, caption='Detected Image', use_column_width=True)

            # Display the bounding box coordinates
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)

    except Exception as e:
        st.error(f"Error: {str(e)}")
