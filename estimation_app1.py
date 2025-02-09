import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Input dimensions for the neural network
width = 368
height = 368
inWidth = width
inHeight = height

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# App title and header setup with custom styles
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #FF5733;
            text-align: center;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #3498DB;
        }
        .sidebar-header {
            font-size: 1.2rem;
            color: #2ECC71;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>🤸‍♀️ Human Pose Estimation</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Estimate human poses in images using OpenCV and Streamlit</p>", unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.markdown("<h2 class='sidebar-header'>📁 Upload Your Image</h2>", unsafe_allow_html=True)
    img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    thres = st.slider('🔍 Detection Threshold', min_value=0, value=20, max_value=100, step=5) / 100
    st.markdown("<p style='text-align: center; color: #9B59B6;'>Made by Varshni</p>", unsafe_allow_html=True)

# Image processing and display
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    st.sidebar.info("Using default demo image.")

st.markdown("### Uploaded Image")
st.image(image, caption="Original Image", use_column_width=True)

@st.cache_data
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

output = poseDetector(image)

st.markdown("### Estimated Pose")
st.image(output, caption="Pose Estimation Output", use_column_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #E74C3C;'>Built using Streamlit and OpenCV</p>",
    unsafe_allow_html=True
)
