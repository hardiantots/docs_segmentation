import os
import cv2
import sys
import time
import subprocess
import numpy as np
import streamlit as st

try:
    import torch
except ModuleNotFoundError as e:
    subprocess.Popen([f"{sys.executable} -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu"], shell=True)
    time.sleep(30)

try: 
    import cv2
except ModuleNotFoundError as e:
    subprocess.Popen([f"{sys.executable} -m pip install opencv-python"], shell=True)
    time.sleep(30)

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from deeplearnmethod import deep_learning_scan, get_image_download_link

# Streamlit Components
st.set_page_config(
    page_title="Docs Segmentation with Pytorch",
)

# Load model
@st.cache_data()
def load_dlModel(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "Mobilev3Model.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model

# Document Processing
def main(input_file, procedure, image_size=384):

    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]
    output = None

    st.write("Input image size:", image.shape)

    col1, col2 = st.columns((1, 1))

    with col1:
        st.title("Input")
        st.image(image, channels="RGB", use_column_width=True)

    with col2:
        st.title("Scanned")

        if procedure == "Deep Learning":
            model = model_mbv3
            output = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)

        st.image(output, channels="RGB", use_column_width=True)

    if output is not None:
        st.markdown(get_image_download_link(output, f"scanned_{input_file.name}", "Download scanned File"), unsafe_allow_html=True)

    return output

IMAGE_SIZE = 384
model_mbv3 = load_dlModel(img_size=IMAGE_SIZE)

st.title("Docs Segmentation")
st.write("Using DeepLabv3 with MobileNetv3Large backbone Model")

with st.sidebar:
    st.title("Options :")
    procedure_selected = st.radio("Select Scanning Procedure:", ("None", "Deep Learning"), index=1, horizontal=True)

    st.title("Reach Me")
    st.write("LinkedIn : [Hardianto Tandi Seno](https://www.linkedin.com/in/hardianto-ts/)")
    st.write("Gmail : [hardiantotandiseno@gmail.com](https://mail.google.com/mail/?view=cm&to=hardiantotandiseno@gmail.com&su=SUBJECT&body=BODY)")


tab1, tab2 = st.tabs(["Upload a Document", "Capture Document"])

with tab1:
    file_upload = st.file_uploader("Upload Document Image :", type=["jpg", "jpeg", "png"])
    if file_upload is not None:
        _ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)
        
with tab2:
    run = st.checkbox("Start Camera")
    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload is not None:
            _ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)