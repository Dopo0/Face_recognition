import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import streamlit as st

# Path to this file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(project_root, 'data/faces_training')

# Relative number of predicted classes
NUM_CLASSES = len(os.listdir(folder_path)) - 1

# definition of the class Network same as the training
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(12),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(48),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(48 * 14 * 14, out_features=1024),
            #nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, out_features=512),
            # nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=NUM_CLASSES)
        )

    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x


# Process frames in the video to predict
def process_image(frame):

    cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_frame)

    # Draw bounding boxes around the detected faces and extract ROI
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Calculate the centered bounding box dimensions
            bbox_width = int(bboxC.width * iw * 1.2)
            bbox_height = int(bboxC.height * ih * 1.2)
            bbox_x = max(0, int(bboxC.xmin * iw - (bbox_width - int(bboxC.width * iw)) / 2))
            bbox_y = max(0, int(bboxC.ymin * ih - (bbox_height - int(bboxC.height * ih)) / 2))

            # Extract the region of interest (ROI) from the frame
            face_roi = frame[bbox_y:min(ih, bbox_y + bbox_height),
                       bbox_x:min(iw, bbox_x + bbox_width)]

            person, confidence= predict(face_roi)
            confidence = round(confidence, 2)
            confidence = '{:.2}'.format(confidence)


            # Draw bounding box around the detected face
            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (0, 255, 0), 2)

            cv2.putText(frame, f'{person}-{confidence}', (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Predict frames
def predict(frame):
    model.eval()

    IMAGE_SIZE = (112, 112)
    test_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1,), (0.3,))])
    
    # Use the list of folder names as custom_class_names
    items = os.listdir(folder_path)
    custom_class_names = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

    # Tranform the image
    image = Image.fromarray(frame)
    img_tranform = test_transform(image)
    img = img_tranform
    img = img.view(-1, 3, 112, 112)

    # Predict
    with torch.no_grad():
        logits = model.forward(img)
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    person_idx = np.where(probabilities == probabilities.max())[0][0]

    confidence = round(probabilities[person_idx],1)
    if confidence <= 0.65:
        person = "unknow"
    else:
        person = custom_class_names[person_idx]

    #print(f'The person is {person}')
    #print(f'Porbability {probabilities}')

    return person, confidence

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_image(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Face Recognition")

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {
            "width": 640,  # Set the desired width
            "height": 480,  # Set the desired height
        }, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    folder_path = os.path.join(project_root, 'data/faces_training')

    items = os.listdir(folder_path)
    custom_class_names = sorted([item for item in items if os.path.isdir(os.path.join(folder_path, item))])

    st.write(custom_class_names)


if __name__ == "__main__":
    # loading model
    model = Network()
    file_names = os.listdir('obj')
    model_path = st.sidebar.selectbox("Select the model to test: ",file_names)

    state_dict = torch.load(os.path.join("obj",model_path), map_location='cpu')
    model.load_state_dict(state_dict)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    main()
