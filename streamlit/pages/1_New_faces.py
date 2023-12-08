import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import pyttsx3

def announce_recognition():
    engine = pyttsx3.init()
    engine.say("Your face has been recognized.")
    engine.runAndWait()


def crop_photo(img):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9)

    # Convert the image to RGB (required for MediaPipe)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection using MediaPipe
    results = face_detection.process(rgb_image)

    # List to store cropped faces_try
    cropped_faces = []

    # Display and save each detected face
    for i, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = rgb_image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        # Crop the region of interest (ROI) from the original image
        face_roi = img[y:y + h, x:x + w]
        cropped_faces.append(face_roi)

        # Draw rectangles around the detected faces_try
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display all cropped faces_try
    for i, face in enumerate(cropped_faces):
        st.write('faces_try cropped')
        #st.image(face, caption=f"Detected Face {i+1}", channels="BGR")

    # Display the image with rectangles
    #st.image(img, caption="Detected Faces", channels="BGR")

    face_detection.close()
    return cropped_faces
def main():



    person_name = st.text_input("New person to add:")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Show the picture taken
        #st.image(cv2_img, caption="Original Image", channels="BGR")

        # Perform face detection and display the results
        crop_faces = crop_photo(cv2_img)

        for i, face in enumerate(crop_faces):
            st.image(face, caption=f"Detected Face {i + 1}", channels="BGR")
            announce_recognition()
            btn = st.button('Save this image', key=f'save {i}')
            if btn:
                save_folder = os.path.join("../faces_crop", person_name)
                os.makedirs(save_folder, exist_ok=True)

                save_path = os.path.join(save_folder, f"face_{i + 1}.jpg")
                cv2.imwrite(save_path, face)

                # Voice notification

if __name__ == "__main__":
    main()
