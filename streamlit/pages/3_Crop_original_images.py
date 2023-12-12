import streamlit as st
import base64
import time
import os
import cv2

import mediapipe as mp
st.title('Unlocking the Mysteries of Your Face')

def crop_originals():
    # Load the MediaPipe face detection model
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    # Input and output folders
    input_folder = './data/original_faces'
    output_folder = './data/faces_training'
    error_log_file = './data/error_log.txt'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the error log file
    with open(error_log_file, 'w') as log_file:
        log_file.write("Error Log:\n")

    # Loop through each subfolder (person) in the input folder
    for person_folder in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_folder)

        # Create a subfolder in the output folder for each person
        output_person_folder = os.path.join(output_folder, person_folder)
        if not os.path.exists(output_person_folder):
            os.makedirs(output_person_folder)
        if person_path != '../original_faces/.DS_Store':
            # Loop through each image in the person's folder
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)

                # Check if the file has a valid image extension
                valid_extensions = {'.png', '.jpg', '.jpeg'}
                _, extension = os.path.splitext(image_file.lower())

                if extension in valid_extensions:
                    try:
                        # Read the input image
                        image = cv2.imread(image_path)

                        # Convert the image to RGB (required for MediaPipe)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Perform face detection using MediaPipe
                        results = face_detection.process(rgb_image)

                        # Loop through each detected face
                        for i, detection in enumerate(results.detections):
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = rgb_image.shape
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(
                                bboxC.height * ih)

                            # Crop the region of interest (ROI) from the original image
                            face_roi = image[y:y + h, x:x + w]

                            # Save the cropped face in the output folder with a unique filename
                            output_filename = os.path.join(output_person_folder,
                                                           f'{os.path.splitext(image_file)[0]}_{i + 1}_face.jpg')
                            cv2.imwrite(output_filename, face_roi)
                            print(f"saving {output_filename}")

                    except Exception as e:
                        # Log the path of the image that couldn't be processed
                        with open(error_log_file, 'a') as log_file:
                            log_file.write(f"Error processing image: {image_path}\n")
                        print(f"Error processing image: {image_path}")

    # Release resources
    face_detection.close()


def main():
    st.title('CNN')
    gif_path = "icons/GIF_FaceRecognitionn.gif"


    with open(gif_path, "rb") as file_:
        contents = file_.read()

    # Encoding the GIF file
    data_url = base64.b64encode(contents).decode("utf-8")

    # Center the GIF using HTML and CSS
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/gif;base64,{data_url}" alt="Face Recognition GIF" style="max-width: 100%; max-height: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )
    # Create an empty container for dynamic content
    status_container = st.empty()

    # Simulate a time-consuming operation
    status_container.text("Classifying your face features... Please wait.")
    time.sleep(3)  # Simulating a 5-second delay

    # Update the container with the result or new message
    status_container.text("Mapping your face features to a name. Almost there...")
    #time.sleep(5)  # Simulating another 5-second delay
    if st.button("Start cropping faces"):
        crop_originals()
    # Final message after the process is complete
    status_container.text("Face classification and mapping completed!")

    #st.markdown(
     ##  unsafe_allow_html=True,
   # )

if __name__ == "__main__":
    main()
