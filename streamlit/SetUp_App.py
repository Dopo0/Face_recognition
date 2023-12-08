
import streamlit as st
import pyttsx3
# Page Configurations
def initialize_app():

    welcome_message = "Welcome to Cyber App ID. This is your virtual assistant."

    # Play welcome message on page load
    st.text(welcome_message)
    text_to_speech(welcome_message)
def announce_recognition():
    engine = pyttsx3.init()
    engine.say("Your face has been recognized.")
    engine.runAndWait()
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
st.set_page_config(
    page_title="Face Recognition App",
    page_icon='face_with_spiral_eyes',
    layout= 'wide',
    #initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Exo:wght@400&display=swap');
        body {
            background-color: #000000;  /* Set to black */
            color: #ffffff;  /* Set text color to white */
            text-align: center;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            font-family: 'Exo', sans-serif;  /* Use Exo font */
        }

        h1 {
            margin-top: 10vh;  /* Adjust the margin to center the title higher or lower */
            font-size: 3em;  /* Adjust the font size as needed */
            font-family: 'Exo', sans-serif;  /* Specify the font for the title */
        }

        img {
            width: 100%;  /* Make the image width 100% of the viewport */
            height: 100vh;  /* Make the image height 100% of the viewport */
            object-fit: cover;  /* Maintain aspect ratio while covering the entire viewport */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
