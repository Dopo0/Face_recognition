# Libraries
import streamlit as st
# from PIL import Image
import base64
import SetUp_App
import time
from SetUp_App import initialize_app, announce_recognition

time.sleep(3)
initialize_app()

def main():
    gif_path = "../streamlit/icons/GIF_Face.gif"

    with open(gif_path, "rb") as file_:
        contents = file_.read()

    # Encoding the GIF file
    data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="face gif">',
        unsafe_allow_html=True,
    )
    st.title("cyber  face ID")
    st.text("Digital Portraits of the Next Era")

if __name__ == "__main__":
    main()










