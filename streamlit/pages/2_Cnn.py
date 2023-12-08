import streamlit as st
import base64
import time
st.title('Unlocking the Mysteries of Your Face')

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
    time.sleep(5)  # Simulating a 5-second delay

    # Update the container with the result or new message
    status_container.text("Mapping your face features to a name. Almost there...")
    time.sleep(5)  # Simulating another 5-second delay

    # Final message after the process is complete
    status_container.text("Face classification and mapping completed!")

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="face gif">',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
