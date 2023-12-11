import streamlit as st
import os 

st.title('Upload Images')

def main():

    st.title('Upload Images')
    
    # Function to check if a folder with the given name already exists
    def is_name_taken(name):
        folder_path = os.path.join("../streamlit/data/faces_training", name)
        if name == '': return False
        else: return os.path.exists(folder_path)

    # Get user's name input
    user_name = st.text_input("Enter your name")

    # Check if the name is taken
    name_taken = is_name_taken(user_name)

    # Display feedback to the user
    if name_taken:
        st.warning(f"The name '{user_name}' is already taken. Please choose a different name.", icon="⚠️")
    else:
        st.success(f"The name '{user_name}' is available.")
    
    # Function to handle multiple file uploads
    def multi_file_uploader(label, key, file_types=None):
        uploaded_files = st.file_uploader(label, key=key, type=file_types, accept_multiple_files=True)
        return uploaded_files

    # Example usage
    uploaded_files = multi_file_uploader("Choose multiple files", key="multi_files")

    if uploaded_files:
        # Create a folder named "images" if it doesn't exist
        os.makedirs(f"../streamlit/data/faces_training/{user_name}", exist_ok=True)

        for file in uploaded_files:
            # Save each file to the "images" folder
            file_path = os.path.join(f"../streamlit/data/faces_training/{user_name}", file.name)

            with open(file_path, "wb") as f:
                f.write(file.read())

            st.write(f"File '{file.name}' saved successfully to 'faces_training' folder.")

if __name__ == "__main__":
    main()
