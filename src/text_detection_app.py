import streamlit as st
import cv2  # Optional for image display (if needed)
import easyocr
import numpy as np  # Optional for image display (if needed)

from text_detection import detect_text  # Import your function


def main():
    # Title and description
    st.title("Text Detection App")
    st.write("Upload an image to detect text.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    # Language selection (optional)
    # language_code = st.selectbox("Language", options=["en", "vi"], index=0)  # Add other languages as needed

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image

        # Detect text (optional language selection)
        text_data = detect_text(image)  # Pass image and adjust arguments as needed

        # Display additional information (optional)
        # st.write(f"Detected Text: {text_data}")  # Assuming text_data is a list of detected text objects

if __name__ == "__main__":
    main()
