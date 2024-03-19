import streamlit as st
import cv2
import easyocr
import numpy as np
import json 
from text_detection import detect_text  # Import your function


def main():
    # Title and description
    st.title("Text Detection App")
    st.write("Upload an image to detect text.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    language_code = st.selectbox("Language", options=["en", "vi"], index=0)  # Add other languages as needed

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        processing_time, text_data = detect_text(image, language_code, threshold=0.3, min_text_length=4)
        if text_data:
            html_text = f"<div style='position: relative;'>"
            st.write("Detected Text:")
            for item in text_data:
                bbox, text, score = item
                x1, y1, x2, y2 = item['bbox']
                image = cv2.rectangle(image, (int(x1[0]), int(y1[1])), (int(x2[0]), int(y2[1])), (0, 255, 0), 2)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Display the image with bounding boxes
                st.image(image, channels="RGB")
                st.write(f"- {item['text']} (Confidence: {item['score']:.2f})")
        else:
            st.write("No text detected above the threshold and minimum length.")
if __name__ == "__main__":
    main()