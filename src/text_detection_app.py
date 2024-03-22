import cv2
import time
import json 
import easyocr
import numpy as np
import pandas as pd
import streamlit as st

language_codes = [
    "abq", "ady", "af", "ang", "ar", "as", "ava", "az", "be", "bg", "bh", "bho", "bn", "bs",
    "ch_sim", "ch_tra", "che", "cs", "cy", "da", "dar", "de", "en", "es", "et", "fa", "fr",
    "ga", "gom", "hi", "hr", "hu", "id", "inh", "is", "it", "ja", "kbd", "kn", "ko", "ku",
    "la", "lbe", "lez", "lt", "lv", "mah", "mai", "mi", "mn", "mr", "ms", "mt", "ne", "new",
    "nl", "no", "oc", "pi", "pl", "pt", "ro", "ru", "rs_cyrillic", "rs_latin", "sck", "sk",
    "sl", "sq", "sv", "sw", "ta", "tab", "te", "th", "tjk", "tl", "tr", "ug", "uk", "ur", "uz",
    "vi"
]

def detect_text(image_path, language='en', threshold=0.25, min_text_length=3):  
    reader = easyocr.Reader([language], gpu=True)
    start_time = time.time()

    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path

    progress_bar = st.progress(0)

    def readtext_with_progress(img):
        text_result = reader.readtext(img)
        for i, _ in enumerate(text_result):
            progress_bar.progress((i + 1) / len(text_result))
        return text_result

    text = readtext_with_progress(img)

    detected_text = []
    processing_time = float(time.time() - start_time)

    for t_, t in enumerate(text):
        bbox, text, score = t
        if score > threshold and len(text) >= min_text_length:
            detected_text.append({'bbox': bbox, 'text': text, 'score': score})

    return processing_time, detected_text

def main():
    st.title("Text Detection App")
    st.write("Choose input type:")
    input_type = st.radio("", ("Camera", "File"))
    language_code = st.selectbox("Language", options=language_codes, index=0)
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    min_text_length = st.number_input("Minimum Text Length", min_value=1, value=4)

    if input_type == "Camera":
        cam_input = st.camera_input("Use camera")
        if cam_input:
            if cam_input is not None:
                image = np.array(bytearray(cam_input.read()), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processing_time, text_data = detect_text(image_rgb, language_code, threshold=confidence_threshold, min_text_length=min_text_length)

                if text_data:
                    st.write("Detected Text:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_rgb, caption="Original Image")
                    with col2:
                        image_with_boxes = image.copy()
                        for item in text_data:
                            bbox, text, score = item
                            x1, y1, x2, y2 = item['bbox']
                            image_with_boxes = cv2.rectangle(image_with_boxes, (int(x1[0]), int(y1[1])), (int(x2[0]), int(y2[1])), (0, 255, 0), 2)
                        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                        st.image(image_with_boxes, caption="Image with Detected Text(s)")

                    df = pd.DataFrame(text_data)[['text', 'score']]
                    df.columns = ["Detected Text", "Confidence Score"]
                    df = df.sort_values(by='Confidence Score', ascending=False, ignore_index=True)
                    st.table(df)
                else:
                    st.write("No text detected above the threshold and minimum length.")
            else:
                st.write("No camera input provided.")
    elif input_type == "File":
        uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            processing_time, text_data = detect_text(image_rgb, language_code, threshold=confidence_threshold, min_text_length=min_text_length)

            if text_data:
                st.write("Detected Text:")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_rgb, caption="Original Image")
                with col2:
                    image_with_boxes = image.copy()
                    for item in text_data:
                        bbox, text, score = item
                        x1, y1, x2, y2 = item['bbox']
                        image_with_boxes = cv2.rectangle(image_with_boxes, (int(x1[0]), int(y1[1])), (int(x2[0]), int(y2[1])), (0, 255, 0), 2)
                    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                    st.image(image_with_boxes, channels="RGB", caption="Image with Detected Text(s)")
                df = pd.DataFrame(text_data)[['text', 'score']]
                df.columns = ["Detected Text", "Confidence Score"]
                df = df.sort_values(by='Confidence Score', ascending=False, ignore_index=True)
                st.table(df)
            else:
                st.write("No text detected above the threshold and minimum length.")

if __name__ == "__main__":
    main()