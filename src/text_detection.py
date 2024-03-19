import cv2
import easyocr
import numpy as np
import time

def detect_text(image_path, language='en', threshold=0.25, min_text_length=3):  
    reader = easyocr.Reader([language], gpu=True)
    start_time = time.time()
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
    text = reader.readtext(img)
    detected_text = []

    processing_time = float(time.time() - start_time)
    for t_, t in enumerate(text):
        bbox, text, score = t
        if score > threshold and len(text) >= min_text_length:

            detected_text.append({'bbox': bbox, 'text': text, 'score': score})
    return processing_time, detected_text