import cv2
import easyocr
import numpy as np

def detect_text(image_path, language='en', threshold=0.25):
    # Initialize text detector
    reader = easyocr.Reader([language], gpu=True)

    # Load the image
    img = cv2.imread(image_path)

    # Detect text on the image
    text = reader.readtext(img)

    # Draw bounding boxes and text (if above threshold)
    for t_, t in enumerate(text):
        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
            cv2.putText(img, text, (int(bbox[0][0]), int(bbox[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image with detected text
    cv2.imshow('Text Detection Result', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
