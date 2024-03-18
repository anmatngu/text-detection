import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

reader = easyocr.Reader(['en'], gpu=True)

def detect_text(cap, threshold=0.25, font=cv2.FONT_HERSHEY_SIMPLEX):
    while True:
        ret, frame = cap.read()
        text = reader.readtext(frame)
        for t_, t in enumerate(text):
            bbox, text, score = t
            if score > threshold:
                cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1]) - 5), font, 1, (255, 0, 0), 2)
                print(text)
        cv2.imshow('Real-time Text Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

detect_text(cap, cv2.FONT_HERSHEY_SIMPLEX)