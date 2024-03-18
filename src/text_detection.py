def contours_text(orig, img, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('cnt',rect)
        cv2.waitKey()
        cropped = orig[y:y + h, x:x + w]
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config)
        print(text)

import cv2
import pytesseract
# read image
im = cv2.imread('./testimg.jpg')
# configurations
config = ('-l eng --oem 1 --psm 3')
# pytesseract
text = pytesseract.image_to_string(im, config=config)
# print text
text = text.split('n')
text