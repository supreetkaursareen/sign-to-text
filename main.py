import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
folder = "images/2"

counter = 0

# Create the folder if it doesn't exist


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Calculate the dimensions for imgCrop to ensure the entire hand is captured
        crop_x = max(x - offset, 0)
        crop_y = max(y - offset, 0)
        crop_w = min(x + w + offset, img.shape[1]) - crop_x
        crop_h = min(y + h + offset, img.shape[0]) - crop_y

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgCrop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgsize / h
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hgap = math.ceil((imgsize - hcal) / 2)
            imgWhite[hgap:hgap + hcal, :] = imgResize  # Ensure dimensions match
        else:
            k = imgsize / w
            wcal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wcal, imgsize))
            wgap = math.ceil((imgsize - wcal) / 2)
            imgWhite[:, wgap:wgap + wcal] = imgResize  # Ensure dimensions match

        cv2.imshow("imagecrop", imgCrop)
        cv2.imshow("imagewhite", imgWhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        image_path = os.path.join(folder, f'Image_{time.time()}.jpg')
        cv2.imwrite(image_path, imgCrop)  # Save imgCrop
        print(f"Image saved as {image_path}")
