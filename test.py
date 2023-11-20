import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgsize = 300
folder = "images/0"
counter = 0
labels = ["1", "2", "beautiful"]

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            wcal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wcal, imgsize))
            wgap = math.ceil((imgsize - wcal) / 2)
            imgWhite[:, wgap:wcal + wgap] = imgResize  # Ensure dimensions match
            prediction, index = classifier.getPrediction(img)

        else:
            k = imgsize / w
            hcal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (imgsize, hcal))
            hgap = math.ceil((imgsize - hcal) / 2)
            imgWhite[hgap:hcal + hgap, :] = imgResize  # Ensure dimensions match
            predicted_label, index = classifier.getPrediction(img)

        predicted_label = labels[index]
        cv2.putText(imgOutput, predicted_label, (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("imagecrop", imgCrop)
        cv2.imshow("imagewhite", imgWhite)

        # Save the cropped image with the text message as the file name
        save_path = os.path.join(folder, f"{predicted_label}_{counter}.png")
        cv2.putText(imgOutput, predicted_label, (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 2)

        cv2.imwrite(save_path, imgCrop)

        counter += 1

    cv2.imshow("image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('q'):  # Press 'q' to close the camera and exit
        cap.release()  # Close the camera
        break
