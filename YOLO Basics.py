from ultralytics import YOLO
import cv2

model = YOLO('Object Detection (YOLO)\YOLO Weights\yolov8l.pt')

# For resizing the image where the height is 500 for every image and the width is accordingly given
img = cv2.imread('Object Detection (YOLO)\Images/1.png')
ratio = float(img.shape[0] / img.shape[1])
imgHeight = 500
imgWidth = int(imgHeight / ratio)
imgResized = cv2.resize(img, (imgWidth, imgHeight))

results = model(imgResized, show = True)

cv2.waitKey(0)