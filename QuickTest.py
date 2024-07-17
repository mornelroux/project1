
#Include Yolov10 in the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))

#Import YOLOv10 and cv2
from yolov10.ultralytics import YOLOv10
import cv2

#Load model with pretrained weights
model = YOLOv10('models/pretrained/yolov10n.pt')

#Load Camera
cap = cv2.VideoCapture(0)

while True:
    #Read frames
    ret,frame = cap.read()

    #Detect objects
    results = model.predict(frame)
    
    #Plot results
    frame_ = results[0].plot()

    #Visualize
    cv2.imshow('Output',frame_)

    #Exit loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
