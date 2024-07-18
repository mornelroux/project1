
#Include Yolov10 in the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))

#Import YOLOv10 and cv2
from yolov10.ultralytics import YOLOv10
import cv2
from src.editor.Editor import ImageEditor

#Load model with pretrained weights
model = YOLOv10('models/pretrained/yolov10n.pt')
editor = ImageEditor()

#Load Camera
cap = cv2.VideoCapture(2)

while True:
    #Read frames
    ret,frame = cap.read()

    #Detect objects
    results = model.predict(frame)

    original_image = results[0].orig_img
    for r in results:
            boxes = r.boxes
            for box in boxes:
                b = (box.xyxy[0])  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                editor.fill_bounding_box(original_image, [0,255,0], b[0], b[1], b[2], b[3],0.8)
                editor.draw_label(original_image, "dog", [0,255,0], b[0], b[1])

    #Visualize
    editor.add_title_to_image(original_image, "Image Detections")
    cv2.imshow('Output',original_image)

    #Exit loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
