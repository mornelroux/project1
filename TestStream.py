
#Include Yolov10 in the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))

#Include Boxmot in the path
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'boxmot'))

#Import YOLOv10 and cv2
from yolov10.ultralytics import YOLOv10
import cv2
from src.editor.Editor import ImageEditor
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker as ByteTracker

#Load model with pretrained weights
model = YOLOv10('models/pretrained/yolov10n.pt')
editor = ImageEditor()
tracker = ByteTracker()

#Load Camera
cap = cv2.VideoCapture(0)

while True:
    #Read frames
    ret,frame = cap.read()

    #Detect objects
    results = model.predict(frame)

    #Save orignal image
    original_image = results[0].orig_img

    # Convert the detections to he required format for tracking
    dets = []
    for result in results:
        for detection in result.boxes.data.cpu().numpy():
            x1,y1,x2,y2,conf,cls = detection
            dets.append([x1,y1,x2,y2,conf,int(cls)])
    dets = np.array(dets)

    # Update tracker
    if dets.shape[0] > 0:
        #Tracker crashes with zero detections
        tracked = tracker.update(dets,frame)
        for t in tracked:
            x1,y1,x2,y2,id,conf,cls,det_ind = t
            # editor.fill_bounding_box(original_image, [0,255,0], x1, y1, x2, y2,0.8)
            # editor.draw_label(original_image, str(id), [0,255,0], x1, y1)
            editor.draw_line(original_image, x1, y1, x2, y2, id)

    # Plain detection
    # for r in results:
    #         boxes = r.boxes
    #         for box in boxes:
    #             b = (box.xyxy[0])  # get box coordinates in (left, top, right, bottom) format
    #             c = box.cls
    #             editor.fill_bounding_box(original_image, [0,255,0], b[0], b[1], b[2], b[3],0.8)
    #             editor.draw_label(original_image, "dog", [0,255,0], b[0], b[1])

    #Visualize
    editor.add_title_to_image(original_image, "Image Detections")
    cv2.imshow('Output',original_image)

    #Exit loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
