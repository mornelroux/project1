
#Include Yolov10 in the path
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ByteTrack'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'boxmot'))

#Import YOLOv10 and cv2
from yolov10.ultralytics import YOLOv10
import cv2
#from ByteTrack.yolox.tracker.byte_tracker import BYTETracker as ByteTracker
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker as ByteTracker

#Load model with pretrained weights
model = YOLOv10('models/pretrained/yolov10n.pt')

# class TrackerArgs:
#     def __init__(self, track_thresh: float, track_buffer: int):
#         self.track_thresh = track_thresh
#         self.track_buffer = track_buffer

# args = TrackerArgs(track_thresh=0.3, track_buffer=30)
# print(args.track_thresh)

tracker = ByteTracker()

#Load Camera
cap = cv2.VideoCapture(0)


while True:
    #Read frames
    ret,frame = cap.read()

    #Detect objects
    results = model.predict(frame)

    # Convert the detections to he required format
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

    print(tracked)
    
    tracker.plot_results(frame,show_trajectories=False)
    
    #Plot results
    #frame_ = results[0].plot()

    #Visualize
    cv2.imshow('Output',frame)

    #Exit loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
