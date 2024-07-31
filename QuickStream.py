
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
from src.editor.InOutCounter import InOutCounter

#Load model with pretrained weights
model = YOLOv10('models/pretrained/yolov10n.pt')

# class TrackerArgs:
#     def __init__(self, track_thresh: float, track_buffer: int):
#         self.track_thresh = track_thresh
#         self.track_buffer = track_buffer

# args = TrackerArgs(track_thresh=0.3, track_buffer=30)
# print(args.track_thresh)

tracker = ByteTracker()
line1 = ((300,0),(300,480))
line2 = ((340,0),(340,480))


counter = InOutCounter(line1,line2)

#Load Camera
cap = cv2.VideoCapture(0)

#Declare memory
memory = {}
count = 0

def intersection(line1,line2):
    #Extract the coordinates of the lines
    A,B = line1
    C,D = line2

    #Calculate the direction vectors
    vector1 = (B[0] - A[0], B[1] - A[1])
    vector2 = (D[0] - C[0], D[1] - C[1])

    #Calculate the determinant of direction vectors
    det = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    print(det) # 0

    #Calculate the intersection point
    if det != 0:
        t = ((C[0] - A[0])*(D[1] - C[1]) - (C[1] - A[1])*(D[0] - C[0]))/det
        u = ((C[0] - A[0])*(B[1] - A[1]) - (C[1] - A[1])*(B[0] - A[0]))/det

        #Check if the intersection point is within the line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
        else:
            return False
    return False


def intersection_angle(line1,line2):
    A,B = line1
    C,D = line2

    #Use C as the reference point


def CounttrackedObjects(img,tracked,line_start,line_stop):
    for t in tracked:
        x1,y1,x2,y2,id,conf,cls,det_ind = t

        #get midpoint of object
        x = 0.5*(x1 + x2)
        y = 0.5*(y1 + y2)
        midpoint = (int(x),int(y))

        #check if object is in memory
        if id not in memory:
            memory[id] = []
            memory[id].append((0,0))

        #Swap the new position to the memory
        previous_midpoints = memory[id][0]
        memory[id][0] = midpoint

        # Draw line connecting previous midpoint and current midpoint
        cv2.line(img, previous_midpoints, midpoint, (255,255,255), 2)#cv2.line(img, (x1, y1), (x1 + w, y1), corner_clr, 2)
        
        if intersection((line_start,line_stop),(previous_midpoints,midpoint)):
            intersection_angle
            count += 1
            print(count)
            print(f"Object {id} has crossed the line")

        print(intersection((line_start,line_stop),(previous_midpoints,midpoint)))
        print(f"Object {id} is being tracked")







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

    #Draw counting line
    #cv2.line(frame, (0,0), (640,480), (255,255,255), 2)#cv2.line(img, (x1, y1), (x1 + w, y1), corner_clr, 2)    

    # Update tracker
    if dets.shape[0] > 0:
        #Tracker crashes with zero detections
        tracked = tracker.update(dets,frame)
        for t in tracked:
                x1,y1,x2,y2,id,conf,cls,det_ind = t

                #get midpoint of object
                x = 0.5*(x1 + x2)
                y = 0.5*(y1 + y2)
                midpoint = (int(x),int(y))
                counter.trackObject(frame,midpoint,id)
    
    tracker.plot_results(frame,show_trajectories=False)
    
    #Plot results
    #frame_ = results[0].plot()

    #Visualize
    cv2.imshow('Output',frame)

    #Exit loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
