import asone
from asone import ASOne
import time
import copy
import numpy as np
from PIL import Image
from classifier.predict import predict
from djitellopy import Tello
import threading

import movement as mv

#global variables
CLASSIFY = False
RESEARCH_MODE = True
MOVE_DRONE = True
TRACKED_ID = -1
TRACKED_BOX = None

def reshape_bbox(bbox_xyxy, x_max, y_max):
    [x0,y0,x1,y1] = [max(0,int(bbox_xyxy[j])) for j in range(4)]
    if y1 >= y_max:
        y1 = y_max-1
    if y0 >= y_max:
        y0 = y_max-1
    if x1 >= x_max:
        x1 = x_max-1
    if x0 >= x_max:
        x0 = x_max-1
    return [x0,y0,x1,y1]

def get_surface(box):
    return abs(box[0]-box[2])*abs(box[1]-box[3])

try:
    tello = Tello()
    """if tello.get_battery() < 10:
        print('error: battery low, please charge the battery')
        exit()"""
except:
    exit()

#Takeoff
tello.connect()
tello.takeoff()
tello.move_up(60)
tello.streamon()


stream_path = 'udp:/0.0.0.0:11111'

#get the model (detect)
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, use_cuda=True)

# Get tracking function
track = detect.track_stream(stream_url=stream_path, output_dir='data/results', save_result=True, display=True, filter_classes=['person'], target_fps=None)
#track = detect.track_webcam(cam_id=0, output_dir='data/results', save_result=True, display=True, filter_classes=['person'], target_fps=None)

#create a thread that moves the drone according to the last position of the tracked person

def drone_mover(tello:Tello,lock1:threading.Lock):
    while MOVE_DRONE:
        time.sleep(0.05)
        lock1.acquire()
        box = copy.copy(TRACKED_BOX)
        lock1.release()
        if box is not None:
            if mv.rotate(box, 960, 5, tello):
                continue
            mv.moveZ(box, 1.90, tello)
    
            
lock = threading.Lock()

drone_thread = threading.Thread(target=drone_mover,args=[tello,lock])

drone_thread.start()

# Loop over track to retrieve outputs of each frame 
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    frame = np.asarray(frame)
    #time.sleep(0.2)
    if RESEARCH_MODE:
        for i in range(len(ids)-1,-1,-1): #read in the inverse order because when the tracked box is lost and other persons are detected,
            if CLASSIFY: #the last box that appeared is the box of the person we tracked before, if it is still on screen
                y_max,x_max,_ = np.shape(frame)
                [x0,y0,x1,y1] = reshape_bbox(bbox_xyxy[i],x_max,y_max)
                BOX_SURFACE = get_surface([x0,y0,x1,y1])
                crop = frame[y0:y1,x0:x1,::-1]
                try:
                    crop = Image.fromarray(crop)
                    prediction = predict(crop)
                except:
                    prediction = 0
                print("Person number " + str(ids[i]) + " detected as ", prediction)
            else:
                prediction = 1
            if prediction:
                RESEARCH_MODE = False
                TRACKED_ID = ids[i]
                break
    else:
        if TRACKED_ID not in ids: #if the tracked person is out of the field of view, stop tracking
            print("entering back research mode")
            lock.acquire()
            TRACKED_BOX = None
            lock.release()
            RESEARCH_MODE = True
            TRACKED_ID = -1
            continue
        #Track the TRACKED_ID
        print("tracking ", TRACKED_ID)
        idx = ids.index(TRACKED_ID)
        y_max,x_max,_ = np.shape(frame)
        lock.acquire()
        TRACKED_BOX = reshape_bbox(bbox_xyxy[idx],x_max,y_max)
        lock.release()
        
lock.acquire()
time.sleep(0.5)
MOVE_DRONE = False
lock.release()

drone_thread.join()

tello.connect()
tello.streamoff()
tello.land()