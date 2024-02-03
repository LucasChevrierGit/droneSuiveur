import time
import cv2
import threading
from copy import copy

cap = cv2.VideoCapture(0)
k_read = True
result = None

def task(lock:threading.Lock,cap):
    #print('thread started')
    global k_read
    global result
    lock.acquire()
    while k_read:
        times += 1
        lock.release()
        result = cap.read()
        lock.acquire()
    lock.release()

lock = threading.Lock()

reading_thread = threading.Thread(target=task, args=(lock,cap))
reading_thread.start()

while True:

    time.sleep(1)

    #print('stop reading')
    lock.acquire()
    k_read = False
    lock.release()

    #print('wait for thread to die')
    reading_thread.join()
    k_read = True

    #print('take the results')
    ret,frame = copy(result) #no need for a lock here, result is accessed only when the thread is dead

    #print('start a new thread')
    reading_thread = threading.Thread(target=task, args=(lock,cap))
    reading_thread.start()

    if not ret:
        break

    #print('show the image')
    cv2.imshow('test', frame)
    
    #print("test if 'q' is pressed")
    if cv2.waitKey(25) & 0xFF == ord('q'):
        lock.acquire()
        k_read = False
        lock.release()
        break

