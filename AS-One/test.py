import socket
import time
import cv2

tello_ip = '192.168.10.1'
tello_port = 8889
tello_address = (tello_ip, tello_port)
host ='192.168.1.15'
#mypc_address = (host, port)
local_ip=''
socket = socket.socket (socket.AF_INET, socket.SOCK_DGRAM)
socket.bind ((local_ip,8889))
socket.sendto ('command'.encode (' utf-8 '), tello_address)
socket.sendto ('streamon'.encode (' utf-8 '), tello_address)
print ("Start streaming")
capture = cv2.VideoCapture ('udp:/0.0.0.0:11111',cv2.CAP_FFMPEG)
if not capture.isOpened():
    capture.open('udp:/0.0.0.0:11111')

while True:
    ret, frame =capture.read()
    print(ret)
    if(ret):
        cv2.imshow('frame', frame)
    if cv2.waitKey (1)&0xFF == ord ('q'):
        break
capture.release ()
cv2.destroyAllWindows ()
socket.sendto ('streamoff'.encode (' utf-8 '), tello_address)
