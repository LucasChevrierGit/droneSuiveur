from djitellopy import Tello
import math
import numpy as np
import time
from numpy.linalg import inv


f = 0.020
X_sensor = 0.0027654763098670576
F = 752.63 #apparent focal length (in pixels)

def rotate(bbox_xyxy,x_max,tolerance,tello:Tello):
    [x0,_,x1,_] = bbox_xyxy
    x = (x0+x1)/2
    xo = x_max/2
    dpx = abs(xo-x)
    t_pixel = X_sensor/x_max
    dpx = dpx*t_pixel
    alpha = round(360*math.atan(dpx/f))
    if alpha <= tolerance:
        return False
    if x > xo:
        print('attempting to rotate ', alpha, ' degrees cw')
        tello.rotate_clockwise(alpha)
        return True
    else:
        print('attempting to rotate ', alpha, ' degrees counter cw')
        tello.rotate_counter_clockwise(alpha)
        return True

def getZ(width_px, true_size):
    return (true_size*F)/width_px
    
def moveZ(bbox_xyxy,H,tello:Tello,hpx_interval=[450,550]):
    [_,y0,_,y1] = bbox_xyxy
    Zmin = getZ(hpx_interval[1],H)
    Zmax = getZ(hpx_interval[0],H)
    Zmid = (Zmin+Zmax)/2
    hpx = abs(y0-y1)
    if hpx > hpx_interval[1]:
        print('too close ! moving back 20cm')
        delta_Z = abs(Zmid-getZ(hpx,H))*100
        if delta_Z < 30:
            tello.move_back(30)
        elif delta_Z > 200:
            tello.move_back(200)
        else:
            tello.move_back(round(delta_Z))
    elif hpx < hpx_interval[0]:
        print('too far ! moving forward 20cm.')
        delta_Z = abs(Zmid-getZ(hpx,H))*100
        if delta_Z < 30:
            tello.move_forward(30)
        elif delta_Z > 200:
            tello.move_forward(200)
        else:
            tello.move_forward(round(delta_Z))

#method using the pixel jacobian

def jacobian(u,v,Z):
    mat = [[-f/Z,0,u/Z,u*v/f,-(f+(u**2)/f),v],[0,-f/Z,v/Z,f+(v**2)/f,-u*v/f,-u]]
    return mat

def get_velocities(box,r_box,H,lamb):
    [_,y0,_,y1] = box
    hpx = abs(y0-y1)
    Z = getZ(hpx,H)
    tri_jac = np.matrix(jacobian(box[0],box[1],Z) + jacobian(box[0],box[3],Z) + jacobian(box[2],box[3],Z))
    inv_jac = inv(tri_jac)
    px_velocities = lamb*np.matrix([[box[0]-r_box[0]],
                                    [box[1]-r_box[1]],
                                    [box[0]-r_box[0]],
                                    [box[3]-r_box[3]],
                                    [box[2]-r_box[2]],
                                    [box[3]-r_box[3]]])
    velocities = np.matmul(inv_jac,px_velocities)
    return velocities