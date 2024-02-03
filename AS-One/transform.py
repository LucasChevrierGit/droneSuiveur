import numpy as np
from classifier.predict import predict


#model = YOLO("yolov5s.pt")

model = None

class Jacobian():

   

    def __init__(self,u,v,Z,f):

        self.jacobian(u,v,Z,f)
        self.jac = self.jacobian(u,v,Z,f)

    def jacobian(self,u,v,Z,f):
        j = np.zeros(12).reshape(2,6)
        j[0] = np.array([-f/Z, 0, u/Z, u*v/f, -(f+ u**2/f), v])
        j[1] = np.array([0, -f/Z, v/Z, f+v**2/f, -u*v/f, -u])
        return j


class Transform():

    def __init__(self,points,u,v,Z,f,lamb):
        self.points = points
        self.jac = []
        self.u = u
        self.v = v
        for i in range(len(u)):
             matrix = Jacobian(u[i],v[i],Z,f).jac
             self.jac.append(matrix)
        mat = np.array(self.jac).reshape(6,6)
        print(mat)

        self.jac = np.linalg.inv(mat)
        self.lamb = lamb
    


    def transform(self):

        self.points = np.matrix(self.points).reshape(6,1)

        uv = np.matrix(np.subtract(self.points, np.matrix([self.u,self.v]).reshape(6,1)))

        return self.lamb* np.matmul( np.matrix(self.jac) , uv)
    



from PIL import Image
from PIL.ExifTags import TAGS


def getRoanCrop(img,model):
    
    res = model([img,img])
    size = img.size
    boxs = res[0].boxes.xyxy.cpu().numpy()

    if len(boxs) ==0 :
        for box in boxs :
            print(box)
            img = img.crop(box)
            if predict(img) :
                return size , box

    print("No Roan found")
    return (-1,-1), []


    

 


def getDistance(img):
    

    #xy xy
    (xSize, ySize), box = getRoanCrop(img, model)


    print(box)

    if not len(box) == 0 :

        # self,points,u,v,Z,f,lamb):
        u = np.array([box[0],box[2],box[0]])
        v = np.array([box[1],box[3],box[3]])


        f = 0.020
        H = 1.9

        sensor_width = 0.0028
        h = np.abs(box[3] - box[1]) * sensor_width/ySize
        Z = f * H/h
        print(Z)

        ratio =  h/np.abs(box[2]-box[0])

        # points x y x y x y
        y = np.abs(box[3]-box[1])
        x = np.abs(box[2]-box[0])

        points = np.array([(xSize-h)/2, (ySize-ratio*y)/2 , (xSize+h)/2 , (ySize+ratio*y)/2, (xSize-h)/2, (ySize+ratio*y)/2 ])

        lamb = 1/10 # 1/x, x repetition to get to the desired pixel
        tf = Transform(points,u,v,Z,f,lamb )
        res = tf.transform() #x,y,z , wx, wy, wz

        print(res)
        return res,box,points
    print('no Roan')
    return []
    

def transform(box,xSize, ySize):
    print('box is: ', box)
    f = 0.020
    H = 1.9
    sensor_width = 0.0028

    if not len(box) == 0 :

        # self,points,u,v,Z,f,lamb):
        u = np.array([box[0],box[2],box[0]])
        v = np.array([box[1],box[3],box[3]])


    
        h = np.abs(box[3] - box[1])# * sensor_width/ySize
        Z = f * H/h
        print(Z)

        ratio =  1

        # points x y x y x y
        y = np.abs(box[3]-box[1])
        x = np.abs(box[2]-box[0])

        points = np.array([xSize/4, ySize/4, xSize*(3/4), ySize*(3/4), xSize/4, ySize*(3/4)])
        #points = np.array([(xSize-h)/2, (ySize-ratio*y)/2 , (xSize+h)/2 , (ySize+ratio*y)/2, (xSize-h)/2, (ySize+ratio*y)/2 ])

        print("points are: ", points)
        lamb = 1/10 # 1/x, x repetition to get to the desired pixel
        tf = Transform(points,u,v,Z,f,lamb )
        res = tf.transform() #vx,vy,vz , wx, wy, wz

        print('the required velocities are: ', res)
        return res
    print('no Roan')
    return []








