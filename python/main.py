import subprocess as sp
import os

FOLLOW_ANYONE = False

def relocate_drone(initial_coordinates=[0.0, 0.0, 0.0, 0.0]):
    print(initial_coordinates)

def floatVect(vect):
    for i in range(len(vect)):
        vect[i] = float(vect[i])

sp.call(["python", "../yolov5/detect_custom.py", "--weights", "yolov5s.pt", "--img", "640", "--conf", "0.25", "--source", "../datasets/custom/test/overtest", "--save-crop", "--save-txt"])
print("yolo finished")

if not FOLLOW_ANYONE:
    sp.call(["python", "predict.py"]) #look for the person to follow
    f = open("results/prediction.txt", 'r') #open the results of the person to follow detection
    pred_lines = f.readlines()
    f.close()
    #os.remove("results/prediction.txt")

f = open("results/placements.txt", 'r') #open the location of boxes file
place_lines = f.readlines()
f.close()
os.remove("results/placements.txt")

crop_dir = os.listdir("./results/crops")
for path in crop_dir:
    #os.remove("./results/crops/" + path)
    None

if len(pred_lines) == 0 or len(place_lines) == 0: #no person detected
    exit()

for i in range(len(pred_lines)):
    if pred_lines[i] or FOLLOW_ANYONE:
        coordinates = place_lines[i].split(" ")
        floatVect(coordinates)
        coordinates.pop(0)
        relocate_drone(initial_coordinates=coordinates)
        break
