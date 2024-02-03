import subprocess as sp

sp.call(["python", "detect_custom.py", "--weights", "yolov5s.pt", "--img", "640", "--conf", "0.25", "--source", "../datasets/coco_person/valid/images", "--save-crop"])