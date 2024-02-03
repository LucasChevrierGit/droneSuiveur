import subprocess as sp
import os

data_dir = "./data/images" # à personnaliser
results_dir = "./runs/detect/exp" #ne pas toucher

sp.call(["python", "detect.py", "--weights", "yolov5n.pt", "--img", "640", "--conf", "0.25", "--source", data_dir, "--save-crop"])

res = os.listdir(results_dir)
if 'crops' in res:
    crop_dir = results_dir + '/crops'
    crop_types = os.listdir(crop_dir)
    for crop_type in crop_types:
        if crop_type == 'person':
            persons = os.listdir(crop_dir + '/person')
            print(len(persons), "personnes ont été détectées. Voir dans le répertoire runs/detect/exp/crops/person du dossier courrant (yolov5)")