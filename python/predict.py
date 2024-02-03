import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import os
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 2
BATCH_SIZE = 32

class LeNet5(nn.Module):

        def __init__(self, n_classes):
            super(LeNet5, self).__init__()
            
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=3, out_channels=60, kernel_size=11, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=60, out_channels=132, kernel_size=7, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=132, out_channels=360, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
            )

            self.classifier = nn.Sequential(
                nn.Linear(in_features=1440, out_features=360),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=360, out_features=84),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=84, out_features=n_classes),
            )


        def forward(self, x):
            #print("x is :\n", x)
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)
            return logits, probs
        
        def toString(self):
            return "LeNet5"


model = LeNet5(N_CLASSES).to(DEVICE)


model.load_state_dict(torch.load("./train1_weights/epoch10.pt"))

transforms = transforms.Compose([transforms.Resize((128,128)),
                                 transforms.ToTensor()])

class TestDataset(Dataset):
     
    def __init__(self, img_dir, label): #roan = 1, other = 0 for label
          self.img_dir = img_dir
          self.label = label
          self.img_names = os.listdir(img_dir)
    
    def __len__(self):
         return len(self.img_names)
    
    def __getitem__(self, idx):
         image = Image.open(self.img_dir + '/' + self.img_names[idx])
         image = transforms(image)
         return image, self.label
     
valid_loader = DataLoader(dataset=TestDataset('../datasets/custom/val/roan', 1), batch_size=32, shuffle=False) #label doesn't matter for detection.

ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()

res = []
#input batch de taill 32 avec des image de taille 32*32
for data in valid_loader:
    inputs, labels = data    
                   # this is what you had
    inputs, labels = inputs.cuda(), labels.cuda()  

    with torch.no_grad():
        model.eval()
        _, probs = model(inputs) #resultat c'est argmax(res[i])
        res.append(probs)

acc = 0
tot = 0
#print the predictions in a txt file
f = open('results/prediction.txt', 'w')
for batch_res in res:
    for tensor in batch_res:
        tot += 1
        detected_class = int(torch.argmax(tensor))
        acc += detected_class
        f.write(str(detected_class) + '\n')
f.close()
if tot:
    acc = acc/tot
print("accuracy :", acc)
        


