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
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(in_channels=132, out_channels=360, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.Dropout(0.5),
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
model.load_state_dict(torch.load("./classifier/no_false_positives.pt"))

class OneDataset(Dataset):
     
    def __init__(self, img): #roan = 1, other = 0 for label
          self.img = img
          self.label = 0
    
    def __len__(self):
         return 1
    
    def __getitem__(self, idx):
        return transforms(self.img), self.label
     
valid_loader = DataLoader(dataset=OneDataset('../datasets/custom/val/roan'), batch_size=32, shuffle=False)

transforms = transforms.Compose([transforms.Resize((128,128)),
                                 transforms.ToTensor()])

def predict(image):
    loader = DataLoader(dataset=OneDataset(image), batch_size=1, shuffle=False) #load the data in a 1 image batch
    for data in loader:
        inputs, labels = data    
        inputs, labels = inputs.cuda(), labels.cuda() 

        with torch.no_grad():
            model.eval()
            _, probs = model(inputs) #get the prediction on the batch
        return int(torch.argmax(probs[0]))
            


