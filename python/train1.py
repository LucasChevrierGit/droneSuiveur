import numpy as np

from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from torchvision import datasets,transforms

import os

import pandas as pd

from PIL import Image,ImageOps

import matplotlib.pyplot as plt

from random import shuffle


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
DATASET_SIZE = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15
CUSTOM_DATA = True
ALEXNET_OVER_LENET5 = False
IMG_SIZE = 32
N_CLASSES = 10

#######################################################
#               Define Dataset Class
#######################################################

# define transforms
class AddGaussianNoise(object): #add noise to the dataset to make the model more robust
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if ALEXNET_OVER_LENET5:
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])
else:
    noisy_transform = transforms.Compose([transforms.Resize((128,128)),
                                 transforms.ToTensor(),
                                 AddGaussianNoise(0,0.01)])
    transforms = transforms.Compose([transforms.Resize((128,128)),
                                 transforms.ToTensor()])
if CUSTOM_DATA:
    def class_to_idx(label):
        if label == "roan":
            return 1
        else:
            return 0

# define the datasets

    class CustomImageDataset(Dataset):
        def __init__(self, img_dir, transform=None, target_transform=None):
            self.img_dir = img_dir
            self.labels = os.listdir(self.img_dir)
            img_paths_temp = []
            for label in self.labels:
                path = self.img_dir + '/' + label
                class_paths = os.listdir(path)
                for im_name in class_paths:
                    img_paths_temp.append(path + '/' + im_name)
            self.img_paths = []
            shuffle(img_paths_temp) #randomize the training images chosen
            i = 0
            for img_path in img_paths_temp:
                self.img_paths.append(img_path)
                i += 1
                if i > DATASET_SIZE: break #select only a certain amount
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]

            image = Image.open(img_path)
            if not ALEXNET_OVER_LENET5:
                #image = ImageOps.grayscale(image)
                None

            label = img_path.split('/')[-2]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, class_to_idx(label)
        
    print("---------------------------\nLoading custom dataset...\n---------------------------")
        
    train_set = CustomImageDataset('../datasets/custom/train', transform=noisy_transform)
    valid_set = CustomImageDataset('../datasets/custom/val', transform=transforms)

    print("Dataset created, creating loader...")

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False)
    print("---------------------------\nLoading successful.\n---------------------------")

#######################################################
#               Define train functions
#######################################################

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    plt.show()
    
    # change the plot style to default
    plt.style.use('default')


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    print("---------------------------\nStarting training...\n---------------------------")
    # set objects for storing metrics
    valid_accuracies = [0]
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            if max(valid_accuracies) < valid_acc:
                torch.save(model.state_dict(), "./best.pt")
            valid_accuracies.append(valid_acc)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    print("---------------------------\nTraining ended successfuly!\n---------------------------")
    return model, optimizer, (train_losses, valid_losses)

if not(CUSTOM_DATA):

    # download and create datasets
    train_dataset = datasets.MNIST(root='mnist_data', 
                                train=True, 
                                transform=transforms,
                                download=True)

    valid_dataset = datasets.MNIST(root='mnist_data', 
                                train=False, 
                                transform=transforms)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)


if ALEXNET_OVER_LENET5:
    class AlexNet(nn.Module):

        def __init__(self, n_classes):
            super(AlexNet, self).__init__()

            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Linear(in_features=6400, out_features=4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=n_classes)
            )
        
        def forward(self, x):
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)
            return logits, probs

        def toString(self):
            return "AlexNet"
else:
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
    
print("---------------------------\nCreating Model...\n---------------------------")

torch.manual_seed(RANDOM_SEED)
model = None
if ALEXNET_OVER_LENET5:
    model = AlexNet(N_CLASSES).to(DEVICE)
    LEARNING_RATE = 0.01
else:
    model = LeNet5(N_CLASSES).to(DEVICE)
    LEARNING_RATE = 0.00003
    #last best lr = 0.00003

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
print("Model Created...\n---------------------------")


model, optimizer, (train_losses, valid_losses) = training_loop(model, criterion, optimizer, train_loader, valid_loader, 15, DEVICE)

torch.save(model.state_dict(), "./weights" + model.toString() + ".pt")
plot_losses(train_losses, valid_losses)




