# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import platform


# %%
if platform.platform()[:7] == 'Windows':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

# %%
# DONE download and train
# download=False & train=False
# train (bool, optional) – If True, creates dataset from training set, otherwise creates from test set.
# https://pytorch.org/vision/0.8/datasets.html
# train_dataset  = torchvision.datasets.FashionMNIST("./data/", 
#                                                    download=False, 
#                                                    transform=transforms.Compose([transforms.ToTensor()]))
# test_dataset  = torchvision.datasets.FashionMNIST("./data/", 
#                                                   download=False, 
#                                                   train=False, 
#                                                   transform=transforms.Compose([transforms.ToTensor()])) 

train_dataset  = torchvision.datasets.FashionMNIST("./chap05/data/", 
                                                   download=False, 
                                                   transform=transforms.Compose([transforms.ToTensor()]))
test_dataset  = torchvision.datasets.FashionMNIST("./chap05/data/", 
                                                  download=False, 
                                                  train=False, 
                                                  transform=transforms.Compose([transforms.ToTensor()])) 

# %%
print("Train dataset")
print(f"type: {type(train_dataset)}")
print(train_dataset)
print()
print("Test dataset")
print(f"type: {type(test_dataset)}")
print(test_dataset)

# %%
# DONE DataLoader
# - map-style and iterable-style datasets,
# - customizing data loading order,
# - automatic batching,
# - single- and multi-process data loading,
# - automatic memory pinning.
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# %%
print(f"type(train_loader): {type(train_loader)}")
print(f"len(train_loader): {len(train_loader)}")
print(dir(train_loader))
print()
print(f"len(test_loader): {len(test_loader)}")

# %%
# DONE labels_map - train_dataset
# train_dataset -> (image, target) X N
# train_dataset[img_xy] -> (image, target)
# train_dataset[img_xy][1] -> target
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 
              5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns * rows +1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')

plt.show()

# %%
# DONE __init__ & forward
# __init__: Network model definition
# forward: forward propagation
class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN,self).__init__()
        # DONE in_features and out_features
        # input_data.size() -> torch.Size([100, 1, 28, 28])
        # out.size() -> torch.Size([100, 784])
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self,input_data):
        # DONE view - reshape
        # the size -1 is inferred from other dimensions
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# %%
learning_rate = 0.001
model = FashionDNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
# TODO model.parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# %%
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
    
        if not (count % 50):    
            total = 0
            correct = 0        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)            
                test = Variable(images.view(100, 1, 28, 28))            
                outputs = model(test)            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

# %%
class FashionCNN(nn.Module):    
    def __init__(self):
        super(FashionCNN, self).__init__()       
        # TODO nn.xx / nn.functional / nn.Sequential 
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(32), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size=2, stride=2))       
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 
                                    nn.BatchNorm2d(64), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(2))        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)       
        return out

# %%
learning_rate = 0.001
model = FashionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# %%
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
    
        if not (count % 50):    
            total = 0
            correct = 0        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)            
                test = Variable(images.view(100, 1, 28, 28))            
                outputs = model(test)            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
