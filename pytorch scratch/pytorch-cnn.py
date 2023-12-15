import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create FC network
class NN(nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x   
    
# Create simple CNN    
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super(CNN, self).__init__()
        """
        n_out = floor((n_in + 2*padding - kernel_size)/stride) + 1
        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        # print("original :", x.shape)    # torch.Size([16, 1, 28, 28]) 16 images, 1 channel per image of size 28x28
        x = F.relu(self.conv1(x))
        # print("conv1    :", x.shape)    # torch.Size([16, 8, 28, 28]) 16 images, 8 channels per image of size 28x28
        x = self.pool(x)
        # print("pool1    :", x.shape)    # torch.Size([16, 8, 14, 14]) 16 images, 8 channels per image of size 14x14. Image size is reduced due to stride = 2
        x = F.relu(self.conv2(x))
        # print("conv2    :", x.shape)    # torch.Size([16, 16, 14, 14]) 16 images, 16 channels per image of size 14x14
        x = self.pool(x)
        # print("pool2    :", x.shape)    # torch.Size([16, 16, 7, 7]) 16 images, 8 channels per image of size 7x7. Image size is further reduced due to stride = 2
        x = x.reshape(x.shape[0], -1)
        # print("reshape  :", x.shape)    # torch.Size([16, 784]) 16 images, each being a vector of shape 784 (16x7x7)
        x = self.fc1(x)
        # print("fc1      :", x.shape)    # torch.Size([16, 10]) 16 images, each being a vector of shape 10 (num_classes)
        
        return x
    
      
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
# hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 10
load_model = True


def save_checkpoint(state, epoch=0):
    print("=> saving checkpoint")
    filename="my_checkpoint"+str(epoch)+".pth.tar"
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if load_model:
    load_checkpoint(torch.load("my_checkpoint9.pth.tar"))

    
for epoch in range(num_epochs):
    losses = []
    
    if epoch % 3 == 0:
        check_point = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(check_point, epoch)
        
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data = data.to(device)
        target = target.to(device)
        
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch+1} was {mean_loss:.5f}')
        
        
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking on train data")
    else:
        print("Checking on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        acc = (float(num_correct)/float(num_samples))*100
    model.train()
    return acc

print(check_accuracy(train_loader, model))
print(check_accuracy(test_loader, model))
