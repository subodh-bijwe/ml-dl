import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision, sys
      
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 10


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
# load pretrain model & modify it

model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
model.to(device)
print(model)
# sys.exit()
# load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
for epoch in range(num_epochs):
    losses = []
        
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
