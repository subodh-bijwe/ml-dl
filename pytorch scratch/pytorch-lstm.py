import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms


      
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
# hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 2

# Create a RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_seq x features
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # forward propagation
        
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(x.shape[0], -1)
        out = self.fc1(out)
        
        return out
        
# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print("Epoch: ", epoch+1)
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data = data.to(device).squeeze(1)
        target = target.to(device)
        
        # data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        loss = criterion(scores, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        
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
            x = x.to(device).squeeze(1)
            y = y.to(device)
            
            # x = x.reshape(x.shape[0], -1)
            # print(x.shape)
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
