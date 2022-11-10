import torch
import torchvision
import torchvision.transforms as transforms

'''
    Data
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = torchvision.datasets.CIFAR10(root='./_data', train=True, download=True, transform=transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
tests_ds = torchvision.datasets.CIFAR10(root='./_data', train=False, download=True, transform=transform)
tests_dl = torch.utils.data.DataLoader(tests_ds, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
    Network
'''
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Network()

'''
    Accuracy
'''
class Accuracy(nn.Module):
    def forward(self, dz, dy):
        return dy == dz.max(axis=1)[1] / len(dz)

'''
    TRAINER
'''
import torch.optim as optim

from firetorch import Model

network = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

model = Model(network, criterion, optimizer, metrics=[Accuracy()])

model.train_epoch(train_dl)
