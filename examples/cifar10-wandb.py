import torch
import torchvision
import torchvision.transforms as transforms

'''
    Data
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = torchvision.datasets.CIFAR10(root='./_data', train=True, download=True, transform=transform)
valid_ds = torchvision.datasets.CIFAR10(root='./_data', train=False, download=True, transform=transform)

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
        return torch.sum(dy == dz.max(axis=1)[1]) / len(dz)

'''
    Wandb
'''
import os
import wandb

os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_MODE'] = 'offline'

wandb.login()

'''
    Train
'''
import torch.optim as optim

from firetorch import Model

EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SCHEDULER_LAMBDA = 0.90
STEPS = 100

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=0, drop_last=True)

network = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: SCHEDULER_LAMBDA ** epoch)

model = Model(network, criterion, optimizer, metrics=[Accuracy()])

wandb.init(project='Cifar10', entity='devprojects', config={'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'scheduler_lambda': SCHEDULER_LAMBDA})

for epoch in range(EPOCHS):
    print('epoch:', epoch)

    train_metrics = model.train_epoch(train_dl, max_steps=STEPS)
    valid_metrics = model.valid_epoch(valid_dl, max_steps=STEPS)

    preds = model.predict(valid_dl, max_steps=1)
    images, labels = next(iter(valid_dl))
    samples = []
    for pred, image, label in zip(preds, images, labels):
        p = pred.argmax(axis=0)
        l = label.item()
        sample = wandb.Image(image, caption=f'{classes[p]}-{classes[l]}-{p}-{l}')
        samples.append(sample)
    print(*preds.argmax(axis=1), sep='\t')
    print(*labels.cpu().numpy(), sep='\t')

    wandb.log({'samples': samples, 'valid': valid_metrics, 'train': train_metrics })

    print()

wandb.finish()
