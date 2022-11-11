import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

valid_ds = torchvision.datasets.CIFAR10(root='./_data', train=False, download=True, transform=transform)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=0, drop_last=True)

batch = next(iter(valid_dl))
print(batch)

