import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F

torch.manual_seed(1)
Record = []
Loss = []

def Gradientrecord(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
    grad_norm = grad_all ** 0.5
    return grad_norm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.out(x)
        return output

if __name__ == '__main__':
    EPOCH = 1000
    BATCH_SIZE = 150
    LR = 0.001
    DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    net = Net()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28*28))
            b_y = Variable(y, requires_grad = False)
            output = net(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, EPOCH, step+1, len(train_data)//BATCH_SIZE, loss.data[0]))
            Record.append(Gradientrecord(net))
            Loss.append(loss.data[0])

    np.save("Record.npy", Record)
    np.save("Loss.npy", Loss)

    print('Loading the Loss...')
    Loss = np.load("Loss.npy")
    print('Loading the Gradient...')
    Gradient = np.load("Record.npy")

    print('Drawing the Gradient curve...')
    plt.plot(Gradient, 'r', label = 'Gradient norm')
    plt.legend(loc = 'upper right')
    plt.title('Q1-2-2-1')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('grad')
    plt.savefig('Q1-2-2-1.png')
    plt.show()

    print('Drawing the loss...')
    plt.plot(Loss, 'b', label = 'Loss')
    plt.legend(loc = 'upper right')
    plt.title('Q1-2-2-2')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig('Q1-2-2-2.png')
    plt.show()
