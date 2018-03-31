import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
download = True  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

## networks
class MLPNet1(nn.Module):
    def __init__(self):
        super(MLPNet1, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP1"

class MLPNet2(nn.Module):
    def __init__(self):
        super(MLPNet2, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP2"

class MLPNet3(nn.Module):
    def __init__(self):
        super(MLPNet3, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP3"

class MLPNet4(nn.Module):
    def __init__(self):
        super(MLPNet4, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP4"

class MLPNet5(nn.Module):
    def __init__(self):
        super(MLPNet5, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP5"

class MLPNet6(nn.Module):
    def __init__(self):
        super(MLPNet6, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP6"

class MLPNet7(nn.Module):
    def __init__(self):
        super(MLPNet7, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP7"

class MLPNet8(nn.Module):
    def __init__(self):
        super(MLPNet8, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP8"

class MLPNet9(nn.Module):
    def __init__(self):
        super(MLPNet9, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP9"

class MLPNet10(nn.Module):
    def __init__(self):
        super(MLPNet10, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def name(self):
        return "MLP10"

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
parameters = []

def train(model):
    if use_cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ceriation = nn.CrossEntropyLoss()
    # count number of parameters
    para = []
    for parameter in model.parameters():
        para.append(parameter)
    parameters.append(para[0].numel()+para[2].numel()+para[4].numel())
    for epoch in range(10):
        # trainning
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = Variable(x), Variable(target)
            optimizer.zero_grad()
            out = model(x)
            loss = ceriation(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            ave_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, ave_loss/total_cnt))
        if epoch == 9:
            train_loss.append(ave_loss/total_cnt)
            train_accuracy.append(correct_cnt/total_cnt)
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = ceriation(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            ave_loss += loss.data[0] 
            if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
                print( '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(epoch, batch_idx+1, ave_loss/total_cnt, correct_cnt * 1.0 / total_cnt))
        if epoch == 9:
            test_loss.append(ave_loss/total_cnt)
            test_accuracy.append(correct_cnt/total_cnt)


# training
model = MLPNet1()
train(model)
model = MLPNet2()
train(model)
model = MLPNet3()
train(model)
model = MLPNet4()
train(model)
model = MLPNet5()
train(model)
model = MLPNet6()
train(model)
model = MLPNet7()
train(model)
model = MLPNet8()
train(model)
model = MLPNet9()
train(model)
model = MLPNet10()
train(model)


plt.title('model loss')
plt.scatter(parameters,train_loss,label='training loss')
plt.scatter(parameters,test_loss,label='testing loss')
plt.xlabel('parameters')
plt.ylabel('loss')
plt.ylim([0,0.001])
plt.legend()
plt.savefig('hw1_3_2_loss.png')
plt.show()

plt.title('model accuracy')
plt.scatter(parameters,train_accuracy,label='training accuracy')
plt.scatter(parameters,test_accuracy,label='testing accuracy')
plt.xlabel('parameters')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('hw1_3_2_accuracy.png')
plt.show()
