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

## random suffle labels
new_train_x = []
new_train_target = []
for batch_idx, (x, target) in enumerate(train_loader):
    x, target = Variable(x), Variable(target)
    perm_list = torch.randperm(100)
    target_perm = target[perm_list]
    batch_idx
    if batch_idx == 0:
        new_train_x = np.array([x.data.numpy()])
        new_train_target = np.array([target_perm.data.numpy()])
    else:
        new_train_x = np.concatenate((new_train_x,np.array([x.data.numpy()])), axis=0)
        new_train_target = np.concatenate((new_train_target,np.array([target_perm.data.numpy()])), axis=0)

new_test_x = []
new_test_target = []
for batch_idx, (x, target) in enumerate(test_loader):
    x, target = Variable(x), Variable(target)
    perm_list = torch.randperm(100)
    target_perm = target[perm_list]
    batch_idx
    if batch_idx == 0:
        new_test_x = np.array([x.data.numpy()])
        new_test_target = np.array([target_perm.data.numpy()])
    else:
        new_test_x = np.concatenate((new_test_x,np.array([x.data.numpy()])), axis=0)
        new_test_target = np.concatenate((new_test_target,np.array([target_perm.data.numpy()])), axis=0)

## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
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
        return "MLP"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def name(self):
        return "LeNet"

train_loss = []
test_loss = []
model = MLPNet()
if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ceriation = nn.CrossEntropyLoss()

for epoch in range(4000):
    # trainning
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for i in range(len(new_train_x)):
        x = Variable(torch.FloatTensor(new_train_x[i]))
        target = Variable(torch.LongTensor(new_train_target[i]))
        optimizer.zero_grad()
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += len(new_train_x[i])
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 or (i+1) == len(new_train_x):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, i+1, ave_loss/total_cnt))
    train_loss.append(ave_loss/total_cnt)
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for i in range(len(new_test_x)):
        x = Variable(torch.FloatTensor(new_test_x[i]))
        target = Variable(torch.LongTensor(new_test_target[i]))
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += len(new_test_x[i])
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss += loss.data[0] 
        if(i+1) % 100 == 0 or (i+1) == len(new_test_x):
            print( '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(epoch, i+1, ave_loss/total_cnt, correct_cnt * 1.0 / total_cnt))
    test_loss.append(ave_loss/total_cnt)
print('finish training')

plt.title('training loss')
plt.plot(range(1,4001),train_loss,label='training loss')
plt.plot(range(1,4001),test_loss,label='testing loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('hw1_3_1.png')

