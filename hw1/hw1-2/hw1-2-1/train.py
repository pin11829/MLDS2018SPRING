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

## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
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
        return "MLP"

layer_1 = []
whole_layer = []
plot_loss = []
category = np.full(10,1)

## training
for iter in range(8):
    model = MLPNet()
    if use_cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ceriation = nn.CrossEntropyLoss()

    layer_1_para = []
    whole_layer_para = []
    local_plot_loss =[]

    for epoch in range(30):
        # trainning
        if (epoch % 3) == 0:
            para = []
            for parameter in model.parameters():
                para.append(parameter)
            layer_1_para.append(list(para[0].view(para[0].numel()).data.numpy()))
            tmp = para[0].view(para[0].numel()).data.numpy()
            tmp2 = para[2].view(para[2].numel()).data.numpy()
            tmp3 = para[4].view(para[4].numel()).data.numpy()
            tmp = np.concatenate((tmp, tmp2), axis=0)
            tmp = np.concatenate((tmp, tmp3), axis=0)
            whole_layer_para.append(list(tmp))
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = ceriation(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            ave_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, accuracy: {:.6f}'.format(epoch, batch_idx+1, correct_cnt * 1.0 / total_cnt))
        if (epoch % 3) == 0:
            local_plot_loss.append(correct_cnt * 1.0 / total_cnt)
    if iter==0:
        layer_1 = np.array(layer_1_para)
        whole_layer = np.array(whole_layer_para)
        plot_loss = np.array(local_plot_loss)
    else:
        layer_1 = np.concatenate((layer_1,layer_1_para),0)
        whole_layer = np.concatenate((whole_layer,whole_layer_para),0)
        plot_loss = np.concatenate((plot_loss,local_plot_loss),0)
        category = np.concatenate((category,np.full(10,iter+1)),0)
print('finish training')

pca = PCA(n_components = 2)
layer_1_pca = pca.fit_transform(layer_1)
whole_layer_pca = pca.fit_transform(whole_layer)


fig = plt.figure()

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

for i in range(80):
    ax.text(layer_1_pca[i][0], layer_1_pca[i][1], str(round(plot_loss[i]*100,2)),color='C'+str(category[i]), style='italic')

ax.axis([-8, 10, -8, 10])

fig.savefig('hw1_2_1.png')