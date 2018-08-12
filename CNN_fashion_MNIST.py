
# coding: utf-8

# In[47]:


import numpy as np
import os
from tqdm import tqdm

import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable as V
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

print('PyTorch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
print('Is GPU available:', torch.cuda.is_available())


# In[48]:


# hyperparameters
n_epochs = 10
batchsize = 128
learning_rate = 0.001
use_gpu = torch.cuda.is_available() # CUDA環境があるかどうか


# In[49]:


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
fashion_mnist_train = datasets.FashionMNIST(root = './data', train = True, transform = tf, download = True)
fashion_mnist_valid = datasets.FashionMNIST(root = './data', train = False, transform = tf)

fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size = batchsize, shuffle = True, num_workers = 4)
fashion_mnist_valid_loader = DataLoader(fashion_mnist_valid, batch_size = batchsize, shuffle = False, num_workers = 4)

print('train_data:', len(fashion_mnist_train))
print('validation_data:', len(fashion_mnist_valid))
print('data shape:', fashion_mnist_train[0][0].size())


# In[50]:


class CNN_FASHION_MNIST(nn.Module):
    def __init__(self):
        super(CNN_FASHION_MNIST, self).__init__()
        self.layers1 = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size = 4, padding = 2),
                            nn.MaxPool2d(kernel_size = 2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Dropout2d(0.25)
        )
        self.layers2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size = 4, padding = 2),
                            nn.MaxPool2d(kernel_size = 2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Dropout2d(0.25)
        )
        self.mlp = nn.Sequential(
                            nn.Linear(64 * 7 * 7, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.25),
                            nn.Linear(256, 10)
        )
    def forward(self, x):
        out = self.layers1(x)
        out = self.layers2(out)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


# In[51]:


net = CNN_FASHION_MNIST()
if use_gpu:
    net.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

print('Model:\n', net)
print('\nLoss function:\n', loss_fn)
print('\nOptimizer:\n', optimizer)


# In[52]:


def train(train_loader):
    net.train()
    running_loss = 0
    for inputs, targets in train_loader:
        if use_gpu:
            inputs = V(inputs.cuda())
            targets = V(targets.cuda())
        else:
            inputs = V(inputs)
            targets = V(targets)
        
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data.item()
        
    train_loss = running_loss / len(train_loader)
    return train_loss


# In[59]:


def valid(valid_loader):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, targets in valid_loader:
        with torch.no_grad():
            if use_gpu:
                inputs = V(inputs.cuda())
                targets = V(targets.cuda())
            else:
                inputs = V(inputs)
                targets = V(targets)
        
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
        
        running_loss += loss.data.item()
        _, preds = torch.max(outputs.data, dim = 1)
        correct += (preds == targets).float().sum()
        total += targets.size(0)
    
    val_loss = running_loss / len(valid_loader)
    val_acc = correct / total
    return val_loss, val_acc


# In[60]:


DIRNAME = './fashion_mnist_result/'
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)

loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in tqdm(range(n_epochs)):
    loss = train(fashion_mnist_train_loader)
    val_loss, val_acc = valid(fashion_mnist_valid_loader)
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print('epoch[%d/%d] loss:%1.4f val_loss:%1.4f val_acc:%1.4f' % (epoch + 1, n_epochs, loss, val_loss, val_acc))

np.save(DIRNAME + 'loss_list.npy', np.array(loss_list))
np.save(DIRNAME + 'val_loss_list.npy', np.array(val_loss_list))
np.save(DIRNAME + 'val_acc_list.npy', np.array(val_acc_list))
torch.save(net.state_dict(), DIRNAME + 'CNN_FASHION_MNIST.pth')

