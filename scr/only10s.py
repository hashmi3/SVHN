import os
import time
#from __future__ import absolute_import
#from __future__ import print_function
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#d_pth = '/net/home/mhashmi/data/svHn/'

d_pth = '/u1/h3/hashmi/public_html/data/svHn/'
tst = 'test_32x32.mat'
trn = 'train_32x32.mat'
extra = 'extra_32x32.mat'

REBUILD_DATA = False    #Make it True to rebuild gray scale data



if __name__ == '__main__':
    if REBUILD_DATA:
        print("Building Data !!!!!!!!!!!!!")
        make_data()

    #model()

    fd = h5py.File('allData.h5', 'r')

    X_train = torch.tensor(fd['X_train'][:]).type(torch.float32)
    y_train =  torch.tensor(fd['y_train'][:]).type(torch.FloatTensor)

    X_test =  torch.tensor(fd['X_test'][:]).type(torch.float32)
    y_test =  torch.tensor(fd['y_test'][:]).type(torch.FloatTensor)

    X_val =  torch.tensor(fd['X_val'][:]).type(torch.float32)
    y_val =  torch.tensor(fd['y_val'][:]).type(torch.FloatTensor)

    fd.close()
    print('Training set', X_train.shape, y_train.shape)
    print('Validation set', X_val.shape, y_val.shape)
    print('Test set', X_test.shape, y_test.shape)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,32,5)
            self.conv2 = nn.Conv2d(32,64,5)
            #self.conv3 = nn.Conv2d(64,128,2)
            #self.dropout = nn.Dropout2d(0.25)

            x = torch.randn(32,32).reshape(-1,1,32,32)
            self._to_linear = None
            self.convs(x)

            self.fc1 = nn.Linear(self._to_linear, 800)
            self.fc2 = nn.Linear(800,10)

        def convs(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x,(2,2))
            #print('After conv1')
            #print('_to_linear shape is = ', x.shape, 'x[0]=',x[0].shape)

            #x =F.max_pool2d(F.relu(x), (2, 2))
            #print('After pool1')
            #print('_to_linear shape is = ', x.shape, 'x[0]=',x[0].shape)

            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            #print('After pool2')
            #print('_to_linear shape is = ', x.shape, 'x[0]=',x[0].shape)

            #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

            #print('After pool3')
            #print('_to_linear shape is = ', x.shape, 'x[0]=',x[0].shape)
            #x = self.dropout(x)

            if self._to_linear is None:
                print('_to_linear shape is = ', x.shape, 'x[0]=',x[0].shape)
                self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            return x

        def forward(self, x):
            x = self.convs(x)
            x = x.view(-1, self._to_linear)

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.softmax(x, dim=1)

    net = Net()
    print(net)

    print('Reading trained Model !!!!!!!!!!')
    net.load_state_dict(torch.load('net.pt'))
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            real_class = torch.argmax(y_test[i])
            net_out = net(X_test[i].view(-1,1,32,32))[0]
            predict_class = torch.argmax(net_out)

            if predict_class == real_class:
                correct += 1
            total += 1

    print('Accuracy: ',round(correct /total, 3 ))






'''
fig, axes = plt.subplots(8,5)
ax = axes.flat
count = 0
with torch.no_grad():
    for i in tqdm(range(len(X_test))):
        real_class = torch.argmax(y_test[i])
        net_out = net(X_test[i].view(-1,1,32,32))[0]
        predict_class = torch.argmax(net_out)

        if count >= 40:
            break
        if predict_class != real_class:
            print("indx ",count,' real = ',real_class, ' predict = ', predict_class)
            ax[count].imshow(X_train[i], cmap = 'gray')
            ax[count].set_xticks([])
            ax[count].set_yticks([])
            #ax[i].set_title(int (torch.argmax( y_train[i]  ) ) + 1)
            ax[count].set_title(int(predict_class))
            count =count + 1
'''


