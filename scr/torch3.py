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


def make_data():
    data = loadmat(d_pth+trn)
    X_train, y_train = data['X'], data['y']

    data = loadmat(d_pth+tst)
    X_test, y_test = data['X'], data['y']

    print('Training set = ', X_train.shape, y_train.shape)
    print('Testing set = ', X_test.shape, y_test.shape)

    X_train = X_train.transpose((3,0,1,2))
    X_test = X_test.transpose((3,0,1,2))


    print('Training set = ', X_train.shape, y_train.shape)
    print('Testing set = ', X_test.shape, y_test.shape)

    fig, axes = plt.subplots(2,10)

    for i, ax in enumerate(axes.flat):
             if X_train[i].shape == (32,32,3):
                         ax.imshow(X_train[i])
             else:
                         ax.imshow(X_train[i,:,:,0])
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_title(y_train[i])

    plt.show()

    #X_gry_test =np.expand_dims(np.dot(X_test, [0.2990, 0.5870, 0.114]), axis = 3)
    #print('Shape of Grey test data = ', X_gry_test.shape)

    #X_gry_train = np.expand_dims(np.dot(X_train , [0.2990, 0.5870, 0.114]), axis = 3)
    #print('Shape of Grey test data = ', X_gry_train.shape)


    X_gry_test = np.dot(X_test, [0.2990, 0.5870, 0.114])
    print('Shape of Grey test data = ', X_gry_test.shape)

    X_gry_train = np.dot(X_train , [0.2990, 0.5870, 0.114])
    print('Shape of Grey test data = ', X_gry_train.shape)


    '''
    fig, axes = plt.subplots(2,10)

    for i, ax in enumerate(axes.flat):
             ax.imshow(X_gry_train[i,:,:,0], cmap = 'gray')
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_title(y_train[i])

    plt.show()
    '''
    train_mean = np.mean(X_gry_train, axis=0)
    train_std = np.std(X_gry_train, axis=0)

    X_gryNorm_train = (X_gry_train - train_mean) / train_std
    X_gryNorm_test = (X_gry_test - train_mean) / train_std

    print('Train = ', X_gryNorm_train.shape)
    print('Test =', X_gryNorm_test.shape )

    X_gryNorm_train, X_val, y_train, y_val = train_test_split(X_gryNorm_train, y_train, test_size=0.13, random_state=7)

    print('Train = ', X_gryNorm_train.shape)
    print('Val =', X_val.shape )
    print(y_train)


    #change this in Future !!!!!!!!!!!!!!!

    # Fit the OneHotEncoder
    enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

    # Transform the label values to a one-hot-encoding scheme
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

    print("Training set", y_train.shape)
    print("Validation set", y_val.shape)
    print("Test set", y_test.shape)

    #np.save('../npData/X_train.npy',X_gryNorm_train)
    #np.save('../npData/y_train.npy', y_train)
    #np.save('../npData/X_test.npy', X_gryNorm_test)
    #np.save('../npData/y_test.npy', y_test)
    #np.save('../npData/X_val.npy', X_val)
    #np.save('../npData/y_val.npy', y_val)


    import h5py

    # Create file
    h5f = h5py.File('../data/allData.h5', 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data= X_gryNorm_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data= X_gryNorm_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('X_val', data= X_val)
    h5f.create_dataset('y_val', data= y_val)

    # Close the file
    h5f.close()

#def model():

if __name__ == '__main__':
    if REBUILD_DATA:
        print("Building Data !!!!!!!!!!!!!")
        make_data()

    #model()

    fd = h5py.File('../data/allData.h5', 'r')

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

    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    loss_function = nn.MSELoss()

    BATCH_SIZE = 100
    EPOCHS = 4

    for epoch in range(EPOCHS):
        for i in tqdm(range(0 , len(X_train), BATCH_SIZE )):
            batch_X = X_train[i:i+BATCH_SIZE].view(-1,1,32,32)
            batch_y = y_train[i: i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss}")


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

    print('Accuracy: ', correct /total)


print('END!!!!!!!!!!')



'''
fig, axes = plt.subplots(3,6)
for i, ax in enumerate(axes.flat):
      ax.imshow(X_train[i])
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(int( torch.argmax(  y_train[i])))
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



