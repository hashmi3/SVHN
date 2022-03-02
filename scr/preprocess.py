import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

d_pth = '/net/home/mhashmi/data/svHn/'
tst = 'test_32x32.mat'
trn = 'train_32x32.mat'
extra = 'extra_32x32.mat'



if __name__ == '__main__':
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

    X_gry_test =np.expand_dims(np.dot(X_test, [0.2990, 0.5870, 0.114]), axis = 3)
    print('Shape of Grey test data = ', X_gry_test.shape)

    X_gry_train = np.expand_dims(np.dot(X_train , [0.2990, 0.5870, 0.114]), axis = 3)
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

'''
    fig, axes = plt.subplots(2,10)

    for i, ax in enumerate(axes.flat):
             ax.imshow(X_val[i,:,:,0], cmap = 'gray')
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_title(y_val[i])

    plt.show()
'''
print(y_train)


#change this in Future !!!!!!!!!!!!!!!
from sklearn.preprocessing import OneHotEncoder

# Fit the OneHotEncoder
enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

# Transform the label values to a one-hot-encoding scheme
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

print("Training set", y_train.shape)
print("Validation set", y_val.shape)
print("Test set", y_test.shape)

import h5py

# Create file
h5f = h5py.File('SVHN_grey.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data= X_gryNorm_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data= X_gryNorm_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data= X_val)
h5f.create_dataset('y_val', data= y_val)

# Close the file
h5f.close()








