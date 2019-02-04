import numpy as np
import os
import matplotlib.pyplot as plt
import random
from six.moves import cPickle

random.seed(1)  # each time generate same sample sequence
plt.ion()  # interactive mode

# class names for cifar 10 dataset
class_names = ('airplane', 'automobile', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

row     = 32
column  = 32
img_shape = (row, column, 3)  # cifar dataset image format

cifar_batch_path = "cifar-10-batches-py"
data_batch = "data_batch_{:d}"

train_size = 49000
val_size = 1000
test_size = 10000

def unpickle(filename):
    with open(filename, 'rb') as fd:
        data = cPickle.load(fd, encoding='latin1') # encoding must be added for python 3
    return data


def load_cifar10_batch(filename):
    dataDict = unpickle(filename)
    data = dataDict['data']
    labels = dataDict['labels']
    data = data.reshape(10000, 3072)
    labels = np.array(labels)
    return data, labels


def imshow(inp, label):
    img = inp.reshape(3, row, column).transpose([1, 2, 0])
    plt.imshow(img)
    plt.title(class_names[label])
    plt.pause(1)


def imshow_batch(partition, inp, label, nrows, ncols):
    plt.figure(figsize=(8, 8))
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True)
    samples = random.sample(range(0, len(partition)), nrows*ncols)
    for ind in range(nrows * ncols):
        ID = partition[samples[ind]]
        img_raw = inp[int(ID)]
        img = img_raw.reshape(3, row, column).transpose([1, 2, 0])
        axeslist.ravel()[ind].imshow(img)
        axeslist.ravel()[ind].set_title(class_names[label[ID]])
        axeslist.ravel()[ind].set_axis_off()

    plt.pause(1)


def load_cifar10():

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # train data loading from batches
    for i in range(1, 6):
        fname = os.path.join(cifar_batch_path, data_batch.format(i))
        data, labels = load_cifar10_batch(fname)
        train_data.append(data)
        train_labels.append(labels)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    # test data loading from test batch
    fname = os.path.join(cifar_batch_path, 'test_batch')
    test_data, test_labels = load_cifar10_batch(fname)

    # There are 50000 images for training and 10000 images for testing
    # Divide training image randomly as train --> 49000, validation --> 1000

    val_data = []
    val_labels = []



    mask = range(train_size, train_size+val_size)
    val_data = train_data[mask]
    val_labels = train_labels[mask]
    mask = range(train_size)
    train_data = train_data[mask]
    train_labels = train_labels[mask]

    data = np.concatenate((train_data, val_data, test_data), axis=0)

    labels = np.concatenate((train_labels, val_labels, test_labels), axis=0)

    print('train data shape: ', train_data.shape)
    print('test data shape: ', test_data.shape)
    print('val data shape: ', val_data.shape)

    return data, labels

# test same images whether read correct or not
# for i in range(500):
#   imshow(val_data[i], val_labels[i])


def create_partition_labels():

    data, data_labels = load_cifar10()

    partition = {}
    labels = {}

    ids = "{:d}"
    ids_str = [ids.format(x) for x in range(train_size)]
    partition['train'] = ids_str
    for i in range(len(ids_str)):
        labels[ids_str[i]] = data_labels[int(ids_str[i])]
    ids_str = [ids.format(x) for x in range(train_size, train_size + val_size)]
    partition['val'] = ids_str
    for i in range(len(ids_str)):
        labels[ids_str[i]] = data_labels[int(ids_str[i])]
    ids_str = [ids.format(x) for x in range(train_size + val_size, train_size + val_size + test_size)]
    partition['test'] = ids_str
    for i in range(len(ids_str)):
        labels[ids_str[i]] = data_labels[int(ids_str[i])]

    return partition, labels, data, class_names
