import cifar_DataCreate as cifar
import torch
import time
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from Dataset import Dataset

# ****************************************************************************************************************

# Cuda for pytorch if there is avaliable gpu, use for acceleration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Default Parameters
params = {'batch_size': 25,
          'shuffle': True,
          'num_workers': 4}

# Transforms
# Normalization for all sets and data augmentation for train
# Cifar-10 image size is 32x32, however increasing the size of input image increases accuracy
# normalizing mean and std parameters calculated from all data in cifar-10
# these two methods in above are intuitions for good results
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ]),
}

# Dataset read
partition, labels, all_data, class_names = cifar.create_partition_labels()

# Datasets created
training_set = Dataset(partition['train'], labels, all_data, data_transforms['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['val'], labels, all_data, data_transforms['val'])
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

test_set = Dataset(partition['test'], labels, all_data, data_transforms['test'])
test_generator = torch.utils.data.DataLoader(test_set, **params)

# *****************************************************************************************************************

# This function is used to train any neural network model with given inputs
# size is used for training network with percentage of train data size
# size = 1 --> 100%  && size = 0 --> 0% (no training)
def train_model(model, criterion, optimizer, scheduler, savepath,  size=1, num_epoch=150):
    # measure time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_acc = np.zeros(num_epoch)
    train_acc_top2 = np.zeros(num_epoch)
    val_acc = np.zeros(num_epoch)
    val_acc_top2 = np.zeros(num_epoch)
    train_loss = np.zeros(num_epoch)
    val_loss = np.zeros(num_epoch)

    for epoch in range(num_epoch):
        print('*' * 25)
        print('Epoch {}/{}\n'.format(epoch + 1, num_epoch))
        print('-' * 25)

    # in each epoch, there are training and validation phase
        for phase in ['train', 'val']:

            running_loss = 0.0
            running_corrects = np.zeros(10) # for each class, holds true positives(true predictions)
            running_corrects_top2 = np.zeros(10)
            total_elements = np.zeros(10)   # for each class, holds ground truths


            if phase == 'train':
                scheduler.step()
                model.train()  # model is set to training

                threshold = len(partition['train']) * size  # threshold for train data size
                i = 0;
                # Batch training starts here
                # It finishes at the end of defined percentage of training data processed
                for inputs, labels in training_generator:
                    if i % 10 == 0:
                        print('\rImage processed: {}/{}'.format(i, len(partition['train'])), end='', flush=True)

                    if i >= threshold:
                     break

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward path
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        _, top2 = torch.topk(outputs, 2)# returns the top 2 predictions
                        loss = criterion(outputs, labels)

                        # backward path
                        loss.backward()
                        optimizer.step()

                    # statistics
                    # at the end, running loss should be divided to number of inputs. Thus, loss should be
                    # multiplied by batch size to calculate loss correctly.
                    running_loss += loss.item() * inputs.size(0)

                    # calculate total top2 and top1 correct predictions for each class and total ground truths
                    for k in range(inputs.size(0)):
                        if top2[k][0] == labels.data[k]:
                            running_corrects[labels.data[k]] += 1
                            running_corrects_top2[labels.data[k]] += 1

                        elif top2[k][1] == labels.data[k]:
                            running_corrects_top2[labels.data[k]] += 1

                        total_elements[labels.data[k]] += 1

                    i += params['batch_size']  # increase processed samples in each batch

                # Calculate average loss
                epoch_loss = running_loss / len(partition['train'])
                # Calculate accuracy for each class and add them.
                epoch_acc = 0
                epoch_acc_top2 = 0
                for x in range(len(running_corrects)):
                    epoch_acc += (running_corrects[x] / total_elements[x])
                    epoch_acc_top2 += (running_corrects_top2[x] / total_elements[x])

                # Normalize accuracy dividing to number of classes
                epoch_acc /= 10
                epoch_acc_top2 /= 10
                print('\nTrain Loss: {:.4f} \nTrain Acc: {:.4f} \nTrain Acc Top2: {:.4f}'.format(
                    epoch_loss, epoch_acc, epoch_acc_top2))
                print('-' * 25)
                print('\n')
                train_acc[epoch] = epoch_acc
                train_acc_top2[epoch] = epoch_acc_top2
                train_loss[epoch] = epoch_loss

            else:
                model.eval()   # model is set to evaluate mode

                threshold = len(partition['val']) * size * 10  # threshold for val data size
                i = 0

                # Batch validation starts here
                # It finishes at the end of defined percentage of validation data processed
                for inputs, labels in validation_generator:
                    if i%10 == 0:
                        print('\rImage processed:{}/{}'.format(i, len(partition['val'])), end='', flush=True)

                    if i >= threshold:
                        break

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward path
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        _, top2 = torch.topk(outputs, 2)  # returns the top 2 prediction
                        loss = criterion(outputs, labels)

                    # statistics
                    # at the end, running loss should be divided to number of inputs. Thus, loss should be
                    # multiplied by batch size to calculate loss correctly.
                    running_loss += loss.item() * inputs.size(0)

                    # calculate total top2 and top1 correct predictions for each class and total ground truths
                    for k in range(inputs.size(0)):
                        if top2[k][0] == labels.data[k]:
                            running_corrects[labels.data[k]] += 1
                            running_corrects_top2[labels.data[k]] += 1

                        elif top2[k][1] == labels.data[k]:
                            running_corrects_top2[labels.data[k]] += 1

                        total_elements[labels.data[k]] += 1

                    i += params['batch_size']  # increase processed samples in each batch

                # Calculate average loss
                epoch_loss = running_loss / len(partition['val'])
                # Calculate accuracy for each class and add them.
                epoch_acc = 0
                epoch_acc_top2 = 0
                for x in range(len(running_corrects)):
                    epoch_acc += (running_corrects[x] / total_elements[x])
                    epoch_acc_top2 += (running_corrects_top2[x] / total_elements[x])

                # Normalize accuracy dividing to number of classes
                epoch_acc /= 10
                epoch_acc_top2 /= 10
                print('\nValidation Loss: {:.4f} \nValidation Acc: {:.4f} \nValidation Acc Top2: {:.4f}'.format(
                     epoch_loss, epoch_acc, epoch_acc_top2))
                print('-' * 25)
                print('\n')
                val_acc[epoch] = epoch_acc
                val_acc_top2[epoch] = epoch_acc_top2
                val_loss[epoch] = epoch_loss

                # deep copy the model if validation accuracy is higher then best accuracy(best model)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('-'*25)
    print('\n')

    savename = "Resized(224)_{}_epoch_result_{}_{}"

    np.savetxt((savepath + savename.format(num_epoch, 'train', 'acc')), train_acc, delimiter=',')
    np.savetxt((savepath + savename.format(num_epoch, 'train', 'acc_top2')), train_acc_top2, delimiter=',')
    np.savetxt((savepath + savename.format(num_epoch, 'train', 'loss')), train_loss, delimiter=',')
    np.savetxt((savepath + savename.format(num_epoch, 'val', 'acc')), val_acc, delimiter=',')
    np.savetxt((savepath + savename.format(num_epoch, 'val', 'acc_top2')), val_acc_top2, delimiter=',')
    np.savetxt((savepath + savename.format(num_epoch, 'val', 'loss')), val_loss, delimiter=',')

    model.load_state_dict(best_model_wts)

    return model


def test_model(model, optimizer, size=1):
    # measure time
    since = time.time()

    running_loss = 0.0
    running_corrects = np.zeros(10)  # for each class, holds true positives(true predictions)
    running_corrects_top2 = np.zeros(10)
    total_elements = np.zeros(10)  # for each class, holds ground truths

    model.eval()  # model is set to evaluate mode
    threshold = len(partition['test']) * size  # threshold for val data size
    i = 0
    # Batch test starts here
    # It finishes at the end of defined percentage of validation data processed
    for inputs, labels in test_generator:
        if i % params['batch_size'] == 0:
            print('\rImage processed:{}/{}'.format(i, len(partition['test'])), end='', flush=True)

        if i >= threshold:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward path
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, top2 = torch.topk(outputs, 2)  # returns the top2 prediction
           # loss = criterion(outputs, labels)  # not need to calculate loss

        # calculate total top2 and top1 correct predictions for each class and total ground truths
        for k in range(inputs.size(0)):
            if top2[k][0] == labels.data[k]:
                running_corrects[labels.data[k]] += 1
                running_corrects_top2[labels.data[k]] += 1

            elif top2[k][1] == labels.data[k]:
                running_corrects_top2[labels.data[k]] += 1

            total_elements[labels.data[k]] += 1

        i += params['batch_size']  # increase processed samples in each batch

    # Calculate accuracy for each class and add them.
    test_acc = 0
    test_acc_top2 = 0
    for x in range(len(running_corrects)):
        test_acc += (running_corrects[x] / total_elements[x])
        test_acc_top2 += (running_corrects_top2[x] / total_elements[x])

    # Normalize accuracy dividing to number of classes
    test_acc /= 10
    test_acc_top2 /= 10
    print('\nTest Acc: {:.4f} \nTest Acc Top2: {:.4f}\n'.format(
        test_acc, test_acc_top2))

    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:4f}'.format(test_acc))
    print('-' * 25)
    print('\n')

    return test_acc, test_acc_top2


def resnet50_trainval(train_type='whole', loss_type='cross_entropy_loss', size=1, batch_size=25, num_epoch=15):

    if os.path.exists('outputs') == False:
        os.makedirs('outputs')

    lr = 0.01
    savepath = 'outputs_lr_{}/{}_{}_trainval/size_{}/'

    if train_type != 'only_fc' and train_type != 'fc+layer4' and train_type != 'whole':
        raise ValueError("Training type should be valid!!!\n 'only_fc' or 'fc+layer4' or 'whole' accepted")

    if loss_type != 'hinge_loss' and loss_type != 'cross_entropy_loss':
        raise ValueError("Loss type should be valid!!!\n 'hinge_loss' or 'cross_entropy_loss' accepted")

    params['batch_size'] = batch_size

    savepath = savepath.format(lr, train_type, loss_type, size)
    if os.path.exists(savepath) == False:
        os.makedirs(savepath)

    resnet50 = models.resnet50(pretrained=True)

    if train_type == 'only_fc':
        for param in resnet50.parameters():
            param.requires_grad = False
    elif train_type == 'fc+layer4':
        for param in resnet50.parameters():
            param.requires_grad = False
        resnet50.layer4.requires_grad = True

    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 10)
    resnet50 = resnet50.to(device)

    if loss_type == 'hinge_loss':
        criterion = torch.nn.MultiMarginLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer_ft = torch.optim.SGD(resnet50.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    resnet50 = train_model(resnet50, criterion, optimizer_ft, scheduler, savepath, size, num_epoch)

    savename = "best_model_{}_epoch_{}_{}_result"
    savename = savename.format(num_epoch, train_type, loss_type)
    torch.save(resnet50.state_dict(), (savepath + savename))

    return savepath, savename


def resnet50_test(savepath, savename, size=1):

    resnet50 = models.resnet50(pretrained=False)
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 10)
    resnet50.load_state_dict(torch.load(savepath + savename))

    optimizer_ft = torch.optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)
    resnet50 = resnet50.to(device)

    test_acc, test_acc_top2 = test_model(resnet50, optimizer_ft, size)

    return test_acc, test_acc_top2


