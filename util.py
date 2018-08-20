'''
util.py

Utilities for preprocessing data, creating directories, saving and restoring model checkpoints

jay.urbain@gmail.com
'''


import torch
from torchvision import datasets, transforms, models
import os

def preprocess_data(train_dir, valid_dir, test_dir,
                    norm_mean=[0.485, 0.456, 0.406], norm_stdv=[0.229, 0.224, 0.225]):
    '''
    Inputs:
    train_dir, valid_dir, test_dir - data directories
    norm_mean, norm_stdv - image mean and standard deviation for 3 channels

    Returns:
    trainloader, validloader, testloader - torchvision data loaders
    '''
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_stdv)])

    data_transforms_train = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(p=0.1),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_stdv)])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Save class ids
    class_idx = train_data.class_to_idx
    return trainloader, validloader, testloader, class_idx


# TODO: Save the checkpoint

def save_model_checkpoint(model, input_size, epochs, save_dir, arch, learning_rate, class_idx, optimizer, criterion, output_size):
    """
    Save trained model as checkpoint file.
    Parameters:
        model - Previously trained and tested CNN
        input_size - Input size used on the specific CNN
        epochs - Number of epochs used to train the CNN
        save_dir - Directory to save the checkpoint file(default- current path)
        arch - pass string value of architecture used for loading
    Returns:
        None
    """
    saved_model = {
    'input_size':input_size,
    'epochs':epochs,
    'arch':arch,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': class_idx,
    'optimizer_dict': optimizer.state_dict(),
    'criterion_dict': criterion.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict()
    }
    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(saved_model, save_path)
    print('Model saved at {}'.format(save_path))


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(checkpoint_path):
    trained_model = torch.load(checkpoint_path)
    arch = trained_model['arch']
    class_idx = trained_model['class_to_idx']
    # Only download the model you need, kill program if one of the three models isn't passed
    if arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        load_model = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()

    for param in load_model.parameters():
        param.requires_grad = False

    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])

    return load_model, arch, class_idx

def cd_if_needed(save_dir):

    '''
    Create directory if needed
    Parameters:
        save_dir - string directory path
    Returns:
        None
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
