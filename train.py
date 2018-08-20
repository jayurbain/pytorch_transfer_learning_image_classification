'''
train.py

Main method for training pytorch model on flower dataset.
Dataset consists of 102 flower categories.
Reference:
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

Example:
python train.py flowers --gpu
python train.py flowers --save_dir save_dir_densenet --arch densenet --gpu
python train.py flowers --save_dir save_dir_alexnet --arch alexnet --gpu
Jay Urbain
jay.urbain@gmail.com

'''

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import argparse

import models
import util

# from PIL import Image

# Define command line arguments
def get_input_args():
    '''
    Read and parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Model training arguments:')
    parser.add_argument('data_dir', type=str, help='Required path to dataset ')
    parser.add_argument('--save_dir', type=str, default='save_dir_test', help='Checkpoint directory')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture  {vgg16, alexnet, densenet}')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Nidden layer sizes')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Checkpoint file name')
    print(parser.parse_args())
    return parser.parse_args()


def train_classifier(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    """
    Trains the selected model. Performs validation loop every 40 steps and prints progress.
    Inputs:
        model - CNN architecture to be trained
        trainloader - pytorch data loader of training data
        validloader - pytorch data loader of data to be used to validate.
        criterion - loss function to be executed (default- nn.NLLLoss)
        optimizer - optimizer function to apply gradients (default- adam optimizer)
        epochs - number of epochs to train on
        gpu - boolean that flags GPU use
    Returns:
        model - Trained CNN
    """
    steps = 0
    print_every = 40
    run_loss = 0

    # Selects CUDA processing if gpu == True and if the environment supports CUDA
    if gpu and torch.cuda.is_available():
        print('GPU PROCESSING')
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... Training under CPU.')
    else:
        print('CPU PROCESSING')

    for e in range(epochs):

        model.train()

        # Training forward pass and backpropagation
        for images, labels in iter(trainloader):
            steps += 1
            images, labels = Variable(images), Variable(labels)
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.data.item()

            # Runs validation forward pass and loop at specified interval
            if steps % print_every == 0:
                model.eval()

                acc = 0
                valid_loss = 0

                for images, labels in iter(validloader):
                    images, labels = Variable(images), Variable(labels)
                    if gpu and torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    with torch.no_grad():
                        out = model.forward(images)
                        valid_loss += criterion(out, labels).data.item()

                        ps = torch.exp(out).data
                        equality = (labels.data == ps.max(1)[1])

                        acc += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(run_loss / print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
                      "Valid Accuracy: {:.3f}".format(acc / len(validloader)))

                run_loss = 0
                model.train()

    print('{} EPOCHS COMPLETE. MODEL TRAINED.'.format(epochs))
    return model

# TODO: Do validation on the test set
def test_model(model, testloader, criterion, gpu):
    """
    Test model on a test dataset and print results.
    Parameters:
        model - Trained CNN to test
        testloader - pytorch data loader of test data
        criterion - loss function to be executed (default- nn.NLLLoss)
        gpu - boolean that flags GPU use
    Returns:
        None
    """
    # Selects CUDA if gpu == True and environment supports CUDA
    if gpu and torch.cuda.is_available():
        print('GPU TESTING')
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... testing under CPU.')
    else:
        print('CPU TESTING')

    model.eval()

    acc = 0
    test_loss = 0
    # Forward pass
    for images, labels in iter(testloader):
        images, labels = Variable(images), Variable(labels)
        if gpu and torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            out = model.forward(images)
            test_loss += criterion(out, labels).data.item()

            ps = torch.exp(out).data
            equality = (labels.data == ps.max(1)[1])

            acc += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(acc / len(testloader)))


def main():

    # read and parse input arguments
    args = get_input_args()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Image dataset mean and standard deviation
    norm_mean = [0.485, 0.456, 0.406]
    norm_stdv = [0.229, 0.224, 0.225]

    # load and preprocess the data using torchvision data loaders
    trainloader, validloader, testloader, class_idx = util.preprocess_data(train_dir, valid_dir, test_dir,
                                                                      norm_mean, norm_stdv)

    # load pretrained convnet
    model, input_size = models.load_pretrained_model(args.arch)
    print(model, input_size)

    # replace classification layer(s) with are own for flower dataset
    output_size = 102
    prob_dropout = 0.5
    models.define_feedforward_classification_for_model(model, input_size, output_size=output_size,
                                                       hidden_units=args.hidden_units, prob_dropout=prob_dropout)
    print(model)

    # define training criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    # train the model
    model = train_classifier(model, trainloader, validloader,
                             criterion=criterion, optimizer=optimizer,
                             epochs=args.epochs, gpu=args.gpu)

    # evaluate model on test data
    test_model(model, testloader, criterion, gpu=True)

    # save model checkpoint file
    save_dir = args.save_dir
    util.cd_if_needed(save_dir)
    checkpoint = args.checkpoint
    util.save_model_checkpoint(model, input_size=input_size, epochs=args.epochs, save_dir=save_dir,
                               arch=args.arch, learning_rate=args.learning_rate, class_idx=class_idx,
                               optimizer=optimizer, criterion=criterion, output_size=output_size)

if __name__ == "__main__":
    main()