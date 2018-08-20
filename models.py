'''
models.py

Methods for defining deep learning architecture

jay.urbain@gmail.com
'''

from torch import nn
from torchvision import models

from collections import OrderedDict

def load_pretrained_model(arch = 'vgg16'):

    '''
    select and load pretrained model
    Parameters:
        arch - string identifying network architecture
    Returns:
        model - pytorch model
        input_size - size of input to classification
    '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, input_size


def define_feedforward_classification_for_model(model, input_size, output_size, hidden_units, prob_dropout):
    '''
    select and load pretrained model
    Parameters:
        model - pytorch model
        input_size - size of input to classification
        output_size - size of input to classification
        prob_dropout - dropout probability
    Returns:
        None
    '''
    hidden_sizes = [hidden_units, hidden_units]
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=prob_dropout)),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=prob_dropout)),
        ('output', nn.Linear(hidden_sizes[1], output_size)),
        ('logsoftmax', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

