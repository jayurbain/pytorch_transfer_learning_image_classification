'''
predict.py

Predict flower classification

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

Example:
# vgg16
python predict.py 'flowers/test/1/image_06743.jpg' save_dir/checkpoint.pth --gpu
# densenet
python predict.py 'flowers/test/1/image_06743.jpg' save_dir_densenet/checkpoint.pth --gpu

Jay Urbain
jay.urbain@gmail.com

'''
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

import json
import argparse

import util
import models

def get_input_args():
    """
    Read and parse input parameters
    """
    parser = argparse.ArgumentParser(description='NN prediction arguments:')
    parser.add_argument('input', type=str, help='Input image to classify')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='default top_k results')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Image category file')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
    print(parser.parse_args())
    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

def predict(image, model, top_k, gpu, category_names, arch, class_idx):
    '''
    Predict the class(s) of an image using a trained deep learning model. Returns top_k classes
    and probabilities.

    If name json file is passed, it will convert classes to actual names.
    '''

    image = Variable(image)

    image = image.unsqueeze(0).float()

    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        print('GPU PROCESSING')
    else:
        print('CPU PROCESSING')
    with torch.no_grad():
        out = model.forward(image)
        results = torch.exp(out).data.topk(top_k)
    classes = np.array(results[1][0], dtype=np.int)
    probs = Variable(results[0][0]).data

    # If category file path passed, convert classes to actual names
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        # Creates a dictionary of loaded names based on class_ids from model
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
        # invert dictionary to accept prediction class output
        mapped_names = {v: k for k, v in mapped_names.items()}

        classes = [mapped_names[x] for x in classes]
        probs = list(probs)
    else:
        # Invert class_idx from model to accept prediction output as key search
        class_idx = {v: k for k, v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probs)
    return classes, probs


def print_predict(classes, probs):
    """
    Prints predictions. Returns Nothing
    Parameters:
        classes - list of predicted classes
        probs - list of probabilities associated with class from classes with the same index
    Returns:
        None - Use module to print predictions
    """
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))

def main():

    # read and parse input arguments
    args = get_input_args()

    # load the checkpoint
    model, arch, class_idx = util.load_model(checkpoint_path=args.checkpoint)

    proc_img = process_image(args.input)
    proc_img = torch.FloatTensor(proc_img)
    classes, probs = predict(proc_img, model, top_k=args.top_k, gpu=args.gpu, category_names='cat_to_name.json', arch=arch,
                             class_idx=class_idx)
    print_predict(classes, probs)

if __name__ == "__main__":
    main()
