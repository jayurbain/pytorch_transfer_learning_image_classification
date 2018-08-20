# pytorch_transfer_learning_image_classification

#### PyTorch end to end image classification using transfer learning.

The application is based on a Udacity Data Science assignment.

The dataset consists of 102 flower categories the dataset here:   
[http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  

The application currently supports 3 different base architectures: vgg16, densenet, and alexnet. For the selected architecture, a pre-trained model is downloaded, the parameters are frozen, and the the final category layer(s) of the selected network is replaced by a 2-layer feedforward network with softmax for flower classification. During training, only the classification parameters are updated. This significantly reduces training time.  

#### Results  
For the default configuration, here is the performance of the three network architectures using the default parameters:  
VGG16 - Test Accuracy: 0.771  
Densenet - Test Accuracy: 0.865  
Alexnet - Test Accuracy: 0.723  

The application can be used with one of the notebooks or the command line version.

#### Command line version

Both GPU and CPU are supported for training and testing. Training on a CPU is not practical.

Training examples:  
python train.py flowers --gpu  
python train.py flowers --save_dir save_dir_densenet --arch densenet --gpu  
python train.py flowers --save_dir save_dir_alexnet --arch alexnet --gpu  

Prediction examples:   
# vgg16  
python predict.py 'flowers/test/1/image_06743.jpg' save_dir/checkpoint.pth --gpu  
# densenet  
python predict.py 'flowers/test/1/image_06743.jpg' save_dir_densenet/checkpoint.pth --gpu  

To add additional networks modify models.load_models() or edit one of the notebooks. The two notebooks are identical except one uses densenet.
