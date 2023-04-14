# Neural Network From Scratch

This is a neural network library created using numpy. It allows you to create, train, and save models. The library is inspired by PyTorch but simplified for educational purposes. The library has the following components:


## The library has the following implemented:

Layers

* Dense
* Conv2D
* Flatten
* MaxPooling2D

Activations

* ReLU
* GeLU
* Sigmoid
* Softmax

Loss functions

* MSE
* CategoricalCrossentropy

Optimizers

* SGD
* Adam

Datasets

The library has a Dataset class implemented which allows you to iterate over batches of data and other utlities.

## Usage

Here is an example of how you can create and train a model using the MNIST dataset:



It includes common layers, activations, losses, and optimizers, making it easy to build a variety of models for different tasks. The library also includes helpful utility functions for data preprocessing, training and testing the models, and saving and loading models.

While this library is not meant to replace existing deep learning frameworks, it can be a great tool for educational purposes or for small sized projects without using much external libraries other than Numpy and raw Python. It is also a good starting point for those interested in deep learning to learn about the fundamentals of building neural networks.
