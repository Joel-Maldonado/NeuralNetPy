# Neural Network Using Python and Numpy

NeuralNetPy is a neural network library created using numpy that allows you to create, train, and save deep learning models. The library is inspired by PyTorch but simplified for educational purposes. 

It includes common layers, activations, losses, and optimizers, making it easy to build a variety of models for different tasks. It also includes helpful utility functions for data preprocessing, training and testing the models, and saving and loading models.

While this library is not meant to replace existing deep learning frameworks, it can be a great tool for educational purposes or for small sized projects without using much external libraries other than Numpy and raw Python. It is also a good starting point for those interested in deep learning to learn about the fundamentals of building neural networks.

## The following components have been implemented:

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

Utils

This library has a Dataset class implemented which allows you to iterate over batches of data and other utlities useful for deep learning

## Install
```
pip install NeuralNetPy
```

PyPi: https://pypi.org/project/NeuralNetPy/

## Usage

Creating a Linear Model

Here is an example of how you can create a simple linear model:

```python

import NeuralNetPy as net

class LinearModel(net.utils.BaseModel):
    def __init__(self):
        super().__init__()
        
        self.layers = [
            net.layers.Flatten(),
            net.layers.Dense(28 * 28, 128),
            net.activations.GeLU(),
            net.layers.Dense(128, 10),
            net.activations.Softmax()
        ]

model = LinearModel()

```

The model consists of a Flatten layer to convert the input to a 1D array, a Dense layer with 128 units and a GeLU activation function, another Dense layer with 10 units and a Softmax activation function.

## Loading Data

You can load your own data using the np.load() method, and then pass the data into the Dataset class provided by the library. Here is an example using random data:


```python

import numpy as np

X_train = np.random.rand(1000, 784)
y_train = np.random.rand(1000, 10)

train = net.utils.Dataset(X_train, y_train, batch_size=32, shuffle=True)
```

## Training Loop

To train the model, you can use a for loop and iterate over the batches of data in the train dataset. Here is an example:

```python

loss_fn = net.losses.CategoricalCrossentropy()
optim = net.optimizers.Adam(model.layers, lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss_train = 0
    running_acc_train = 0

    for i, (batch_X, batch_y) in enumerate(train):
        y_pred = model.forward(batch_X)

        loss = loss_fn.forward(y_pred=y_pred, y_true=batch_y).mean()
        acc = net.utils.acc_fn(y_pred=y_pred, y_true=batch_y)

        grad = loss_fn.backward(y_pred=y_pred, y_true=batch_y)
        model.backward(grad)

        optim.step()

        running_loss_train += loss
        running_acc_train += acc

    running_loss_train /= len(train)
    running_acc_train /= len(train)

    print(f"Epoch: {epoch+1} | Loss: {running_loss_train} | Acc: {running_acc_train}")
```

This will train the model for 10 epochs, iterating over the batches of data and updating the model parameters using the Adam optimizer.


## Saving and Loading Models

To save a model, you can use the save method provided by the model. Here is an example:

```python
model.save('linear_model_1')
```
This will save the model parameters to a .npz file with the specified name.

To load a saved model, you can create a new instance of MyModel and then call the load method with the path to the saved model file:

```python

model = LinearModel()
model.load('linear_model_1.npz')

```
