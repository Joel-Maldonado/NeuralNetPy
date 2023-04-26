import matplotlib.pyplot as plt
import numpy as np

import NeuralNetPy as net


# Create a custom neural network class
class MyModel(net.models.BaseModel):
    def __init__(self):
        super().__init__()

        # Define the layers of the neural network
        self.layers = [
            net.layers.Flatten(),
            net.layers.Dense(1, 16),
            net.activations.GeLU(),
            net.layers.Dense(16, 16),
            net.activations.GeLU(),
            net.layers.Dense(16, 1),
        ]


# Create an instance of the neural network class
model1 = MyModel()


# Load the training data
X_train = np.arange(0, 2 * np.pi, 0.01).reshape(-1, 1)
y_train = np.sin(X_train)


# Create a dataset object for the training data
train_data = net.utils.Dataset(X_train, y_train, batch_size=32, shuffle=True)


# Define the loss function and optimizer
loss_fn = net.losses.MSE()
optim = net.optimizers.Adam(model1.layers, lr=0.01)


# Training Loop
epochs = 500
for epoch in range(epochs):
    running_loss_train = 0

    # Iterate over the batches of the training data
    for i, (batch_X, batch_y) in enumerate(train_data):
        # Forward pass
        y_pred = model1.forward(batch_X)

        # Compute the loss
        loss = loss_fn.forward(y_pred=y_pred, y_true=batch_y).mean()

        # Backpropagate the loss
        grad = loss_fn.backward(y_pred=y_pred, y_true=batch_y)
        model1.backward(grad)

        # Update the parameters
        optim.step()

        # Accumulate the running loss
        running_loss_train += loss

    # Average the running loss
    running_loss_train /= len(train_data)

    # Print the epoch number and the loss
    print(f"Epoch: {epoch+1} | Loss: {running_loss_train}")


# Try plotting the results

train_pred = model1.forward(X_train)

# plt.scatter(X_train, train_pred)
plt.plot(X_train, y_train, label='Actual')
plt.plot(X_train, train_pred, label='Prediction')
plt.legend()
plt.show()


# Save the model weights
model1.save('sin-model-1')


# Load model
loaded_model = MyModel() # Random weights
loaded_model.load('sin-model-1.npz') # Loads saved weights into loaded_model
