import torch
from torch import nn
import matplotlib.pyplot as plt
from model import model_0
from data import weight, bias, X_train, y_train, X_test, y_test, plot_predictions

## Loss function	
# Measures how wrong your models predictions (e.g. y_preds) are compared to the truth labels (e.g. y_test). Lower the better.	
# PyTorch has plenty of built-in loss functions in torch.nn.	
# Mean absolute error (MAE) for regression problems (torch.nn.L1Loss()). Binary cross entropy for binary classification problems (torch.nn.BCELoss()).

## Optimizer
# Tells your model how to update its internal parameters to best lower the loss.	
# You can find various optimization function implementations in torch.optim.	
# Stochastic gradient descent (torch.optim.SGD()). Adam optimizer (torch.optim.Adam()).

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params = model_0.parameters(), # parameters of target model to optimize
                            lr = 0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

### PyTorch training loop
# 1	Forward pass	
# The model goes through all of the training data once, performing its forward() function calculations.	
# model(x_train)
# 2	Calculate the loss	
# The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.	
# loss = loss_fn(y_pred, y_train)
# 3	Zero gradients	
# The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step.	
# optimizer.zero_grad()
# 4	Perform backpropagation on the loss	
# Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True). This is known as backpropagation, hence "backwards".	
# loss.backward()
# 5	Update the optimizer (gradient descent)	
# Update the parameters with requires_grad=True with respect to the loss gradients in order to improve them.	
# optimizer.step()



### PyTorch testing loop
# 1	Forward pass	
# The model goes through all of the training data once, performing its forward() function calculations.	
# model(x_test)
# 2	Calculate the loss	
# The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.	
# loss = loss_fn(y_pred, y_test)
# 3	Calculate evaluation metrics (optional)	
# Alongside the loss value you may want to calculate other evaluation metrics such as accuracy on the test set.	
# Custom functions

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 200

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Calculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

## Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

# Find our model's learned parameters
print("\nThe model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}\n")
