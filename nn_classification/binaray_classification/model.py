## Building a model steps 

# 1. Setting up device agnostic code (so our model can run on CPU or GPU if it's available).
# 2. Constructing a model by subclassing nn.Module.
# 3. Defining a loss function and optimizer.
# 4. Creating a training loop (this'll be in the next section).

import torch
from torch import nn
from data import X_train, X_test, y_train, y_test

# Make device agnostic code
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        device = torch.device('cpu')
else:
    device = torch.device('mps') 
    print("==>> device: ", device)
    
## Supervised learning model
# 1. Subclasses nn.Module (almost all PyTorch models are subclasses of nn.Module).
# 2. Creates 2 nn.Linear layers in the constructor capable of handling the input and output shapes of X and y.
# 3. Defines a forward() method containing the forward pass computation of the model.
# 4. Instantiates the model class and sends it to the target device.

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
        
        # self.layer_1 takes 2 input features in_features=2 and produces 5 output features out_features=5.
        # This is known as having 5 hidden units or neurons.
        # The number of hidden units you can use in neural network layers is a hyperparameter 
        # The only rule with hidden units is that the next layer, in our case, self.layer_2 has to take the same in_features as the previous layer out_features.
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
# print("==>> model_0: ", model_0)

# Replicate CircleModelV0 with nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
# print("==>> model_0: ", model_0)

# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
# print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
# print(f"\nFirst 10 predictions:\n{untrained_preds[:10].to('cpu')}")
# print(f"\nFirst 10 test labels:\n{y_test[:10].to('cpu')}")

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Evaluation Metric measuring how right the model is.
# Accuracy is one of several evaluation metrics that can be used for classification problems.
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc