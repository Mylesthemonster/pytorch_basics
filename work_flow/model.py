import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
from data import X_train, y_train, X_test, y_test, plot_predictions

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):                     # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        
        # Initialize the model parameters (weights and bias)
        self.weights = nn.Parameter(torch.randn(1,                      # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float),     # <- PyTorch loves float32 by default
                                                requires_grad=True)     # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1,                         # <- start with random bias (this will get adjusted as the model learns)
                                             dtype=torch.float),        # <- PyTorch loves float32 by default
                                             requires_grad=True)        # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:     # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias                 # <- this is the linear regression formula (y = m*x + b)

# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print("==>> list(model_0.parameters()): ", list(model_0.parameters()))

# List named parameters 
print("==>> model_0.state_dict(): ", model_0.state_dict())

# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)
    
# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# plot_predictions(predictions=y_preds)

print("==>> y_test - y_preds: ", y_test - y_preds)
print('\n')