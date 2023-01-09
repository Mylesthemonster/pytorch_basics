import torch
import matplotlib.pyplot as plt

# Create a toy tensor (similar to the data going into our model(s))
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print("==>> A: ", A)

# Visualize the toy tensor
# plt.plot(A)
# plt.show()

# Create ReLU function by hand 
def relu(x):
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors

# Pass toy tensor through ReLU function
print("==>> relu(A): ", relu(A))

# Plot ReLU activated toy tensor
# plt.plot(relu(A))
# plt.show()

# Create a custom sigmoid function
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

# Test custom sigmoid on toy tensor
sigmoid(A)

# Plot sigmoid activated toy tensor
plt.plot(sigmoid(A))
plt.show()