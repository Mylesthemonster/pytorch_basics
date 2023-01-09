# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Create a tensor
x = torch.arange(0, 100, 10)
print("==>> x: ", x)

print(f"Minimum: {torch.min(x)}")
print(f"Maximum: {torch.max(x)}")
# print(f"Mean: {x.mean()}") # this will error type is int64/LongTensor/Long
print(f"Mean: {torch.mean(x.type(torch.float32))}") # Change x to float datatype first to get mean
print(f"Sum: {torch.sum(x)}")

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
