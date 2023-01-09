# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print("==>> random_tensor: ", random_tensor)
print("==>> random_tensor.dtype: ", random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print("==>> random_image_size_tensor.shape: ", random_image_size_tensor.shape)
print("==>> random_image_size_tensor.ndim: ", random_image_size_tensor.ndim)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print("==>> zeros: ", zeros)
print("==>> zeros.dtype: ", zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print("==>> ones: ", ones)
print("==>> ones.dtype: ", ones.dtype)

# Creating a range and tensors with torch.arange(start, end, step)
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print("==>> zero_to_ten: ", zero_to_ten)

# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print("==>> ten_zeros: ", ten_zeros)
