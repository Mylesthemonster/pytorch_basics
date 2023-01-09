# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=torch.device('mps'), # defaults to None which uses cpu , torch.device('mps') uses Metal GPU
                               requires_grad=False) # if True, operations perfromed on the tensor are recorded 

print("==>> float_32_tensor.shape: ", float_32_tensor.shape)
print("==>> float_32_tensor.dtype: ", float_32_tensor.dtype)
print("==>> float_32_tensor.device: ", float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print("==>> float_16_tensor.dtype: ", float_16_tensor.dtype)

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print("==>> tensor.dtype: ", tensor.dtype)

# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16

# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8
