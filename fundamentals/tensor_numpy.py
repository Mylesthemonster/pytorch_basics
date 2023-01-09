# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# NumPy array to tensor
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
tensor = torch.from_numpy(array).type(torch.float32)

print("==>> array: ", array)
print("==>> tensor: ", tensor)

# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
print("==>> tensor: ", tensor)
print("==>> numpy_tensor: ", numpy_tensor)
