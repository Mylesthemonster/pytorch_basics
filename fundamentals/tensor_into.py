# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Scalar
scalar = torch.tensor(7)
print(f'scalar = {scalar}\n')
print(f'The dimensions of a tensor using the ndim attribute.\nscalar.ndim = {scalar.ndim}\n')

# Get the Python number within a tensor (only works with one-element tensors)
print(f'Use the item() method turn torch.Tensor to a Python integer.\nscalar.item() = {scalar.item()}\n')

# Vector
vector = torch.tensor([7, 7])
print(f'vector = {vector}\nIts of type type torch.Tensor\n')
print(f'Check the number of dimensions of vector.\nvector.ndim = {vector.ndim}\n')
# Dimensions a tensor in PyTorch has by the number of square brackets on the outside ([) and you only need to count one side.

# Check shape of vector
print(f'Check the shape of vector.\nvector.shape = {vector.shape}\n')

# Matrix
MATRIX = torch.tensor([[7, 8],[9, 10]])
print(f'MATRIX = {MATRIX}\n')

# Check number of dimensions
print(f'Check the number of dimensions of MATRIX.\nMATRIX.ndim = {MATRIX.ndim}\n')

# Check shape of MATRIX
MATRIX.shape
print(f'Check the shape of MATRIX.\nMATRIX.shape = {MATRIX.shape}\n')
print('We get output "torch.Size([2, 2])" because MATRIX is two elements deep and two elements wide\n')

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(f'TENSOR = {TENSOR}\n')

# Check number of dimensions for TENSOR
print(f'Check the number of dimensions of TENSOR.\nTENSOR.ndim = {TENSOR.ndim}\n')

# Check shape of TENSOR
print(f'Check the shape of TENSOR.\nTENSOR.shape = {TENSOR.shape}\n')
