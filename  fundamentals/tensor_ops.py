# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print("==>> tensor: ", tensor)
print("==>> tensor + 10: ", tensor + 10)

# Multiply it by 10
print("==>> tensor * 10: ", tensor * 10)

# Tensors don't change unless reassigned
print("==>> tensor: ", tensor)

# Subtract and reassign
tensor = tensor - 10
print("==>> tensor = tensor - 10: ", tensor)

# Add and reassign
tensor = tensor + 10
print("==>> tensor = tensor + 10: ", tensor)

# Can also use torch functions
print("==>> torch.multiply(tensor, 10): ", torch.multiply(tensor, 10))

# Element-wise multiplication	= tensor * tensor
# [1*1, 2*2, 3*3] = [1, 4, 9]
print("==>> Element-wise matrix multiplication: ", tensor * tensor)

# Matrix multiplication       = tensor.matmul(tensor)
# [1*1 + 2*2 + 3*3] = [14]
print("==>> Matrix multiplication: ", torch.matmul(tensor, tensor))

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],

                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# View tensor_A and tensor_B
print("==>> tensor_A: ", tensor_A)
print("==>> tensor_B: ", tensor_B)

# View tensor_A and tensor_B.T
print("==>> tensor_A: ", tensor_A)
print("==>> tensor_B.T: ", tensor_B.T)

# The operation works when tensor_B is transposed
print(f'Original shapes:\n') 
print(f'tensor_A = {tensor_A.shape}')
print(f'tensor_B = {tensor_B.shape}\n')
print(f'New shapes:\n')
print(f'tensor_A = {tensor_A.shape} (same as above)')
print(f'tensor_B.T = {tensor_B.T.shape}\n')
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")

# torch.mm is a shortcut for matmul
print("==>> torch.mm(tensor_A, tensor_B.T): ", torch.mm(tensor_A, tensor_B.T))