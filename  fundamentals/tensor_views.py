# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

x = torch.arange(1., 8.)
print("==>> x: ", x)
print("==>> x.shape: ", x.shape)

# Add an extra dimension
x_reshaped = x.reshape(1, 7)
print("==>> x_reshaped: ", x_reshaped)
print("==>> x_reshaped.shape: ", x_reshaped.shape)

# Change view (keeps same data as original but changes view)
z = x.view(1, 7)
print("==>> z: ", z)
print("==>> z.shape: ", z.shape)

# Changing z changes x
z[:, 0] = 5
print("==>> z: ", z)
print("==>> x: ", x)

# Stack tensors on top of each other
x_stacked_dim0 = torch.stack([x, x, x, x], dim=0) 
print("==>> x_stacked_dim0: ", x_stacked_dim0)
x_stacked_dim1 = torch.stack([x, x, x, x], dim=1)
print("==>> x_stacked_dim1: ", x_stacked_dim1)

# Remove extra dimension from x_reshaped
print(f"\nPrevious tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")
x_squeezed = x_reshaped.squeeze()
print('Squeezed tensor:')
print(f"New tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
print(f"\nPrevious tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print('Unsqueezed tensor:')
print(f"New tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"\nPrevious shape: {x_original.shape}")
print('Permute tensor to rearrange the axis order:')
print(f"New shape: {x_permuted.shape}")