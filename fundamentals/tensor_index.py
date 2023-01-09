# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

x = torch.arange(1, 10).reshape(1, 3, 3)
print("==>> x: ", x)
print("==>> x.shape: ", x.shape)

# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")

# Get all values of 0th dimension and the 0 index of 1st dimension
print("==>> x[:, 0]: ", x[:, 0])

# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
print("==>> x[:, :, 1]: ", x[:, :, 1])

# Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
print("==>> x[:, 1, 1]: ", x[:, 1, 1])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension (same as x[0][0])
print("==>> x[0, 0, :]: ", x[0, 0, :])