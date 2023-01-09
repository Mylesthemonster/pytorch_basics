import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=False,
    transform=ToTensor()
)

# See first training sample
image, label = train_data[0]
# print("==>> image: ", image)
# print("==>> label: ", label)

# What's the shape of the image?
# print("==>> image.shape: ", image.shape)
# [color_channels=1, height=28, width=28]
# Format as NHWC (Number of Images, Height, Width, Color Channels) performs best 

# How many samples are there? 
# print("==>> len(train_data.data): ", len(train_data.data))
# print("==>> len(train_data.targets): ", len(train_data.targets))
# print("==>> len(test_data.data): ", len(test_data.data))
# print("==>> len(test_data.targets): ", len(test_data.targets))

# See classes
class_names = train_data.classes
# print("==>> class_names: ", class_names)
# 10 different classes, it means our problem is multi-class classification

# Visualize the data
image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap="gray") # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(class_names[label])
# plt.show()

# Plot more images
torch.manual_seed(42)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()