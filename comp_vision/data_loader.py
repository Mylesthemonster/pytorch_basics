import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import train_data, test_data, class_names

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
# print(f"Dataloaders: {train_dataloader, test_dataloader}") 
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print("==>> train_features_batch.shape: ", train_features_batch.shape)
# print("==>> train_labels_batch.shape: ", train_labels_batch.shape)

# Show a sample
# torch.manual_seed(42)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# print(f"Image size: {img.shape}")
# print(f"Label: {label}, label size: {label.shape}")
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis("Off")
# plt.show()