import torch
from torch import nn
from data import NUM_CLASSES, NUM_FEATURES, RANDOM_SEED, X_blob_train, X_blob_test, y_blob_train, y_blob_test

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        device = torch.device('cpu')
else:
    device = torch.device('mps') 
    print("==>> device: ", device)
    
# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(), # <- Dataset require non-linear layers
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(), # <- Dataset require non-linear layers
            nn.Linear(in_features=hidden_units, out_features=output_features), # how many classes are there?
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=NUM_FEATURES, 
                    output_features=NUM_CLASSES, 
                    hidden_units=8).to(device)
print("==>> model_4: ", model_4)

# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), 
                            lr=0.1) # exercise: try changing the learning rate here and seeing what happens to the model's performance

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Perform a single forward pass on the data (we'll need to put it to the target device for it to work)
# print("==>> model_4(X_blob_train.to(device))[:5]: ", model_4(X_blob_train.to(device))[:5])
# How many elements in a single prediction sample?
# print("==>> model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES: ", model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES)

# logits -> prediction probabilities -> prediction labels
# Make prediction logits with model
y_logits = model_4(X_blob_test.to(device))

# Perform softmax calculation on logits across dimension 1 to get prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1) 
# print(y_logits[:5].to('cpu'))
# print(y_pred_probs[:5].to('cpu'))

# Sum the first sample output of the softmax activation function 
# print("==>> torch.sum(y_pred_probs[0]): ", torch.sum(y_pred_probs[0]))

# Which class does the model think is *most* likely at the index 0 sample?
# print(y_pred_probs[0])
# print(torch.argmax(y_pred_probs[0]))