import torch
import matplotlib.pyplot as plt

# Check PyTorch version
# print("==>> torch.__version__: ", torch.__version__)

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print("==>> X[:10]: ", X[:10])
# print("==>> y[:10]: ", y[:10])

## Split data into training and validation sets
# Training set     The model learns from this data                                    ~60-80%     Always
# Validation set   The model gets tuned on this data                                  ~10-20%     Often but not always
# Testing set      The model gets evaluated on this data to test what it has learned  ~10-20%	  Always

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print("==>> len(X_train): ", len(X_train))
print("==>> len(y_train): ", len(y_train))
print("==>> len(X_test): ", len(X_test))
print("==>> len(y_test): ", len(y_test))

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
  plt.show()
  
# plot_predictions()