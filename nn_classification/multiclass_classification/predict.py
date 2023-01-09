import torch
from train import model_4, accuracy_fn, X_blob_train, X_blob_test, y_blob_train, y_blob_test, device
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary
from torchmetrics import Accuracy

# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# View the first 10 predictions
print("==>> y_logits[:10]: ", y_logits[:10].to('cpu'))

# Turn predicted logits in prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)

# Turn prediction probabilities into prediction labels
y_preds = y_pred_probs.argmax(dim=1)

# Compare first 10 model preds and test labels
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_4, X_blob_train, y_blob_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_4, X_blob_test, y_blob_test)
# plt.show()

# Setup metric and make sure it's on the target device
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to('cpu')

# Calculate accuracy
print("==>> torchmetrics_accuracy(y_preds.to('cpu'), y_blob_test.to('cpu')): ", torchmetrics_accuracy(y_preds.to('cpu'), y_blob_test.to('cpu')))