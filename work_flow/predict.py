import torch
from torch import nn
import matplotlib.pyplot as plt
from train import model_0
from data import weight, bias, X_train, y_train, X_test, y_test, plot_predictions

# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
print("==>> y_preds: ", y_preds)

# plot_predictions(predictions=y_preds)