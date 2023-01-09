import torch
from train import model_0
from model import LinearRegressionModel
from data import X_test
from predict import y_preds
from pathlib import Path

## Saving and loading a PyTorch model
# torch.save	
# Saves a serialized object to disk using Python's pickle utility. Models, tensors and various other Python objects like dictionaries can be saved using torch.save.
# torch.load	
# Uses pickle's unpickling features to deserialize and load pickled Python object files (like models, tensors or dictionaries) into memory. You can also set which device to load the object to (CPU, GPU etc).
# torch.nn.Module.load_state_dict	
# Loads a model's parameter dictionary (model.state_dict()) using a saved state_dict() object.

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

# Check the saved file path
import subprocess

bashCommand = "ls -l models/01_pytorch_workflow_model_0.pth"
result = subprocess.run(bashCommand.split(), stderr=subprocess.PIPE, text=True)
print(result.stderr)

# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
print("==>> loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)): ", loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model

# Compare previous model predictions with loaded model predictions (these should be the same)
print("==>> y_preds == loaded_model_preds: ", y_preds == loaded_model_preds)