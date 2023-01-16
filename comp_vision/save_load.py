from pathlib import Path
import torch
from data_loader import train_dataloader, test_dataloader, train_features_batch, train_labels_batch, train_data, test_data, class_names
from cnn_ex import FashionMNISTModelV2, model_2, loss_fn, accuracy_fn, eval_model, model_2_results

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        if not torch.cuda.is_available():     
            device = torch.device('cpu')
            print("==>> device: ", device)                      
        else:                                                      
            device = torch.device('cuda')                       
            print("==>> device: ", device)
else:
    # this ensures that the current MacOS version is at least 12.3+
    print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print(torch.backends.mps.is_built())
    device = torch.device('mps') 
    print("==>> device: ", device)
    
# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

# Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# Note: loading model will error if the shapes here aren't the same as the saved version
loaded_model_2 = FashionMNISTModelV2(input_shape=1, 
                                    hidden_units=10, # try changing this to 128 and seeing what happens 
                                    output_shape=10) 

# Load in the saved state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model_2 = loaded_model_2.to(device)

# Evaluate loaded model
torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)

print("==>> loaded_model_2_results: ", loaded_model_2_results)
print("==>> model_2_results: ", model_2_results)

# Check to see if results are close to each other (if they are very far away, there may be an error)
torch.isclose(torch.tensor(model_2_results["model_loss"]), 
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-08, # absolute tolerance
              rtol=0.0001) # relative tolerance