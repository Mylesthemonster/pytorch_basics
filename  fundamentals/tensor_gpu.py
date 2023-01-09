# python 3.10.9

import torch
print("==>> torch.__version__: ", torch.__version__)

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        device = torch.device('cpu')
else:
    # this ensures that the current MacOS version is at least 12.3+
    print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print(torch.backends.mps.is_built())
    device = torch.device('mps') 
    print("==>> device: ", device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print("==>> tensor_on_gpu: ", tensor_on_gpu)

# If tensor is on GPU, can't transform it to NumPy (this will error)
# tensor_on_gpu.numpy()

# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print("==>> tensor_back_on_cpu: ", tensor_back_on_cpu, tensor.device)
print("==>> tensor_on_gpu: ", tensor_on_gpu)