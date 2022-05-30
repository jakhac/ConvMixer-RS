import torch

print("CUDA is available", torch.cuda.is_available())
print("CUDA device", torch.cuda.get_device_name(0))
print("CUDA 0", torch.device("cuda:0"))
print("Torch version", torch.__version__)

print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
