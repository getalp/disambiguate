import torch


cpu_device = torch.device("cpu")

if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
    default_device = gpu_device
else:
    default_device = cpu_device

