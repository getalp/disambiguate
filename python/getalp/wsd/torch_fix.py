import torch


def torch_tensor(*args, **kwargs) -> torch.Tensor:
    return torch.tensor(*args, **kwargs)


def torch_from_numpy(*args, **kwargs) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs)


def torch_empty(*args, **kwargs) -> torch.Tensor:
    return torch.empty(*args, **kwargs)


def torch_zeros(*args, **kwargs) -> torch.Tensor:
    return torch.zeros(*args, **kwargs)


def torch_ones(*args, **kwargs) -> torch.Tensor:
    return torch.ones(*args, **kwargs)


def torch_ones_like(*args, **kwargs) -> torch.Tensor:
    return torch.ones_like(*args, **kwargs)


def torch_full(*args, **kwargs) -> torch.Tensor:
    return torch.full(*args, **kwargs)


def torch_cat(*args, **kwargs) -> torch.Tensor:
    return torch.cat(*args, **kwargs)


def torch_stack(*args, **kwargs) -> torch.Tensor:
    return torch.stack(*args, **kwargs)


def torch_transpose(*args, **kwargs) -> torch.Tensor:
    return torch.transpose(*args, **kwargs)


def torch_squeeze(*args, **kwargs) -> torch.Tensor:
    return torch.squeeze(*args, **kwargs)


def torch_unsqueeze(*args, **kwargs) -> torch.Tensor:
    return torch.unsqueeze(*args, **kwargs)


def torch_max(*args, **kwargs) -> torch.Tensor:
    return torch.max(*args, **kwargs)


def torch_argmax(*args, **kwargs) -> torch.Tensor:
    return torch.argmax(*args, **kwargs)


def torch_arange(*args, **kwargs) -> torch.Tensor:
    return torch.arange(*args, **kwargs)


def torch_exp(*args, **kwargs) -> torch.Tensor:
    return torch.exp(*args, **kwargs)


def torch_sin(*args, **kwargs) -> torch.Tensor:
    return torch.sin(*args, **kwargs)


def torch_cos(*args, **kwargs) -> torch.Tensor:
    return torch.cos(*args, **kwargs)


torch_float32: torch.dtype = torch.float32
torch_long: torch.dtype = torch.long
torch_uint8: torch.dtype = torch.uint8
