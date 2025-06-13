import torch
import torch.nn as nn
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def glorot_orthogonal(tensor, scale):
    # 参数初始化方法
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.requires_grad == True:
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
                # glorot_orthogonal(tensor=param, scale=2)
                # nn.init.xavier_uniform_(param)