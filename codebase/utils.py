import math
import torch 
import torch.nn as nn

def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))

def bias_init_with_prob(prior_prob=0.01):
    """
    Initialize conv/fc bias value according to a given probability value.
    """
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

def get_activation(act, inpace=True):
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU() 
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f"Unknown activation {act} requested")  
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m

def collate_fn(batch):
    targets = []
    images = []
    for image, target in batch:
        targets.append(target)
        images.append(image)
    return torch.stack(images), targets