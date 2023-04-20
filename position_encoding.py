import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    out_dim = 3
    embed_fns = []
    #include x
    embed_fns.append(lambda x : x)
    freq_bands = 2.**torch.linspace(0., multires-1, steps=multires)
    
    ## each sin/cos add 3 dim
    for freq in freq_bands:
        embed_fns.append(lambda x : torch.sin(x * freq))
        embed_fns.append(lambda x : torch.cos(x * freq))
        out_dim += 6
    embed = lambda x:torch.cat([fn(x) for fn in embed_fns], -1)
    return embed, out_dim