import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_rays(H, W, K, c2w):
    dirs=torch.zeros((H,W,3))
    half_H=int(H/2)
    half_W=int(W/2)
    focal=K[0][2]
    # transform pixles to camera frame 
    for x in range(H):
        for y in range(W):
            dirs[x,y]=torch.Tensor([(x-half_H)/focal, -(y-half_W)/focal,-1])
    dirs=dirs.transpose(1,0)      
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    dirs=np.zeros((H,W,3))
    half_H=int(H/2)
    half_W=int(W/2)
    focal=K[0][2]
    # transform pixles to camera frame 
    for x in range(H):
        for y in range(W):
            dirs[x,y]=[(x-half_H)/focal, -(y-half_W)/focal,-1]
    dirs=dirs.transpose(1,0,2)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

