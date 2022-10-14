import torch 
from torch import nn 

#gloabal variables 
n = 100 #number of cameras 
m = 100 #number of points to sample for each ray 
d = 100 #projection space dimension 

"""Step 1: Marching the Camera Rays Through the Scene
Input: A set of camera poses {x_c, y_c, z_c, gamma_c, theta_c} x n 
Output: A bundle of rays for every pose {v_o, v_d} x {H x W x n}
"""
def get_v(init_position): #torch.randn(n, 1, 1, 1, 1, 1)
    v_0 = init_position[:, :, :, :]
    v_d = 

    return v_0, v_d 

"""Step 2: Collecting Query Points
Input: A bundle of rays for every pose {v_o, v_d} x {H x W x n}
Output: A set of 3D query points {x_p, y_p, z_p} x {n x m x H x W}
"""
def get_query_points(v_0 , v_d):
    query_points = 
    return query_points 

"""Step 3: Projecting Query Points to High-Dimensional Space (Positional Encoding)
Input: A set of 3D query points {x_p, y_p, z_p} x {n x m x H x W}
Output: A set of query points embedded into d-dimensional space {x_1, x_2, ... x_d} x {n x m x H x W}
"""
def positional_encoding(query_points):
    projection = 
    return projection 

"""Step 4: Neural Network Inference
Input: A set of query points embedded into d-dimensional space {x_1, x_2, ... x_d} x {n x m x H x W}
Output: RGB colour and volume density for every query point {R, G, B, sigma} x {n x m x H x W}
"""
class NeRF(nn.Module):
    def __init__(self):
        super().__init__()

"""Step 5: Volume Rendering
Input: Query points, RGB colour and volume density {x_1, x_2, ... x_d, R, G, B, sigma} x {n x m x H x W}
Output: A set of rendered images (one per pose) {H, W} x n 
"""
def volume_render(network_output): 

    rendered_images = 
    return rendered_images 


def render_loss(rendered_images, gt_images):
    return nn.MSEloss(rendered_images, gt_images)
