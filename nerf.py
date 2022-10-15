import torch 
from torch import nn 

#gloabal variables 
n = 100 #number of cameras 
m = 100 #number of points to sample for each ray 
d = 100 #projection space dimension 
near_thresh = 2.0
far_thresh = 6.0
H = 
W = 

#data: camera poses, image, 


"""Step 1: Marching the Camera Rays Through the Scene
Input: A set of camera poses {x_c, y_c, z_c, gamma_c, theta_c} x n , height H, width W 
Output: A bundle of rays for every pose {v_o, v_d} x {H x W x n}
"""
def get_v(init_position, H, W): 
    v_0 = init_position[:, :, :, :]
    v_d = 

    return v_0, v_d 

"""Step 2: Collecting Query Points
Input: A bundle of rays for every pose {v_o, v_d} x {H x W x n}
Output: A set of 3D query points {x_p, y_p, z_p} x {H x W x m x n}
"""
def get_query_points(v_0 , v_d):
    depth_values = torch.linspace(near_thresh, far_thresh, m)
    query_points = (
        v_0[:, :, None, :]
        + v_d[:, :, None, :] * depth_values[:, :, :, None]
    )
    return query_points 

"""Step 3: Projecting Query Points to High-Dimensional Space (Positional Encoding)
Input: A set of 3D query points {x_p, y_p, z_p} x {H x W x m x n}
Output: A set of query points embedded into d-dimensional space {x_1, x_2, ... x_d} x {H x W x m x n}
"""
def positional_encoding(query_points):
    query_points = query_points.reshape(-1, 3)
    encoding = []
    frequencies = torch.linspace(2.0 ** 0.0, 2.0 ** (d/3 - 1), d/3)    
    for freq in frequencies:
        encoding.append(torch.sin(query_points * freq))
        encoding.append(torch.cos(query_points * freq))
    return encoding

"""Step 4: Neural Network Inference
Input: A set of query points embedded into d-dimensional space {x_1, x_2, ... x_d} x {H x W x m x n}
Output: RGB colour and volume density for every query point {R, G, B, sigma} x {H x W x m x n}
"""
class NeRF(nn.Module):
    def __init__(self, mid_channels=128):
        super().__init__()
        self.layer1 = torch.nn.Linear(d, mid_channels)
        self.layer2 = torch.nn.Linear(mid_channels, mid_channels)
        self.layer3 = torch.nn.Linear(mid_channels, 4)
    def forward(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        x = self.layer3(x)
        return x 

"""Step 5: Volume Rendering
Input: Depth values, RGB colour and volume density {R, G, B, sigma} x {H x W x m x n}
Output: A set of rendered images (one per pose) {H, W} x n 
"""
def exclusive_cumprod(x):
    """
    [a, b, c, d] -> [1, a, a * b, a * b *c]
    """
    cp = torch.cumprod(x, dim=-1)
    cp = torch.roll(cp, 1, dims=-1)
    cp[..., 0] = 1.
    return cp

def volume_render(network_output, depth_values): 
    sigma= torch.nn.functional.relu(network_output[:, :, :, :, 3])
    rgb = torch.sigmoid(network_output[:, :, :, :, :3])

    dists = torch.ones(m)*(depth_values[..., 1] - depth_values[..., 0])

    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * exclusive_cumprod(1.0 - alpha)

    rendered_images = (weights[:, :, :, :, None] * rgb).sum(dim=-2)
    return rendered_images 

"""
TRAIN
"""
def render_loss(rendered_images, gt_images):
    return nn.MSEloss(rendered_images, gt_images)
