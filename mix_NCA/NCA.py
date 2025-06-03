import torch
import torch.nn.functional as F
import torch.nn as nn

# This code as been adapted from the amazing implementation of
# https://github.com/greydanus/studying_growth
# Go check it out with the associated blog post, it is a great introduction to NCAs!

class NCA(nn.Module):
    def __init__(self, update_net, state_dim=16, hidden_dim=128, dropout=0, 
                 device="cuda", use_alive_mask=True, alive_threshold=0.1, 
                 alive_channel=3, maintain_seed=True, residual=True, 
                 grid_type="square", modality="image", filter_type="sobel",
                 distribution=None, random_updates=False, seed_value=1.0):
        super(NCA, self).__init__()
        self.state_dim = state_dim
        self.dropout = dropout
        self.device = device
        self.use_alive_mask = use_alive_mask
        self.alive_threshold = alive_threshold
        self.alive_channel = alive_channel
        self.maintain_seed = maintain_seed
        self.residual = residual
        self.grid_type = grid_type
        self.modality = modality
        self.filter_type = filter_type
        self.distribution = distribution
        self.update = update_net
        self.random_updates = random_updates
        self.seed_value = seed_value
        # Setup perception filters based on grid type
        if grid_type == "square":
            self._setup_perception_filters_square(device)
        elif grid_type == "hexagonal":
            self._setup_perception_filters_hex(device)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
            
        # Set perception function based on modality
        if modality == "image":
            self.perceive = self._perceive_image
        elif modality == "tensor":
            self.perceive = self._perceive_tensor
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _setup_perception_filters_hex(self, device):
        self.identity = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]], dtype=torch.float32).to(device)
        
        if self.filter_type == "sobel":
            self.sobel_x = torch.tensor([
                [0, 0, 0], 
                [-1, 0, 1], 
                [-1, 0, 1],
                [0, 0, 0]
                ], dtype=torch.float32).to(device)
            
            self.sobel_y = torch.tensor([
                [0, 2, 0], 
                [1, 0, 1], 
                [-1, 0, -1],
                [0, -2, 0]
                ], dtype=torch.float32).to(device)
            
            self.register_buffer('sobel_x_kernel', 
                self.sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_y_kernel', 
                self.sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
        else:  # laplacian
            self.laplacian = torch.tensor([
                [0, 1, 0],
                [1, -6, 1],
                [1, 0, 1],
                [0, 1, 0]
            ], dtype=torch.float32).to(device)
            
            self.register_buffer('laplacian_kernel', 
                self.laplacian.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
        
        self.register_buffer('identity_kernel', 
            self.identity.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

    def _setup_perception_filters_square(self, device):
        self.identity = torch.tensor([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]], dtype=torch.float32).to(device)
        
        if self.filter_type == "sobel":
            self.sobel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).to(device)
            
            self.sobel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).to(device)
            
            self.register_buffer('sobel_x_kernel', 
                self.sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_y_kernel', 
                self.sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
        else:  # laplacian
            self.laplacian = torch.tensor([[1, 1, 1],
                                         [1, -8, 1],
                                         [1, 1, 1]], dtype=torch.float32).to(device)
            
            self.register_buffer('laplacian_kernel', 
                self.laplacian.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
        
        self.register_buffer('identity_kernel', 
            self.identity.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

    def _perceive_image(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=1, groups=self.state_dim)
        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.state_dim)
            return torch.cat([identity, laplacian, laplacian], dim=1)  # Duplicate to maintain same dimensions

    def _perceive_tensor(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)
        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=0, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, laplacian, laplacian], dim=1)  # Duplicate to maintain same dimensions

    def forward(self, x, num_steps, seed_loc=None, return_history=False):
        frames = []
        
        for i in range(num_steps):
            if torch.isnan(x).any():
                print(f"NaN detected in state at step {i}")
                break
                
            # Get pre-update alive mask if using
            if self.use_alive_mask:
                alive_mask_pre = nn.functional.max_pool2d(
                    x[:, self.alive_channel:self.alive_channel+1], 
                    3, stride=1, padding=1
                ) > self.alive_threshold
            
            # Create update mask based on modality
            if self.residual:
                if self.modality == "image":
                    update_mask = torch.rand(*x.shape, device=x.device) > self.dropout
                elif self.modality == "tensor":
                    update_mask = torch.rand(*x[:,:,1:2,1:2].shape, device=x.device) > self.dropout
            
            # Perceive neighborhood and compute update
            perceived = self.perceive(x)
            dx = self.update(perceived)
            
            # Transform network output based on distribution
            if self.distribution == "normal":
                # Split output into mean and log_std
                mean, log_std = torch.chunk(dx, 2, dim=1)
                log_std = torch.clamp(log_std, min=-6, max=6)
                # Ensure positive std dev
                std = torch.clamp(torch.nn.functional.softplus(log_std), min=1e-6, max=10) * 0.06
                # Sample from normal distribution
                if self.random_updates:
                    dx = mean + std * torch.randn_like(mean)
                else:
                    dx = mean
                
            
            # Apply updates
            if self.residual:
                dx = torch.clamp(dx, -5.0, 5.0)
                if self.modality == "image":
                    x = x + update_mask * dx
                elif self.modality == "tensor":
                    x = x[:,:,1:2,1:2] + update_mask * dx
            else:
                x = dx
            
            # Apply alive mask if using
            if self.use_alive_mask:
                alive_mask_post = nn.functional.max_pool2d(
                    x[:, self.alive_channel:self.alive_channel+1], 
                    3, stride=1, padding=1
                ) > self.alive_threshold
                x = x * alive_mask_pre * alive_mask_post
            
            # Maintain seed if specified
            if seed_loc is not None and self.maintain_seed:
                x[..., self.alive_channel, seed_loc[0], seed_loc[1]] = self.seed_value
            
            if return_history:
                frames.append(x.clone())
                
        return torch.stack(frames) if frames else x

