import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Normal

import torch
import torch.nn.functional as F
import torch.nn as nn

class MixtureNCANoise(nn.Module):
    def __init__(self, update_nets, num_rules=5, state_dim=16, hidden_dim=128, 
                 dropout=0, temperature=1.0, device="cuda", num_latent_dims=1,
                 use_alive_mask=True, alive_threshold=0.1, alive_channel=3, 
                 maintain_seed=True, residual=True, grid_type = "square", modality = "image", filter_type="sobel", 
                 save_internal_noise=False, distribution=None, seed_value=1.0):
        super(MixtureNCANoise, self).__init__()
        self.state_dim = state_dim
        self.dropout = dropout
        self.num_rules = num_rules
        self.device = device
        self.temperature = temperature
        self.use_alive_mask = use_alive_mask
        self.alive_threshold = alive_threshold
        self.alive_channel = alive_channel
        self.maintain_seed = maintain_seed
        self.residual = residual
        self.num_latent_dims = num_latent_dims
        self.grid_type = grid_type
        self.filter_type = filter_type
        self.save_internal_noise = save_internal_noise
        self.internal_noise = None
        self.distribution = distribution
        self.seed_value = seed_value
        input_mult = 3 if filter_type == "sobel" else 2
        # Create multiple update networks
        self.update_nets = nn.ModuleList([
            update_nets(state_dim * input_mult + num_latent_dims, hidden_dim, state_dim, device) 
            for _ in range(num_rules)
        ])
        
        # Modified mixture network with batch norm for stability
        self.mixture_net = nn.Sequential(
            nn.Conv2d(state_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_rules, 1),
        ).to(device)
        
        if grid_type == "square":
            self._setup_perception_filters_square(device)
        elif grid_type == "hexagonal":
            self._setup_perception_filters_hex(device)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")

        self.modality = modality
        if modality == "image":
            self.perceive = self._perceive_image
        elif modality == "tensor":
            self.perceive = self._perceive_tensor
        
    def straight_through_sample(self, logits, sample_non_differentiable=False, temperature=None, straight_through=True):
        """
        Sample from logits using either Gumbel-Softmax (with optional straight-through estimator) or actual sampling.

        Args:
            logits: Input logits
            sample_non_differentiable: If True, actually sample from the distribution (multinomial). If False, use Gumbel-Softmax.
            temperature: Optional temperature for Gumbel-Softmax. If None, uses self.temperature.
            straight_through: If True, use straight-through estimator. If False, return soft Gumbel-Softmax sample.
        """
        if temperature is None:
            temperature = self.temperature

        probs = F.softmax(logits / temperature, dim=1)

        if sample_non_differentiable:
            # Sample from the multinomial distribution
            shape = probs.shape
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, shape[1])
            samples_idx = torch.multinomial(flat_probs, 1)
            hard_samples = torch.zeros_like(flat_probs)
            hard_samples.scatter_(1, samples_idx, 1.0)
            hard_samples = hard_samples.reshape(shape[0], shape[2], shape[3], shape[1])
            hard_samples = hard_samples.permute(0, 3, 1, 2)
            return hard_samples
        else:
            # Gumbel-Softmax (optionally with straight-through)
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            y = (logits + gumbel_noise) / temperature
            y_soft = F.softmax(y, dim=1)

            if straight_through:
                # Hard one-hot
                _, max_idx = y_soft.max(dim=1, keepdim=True)
                y_hard = torch.zeros_like(logits).scatter_(1, max_idx, 1.0)
                # Straight-through estimator
                samples = y_hard.detach() - y_soft.detach() + y_soft
                return samples
            else:
                # Just return the soft Gumbel-Softmax sample
                return y_soft
            
            

    def forward(self, x, num_steps, seed_loc=None, return_history=False, sample_non_differentiable = False, temperature = None, straight_through = True,  weights = None):
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
            
            # Update state
            if self.residual:
                if self.modality == "image":
                    update_mask = torch.rand(*x.shape, device=x.device) > self.dropout
                elif self.modality == "tensor":
                    update_mask = torch.rand(*x[:,:,1:2,1:2].shape, device=x.device) > self.dropout
            
            # Compute mixture weights and get selected rules
            if self.modality == "image":
                identity = F.conv2d(x, self.identity_kernel, padding=1, groups=self.state_dim)
            elif self.modality == "tensor":
                identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)
            
            mixture_logits = self.mixture_net(identity)
            mixture_weights = self.straight_through_sample(mixture_logits, sample_non_differentiable = sample_non_differentiable, temperature = temperature, straight_through = straight_through)  # [batch, num_rules, height, width]
            

            if weights is not None:
                # multiply the mixture weights by the weights
                mixture_weights = mixture_weights * weights

            # Perceive input once
            perceived = self.perceive(x)  # [batch, state_dim*3, height, width]
            
            x_k = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)).sample((mixture_weights.shape[0], self.num_rules, self.num_latent_dims, perceived.shape[-2], perceived.shape[-1]))

            # Apply all rules at once and combine
            all_updates = torch.stack([
                update_net(torch.cat([perceived, x_k[:, i]], dim=1)) for i, update_net in enumerate(self.update_nets)
            ], dim=1)  # [batch, num_rules, state_dim, height, width]

            if self.save_internal_noise:
                self.internal_noise = x_k * mixture_weights.unsqueeze(2)
            
            # Combine updates using mixture weights
            combined_update = (all_updates * mixture_weights.unsqueeze(2)).sum(dim=1)
            

            if self.residual:
                # Apply updates with clamping
                combined_update = torch.clamp(combined_update, -5.0, 5.0)
                if self.modality == "image":
                    x = x + update_mask * combined_update
                elif self.modality == "tensor":
                    x = x[:,:,1:2,1:2] + update_mask * combined_update
            else:
                x = combined_update
            
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
            
            self.register_buffer('identity_kernel', 
                self.identity.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_x_kernel', 
                self.sobel_x.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('sobel_y_kernel', 
                self.sobel_y.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
        
        elif self.filter_type == "laplacian":
            self.laplacian = torch.tensor([[1, 1, 1],
                                         [1, -8, 1],
                                         [1, 1, 1]], dtype=torch.float32).to(device)
            
            self.register_buffer('identity_kernel', 
                self.identity.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))
            self.register_buffer('laplacian_kernel', 
                self.laplacian.unsqueeze(0).unsqueeze(0).repeat(self.state_dim, 1, 1, 1))

    def _perceive_image(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=1, groups=self.state_dim)
        
        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=1, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=1, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.state_dim)
            return torch.cat([identity, laplacian], dim=1)
    
    def _perceive_tensor(self, x):
        identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)
        
        if self.filter_type == "sobel":
            sobel_x = F.conv2d(x, self.sobel_x_kernel, padding=0, groups=self.state_dim)
            sobel_y = F.conv2d(x, self.sobel_y_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, sobel_x, sobel_y], dim=1)
        else:  # laplacian
            laplacian = F.conv2d(x, self.laplacian_kernel, padding=0, groups=self.state_dim)
            return torch.cat([identity, laplacian], dim=1)  
    
    def get_rule_probabilities(self, x):
        """Get rule assignment probabilities for each position
        
        Args:
            x: Input state tensor of shape (batch_size, channels, height, width)
            
        Returns:
            probabilities: Tensor of shape (batch_size, num_rules, height, width)
                        containing assignment probability for each rule at each position
        """
        with torch.no_grad():
            # Get mixture logits
            if self.modality == "image":
                identity = F.conv2d(x, self.identity_kernel, padding=1, groups=self.state_dim)
            elif self.modality == "tensor":
                identity = F.conv2d(x, self.identity_kernel, padding=0, groups=self.state_dim)
            mixture_logits = self.mixture_net(identity)
            
            # Convert to probabilities
            probabilities = F.softmax(mixture_logits / self.temperature, dim=1)
            
        return probabilities

    def visualize_rule_assignments(self, x, rule_idx=None):
        """Visualize rule assignment probabilities with improved colorbar formatting
        
        Args:
            x: Input state tensor
            rule_idx: Optional specific rule index to visualize. If None, shows all rules.
        """
        import matplotlib.pyplot as plt
        
        probs = self.get_rule_probabilities(x)
        batch_size, num_rules, height, width = probs.shape
        
        if rule_idx is not None:
            # Show single rule
            plt.figure(figsize=(6, 6))
            im = plt.imshow(probs[0, rule_idx].cpu().numpy())
            plt.colorbar(im, format='%.2f')  # Limit decimal places
            plt.title(f'Rule {rule_idx} Assignment Probability')
        else:
            # Show all rules
            fig, axes = plt.subplots(1, num_rules, figsize=(4*num_rules, 4))
            for i in range(num_rules):
                im = axes[i].imshow(probs[0, i].cpu().numpy())
                axes[i].set_title(f'Rule {i}')
                # Add colorbar with formatting
                cbar = plt.colorbar(im, ax=axes[i], format='%.6f')
                # Adjust number of ticks
                cbar.locator = plt.MaxNLocator(5)  # Limit number of ticks
                cbar.update_ticks()
        
        plt.tight_layout()  # Adjust spacing between subplots
        return fig

