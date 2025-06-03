import torch
import numpy as np
import matplotlib.pyplot as plt

class RobustnessAnalysis:
    def __init__(self, standard_nca, mixture_nca, stochastic_mixture_nca, device="cuda"):
        """Initialize with three NCA models for comparison
        
        Args:
            standard_nca: Standard NCA model
            mixture_nca: Deterministic mixture NCA model
            stochastic_mixture_nca: Stochastic Gaussian mixture NCA model
            device: Computing device
        """
        self.standard_nca = standard_nca
        self.mixture_nca = mixture_nca
        self.stochastic_mixture_nca = stochastic_mixture_nca
        self.device = device
        
    def apply_perturbation(self, state, perturbation_type, seed=None, **kwargs):
        """Apply different types of perturbations to the state
        
        Args:
            state: Input state tensor
            perturbation_type: String indicating type of perturbation
                'deletion': Remove square patch of cells
                'noise': Add random noise
                'corruption': Set random cells to random values
                'masking': Mask out specific channels
                'random_pixels': Mask n random pixels
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for specific perturbations
        """
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        perturbed_state = state.clone()
        
        if perturbation_type == 'random_pixels':
            # Get number of pixels to mask
            n_pixels = kwargs.get('n_pixels', 100)
            
            # Get dimensions
            B, C, H, W = state.shape
            
            # Generate random pixel locations
            total_pixels = H * W
            flat_indices = torch.randperm(total_pixels)[:n_pixels]
            
            # Convert to 2D indices
            y_indices = flat_indices // W
            x_indices = flat_indices % W
            
            # Create mask for all channels
            for b in range(B):
                for c in range(C):
                    perturbed_state[b, c, y_indices, x_indices] = 0
                
        elif perturbation_type == 'deletion':
            # Get parameters
            size = kwargs.get('size', 10)
            
            # Get dimensions
            B, C, H, W = state.shape
            
            # Create binary mask of non-empty cells (assuming empty is when all channels are 0)
            non_empty = (state.sum(dim=1) > 0)  # [B, H, W]
            
            for b in range(B):
                # Get valid positions (non-empty cells with enough margin for the square)
                valid_y, valid_x = torch.where(non_empty[b][size//2:-(size//2), size//2:-(size//2)])
                
                if len(valid_y) > 0:
                    # Randomly select one valid position
                    idx = torch.randint(0, len(valid_y), (1,))
                    center_y = valid_y[idx] + size//2
                    center_x = valid_x[idx] + size//2
                    
                    # Apply deletion
                    x1, x2 = center_x - size//2, center_x + size//2
                    y1, y2 = center_y - size//2, center_y + size//2
                    perturbed_state[b, :, x1:x2, y1:y2] = 0
            
        elif perturbation_type == 'noise':
            # Add Gaussian noise
            noise_level = kwargs.get('noise_level', 0.1)
            noise = torch.randn_like(state) * noise_level
            perturbed_state = (state + noise).clamp(0, 1)
            
        elif perturbation_type == 'corruption':
            # Randomly corrupt cells
            corruption_prob = kwargs.get('corruption_prob', 0.1)
            mask = torch.rand_like(state) < corruption_prob
            random_values = torch.rand_like(state)
            perturbed_state = torch.where(mask, random_values, state)
            
        elif perturbation_type == 'masking':
            # Mask specific channels
            channels = kwargs.get('channels', [])
            for c in channels:
                perturbed_state[:, c] = 0
                
        # Reset seed if it was set
        if seed is not None:
            torch.manual_seed(torch.seed())
        
        return perturbed_state
    
    def measure_recovery(self, original_state, perturbed_state, model, steps=100, seed_loc = None):
        """Measure how well the model recovers from perturbation"""
        # Run model on original and perturbed states
        with torch.no_grad():
            # check if attribute temperature is present
            if hasattr(model, 'temperature'):
                original_trajectory = model(original_state, steps, return_history=True, seed_loc = seed_loc, sample_non_differentiable = True)
                perturbed_trajectory = model(perturbed_state, steps, return_history=True, seed_loc = seed_loc, sample_non_differentiable = True)
            else:
                original_trajectory = model(original_state, steps, return_history=True, seed_loc = seed_loc)
                perturbed_trajectory = model(perturbed_state, steps, return_history=True, seed_loc = seed_loc)
            
        # Compute recovery metrics
        recovery_error = torch.mean((original_trajectory - perturbed_trajectory)**2, dim=(2,3,4))
        final_error = recovery_error[-1]
        recovery_time = torch.where(recovery_error < 0.1)[0]
        
        if len(recovery_time) > 0:
            recovery_time = recovery_time[0].item()
        else:
            recovery_time = steps


        
        return {
            'recovery_error': recovery_error.cpu().numpy(),
            'final_error': final_error.item(),
            'recovery_time': recovery_time
        }
    
    def compare_robustness(self, init_state, perturbation_type, n_runs=5, steps=100, seed=None, **kwargs):
        """Compare robustness of all three NCA variants"""
        standard_metrics = []
        mixture_metrics = []
        stochastic_metrics = []
        
        for i in range(n_runs):
            # Apply perturbation
            perturbed_state = self.apply_perturbation(init_state, perturbation_type, i, **kwargs)
            
            # Measure recovery for all three models
            standard_recovery = self.measure_recovery(init_state, perturbed_state, 
                                                    self.standard_nca, steps, seed_loc=seed)
            mixture_recovery = self.measure_recovery(init_state, perturbed_state, 
                                                   self.mixture_nca, steps, seed_loc=seed)
            stochastic_recovery = self.measure_recovery(init_state, perturbed_state, 
                                                      self.stochastic_mixture_nca, steps, seed_loc=seed)
            
            standard_metrics.append(standard_recovery)
            mixture_metrics.append(mixture_recovery)
            stochastic_metrics.append(stochastic_recovery)
            
        return standard_metrics, mixture_metrics, stochastic_metrics
    
    def compute_robustness_metrics(self, init_state, perturbation_type, n_runs=5, steps=100, seed=None, **kwargs):
        """Compute robustness metrics for all three models"""
        standard_metrics, mixture_metrics, stochastic_metrics = self.compare_robustness(
            init_state, perturbation_type, n_runs, steps, seed, **kwargs
        )
        
        # Store trajectories for visualization
        perturbed_state = self.apply_perturbation(init_state, perturbation_type, seed=3, **kwargs)
        with torch.no_grad():
            standard_recovery = self.standard_nca(perturbed_state, steps, return_history=True, seed_loc=seed)
            mixture_recovery = self.mixture_nca(perturbed_state, steps, return_history=True, seed_loc=seed, sample_non_differentiable = True)
            stochastic_recovery = self.stochastic_mixture_nca(perturbed_state, steps, return_history=True, seed_loc=seed, sample_non_differentiable = True)
            
        return {
            'standard_metrics': standard_metrics,
            'mixture_metrics': mixture_metrics,
            'stochastic_metrics': stochastic_metrics,
            'init_state': init_state,
            'perturbed_state': perturbed_state,
            'standard_recovery': standard_recovery.cpu(),
            'mixture_recovery': mixture_recovery.cpu(),
            'stochastic_recovery': stochastic_recovery.cpu(),
            'perturbation_type': perturbation_type,
            'parameters': kwargs
        }

    def visualize_stored_results(self, results, plot_type='both', ax=None, figsize=(20, 14), replicate = 0):
        """Visualize results with clear separation between labels and images"""
        if plot_type in ['trajectories', 'both']:
            # Create figure with GridSpec
            fig = plt.figure(figsize=figsize)

            gs = plt.GridSpec(3, 5, width_ratios=[0.4, 1, 1, 1, 1], 
                             hspace=0.4, wspace=0.3,
                             left=0.05, right=0.95, top=0.9, bottom=0.1)
            
            # Define models in consistent order with clear labels
            model_data = [
                (results['standard_recovery'], 'Standard NCA', '#1f77b4'),
                (results['mixture_recovery'], 'Mixture NCA', '#d62728'),
                (results['stochastic_recovery'], 'Stochastic Mixture NCA', '#2ca02c')
            ]
            
            # Get stored trajectories
            perturbed_state = results['perturbed_state']
            
            # Add main title
            fig.suptitle(f'Recovery from {results["perturbation_type"]} perturbation', 
                        fontsize=14, fontweight='bold', y=0.95)
            
            # Column titles
            titles = ['Original', 'Perturbed', 'Mid Recovery', 'Final']
            for col, title in enumerate(titles):
                ax = fig.add_subplot(gs[0, col+1])
                ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
                ax.axis('off')
            
            # Plot for all three models
            for row, (recovery, label, color) in enumerate(model_data):
                # Create label column with minimal box
                label_ax = fig.add_subplot(gs[row, 0])
                label_ax.text(0.5, 0.5, label,
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize=12,
                             fontweight='bold',
                             transform=label_ax.transAxes,
                             bbox=dict(
                                 facecolor=color,
                                 alpha=0.2,
                                 edgecolor=color,
                                 pad=3,  # Minimal padding
                                 boxstyle='round,pad=0.2'  # Minimal round corners
                             ))
                label_ax.axis('off')
                
                # Create image plots
                images = [
                    self.to_rgb(results['init_state'][0].cpu()),  # Original
                    self.to_rgb(perturbed_state[0].cpu()),  # Perturbed
                    self.to_rgb(recovery[recovery.shape[0]//2, replicate]),  # Mid Recovery
                    self.to_rgb(recovery[-1, replicate])  # Final
                ]
                
                for col, img in enumerate(images):
                    ax = fig.add_subplot(gs[row, col+1])
                    ax.imshow(img.permute(1,2,0))
                    ax.axis('off')
                    ax.set_ylabel('')
            
            return fig, ax

        if plot_type in ['error', 'both']:
            fig = plt.figure(figsize=figsize)

            # Get recovery errors
            std_errors = np.array([m['recovery_error'] for m in results['standard_metrics']])
            mix_errors = np.array([m['recovery_error'] for m in results['mixture_metrics']])
            stoch_errors = np.array([m['recovery_error'] for m in results['stochastic_metrics']])
            
            # Calculate statistics (maintaining consistent order)
            model_errors = [
                (std_errors, 'Standard NCA', 'blue'),
                (mix_errors, 'Mixture NCA', 'red'),
                (stoch_errors, 'Stochastic Mixture NCA', 'green')
            ]

            if ax is None:
                ax = fig.gca()
            
            # Plot all three models in consistent order
            for errors, label, color in model_errors:
                mean = np.mean(errors, axis=0).squeeze()
                std = np.std(errors, axis=0).squeeze()
                steps = np.arange(len(mean))
                
                ax.plot(steps, mean, label=label, color=color, alpha=0.8)
                ax.fill_between(steps, mean-std, mean+std, color=color, alpha=0.2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Recovery Error')
            ax.set_title(f'Recovery from {results["perturbation_type"]} perturbation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig, ax

    def print_summary_statistics(self, results):
        """Print summary statistics for all three models"""
        print(f"\nSummary Statistics for {results['perturbation_type']} perturbation:")
        
        for model_name, metrics in [
            ('Standard NCA', results['standard_metrics']),
            ('Mixture NCA', results['mixture_metrics']),
            ('Stochastic Mixture NCA', results['stochastic_metrics'])
        ]:
            final_error = np.mean([m['final_error'] for m in metrics])
            recovery_time = np.mean([m['recovery_time'] for m in metrics])
            
            print(f"\n{model_name}:")
            print(f"Average final error: {final_error:.4f}")
            print(f"Average recovery time: {recovery_time:.1f} steps")
        
    def plot_recovery_comparison(self, init_state, perturbation_type, **kwargs):
        """Plot recovery error over time for all three models"""
        standard_metrics, mixture_metrics, stochastic_metrics = self.compare_robustness(
            init_state, perturbation_type, n_runs=5, **kwargs
        )
        
        # Plot average recovery error
        std_errors = np.mean([m['recovery_error'] for m in standard_metrics], axis=0)
        mix_errors = np.mean([m['recovery_error'] for m in mixture_metrics], axis=0)
        stoch_errors = np.mean([m['recovery_error'] for m in stochastic_metrics], axis=0)
        
        plt.figure(figsize=(10, 5))
        plt.plot(std_errors, label='Standard NCA', alpha=0.8)
        plt.plot(mix_errors, label='Mixture NCA', alpha=0.8)
        plt.plot(stoch_errors, label='Stochastic Mixture NCA', alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Recovery Error')
        plt.title(f'Recovery from {perturbation_type} perturbation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show() 

    def to_rgb(self, x):
        """Convert RGBA tensor to RGB"""
        rgb, a = x[..., :3,:,:], x[..., 3:4,:,:].clip(0,1)
        return (1.0-a+rgb).clip(0,1) 