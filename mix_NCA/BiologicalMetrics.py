import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd
from typing import List, Tuple
from mix_NCA.TissueModel import ComplexCellType
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from mix_NCA.utils_simulations import grid_to_channels_batch
from scipy.stats import wasserstein_distance

class BiologicalMetrics:
    def __init__(self, true_dataset: torch.Tensor, generated_dataset: torch.Tensor, 
                 cell_types: List[int], device="cuda"):
        """
        Args:
            true_dataset: Original dataset [N_samples, channels, H, W]
            generated_dataset: Generated dataset [N_samples, channels, H, W]
            cell_types: List of possible cell types
            device: Computing device
        """
        self.true_dataset = true_dataset
        self.generated_dataset = generated_dataset
        self.cell_types = cell_types
        self.device = device
        #print(f"Dataset shapes - True: {true_dataset.shape}, Generated: {generated_dataset.shape}")

    def categorical_mmd(self) -> float:
        """
        Maximum Mean Discrepancy with a specialized categorical kernel for integer-encoded cell types
        
        Returns:
            float: MMD value between true and generated distributions
        """
        def categorical_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Compute categorical kernel between two cell type distributions
            
            Args:
                x: First distribution [batch_size, height, width] with integer cell types
                y: Second distribution [batch_size, height, width] with integer cell types
                
            Returns:
                Kernel value measuring similarity between distributions
            """
            # Reshape tensors to 2D: [batch_size, height*width]
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)
            
            # Compute pairwise equality matrix
            # For each pair of samples, count matching cell types
            x_flat = x_flat.unsqueeze(1)  # [batch_size, 1, height*width]
            y_flat = y_flat.unsqueeze(0)  # [1, batch_size, height*width]
            
            # Delta kernel: K(x,y) = 1 if cell types match, 0 otherwise
            matches = (x_flat == y_flat).float()  # [batch_size_x, batch_size_y, height*width]
            
            # Average over spatial dimensions
            return matches.mean(dim=-1)  # [batch_size_x, batch_size_y]
        
        # Compute kernel matrices
        xx = categorical_kernel(self.true_dataset, self.true_dataset)
        yy = categorical_kernel(self.generated_dataset, self.generated_dataset)
        xy = categorical_kernel(self.true_dataset, self.generated_dataset)
        
        # Compute MMD
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        
        return mmd.item()

    def distribution_metrics(self) -> dict:
        """
        Calculate multiple distribution similarity metrics
        """
        # Get cell type distributions
        true_counts = torch.zeros(len(self.cell_types), device=self.device)
        gen_counts = torch.zeros(len(self.cell_types), device=self.device)
        
        for ct in self.cell_types:
            if ct == ComplexCellType.EMPTY:
                continue
            true_mask = torch.tensor(self.true_dataset == ct.value, device=self.device)
            gen_mask = torch.tensor(self.generated_dataset == ct.value, device=self.device)
            
            true_count = torch.sum(true_mask.float())
            gen_count = torch.sum(gen_mask.float())
            
            true_counts[ct.value] = true_count
            gen_counts[ct.value] = gen_count
        
        # Normalize to get probability distributions
        true_dist = true_counts / (true_counts.sum() + 1e-8)
        gen_dist = gen_counts / (gen_counts.sum() + 1e-8)
        
        # KL divergence
        kl_div = torch.sum(true_dist * torch.log((true_dist + 1e-8)/(gen_dist + 1e-8)))
        
        # Chi-square distance
        chi_square = torch.sum((true_dist - gen_dist)**2 / (true_dist + gen_dist + 1e-8))
        
        # Categorical MMD
        cat_mmd = self.categorical_mmd()
        
        #print(f"\nDistribution metrics:")
        #print(f"True distribution: {true_dist}")
        #print(f"Generated distribution: {gen_dist}")
        #print(f"KL divergence: {kl_div:.4f}")
        #print(f"Chi-square distance: {chi_square:.4f}")
        #print(f"Categorical MMD: {cat_mmd:.4f}")
        
        return {
            'kl_divergence': kl_div.item(),
            'chi_square': chi_square.item(),
            'categorical_mmd': cat_mmd
        }

    def tumor_size_distribution(self) -> float:
        """
        Compare tumor size distributions using Wasserstein distance
        """
        def get_tumor_sizes(data):
            sizes = []
            for i in range(data.shape[0]):
                tumor_mask = torch.tensor(data[i] > 0, device=self.device)
                size = tumor_mask.float().sum().item()
                sizes.append(size)
            return torch.tensor(sizes, device=self.device)

        true_sizes = get_tumor_sizes(self.true_dataset)
        gen_sizes = get_tumor_sizes(self.generated_dataset)

        # Sort sizes for Wasserstein distance calculation
        true_sorted = torch.sort(true_sizes)[0]
        gen_sorted = torch.sort(gen_sizes)[0]
        
        # Calculate 1D Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = torch.mean(torch.abs(true_sorted - gen_sorted))
        
        # Normalize by mean tumor size
        mean_size = torch.mean(true_sizes)
        normalized_dist = wasserstein_dist / (mean_size + 1e-8)

        #print(f"\nTumor size distributions:")
        #print(f"True - Mean: {true_sizes.mean():.2f}, Std: {true_sizes.std():.2f}")
        #print(f"Generated - Mean: {gen_sizes.mean():.2f}, Std: {gen_sizes.std():.2f}")
        #print(f"Wasserstein distance: {normalized_dist:.4f}")
        
        return normalized_dist.item()

    def spatial_correlation(self) -> dict:
        """
        Compare spatial patterns using border size and spatial variance distributions
        """
        def compute_spatial_metrics(data):
            """Compute border size and spatial variance for each sample in dataset"""
            border_sizes = []
            spatial_variances = []
            
            # Create convolution kernel for edge detection
            kernel = torch.tensor([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], device=self.device).float() / 8
            kernel = kernel.view(1, 1, 3, 3)
            
            for i in range(len(data)):  # For each sample
                # Convert to binary mask (any non-empty cell)
                mask = (data[i][1:].sum(dim=0) > 0).float()  # Now [H, W]
                
                # Add batch and channel dimensions for convolution
                mask = mask.unsqueeze(0).unsqueeze(0)  # Now [1, 1, H, W]
                
                # Compute border size using convolution with padding
                edges = torch.abs(F.conv2d(mask, kernel, padding=1)) > 0.25  # Add padding=1
                border_sizes.append(edges.sum().item())
                
                # Compute spatial variance
                non_empty_positions = torch.nonzero(mask.squeeze())
                if len(non_empty_positions) > 0:
                    mean_pos = non_empty_positions.float().mean(dim=0)
                    variance = ((non_empty_positions.float() - mean_pos)**2).sum(dim=1).mean()
                    spatial_variances.append(variance.item())
                else:
                    spatial_variances.append(0.0)
            
            return {
                'border_sizes': border_sizes,
                'spatial_variances': spatial_variances
            }
        
        # Compute metrics for both datasets
        true_metrics = compute_spatial_metrics(self.true_dataset)
        gen_metrics = compute_spatial_metrics(self.generated_dataset)
        
        # Compute Wasserstein distances
        border_dist = wasserstein_distance(
            np.array(true_metrics['border_sizes']), 
            np.array(gen_metrics['border_sizes'])
        )
        
        spatial_var_dist = wasserstein_distance(
            np.array(true_metrics['spatial_variances']), 
            np.array(gen_metrics['spatial_variances'])
        )
        
        # Normalize by the mean of true distributions to get relative differences
        border_dist_norm = border_dist / (np.mean(true_metrics['border_sizes']) + 1e-8)
        spatial_var_dist_norm = spatial_var_dist / (np.mean(true_metrics['spatial_variances']) + 1e-8)
        
        return {
            'border_size_diff': border_dist_norm,
            'spatial_variance_diff': spatial_var_dist_norm
        }

def compare_generated_distributions(histories, standard_nca, mixture_nca, stochastic_nca, nca_with_noise,
                                 n_steps=35, n_evaluations=10, device="cuda", deterministic_rule_choice = False):
    """
    Compare generated distributions against the true dataset with multiple evaluations
    
    Args:
        histories: List of true state histories
        standard_nca: Standard NCA model
        mixture_nca: Mixture NCA model
        stochastic_nca: Stochastic Mixture NCA model
        nca_with_noise: NCA model with normal distribution output
        n_steps: Number of steps for generation
        n_evaluations: Number of times to evaluate stochastic models
        device: Computing device
    """
    # Collect all true states
    initial_states = []
    for hist in histories:
        grid_state = hist[0]
        encoded_state = grid_to_channels_batch(grid_state, len(ComplexCellType), device)
        initial_states.append(encoded_state)
    
    # Stack all true states
    true_dataset = torch.cat([torch.tensor(ts[-1]).to(device).unsqueeze(0) for ts in histories], dim=0)
    
    metrics = {
        'Model Type': [],
        'KL Divergence': [],
        'KL Divergence SD': [],
        'Chi-Square': [],
        'Chi-Square SD': [],
        'Categorical MMD': [],
        'Categorical MMD SD': [],
        'Tumor Size Diff': [],
        'Tumor Size Diff SD': [],
        'Border Size Diff': [],
        'Border Size Diff SD': [],
        'Spatial Variance Diff': [],
        'Spatial Variance Diff SD': []
    }
    
    # Generate from standard NCA (deterministic, only once)
    with torch.no_grad():
        standard_samples = []
        for true_state in initial_states:
            sample = standard_nca(true_state, n_steps, return_history=True)[-1]
            standard_samples.append(sample.argmax(dim=1))
        standard_gen = torch.stack(standard_samples).squeeze(1)
    
    # Create mapping for metric names
    metric_mapping = {
        'kl_divergence': 'KL Divergence',
        'chi_square': 'Chi-Square',
        'categorical_mmd': 'Categorical MMD',
        'tumor_size': 'Tumor Size Diff',
        'border_size': 'Border Size Diff',
        'spatial_variance': 'Spatial Variance Diff'
    }
    
    # Compare each model's generated distribution
    for name, model in [
        ('Standard NCA', standard_nca),
        ('Mixture NCA', mixture_nca),
        ('Stochastic Mixture NCA', stochastic_nca),
        ('NCA with Noise', nca_with_noise)
    ]:
        # For stochastic models and NCA with noise, evaluate multiple times
        if name != 'Standard NCA':
            all_metrics = {
                'kl_divergence': [],
                'chi_square': [],
                'categorical_mmd': [],
                'tumor_size': [],
                'border_size': [],
                'spatial_variance': []
            }
            
            for eval_idx in range(n_evaluations):
                # set the seed for reproducibility
                torch.manual_seed(eval_idx)
                with torch.no_grad():
                    samples = []
                    for true_state in initial_states:
                        if name == 'NCA with Noise':
                            if deterministic_rule_choice:
                                current_state = true_state
                                for _ in range(n_steps):
                                    class_assignment = torch.argmax(current_state, dim=1)
                                    class_assignment = torch.nn.functional.one_hot(class_assignment, num_classes=6)
                                    class_assignment = class_assignment.transpose(1,3).transpose(2, 3)
                                    current_state = model(current_state, 1, return_history=False, class_assignment = class_assignment)
                                sample = current_state
                            else:

                                sample = model(true_state, n_steps, return_history=True)[-1]
                            sample = sample.argmax(dim=1)
                        else: 
                            sample = model(true_state, n_steps, return_history=True, sample_non_differentiable = True)[-1]
                            sample = sample.argmax(dim=1)
                        samples.append(sample)
                    generated = torch.stack(samples).squeeze(1)
                
                bio_metrics = BiologicalMetrics(true_dataset, generated, list(ComplexCellType), device)
                
                dist_metrics = bio_metrics.distribution_metrics()
                all_metrics['kl_divergence'].append(dist_metrics['kl_divergence'])
                all_metrics['chi_square'].append(dist_metrics['chi_square'])
                all_metrics['categorical_mmd'].append(dist_metrics['categorical_mmd'])
                all_metrics['tumor_size'].append(bio_metrics.tumor_size_distribution())
                
                spatial_metrics = bio_metrics.spatial_correlation()
                all_metrics['border_size'].append(spatial_metrics['border_size_diff'])
                all_metrics['spatial_variance'].append(spatial_metrics['spatial_variance_diff'])
            
            # Calculate means and standard deviations (not standard errors)
            metrics['Model Type'].append(name)
            for metric_name, values in all_metrics.items():
                values = np.array(values)
                mean = np.mean(values)
                sd = np.std(values)
                print(f"{metric_name}: {mean} ± {sd}")
                
                column_name = metric_mapping[metric_name]
                metrics[column_name].append(f"{mean:.3f}")
                metrics[f"{column_name} SD"].append(f"±{sd:.3f}")
        
        else:
            # For standard NCA, just evaluate once
            bio_metrics = BiologicalMetrics(true_dataset, standard_gen, list(ComplexCellType), device)
            
            dist_metrics = bio_metrics.distribution_metrics()
            spatial_metrics = bio_metrics.spatial_correlation()
            
            metrics['Model Type'].append(name)
            metrics['KL Divergence'].append(f"{dist_metrics['kl_divergence']:.3f}")
            metrics['KL Divergence SD'].append("±0.000")
            metrics['Chi-Square'].append(f"{dist_metrics['chi_square']:.3f}")
            metrics['Chi-Square SD'].append("±0.000")
            metrics['Categorical MMD'].append(f"{dist_metrics['categorical_mmd']:.3f}")
            metrics['Categorical MMD SD'].append("±0.000")
            metrics['Tumor Size Diff'].append(f"{bio_metrics.tumor_size_distribution():.3f}")
            metrics['Tumor Size Diff SD'].append("±0.000")
            metrics['Border Size Diff'].append(f"{spatial_metrics['border_size_diff']:.3f}")
            metrics['Border Size Diff SD'].append("±0.000")
            metrics['Spatial Variance Diff'].append(f"{spatial_metrics['spatial_variance_diff']:.3f}")
            metrics['Spatial Variance Diff SD'].append("±0.000")
    
    # Create DataFrame with metrics and standard deviations
    df = pd.DataFrame(metrics)
    
    return df

column_renames_spatial = {
    'Model Type': 'Model',
    'Tumor Size Diff': 'Size',
    'Border Size Diff': 'Border W-dist',
    'Spatial Variance Diff': 'Spatial W-dist'
}

def compare_abm_distributions(histories, abm_models, n_steps=35, n_evaluations=10, device="cuda"):
    """
    Compare generated distributions against the true dataset with multiple evaluations
    
    Args:
        histories: List of true state histories
        abm_models: List of trained ABM models
        n_steps: Number of steps for generation
        n_evaluations: Number of times to evaluate models
        device: Computing device
    """
    # Stack all true states
    true_dataset = torch.cat([torch.tensor(ts[-1]).to(device).unsqueeze(0) for ts in histories], dim=0)
    print(f"True dataset shape after stacking: {true_dataset.shape}")
    
    metrics = {
        'Model Type': [],
        'KL Divergence': [],
        'KL Divergence SD': [],
        'Chi-Square': [],
        'Chi-Square SD': [],
        'Categorical MMD': [],
        'Categorical MMD SD': [],
        'Tumor Size Diff': [],
        'Tumor Size Diff SD': [],
        'Border Size Diff': [],
        'Border Size Diff SD': [],
        'Spatial Variance Diff': [],
        'Spatial Variance Diff SD': []
    }
    
    # Create mapping for metric names
    metric_mapping = {
        'kl_divergence': 'KL Divergence',
        'chi_square': 'Chi-Square',
        'categorical_mmd': 'Categorical MMD',
        'tumor_size': 'Tumor Size Diff',
        'border_size': 'Border Size Diff',
        'spatial_variance': 'Spatial Variance Diff'
    }
    
    # Compare each ABM model's generated distribution
    for i, model in enumerate(abm_models):
        model_name = f"ABM Model {i+1}"
        print(f"\nProcessing {model_name}")
        
        all_metrics = {
            'kl_divergence': [],
            'chi_square': [],
            'categorical_mmd': [],
            'tumor_size': [],
            'border_size': [],
            'spatial_variance': []
        }
        
        for eval_idx in range(n_evaluations):
            # Set random seed for reproducibility
            np.random.seed(eval_idx)
            
            # Generate samples using ABM model
            samples = []
            for _ in range(len(histories)):  # Generate same number of samples as histories
                history = model.simulate(n_steps)
                samples.append(history[-1])  # Take final state
            generated = torch.tensor(np.stack(samples)).to(device)
            
            # Compute metrics
            bio_metrics = BiologicalMetrics(true_dataset, generated, list(ComplexCellType), device)
            
            dist_metrics = bio_metrics.distribution_metrics()
            all_metrics['kl_divergence'].append(dist_metrics['kl_divergence'])
            all_metrics['chi_square'].append(dist_metrics['chi_square'])
            all_metrics['categorical_mmd'].append(dist_metrics['categorical_mmd'])
            all_metrics['tumor_size'].append(bio_metrics.tumor_size_distribution())
            
            spatial_metrics = bio_metrics.spatial_correlation()
            all_metrics['border_size'].append(spatial_metrics['border_size_diff'])
            all_metrics['spatial_variance'].append(spatial_metrics['spatial_variance_diff'])
        
        # Calculate means and standard deviations
        metrics['Model Type'].append(model_name)
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            sd = np.std(values)
            
            column_name = metric_mapping[metric_name]
            metrics[column_name].append(f"{mean:.3f}")
            metrics[f"{column_name} SD"].append(f"±{sd:.3f}")
    
    # Create DataFrame with metrics and standard deviations
    df = pd.DataFrame(metrics)
    
    # Print formatted table
    print("\nResults with Standard Deviations:")
    print(df.to_markdown(index=False))
    
    return df



