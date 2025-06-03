import numpy as np
import torch
from mix_NCA.TissueModel import ComplexCellType, TissueModel
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
import multiprocessing as mp
from functools import partial



class ABCModel:
    def __init__(self, grid_size=50, metric_type='cell_distribution'):
        """
        Initialize ABM model with chosen metric
        Args:
            grid_size: size of simulation grid
            metric_type: one of ['cell_distribution', 'neighborhood', 'covariance']
        """
        self.grid_size = grid_size
        self.metric_type = metric_type
        
        # Parameters to learn with priors
        self.params = {
            'stem_division_rate': 1,
            'intermediate_division_rate': 1,
            'intermediate2_division_rate': 1,
            'death_rates': np.ones(5) * 0.1,
            'survival_rates': np.ones(5) * 1,
            'differentiation_probs': np.ones((5, 5)) * 1,
            'interaction_matrix': np.ones((5, 5)) * 1
        }
    
    def simulate(self, steps):
        """Run agent-based simulation with current parameters"""
        model = TissueModel(
            grid_size=30,
            initial_stem_cells=5,
            stem_division_rate=self.params['stem_division_rate'],
            intermediate_division_rate=self.params['intermediate_division_rate'],
            intermediate2_division_rate=self.params['intermediate2_division_rate']
        )
        
        model.death_rates = {ct: rate for ct, rate in zip(list(ComplexCellType)[1:], self.params['death_rates'])}
        model.survival_rates = {ct: rate for ct, rate in zip(list(ComplexCellType)[1:], self.params['survival_rates'])}
        model.base_differentiation = self.params['differentiation_probs']
        model.interaction_matrix = self.params['interaction_matrix']
        
        history, _ = model.simulate(steps)
        return history
    
    def compute_cell_distribution(self, grid):
        """Compute overall cell type distribution"""
        return np.bincount(grid.flatten(), minlength=6) / grid.size
    
    def compute_neighborhood_distribution(self, grid):
        """
        Compute cell type distribution in 3x3 neighborhoods, including empty cells
        Returns distribution of shape (6,) including empty type
        """
        neighborhood_dist = np.zeros(6)  # Include empty type
        count = 0
        
        # Create padded grid to handle boundaries
        padded = np.pad(grid, pad_width=1, mode='constant', constant_values=ComplexCellType.EMPTY.value)
        
        # For each cell in the grid
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                # Extract 3x3 neighborhood
                neighborhood = padded[i-1:i+2, j-1:j+2].flatten()
                # Count all cell types including empty
                dist = np.bincount(neighborhood, minlength=6)
                neighborhood_dist += dist
                count += 1
        
        return neighborhood_dist / count if count > 0 else neighborhood_dist
    
    
    def compute_type_correlations(self, grid):
        """
        Compute correlation matrix between cell type distributions
        Returns correlation matrix of shape (6, 6) including empty type
        """
        # Create binary masks for each cell type
        type_masks = np.array([(grid == i).astype(float) for i in range(6)])  # shape: (6, H, W)
        
        # Flatten masks for correlation computation
        flat_masks = type_masks.reshape(6, -1)  # shape: (6, H*W)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(flat_masks)  # shape: (6, 6)
        
        # Replace NaN values that occur when a cell type is not present
        corr_matrix = np.nan_to_num(corr_matrix, 0.0)
        
        return corr_matrix
    
    def compute_distance(self, sim_data, observed_data):
        """
        Compute distance based on chosen metric
        Args:
            sim_data: simulated data array [timesteps, grid_size, grid_size]
            observed_data: observed data array [batch, timesteps, grid_size, grid_size]
        """
        total_dist = 0
        
        for batch_idx in range(observed_data.shape[0]):
            batch_dist = 0
            
            if self.metric_type == 'cell_distribution':
                for t in range(len(sim_data)):
                    sim_dist = self.compute_cell_distribution(sim_data[t])
                    obs_dist = self.compute_cell_distribution(observed_data[batch_idx, t])
                    batch_dist += wasserstein_distance(sim_dist, obs_dist)
                    
            elif self.metric_type == 'neighborhood':
                for t in range(len(sim_data)):
                    sim_dist = self.compute_neighborhood_distribution(sim_data[t])
                    obs_dist = self.compute_neighborhood_distribution(observed_data[batch_idx, t])
                    batch_dist += wasserstein_distance(sim_dist, obs_dist)
                    
            else:  # type_correlations
                for t in range(len(sim_data)):
                    sim_corr = self.compute_type_correlations(sim_data[t])
                    obs_corr = self.compute_type_correlations(observed_data[batch_idx, t])
                    # Normalize by maximum possible Frobenius norm
                    batch_dist += np.linalg.norm(sim_corr - obs_corr) / np.sqrt(2)
            
            total_dist += batch_dist / len(sim_data)
            
        return total_dist / observed_data.shape[0]


    def plot_statistics(self, stats, figsize=(10, 4)):
        """
        Plot tumor size and clustering coefficient over time
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Tumor size
        ax1.plot(stats['tumor_size'])
        ax1.set_title('Tumor Size')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Cells')
        
        # Clustering coefficient
        ax2.plot(stats['clustering_coefficient'])
        ax2.set_title('Clustering Coefficient')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coefficient')
        
        plt.tight_layout()
        return fig

    def _simulate_particle(self, params, steps=35):
        """Helper function for parallel simulation"""
        model = TissueModel(
            grid_size=30,
            initial_stem_cells=5,
            stem_division_rate=params['stem_division_rate'],
            intermediate_division_rate=params['intermediate_division_rate'],
            intermediate2_division_rate=params['intermediate2_division_rate']
        )
        model.death_rates = {ct: rate for ct, rate in zip(list(ComplexCellType)[1:], params['death_rates'])}
        model.base_differentiation = params['differentiation_probs']
        model.interaction_matrix = params['interaction_matrix']
        model.survival_rates = {ct: rate for ct, rate in zip(list(ComplexCellType)[1:], params['survival_rates'])}
        history, _ = model.simulate(steps)
        return history

    def _process_particle(self, particle_idx, observed_data, epsilon):
        """Helper function for parallel processing"""
        # Sample parameters from priors
        proposed_params = {
            'stem_division_rate': np.random.gamma(1, .1),
            'intermediate_division_rate': np.random.gamma(1, .1),
            'intermediate2_division_rate': np.random.gamma(1, .1),
            'death_rates': np.random.gamma(1, 0.01, size=5),
            'survival_rates': np.random.gamma(1, .1, size=5),
            'differentiation_probs': np.random.gamma(1, .1, size=(5, 5)),
            'interaction_matrix': np.random.normal(0, 1, size=(5, 5))
        }
        
        # Create temporary model with proposed parameters
        temp_model = ABCModel(grid_size=self.grid_size, metric_type=self.metric_type)
        temp_model.params = proposed_params
        
        # Simulate with these parameters
        sim_data = temp_model.simulate(steps=35)
        
        # Compute distance
        dist = temp_model.compute_distance(sim_data, observed_data)
        
        return (proposed_params, dist) if dist < epsilon else None

    def fit(self, observed_data, n_particles=1000, epsilon=0.1, n_processes=None):
        """
        Parallel fit using ABC
        Args:
            observed_data: observed data array [batch, timesteps, grid_size, grid_size]
            n_particles: number of particles to simulate
            epsilon: acceptance threshold
            n_processes: number of processes for parallel computation
        """
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        print(f"Using {n_processes} processes")
        
        # Create partial function with fixed arguments
        process_func = partial(self._process_particle, 
                             observed_data=observed_data,
                             epsilon=epsilon)
        
        # Run parallel simulations
        with mp.Pool(processes=n_processes) as pool:
            results = list(filter(None, pool.map(process_func, range(n_particles))))
        
        # Process results
        if not results:
            print("No particles accepted. Try increasing epsilon.")
            return
        
        accepted_params, accepted_distances = zip(*results)
        
        # Update parameters using weighted mean
        weights = 1 / (np.array(accepted_distances) + 1e-10)
        weights /= weights.sum()
        
        for key in self.params:
            self.params[key] = np.average(
                [p[key] for p in accepted_params],
                weights=weights,
                axis=0
            )
        
        print(f"Accepted {len(results)} particles")
        print(f"Mean distance: {np.mean(accepted_distances):.4f}")

    def plot_simulation(self, sim_data, n_timesteps=10, figsize=(20, 4), plot_every=None):
        """
        Plot the tissue simulation evolution over time
        Args:
            sim_data: numpy array of shape [timesteps, grid_size, grid_size]
            n_timesteps: number of timesteps to show (evenly spaced)
            figsize: figure size
            plot_every: if set, plot every N steps instead of evenly spaced steps
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Get colors directly from ComplexCellType
        colors = [cell_type.get_color() for cell_type in ComplexCellType]
        cmap = ListedColormap(colors)
        
        # Calculate timestep indices to show
        total_steps = len(sim_data)
        if plot_every is not None:
            # Show every N steps
            step_indices = np.arange(0, total_steps, plot_every)
            n_timesteps = len(step_indices)
        else:
            # Show evenly spaced steps
            step_indices = np.linspace(0, total_steps-1, n_timesteps, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(1, n_timesteps, figsize=figsize)
        
        # Plot each timestep
        for i, step in enumerate(step_indices):
            axes[i].imshow(sim_data[step], cmap=cmap, vmin=0, vmax=5)
            axes[i].axis('off')
            axes[i].set_title(f't={step}')
        
        # Add colorbar to show cell types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ct.get_color(), label=ct.name.replace('_', ' ').title())
            for ct in list(ComplexCellType)[1:]  # Skip EMPTY
        ]
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
        
        plt.tight_layout()
        return fig

    