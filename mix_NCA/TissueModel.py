import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # Add this import


class TumorImmuneCellType(Enum):
    EMPTY = 0
    SENSITIVE_TUMOR = 1
    RESISTANT_TUMOR = 2
    IMMUNE = 3
    
    def get_color(self):
        colors = {
            TumorImmuneCellType.EMPTY: 'white',
            TumorImmuneCellType.SENSITIVE_TUMOR: 'red',
            TumorImmuneCellType.RESISTANT_TUMOR: 'darkred',
            TumorImmuneCellType.IMMUNE: 'blue'
        }
        return colors[self]


class ComplexCellType(Enum):
    EMPTY = 0
    STEM = 1
    INTERMEDIATE_1 = 2
    INTERMEDIATE_2 = 3
    DIFFERENTIATED_1 = 4
    DIFFERENTIATED_2 = 5
    
    def get_color(self):
        colors = {
            ComplexCellType.EMPTY: 'white',
            ComplexCellType.STEM: '#99CCFF',          # Light but visible blue
            ComplexCellType.INTERMEDIATE_1: '#6699FF', # Medium-light blue
            ComplexCellType.INTERMEDIATE_2: '#3366CC', # Medium blue
            ComplexCellType.DIFFERENTIATED_1: '#000066', # Darkest blue
            ComplexCellType.DIFFERENTIATED_2: '#8B0000' # Dark red
        }
        return colors[self]


class TissueModel:
    def __init__(self, 
                 grid_size=50, 
                 initial_stem_cells=5,
                 stem_division_rate=0.5,
                 intermediate_division_rate=0.5,
                 intermediate2_division_rate=0.5):
        
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Base rates
        self.death_rates = {
            ComplexCellType.STEM: 0.01,
            ComplexCellType.INTERMEDIATE_1: 0.01,
            ComplexCellType.INTERMEDIATE_2: 0.01,
            ComplexCellType.DIFFERENTIATED_1: 0.05,
            ComplexCellType.DIFFERENTIATED_2: 0.05
        }
        
        self.division_rates = {
            ComplexCellType.STEM: stem_division_rate,
            ComplexCellType.INTERMEDIATE_1: intermediate_division_rate,
            ComplexCellType.INTERMEDIATE_2: intermediate2_division_rate,
            ComplexCellType.DIFFERENTIATED_1: 0.0,
            ComplexCellType.DIFFERENTIATED_2: 0.0
        }
        
        # Base differentiation probabilities (from type i to type j)
        self.base_differentiation = np.array([
            #STEM  INT1  INT2  DIFF  DIFF2
            [0.0,  0.4,  0.0,  0.0,  0.0],    # STEM
            [0.05, 0.0,  0.4,  0.0,  0.0],    # INTER1
            [0.0,  0.0,  0.0,  0.3,  0.3],    # INTER2 (equal base prob for DIFF/DIFF2)
            [0.0,  0.0,  0.0,  0.0,  0.0],    # DIFF
            [0.0,  0.0,  0.0,  0.0,  0.0]     # DIFF2
        ])
        
        # How each cell type affects differentiation of others
        self.interaction_matrix = np.array([
            #STEM  INT1  INT2  DIFF  DIFF2
            [0.2,  0.1,  -0.1, -0.2, -0.2],   # Effect on STEM
            [0.1,  0.2,  0.1,  0.0,  0.0],    # Effect on INTER1
            [-0.1, 0.1,  0.2,  0.1,  0.1],    # Effect on INTER2
            [-0.2, 0.0,  0.1,  0.2,  0.0],    # Effect on DIFF
            [-0.2, 0.0,  0.1,  0.3,  0.2]     # Effect on DIFF2 (enhanced by DIFF)
        ])
        
        self.tissue_params = {
            'overcrowding_threshold': 10,
            'isolation_threshold': 10
        }

        self.initial_stem_cells = initial_stem_cells
        
        # Initialize tissue
        self._initialize_tissue()
        
        # Track tissue statistics
        self.statistics = {
            'cell_counts': [],
            'spatial_organization': [],
            'cell_death_events': []
        }

        # Add survival rates dictionary
        self.survival_rates = {
            ComplexCellType.STEM: 0.7,           # High survival rate for stem cells
            ComplexCellType.INTERMEDIATE_1: 0.6,  # Moderate-high survival for early progenitors
            ComplexCellType.INTERMEDIATE_2: 0.5,  # Moderate survival for late progenitors
            ComplexCellType.DIFFERENTIATED_1: 0.4,  # Lower survival for DIFFERENTIATED_1 cells
            ComplexCellType.DIFFERENTIATED_2: 0.4 # Lower survival for specialized cells
        }

    
    def _initialize_tissue(self):
        """Initialize tissue with stem cells in a biologically plausible pattern"""
        # Create initial stem cell cluster near center
        center = self.grid_size // 2
        radius = int(np.sqrt(self.initial_stem_cells))
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:  # Circular pattern
                    x, y = center + i, center + j
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        if np.random.random() < 0.8:  # Add some randomness
                            self.grid[x, y] = ComplexCellType.STEM.value
    
    
    def calculate_cell_fate(self, cell_type):
        """
        First decide if cell survives, dies, or divides.
        Returns:
        - 'death': cell dies
        - 'division': cell divides
        - 'survive': cell stays alive without dividing
        """
        # Get base rates
        death_rate = self.death_rates[cell_type]
        division_rate = self.division_rates[cell_type]
        survival_rate = self.survival_rates[cell_type]
        
        # Normalize rates to probabilities
        total_rate = death_rate + division_rate + survival_rate
        
        if total_rate > 0:
            death_prob = death_rate / total_rate
            division_prob = division_rate / total_rate
            # survival_prob = survival_rate / total_rate (not needed explicitly)
            
            r = np.random.random()
            if r < death_prob:
                return 'death'
            elif r < (death_prob + division_prob):
                return 'division'
            else:
                return 'survive'
        
        return 'survive'  # Default to survival if no rates are specified

    def calculate_differentiation(self, cell_type, neighbors):
        """
        Calculate differentiation fate based on current type and neighbors.
        Only called after a division event for the new cell.
        """
        # Get differentiation rates from base matrix
        differentiation_rates = self.base_differentiation[cell_type.value - 1].copy()
        
        # Apply neighbor effects to differentiation rates
        if len(neighbors) > 0:
            neighbor_types = [self.grid[ni, nj] for ni, nj in neighbors]
            for n_type in neighbor_types:
                if n_type != ComplexCellType.EMPTY.value:
                    differentiation_rates += self.interaction_matrix[n_type - 1]
        
        # Ensure no negative rates
        differentiation_rates = np.maximum(differentiation_rates, 0)
        
        # Calculate differentiation probability
        total_diff_rate = differentiation_rates.sum()
        
        if total_diff_rate > 0:
            # Normalize differentiation rates
            differentiation_probs = differentiation_rates / total_diff_rate
            
            # Choose new cell type
            new_type = np.random.choice(len(differentiation_probs), p=differentiation_probs) + 1
            return new_type
        
        # If no differentiation occurs, keep same type
        return cell_type.value

    def get_neighbors(self, i, j):
        """Get all neighboring coordinates within grid bounds"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.grid_size and 
                    0 <= nj < self.grid_size):
                    neighbors.append((ni, nj))
        return neighbors
    
    def step(self):
        """Execute one step of the simulation"""
        new_grid = self.grid.copy()
        cell_deaths = 0
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                current_cell = self.grid[x, y]
                if current_cell != ComplexCellType.EMPTY.value:
                    # Get current cell type as enum
                    current_type = ComplexCellType(current_cell)
                    
                    # First determine if cell dies or divides
                    fate = self.calculate_cell_fate(current_type)
                    
                    if fate == 'death':
                        new_grid[x, y] = ComplexCellType.EMPTY.value
                        cell_deaths += 1
                    
                    elif fate == 'division':
                        # Get empty neighbors
                        empty_neighbors = self.get_empty_neighbors(x, y)
                        if empty_neighbors:
                            # Choose random empty neighbor for new cell
                            ni, nj = empty_neighbors[np.random.randint(len(empty_neighbors))]
                            
                            # Get neighbors for the new cell position
                            new_cell_neighbors = self.get_neighbors(ni, nj)
                            
                            # Calculate differentiation fate for the new cell
                            new_cell_type = self.calculate_differentiation(current_type, new_cell_neighbors)
                            
                            # Place the new cell with its determined type
                            new_grid[ni, nj] = new_cell_type
    
        self.grid = new_grid
        self._update_statistics(cell_deaths)
    
    def _update_statistics(self, cell_deaths):
        """Update tissue statistics"""
        cell_counts = {
            cell_type: np.sum(self.grid == cell_type.value)
            for cell_type in ComplexCellType
        }
        self.statistics['cell_counts'].append(cell_counts)
        self.statistics['cell_death_events'].append(cell_deaths)
    
    def plot_tissue(self, show_statistics=False, figsize=(15, 6)):
        """Visualize tissue state and optionally show statistics
        
        Args:
            show_statistics (bool): Whether to show statistics
            figsize (tuple): Figure size in inches (width, height)
        """
        if show_statistics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Use colors from ComplexCellType.get_color()
            colors = [cell_type.get_color() for cell_type in ComplexCellType]
            cmap = plt.cm.colors.ListedColormap(colors)
            ax1.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(ComplexCellType)-1)
            ax1.set_title('Tissue Organization')
            
            # Create custom legend for tissue plot
            legend_elements = [
                Patch(facecolor=cell_type.get_color(), label=f"{cell_type.name} ({cell_type.value})")
                for cell_type in ComplexCellType
            ]
            ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot statistics
            counts = self.statistics['cell_counts'][-1]
            cell_types = [ct.name for ct in ComplexCellType]
            values = [counts[ct] for ct in ComplexCellType]
            ax2.bar(cell_types, values, color=[ct.get_color() for ct in ComplexCellType])
            ax2.set_title('Cell Type Distribution')
            ax2.tick_params(axis='x', rotation=45)
            
        else:
            plt.figure(figsize=(8, 8))
            colors = [cell_type.get_color() for cell_type in ComplexCellType]
            cmap = plt.cm.colors.ListedColormap(colors)
            plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(ComplexCellType)-1)
            plt.title('Tissue Organization')
            
            # Create custom legend
            legend_elements = [
                Patch(facecolor=cell_type.get_color(), label=f"{cell_type.name} ({cell_type.value})")
                for cell_type in ComplexCellType
            ]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def get_empty_neighbors(self, i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.grid_size and 
                    0 <= nj < self.grid_size and 
                    self.grid[ni, nj] == ComplexCellType.EMPTY.value):
                    neighbors.append((ni, nj))
        return neighbors

    def simulate(self, steps):
        """Run simulation for given number of steps
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            history: List of grid states at each time step
            statistics: Dictionary containing evolution of:
                - cell_counts
                - cell_death_events
        """
        history = [self.grid.copy()]
        statistics = {
            'cell_counts': [self.statistics['cell_counts'][-1]] if self.statistics['cell_counts'] else [],
            'cell_death_events': []
        }
        
        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())
            statistics['cell_counts'].append(self.statistics['cell_counts'][-1])
            statistics['cell_death_events'].append(self.statistics['cell_death_events'][-1])
            
        return history, statistics
    
    def plot_grid(self, separate_types=False, time_point=None, history=None, ax=None, skip_empty=False):
        """Plot the grid either as a single plot or separate facets for each cell type
        
        Args:
            separate_types: If True, creates separate subplot for each cell type
            time_point: Optional int, specific time point to plot from history
            history: Optional list of grid states. If None, plots current state
            ax: Optional matplotlib axes object to plot on. If None, creates new figure
        """
        # Determine which grid to plot
        if history is not None and time_point is not None:
            if time_point >= len(history):
                raise ValueError(f"Time point {time_point} exceeds history length {len(history)}")
            grid_to_plot = history[time_point]
        else:
            grid_to_plot = self.grid
            
        if not separate_types:
            # Single plot with all cell types
            if ax is None:
                plt.figure(figsize=(8, 8))
                ax = plt.gca()
            
            # Use colors from ComplexCellType enum
            colors = [cell_type.get_color() for cell_type in ComplexCellType]
            cmap = plt.cm.colors.ListedColormap(colors)
            
            im = ax.imshow(grid_to_plot, cmap=cmap, vmin=0, vmax=len(ComplexCellType)-1)
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=cell_type.get_color(), label=cell_type.name)
                for cell_type in ComplexCellType
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            ax.set_xticks([])
            ax.set_yticks([])
            title = 'Complex Organoid Model'
            if time_point is not None:
                title += f' (t={time_point})'
            ax.set_title(title)
            
        else:
            if ax is None:
                if skip_empty:
                    fig, axes = plt.subplots(1, len(ComplexCellType) - 1, 
                                       figsize=(4*(len(ComplexCellType) - 1), 4))
                else:
                    fig, axes = plt.subplots(1, len(ComplexCellType), 
                                       figsize=(4*len(ComplexCellType), 4))
            else:
                # If ax is provided, it should be an array of axes
                if not isinstance(ax, np.ndarray) or len(ax) != len(ComplexCellType):
                    raise ValueError("For separate_types=True, ax must be an array of axes "
                                   f"with length {len(ComplexCellType)}")
                axes = ax
            
            for idx, cell_type in enumerate(ComplexCellType):
                if skip_empty:
                    idx -= 1
                    if cell_type == ComplexCellType.EMPTY:
                        continue
                # Create binary mask for this cell type
                mask = (grid_to_plot == cell_type.value).astype(float)
                
                # Plot mask using cell type's color for consistency
                color_map = plt.cm.colors.LinearSegmentedColormap.from_list(
                    f"custom_{cell_type.name}",
                    ['white', cell_type.get_color()]
                )
                
                im = axes[idx].imshow(mask, cmap=color_map)
                axes[idx].set_title(f'{cell_type.name}\n(t={time_point if time_point is not None else "current"})')
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                
                # Add colorbar
                plt.colorbar(im, ax=axes[idx])
        
        if ax is None:
            plt.tight_layout()
            plt.show()
        
        return fig

# Visualization helper
def plot_cell_type_evolution(history, figsize=(12, 6)):
    """Plot evolution of cell types over time
    
    Args:
        history: List of grid states
        figsize (tuple): Figure size in inches (width, height)
    """
    plt.figure(figsize=figsize)
    cell_counts = {cell_type: [] for cell_type in ComplexCellType}
    
    for grid in history:
        for cell_type in ComplexCellType:
            count = np.sum(grid == cell_type.value)
            cell_counts[cell_type].append(count)
    
    for cell_type, counts in cell_counts.items():
        plt.plot(counts, 
                label=f"{cell_type.name}", 
                color=cell_type.get_color(),
                linewidth=2)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Cell Count')
    plt.title('Evolution of Cell Types')
    plt.legend(title="Cell Types")
    plt.grid(True)
    plt.show()


def create_complex_model_example(n_stems):
    model = TissueModel(
        grid_size=30,
        initial_stem_cells=n_stems,
        stem_division_rate=0.8,      # Reduced to account for survival
        intermediate_division_rate=0.5,
        intermediate2_division_rate=0.5
    )
    
    # Adjust death rates to be lower to account for survival
    model.death_rates = {
        ComplexCellType.STEM: 0.,
        ComplexCellType.INTERMEDIATE_1: 0.,
        ComplexCellType.INTERMEDIATE_2: 0.,
        ComplexCellType.DIFFERENTIATED_1: 0.001,
        ComplexCellType.DIFFERENTIATED_2: 0.001
    }
    
    # Set survival rates
    model.survival_rates = {
        ComplexCellType.STEM: 0.,
        ComplexCellType.INTERMEDIATE_1: 0.,
        ComplexCellType.INTERMEDIATE_2: 0.01,
        ComplexCellType.DIFFERENTIATED_1: 1,
        ComplexCellType.DIFFERENTIATED_2: 1
    }
    
    # Base differentiation probabilities
    model.base_differentiation = np.array([
        #STEM  INT1  INT2  DIFF  DIFF2
        [0.3,  0.8,  0.0,  0.0,  0.0],    # STEM -> mostly self-renewal
        [0.1, 0.2,  0.8,  0.0,  0.0],    # INT1 -> small back to STEM, mostly to INT2
        [0.0,  0.0,  0.2,  1.0,  0.0],    # INT2 -> small self-renewal, go to DIFF
        [0.0,  0.0,  0.0,  1.0,  0.0],    # DIFF -> stays DIFF
        [0.0,  0.0,  0.0,  0.0,  1.0]     # DIFF2 -> stays DIFF2
    ])
    
    # Interaction matrix
    model.interaction_matrix = np.array([
        #STEM  INT1  INT2  DIFF  DIFF2
        [0.0,  0.0, 0.0, 0.0, 0.0],  # Effect on STEM
        [0.0, 0.0,  0.0, 0.0, 0.0], # Effect on INT1
        [0.0, 0.0, 0.0,  0.0,  0.0],   # Effect on INT2
        [0.0,  0.0,  0.0, 0.0,  0.3],   # Effect on DIFF
        [0.0,  0.0,  0.0, 0.0,  0.0]   # Effect on DIFF2 
    ])
    
    return model


def create_simple_model_example(n_stems):
    model = TissueModel(
        grid_size=30,
        initial_stem_cells=n_stems,
        stem_division_rate=0.6,      # Reduced to account for survival
        intermediate_division_rate=0.5,
        intermediate2_division_rate=0.5
    )
    
    # Adjust death rates to be lower to account for survival
    model.death_rates = {
        ComplexCellType.STEM: 0.05,
        ComplexCellType.INTERMEDIATE_1: 0.0,
        ComplexCellType.INTERMEDIATE_2: 0.0,
        ComplexCellType.DIFFERENTIATED_1: 0.0,
        ComplexCellType.DIFFERENTIATED_2: 0.0
    }
    
    # Set survival rates
    model.survival_rates = {
        ComplexCellType.STEM: 1,
        ComplexCellType.INTERMEDIATE_1: 0.,
        ComplexCellType.INTERMEDIATE_2: 0.01,
        ComplexCellType.DIFFERENTIATED_1: 1,
        ComplexCellType.DIFFERENTIATED_2: 1
    }
    
    # Base differentiation probabilities
    model.base_differentiation = np.array([
        #STEM  INT1  INT2  DIFF  DIFF2
        [0.0,  0.0,  0.0,  0.1,  0.0],    # STEM -> mostly self-renewal
        [0.0, 0.0,  0.0,  0.0,  0.0],    # INT1 -> small back to STEM, mostly to INT2
        [0.0,  0.0,  0.0,  0.0,  0.0],    # INT2 -> small self-renewal, go to DIFF
        [0.0,  0.0,  0.0,  1.0,  0.],    # DIFF -> stays DIFF
        [0.0,  0.0,  0.0,  0.0,  0.0]     # DIFF2 -> stays DIFF2
    ])
    
    # Interaction matrix
    model.interaction_matrix = np.array([
        #STEM  INT1  INT2  DIFF  DIFF2
        [0.0,  0.0, 0.0, 0.0, 0.0],  # Effect on STEM
        [0.0, 0.0,  0.0, 0.0, 0.0], # Effect on INT1
        [0.0, 0.0, 0.0,  0.0,  0.0],   # Effect on INT2
        [0.0,  0.0,  0.0, 0.0,  0.0],   # Effect on DIFF
        [0.0,  0.0,  0.0, 0.0,  0.0]   # Effect on DIFF2 
    ])
    
    return model

class TumorImmuneModel(TissueModel):
    def __init__(self, 
                 grid_size=50,
                 initial_tumor_cells=5,
                 initial_immune_cells=10,
                 p_div_ts=0.1,  # Division probability for sensitive tumor cells
                 p_div_tr=0.1,  # Division probability for resistant tumor cells
                 p_resist=0.001,  # Mutation probability to resistance
                 p_kill=0.6,  # Probability of immune cell killing sensitive tumor cell
                 p_kill_tr=0.01,  # Probability of immune cell killing resistant tumor cell
                 p_move_i=0.3,  # Probability of immune cell movement
                 p_death_i=0.01,  # Probability of immune cell death
                 p_death_ts=0.0005,  # Natural death probability for sensitive tumor cells
                 p_death_tr=0.0005):  # Natural death probability for resistant tumor cells
        
        super().__init__(grid_size=grid_size)
        
        # Store parameters
        self.p_div_ts = p_div_ts
        self.p_div_tr = p_div_tr
        self.p_resist = p_resist
        self.p_kill = p_kill
        self.p_kill_tr = p_kill_tr
        self.p_move_i = p_move_i
        self.p_death_i = p_death_i
        self.p_death_ts = p_death_ts
        self.p_death_tr = p_death_tr
        
        # Initialize grid with tumor and immune cells
        self._initialize_tumor_immune(initial_tumor_cells, initial_immune_cells)
        
    def _initialize_tumor_immune(self, n_tumor, n_immune):
        """Initialize grid with tumor and immune cells"""
        # Clear grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Place tumor cells near center
        center = self.grid_size // 2
        radius = int(np.sqrt(n_tumor))
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:  # Circular pattern
                    x, y = center + i, center + j
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        if np.random.random() < 0.8:  # Add some randomness
                            self.grid[x, y] = TumorImmuneCellType.SENSITIVE_TUMOR.value
        
        # Place immune cells randomly
        empty_cells = list(zip(*np.where(self.grid == TumorImmuneCellType.EMPTY.value)))
        if empty_cells:
            immune_positions = np.random.choice(len(empty_cells), size=min(n_immune, len(empty_cells)), replace=False)
            for pos in immune_positions:
                x, y = empty_cells[pos]
                self.grid[x, y] = TumorImmuneCellType.IMMUNE.value
    
    def step(self):
        """Execute one step of the simulation"""
        new_grid = self.grid.copy()
        
        # Process each cell in random order
        cells = list(zip(*np.where(self.grid != TumorImmuneCellType.EMPTY.value)))
        np.random.shuffle(cells)
        
        for x, y in cells:
            current_cell = self.grid[x, y]
            
            # Skip if cell was already changed
            if new_grid[x, y] != current_cell:
                continue
                
            if current_cell == TumorImmuneCellType.SENSITIVE_TUMOR.value:
                self._process_sensitive_tumor(x, y, new_grid)
            elif current_cell == TumorImmuneCellType.RESISTANT_TUMOR.value:
                self._process_resistant_tumor(x, y, new_grid)
            elif current_cell == TumorImmuneCellType.IMMUNE.value:
                self._process_immune(x, y, new_grid)
        
        self.grid = new_grid

    def simulate(self, steps):
        """Run simulation for given number of steps
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            history: List of grid states at each time step
            statistics: Dictionary containing evolution of:
                - cell_counts
                - cell_death_events
        """
        history = [self.grid.copy()]

        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())

            
        return history
    
    def _process_sensitive_tumor(self, x, y, new_grid):
        """Process a sensitive tumor cell"""
        # Check for immune cell neighbors
        immune_neighbors = self._get_neighbors_of_type(x, y, TumorImmuneCellType.IMMUNE.value)
        if immune_neighbors:
            # Probability of being killed increases with number of immune neighbors
            if np.random.random() < self.p_kill * len(immune_neighbors):
                new_grid[x, y] = TumorImmuneCellType.EMPTY.value
                return
        
        # Check for natural death
        if np.random.random() < self.p_death_ts:
            new_grid[x, y] = TumorImmuneCellType.EMPTY.value
            return
        
        # Check for division
        empty_neighbors = self._get_neighbors_of_type(x, y, TumorImmuneCellType.EMPTY.value)
        if empty_neighbors and np.random.random() < self.p_div_ts:
            # Choose random empty neighbor for division
            nx, ny = empty_neighbors[np.random.randint(len(empty_neighbors))]
            new_grid[nx, ny] = TumorImmuneCellType.SENSITIVE_TUMOR.value
            
            # Check for mutation to resistance
            if np.random.random() < self.p_resist:
                new_grid[x, y] = TumorImmuneCellType.RESISTANT_TUMOR.value
    
    def _process_resistant_tumor(self, x, y, new_grid):
        """Process a resistant tumor cell"""
        # Check for immune cell neighbors (with reduced kill probability)
        immune_neighbors = self._get_neighbors_of_type(x, y, TumorImmuneCellType.IMMUNE.value)
        if immune_neighbors:
            if np.random.random() < self.p_kill_tr * len(immune_neighbors):
                new_grid[x, y] = TumorImmuneCellType.EMPTY.value
                return
        
        # Check for natural death
        if np.random.random() < self.p_death_tr:
            new_grid[x, y] = TumorImmuneCellType.EMPTY.value
            return
        
        # Check for division
        empty_neighbors = self._get_neighbors_of_type(x, y, TumorImmuneCellType.EMPTY.value)
        if empty_neighbors and np.random.random() < self.p_div_tr:
            # Choose random empty neighbor for division
            nx, ny = empty_neighbors[np.random.randint(len(empty_neighbors))]
            new_grid[nx, ny] = TumorImmuneCellType.RESISTANT_TUMOR.value
    
    def _process_immune(self, x, y, new_grid):
        """Process an immune cell"""
        # Check for death
        if np.random.random() < self.p_death_i:
            new_grid[x, y] = TumorImmuneCellType.EMPTY.value
            return
        
        # Check for movement
        empty_neighbors = self._get_neighbors_of_type(x, y, TumorImmuneCellType.EMPTY.value)
        if empty_neighbors and np.random.random() < self.p_move_i:
            # Move towards the center   
            center = self.grid_size // 2

            nx, ny = x, y

            if x < center:
                nx = x + 1
            elif x > center:
                nx = x - 1
            
            if y < center:
                ny = y + 1
            elif y > center:
                ny = y - 1
            
            new_grid[nx, ny] = TumorImmuneCellType.IMMUNE.value
            new_grid[x, y] = TumorImmuneCellType.EMPTY.value
    
    def _get_neighbors_of_type(self, x, y, cell_type):
        """Get coordinates of neighbors of specific type"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    self.grid[nx, ny] == cell_type):
                    neighbors.append((nx, ny))
        return neighbors
    
    def plot_tissue(self, show_statistics=False, figsize=(15, 6)):
        """Visualize tissue state and optionally show statistics"""
        if show_statistics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Use colors from TumorImmuneCellType.get_color()
            colors = [cell_type.get_color() for cell_type in TumorImmuneCellType]
            cmap = plt.cm.colors.ListedColormap(colors)
            ax1.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(TumorImmuneCellType)-1)
            ax1.set_title('Tumor-Immune System')
            
            # Create custom legend
            legend_elements = [
                Patch(facecolor=cell_type.get_color(), label=f"{cell_type.name}")
                for cell_type in TumorImmuneCellType
            ]
            ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot statistics
            counts = np.bincount(self.grid.flatten(), minlength=len(TumorImmuneCellType))
            cell_types = [ct.name for ct in TumorImmuneCellType]
            values = counts / self.grid.size
            ax2.bar(cell_types, values, color=[ct.get_color() for ct in TumorImmuneCellType])
            ax2.set_title('Cell Type Distribution')
            ax2.tick_params(axis='x', rotation=45)
            
        else:
            plt.figure(figsize=(8, 8))
            colors = [cell_type.get_color() for cell_type in TumorImmuneCellType]
            cmap = plt.cm.colors.ListedColormap(colors)
            plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(TumorImmuneCellType)-1)
            plt.title('Tumor-Immune System')
            
            # Create custom legend
            legend_elements = [
                Patch(facecolor=cell_type.get_color(), label=f"{cell_type.name}")
                for cell_type in TumorImmuneCellType
            ]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()