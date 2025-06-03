import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from mix_NCA.TissueModel import  ComplexCellType
import numpy as np
from matplotlib.patches import Patch

def classification_update_net(n_channels, hidden_dims = 128, n_channels_out=None, device = "cuda"):
    if n_channels_out is None:
        n_channels_out = n_channels
    return nn.Sequential(
                      nn.Conv2d(n_channels, hidden_dims, 1),  # process perceptual inputs
                      nn.ReLU(),                              # nonlinearity
                      nn.Conv2d(hidden_dims, n_channels_out, 1),     # output a residual update
                      nn.Softmax(dim=1)
                    ).to(device)

def classification_update_net_unorm(n_channels, hidden_dims = 128, n_channels_out=None, device = "cuda"):
    if n_channels_out is None:
        n_channels_out = n_channels
    return nn.Sequential(
                      nn.Conv2d(n_channels, hidden_dims, 1),  # process perceptual inputs
                      nn.ReLU(),                              # nonlinearity
                      nn.Conv2d(hidden_dims, n_channels_out, 1)     # output a residual update
                    ).to(device)


def grid_to_channels_batch(grids, n_cell_types, device="cpu"):
    """Convert a batch of grids to one-hot encoded channels
    
    Args:
        grids: List of 2D numpy arrays or single 2D array
        n_channels: Number of channels (cell types)
        device: torch device to put tensor on
    
    Returns:
        Tensor of shape [batch_size, n_channels, height, width]
    """
    # Convert list of grids to numpy array if needed
    if isinstance(grids, list):
        grids = np.array(grids)
    elif isinstance(grids, np.ndarray) and grids.ndim == 2:
        grids = np.array([grids])  # Add batch dimension for single grid
        
    # Convert to tensor
    grids_tensor = torch.from_numpy(grids).to(device)
    batch_size, height, width = grids_tensor.shape
    
    # Create one-hot encoded channels using scatter_
    channels = torch.zeros((batch_size, n_cell_types, height, width), device=device)
    grids_tensor = grids_tensor.long()  # Convert to long for indexing
    channels.scatter_(1, grids_tensor.unsqueeze(1), 1.0)
    
    return channels

def train_nca_dyn(model, target_states, n_cell_types=5, time_length=10, n_epochs=300, 
                  device="cuda", update_every=2, lr=0.001, milestones=[500], gamma=0.1,
                  loss_type="mse", temperature=None, min_temperature=0.1, anneal_rate=0.005, class_assignment = None, straight_through = False):
    """Train NCA on dynamic sequences with random time spans
    
    Args:
        model: NCA model to train
        target_states: List of target state sequences
        n_cell_types: Number of cell types
        time_length: Length of training window
        n_epochs: Number of training epochs
        device: Computing device
        update_every: Steps between updates
        lr: Learning rate
        milestones: LR scheduler milestones
        gamma: LR decay factor
        loss_type: Type of loss function ("cross_entropy" or "mse")
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    

    # Set up loss function based on type
    if loss_type.lower() == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Use 'cross_entropy' or 'mse'.")
    
    # Get total sequence length from target states
    total_sequence_length = len(target_states[0])
    
    # Ensure time_length is compatible with update_every
    time_length = (time_length // update_every) * update_every
    
    # Precompute all grid representations
    print("Precomputing grid representations...")

    precomputed_states = torch.stack([
        grid_to_channels_batch([x[t] for x in target_states], n_cell_types=n_cell_types, device=device)
        for t in range(total_sequence_length)
    ])

    # Create outer progress bar for epochs
    pbar = tqdm(range(n_epochs), desc=f'Training NCA ({loss_type})')
    
    for _ in pbar:    
        total_loss = 0

        
        # Sample random start point that allows for time_length window
        max_start = total_sequence_length - time_length
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        end_idx = start_idx + time_length
        
        # Track gradients for the sampled sequence
        for step in range(start_idx, end_idx - update_every, update_every):
            # Get current state from precomputed tensors
            current_state = precomputed_states[step] 
            # Forward pass
            for t in range(update_every):
                if class_assignment is not None:
                    class_assignment = torch.argmax(current_state, dim=1)
                    class_assignment = torch.nn.functional.one_hot(class_assignment, num_classes=n_cell_types)
                    class_assignment = class_assignment.transpose(1,3).transpose(2, 3)
                    if temperature is None:
                        current_state = model(current_state, num_steps=1, return_history=False, class_assignment = class_assignment, straight_through = straight_through)
                    else:
                        current_state = model(current_state, num_steps=1, return_history=False, class_assignment = class_assignment, temperature = temperature, straight_through = straight_through)
                else:
                    if temperature is None:
                        current_state = model(current_state, num_steps=1, return_history=False)
                    else:
                        current_state = model(current_state, num_steps=1, return_history=False, temperature = temperature, straight_through = straight_through)
            

            target_state = precomputed_states[step + update_every]
            pred = current_state
            target = target_state
            
            if loss_type == "cross_entropy":
                target = torch.argmax(target, dim=1)

            # Compute loss
            single_loss = criterion(pred, target)
            
            # Backward pass
            single_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += single_loss.item()


            if temperature is not None:
                temperature = max(min_temperature, temperature - anneal_rate * n_epochs)

        scheduler.step()
        
        # Update progress bar description with current loss
        pbar.set_postfix({
            'loss': f'{total_loss/time_length:.6f}',
            'window': f'{start_idx}-{end_idx}'
        })
    
    return total_loss


def plot_nca_prediction(nca, initial_state, cell_type_enum, n_cell_types=5, steps=30, 
                       show_intermediate=True, device="cuda", random=False, random_seed=3, 
                       figsize=(20, 5), class_assignment = None):
    """
    Plot NCA prediction with customizable cell type structure
    
    Args:
        nca: The NCA model
        initial_state: Initial grid state
        cell_type_enum: Enum class defining the cell types and their order
        n_cell_types: Number of cell types
        steps: Number of simulation steps
        show_intermediate: Whether to show intermediate steps
        device: Computing device
        random: Whether to use random rule selection
        random_seed: Random seed for reproducibility
        figsize (tuple): Figure size in inches (width, height)
    """
    current_state = grid_to_channels_batch([initial_state], n_cell_types=n_cell_types, device=device)
    n_plots = min(5, steps) + 2 if show_intermediate else 3
    plt.figure(figsize=figsize)
    torch.manual_seed(random_seed)
    
    # Get colors from the enum if they exist, otherwise use default gradient
    if hasattr(cell_type_enum, 'get_color'):
        colors = [cell_type.get_color() for cell_type in cell_type_enum]
    else:
        # Default color gradient
        colors = ['white'] + [plt.cm.Blues(i) for i in np.linspace(0.2, 1, n_cell_types-1)]
    
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Plot initial state
    plt.subplot(1, n_plots, 1)
    plt.imshow(initial_state, cmap=cmap, vmin=0, vmax=len(cell_type_enum)-1)
    plt.title('Initial State')
    
    # Create legend using enum names
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=cell_type.name)
        for i, cell_type in enumerate(cell_type_enum)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    nca.to(device)
    nca.eval()
    nca.evaluation = True
    
    # Store states for visualization
    states = [initial_state]
    for step in range(steps):
        if random:
            if class_assignment is not None:
                class_assignment = torch.argmax(current_state, dim=1)
                class_assignment = torch.nn.functional.one_hot(class_assignment, num_classes=n_cell_types)
                class_assignment = class_assignment.transpose(1,3).transpose(2, 3)
                current_state = nca(current_state, num_steps=1, return_history = False, sample_non_differentiable = True, class_assignment = class_assignment)
            else:
                current_state = nca(current_state, num_steps=1, return_history = False, sample_non_differentiable = True)
        else:
            current_state = nca(current_state, num_steps=1, return_history = False)
        
        #current_state = torch.argmax(current_state, dim=1)
        #current_state = torch.nn.functional.one_hot(current_state, num_classes=n_cell_types).transpose(1,3).transpose(2,3).float()

        if step < 4 and show_intermediate:
            states.append(torch.argmax(current_state[0, :n_cell_types], dim=0).copy().cpu().numpy())
        
      
    states.append(torch.argmax(current_state[0, :n_cell_types], dim=0).cpu().numpy())
    
    # Plot intermediate and final states
    for i, state in enumerate(states[1:], start=2):
        plt.subplot(1, n_plots, i)
        plt.imshow(state, cmap=cmap, vmin=0, vmax=len(cell_type_enum)-1)
        if show_intermediate:
            plt.title(f'Step {i-1}' if i <= 5 else 'Final State')
        else:
            plt.title(f'Final State')
    
    plt.tight_layout()
    plt.show()

def analyze_diff_dependencies(nca, initial_states, cell_type_enum=ComplexCellType, 
                            steps=30, device="cuda", figsize=(12, 6)):
    """
    Analyze how DIFF2 probabilities depend on DIFF cell quantities
    
    Args:
        nca: The NCA model
        initial_states: List of initial grid states
        cell_type_enum: Enum class defining the cell types and their order
        steps: Number of simulation steps
        device: Computing device
        figsize (tuple): Figure size in inches (width, height)
    """
    # Convert initial states to NCA format
    current_state = grid_to_channels_batch(initial_states, n_cell_types=len(cell_type_enum), device=device)
    
    # Storage for analysis
    diff_counts = []
    diff2_probs = []
    
    nca.eval()
    with torch.no_grad():
        for step in range(steps):
            # Get current cell type distributions
            cell_dist = torch.argmax(current_state[0, :len(cell_type_enum)], dim=0).cpu().numpy()
            diff_count = np.sum(cell_dist == cell_type_enum.DIFFERENTIATED.value)
            diff_counts.append(diff_count)
            
            # Get rule probabilities
            probs = nca.get_rule_probabilities(current_state)
            
            # Get probability of differentiation to DIFF2
            # Assuming the last channel represents DIFF2 probability
            diff2_prob = probs[0, -1].mean().cpu().item()  # Average over spatial dimensions
            diff2_probs.append(diff2_prob)
            
            # Step forward
            current_state = nca(current_state, num_steps=1, return_history=False)
    
    # Plot relationship
    plt.figure(figsize=figsize)
    
    # Create subplots with specific sizes
    plt.subplot(121)
    plt.scatter(diff_counts, diff2_probs, alpha=0.5)
    plt.xlabel('Number of DIFF cells')
    plt.ylabel('Average DIFF2 probability')
    plt.title('DIFF2 Probability vs DIFF Cell Count')
    
    plt.subplot(122)
    plt.plot(diff_counts, label='DIFF cells', color=cell_type_enum.DIFFERENTIATED.get_color())
    plt.plot(diff2_probs, label='DIFF2 prob', color=cell_type_enum.DIFFERENTIATED_2.get_color())
    plt.xlabel('Time steps')
    plt.ylabel('Count / Probability')
    plt.title('Evolution of DIFF cells and DIFF2 probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation
    correlation = np.corrcoef(diff_counts, diff2_probs)[0,1]
    print(f"Correlation between DIFF cells and DIFF2 probability: {correlation:.3f}")
    
    return diff_counts, diff2_probs


def plot_nca_prediction2(nca, initial_state, steps=30, plot_every=5, device="cuda", 
                       cell_type_enum=ComplexCellType, n_cell_types=6, random=True, seed=3):
    """Plot NCA prediction with intermediate states at specified intervals
    
    Args:
        nca: The NCA model
        initial_state: Initial grid state
        steps: Total number of simulation steps
        plot_every: Plot state every N steps (e.g. 5 means plot step 0,5,10,...)
        device: Computing device
        cell_type_enum: Enum class defining cell types
        n_cell_types: Number of cell types to consider
        random: Whether to use random rule selection
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)

    current_state = grid_to_channels_batch([initial_state], n_cell_types=n_cell_types, device=device)
    
    # Calculate number of plots needed
    n_plots = (steps // plot_every) + 1  # +1 for initial state
    
    # Create figure with dynamic width based on number of plots
    fig_width = min(20, max(12, n_plots * 3))  # Scale width with number of plots
    plt.figure(figsize=(fig_width, 5))
    
    # Define colors
    colors = [cell_type.get_color() for cell_type in cell_type_enum]
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Plot initial state
    plt.subplot(1, n_plots, 1)
    plt.imshow(initial_state, cmap=cmap, vmin=0, vmax=len(cell_type_enum)-1)
    plt.title('Initial State')
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors[i], label=cell_type.name)
        for i, cell_type in enumerate(cell_type_enum)
    ]
    #plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='best')
    
    nca.to(device)
    nca.eval()
    
    # Run simulation and plot at intervals
    plot_idx = 2
    for step in range(1, steps + 1):
        if random:
            current_state = nca(current_state, num_steps=1, return_history=False, sample_non_differentiable = True)
        else:
            current_state = nca(current_state, num_steps=1, return_history=False)
            
        if step % plot_every == 0:
            state = torch.argmax(current_state[0, :n_cell_types], dim=0).cpu().numpy()
            plt.subplot(1, n_plots, plot_idx)
            plt.imshow(state, cmap=cmap, vmin=0, vmax=len(cell_type_enum)-1)
            plt.title(f'Step {step}')
            plot_idx += 1
    
    #plt.tight_layout()
    return plt

def plot_automata_comparison_grid(det_nca, gca, mix_nca, stoch_mix_nca, initial_state, 
                                n_examples=3, n_steps=35, figsize=(10, 18), 
                                cell_type_enum=ComplexCellType, device = "cuda"):
    """
    Plot a grid comparing original tissue with predictions from different NCA models and a GCA baseline.
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.font_manager import FontProperties

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # Now 5 rows: Original, NCA, GCA, MNCA, MNCA+N
    fig, axes = plt.subplots(5, n_examples, figsize=figsize)

    colors = [cell_type.get_color() for cell_type in cell_type_enum]
    custom_cmap = mcolors.ListedColormap(colors)

    models = {
        0: ('Original', None),
        1: ('NCA', det_nca),
        2: ('GCA', gca),
        3: ('MNCA', mix_nca),
        4: ('MNCA+N', stoch_mix_nca)
    }

    model_abbreviations = {
        0: 'O',
        1: 'NCA',
        2: 'GCA',
        3: 'MNCA',
        4: 'MNCA+N'
    }

    for row in range(5):
        model_name, model = models[row]
        for col in range(n_examples):
            ax = axes[row, col]
            label = f'{model_abbreviations[row]}_{col + 1}'
            ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                    fontsize=14, fontweight='bold')
            if row == 0:
                # Plot original state
                ax.imshow(
                    initial_state[col][-1],
                    cmap=custom_cmap,
                    vmin=0,
                    vmax=len(cell_type_enum)-1
                )
                if col == 0:
                    ax.set_ylabel(model_name, fontsize=14, fontweight='bold')
            else:
                with torch.no_grad():
                    torch.manual_seed(col)
                    current_state = grid_to_channels_batch([initial_state[col][0]], n_cell_types = len(cell_type_enum), device = device)
                    for _ in range(n_steps):
                        if row in [3,4]:
                            current_state = models[row][1](current_state, 1, return_history=False, sample_non_differentiable = True)
                        else:
                            current_state = models[row][1](current_state, 1, return_history=False)
                        ax.imshow(
                            torch.argmax(current_state.squeeze(), dim=0).cpu(),
                            cmap=custom_cmap,
                            vmin=0,
                            vmax=len(cell_type_enum)-1
                        )
                    if col == 0:
                        ax.set_ylabel(model_name, fontsize=14, fontweight='bold')
            ax.axis('off')

    norm = plt.Normalize(vmin=-0.5, vmax=len(cell_type_enum)-0.5)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(cell_type_enum)))
    cbar.set_ticklabels([ct.name.replace('_', ' ').title() for ct in cell_type_enum])
    cbar.ax.tick_params(labelsize=14)
    for tick in cbar.ax.yaxis.get_major_ticks():
        tick.label2.set_fontweight('bold')
    cbar.set_label('', fontsize=14, fontweight='bold')

    return fig