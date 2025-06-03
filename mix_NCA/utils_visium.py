import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import scipy

def prepare_input_data_nca(X, ad, cell_idxs, ligand_idx, device, n_neighbours):
    x = generate_neighbour_patch(ad, cell_idxs, ligand_idx, n_neighbours,  device)
    return x

def generate_neighbour_patch(X, ad, device='cuda'):
    """
    Generate patches of nearest neighbours for hexagonal grid.
    Vectorized implementation assuming exact angles (up to rounding).
    """
    # Move data to specified device
    coords = torch.tensor(ad.obsm['spatial'], dtype=torch.float32, device=device)
    connectivities = torch.tensor(ad.obsp['spatial_connectivities'].toarray(), 
                                dtype=torch.float32, device=device)
    x_values = torch.tensor(X, dtype=torch.float32, device=device)
    
    n_cells = X.shape[0]
    n_features = X.shape[1]
    
    # Initialize output tensor
    out_patch = torch.zeros((n_cells, n_features, 4, 3), 
                          dtype=torch.float32, device=device)
    
    # Define hexagonal grid positions and their corresponding angles
    hex_positions = torch.tensor([
        [0, 1],  # top (0°)
        [1, 2],  # top-right (60°)
        [2, 2],  # bottom-right (120°)
        [3, 1],  # bottom (180°)
        [2, 0],  # bottom-left (240°)
        [1, 0],  # top-left (300°)
    ], device=device)
    
    hex_angles = torch.tensor([0, 60, 120, 180, 240, 300], device=device)
    
    # Get relative positions of all connected neighbors
    rel_positions = coords[None, :, :] - coords[:, None, :]  # [n_cells, n_cells, 2]
    
    # Compute angles (in degrees) for all pairs
    angles = torch.atan2(rel_positions[..., 1], rel_positions[..., 0])
    angles = torch.round(torch.rad2deg(angles + 2 * torch.pi) % 360)  # Round to nearest degree
    
    # Create mask for valid connections
    valid_connections = connectivities > 0  # [n_cells, n_cells]
    
    # For each hexagonal direction
    for angle, (i, j) in zip(hex_angles, hex_positions):
        # Find neighbors at this angle (with tolerance for numerical precision)
        angle_mask = (torch.abs(angles - angle) < 1) & valid_connections  # [n_cells, n_cells]
        # Get indices of neighbors at this angle
        neighbor_indices = torch.where(angle_mask)[1]  # [n_matches]
        source_indices = torch.where(angle_mask)[0]   # [n_matches]
        
        # Place features in the correct position
        if len(neighbor_indices) > 0:
            out_patch[source_indices, :, i, j] = x_values[neighbor_indices]
    # Add center cell values
    out_patch[:, :, 1, 1] = x_values
    
    return out_patch

def train_nca_dyn(model, X, ad, patches, time_length=30, n_epochs=300, device="cuda", update_every=1, 
                  lr=0.001, milestones=[1500, 2000, 2500], gamma=0.1, additional_steps = 2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.MSELoss()  # Changed to 'none' to apply mask
    pbar = tqdm(range(n_epochs), desc='Training NCA')
    
    losses = []  # List to store losses
    
    for _ in pbar:    
        total_loss = 0
        # Track gradients for the entire sequence
        for step in range(0, time_length - update_every + additional_steps, update_every):
            
            target_step = step + update_every
            if step >= time_length - update_every:
                step = time_length - update_every
                target_step = step

            current_state = patches[step].to(device)
            new_state = model(current_state, num_steps=update_every, return_history=False)
            target_state = patches[target_step].to(device)
            
            # Create mask for non-zero cells in target state
            # Check if all features are non-zero for each cell
            #nonzero_mask = (target_state[:, :, 1, 1] != 0).all(dim=1)

            # Create subplot
            
            
            # Compute loss only on non-zero cells
            loss_per_cell = criterion(new_state[:, :].squeeze(), 
                                   target_state[:, :, 1, 1].squeeze())
            
            # Average loss over features for each cell
            #single_loss = loss_per_cell.mean(dim=1)
            
            # Apply mask and compute mean loss
            #single_loss = (loss_per_cell * nonzero_mask.float()).sum() / (nonzero_mask.sum() + 1e-8) * 100
            
            loss_per_cell.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            #ad.obs["nonzero_mask"] = nonzero_mask.cpu().detach().numpy()
            #sc.pl.spatial(ad, color=f'nonzero_mask', title=f'Mask', size=5, spot_size=15,
            #           use_raw=False,vmax= "p99", cmap='RdBu_r', vcenter=0, show = False)


            total_loss += loss_per_cell.item()

        scheduler.step()
        
        # Store the average loss for this epoch
        losses.append(total_loss/time_length)
        
        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{losses[-1]:.6f}'})
    
    return losses



def generate_stem_cell_expansion_series(adata, max_steps=50, early_stop_threshold=1, use_pca=False):
    """
    Generate time series data starting from stem cells and expanding to neighbors.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix containing spatial information and stem cell annotations
    max_steps : int
        Maximum number of expansion steps
        
    Returns:
    --------
    list
        List of arrays representing each time step of expansion
    """
    # Get initial stem cell positions
    current_cells = adata.obs['stem_cells'].values
    series = []
    
    # Convert sparse matrix to dense if needed
    if use_pca:
        expression_matrix = adata.obsm['X_pca'].copy()
    else:
        if scipy.sparse.issparse(adata.X):
            expression_matrix = adata.X.toarray().copy()
        else:
            expression_matrix = adata.X.copy()
        
    # Get spatial connectivity matrix
    connectivity = adata.obsp['spatial_connectivities'].toarray()
    
    # Initialize first frame with just stem cells
    frame = np.zeros_like(expression_matrix)
    frame[current_cells] = expression_matrix[current_cells]
    series.append(frame.copy())
    
    # Expand outward until we've covered all cells or reached max_steps
    cells_covered = current_cells.copy()
    for _ in range(max_steps):


        # stop if the percentage of cells covered is greater than 90%
        if sum(cells_covered) > early_stop_threshold * len(expression_matrix):
            break
        # Find neighbors of current cells that haven't been covered yet
        new_neighbors = np.zeros_like(current_cells)
        for idx in np.where(current_cells)[0]:
            neighbors = connectivity[idx] > 0
            new_neighbors = new_neighbors | (neighbors & ~cells_covered)
        
        # If no new neighbors, we're done
        if not np.any(new_neighbors):
            break
            
        # Add new neighbors to current frame
        frame[new_neighbors] = expression_matrix[new_neighbors]
        series.append(frame.copy())
        
        # Update tracking variables
        cells_covered = cells_covered | new_neighbors
        current_cells = new_neighbors
    
    return series

def prepare_stem_cell_nca_data(adata, device='cuda', n_neighbours=6):
    """
    Prepare data for NCA training starting from stem cells.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    device : str
        Device to use ('cuda' or 'cpu')
    n_neighbours : int
        Number of neighbors to consider
        
    Returns:
    --------
    list
        List of tensors for each time step
    int
        Total number of time steps
    """
    # Generate expansion series
    series = generate_stem_cell_expansion_series(adata)
    
    # Convert series to format needed for NCA training
    target_states = []
    for frame in series:
        # Get indices of non-zero cells in this frame
        active_cells = np.where(frame.any(axis=1))[0]
        
        # Prepare input data for these cells
        frame_data = prepare_input_data_nca(
            adata, 
            active_cells,
            np.arange(frame.shape[1]),  # all genes/features
            device,
            n_neighbours
        )
        target_states.append(frame_data)
    
    return target_states, len(series)

# Debug function to visualize patches
def visualize_patches(patches, sample_indices=None, feature_index = 0):
    """
    Visualize the neighborhood patches for specified cells.
    
    Args:
        patches: Output from generate_neighbour_patch
        sample_indices: List of cell indices to visualize. If None, picks 5 random cells.
    """
    import matplotlib.pyplot as plt
    
    if sample_indices is None:
        sample_indices = np.random.choice(patches.shape[0], 
                                        size=min(5, patches.shape[0]), 
                                        replace=False)
    
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(sample_indices, axes):
        # Sum across features for visualization
        patch_sum = patches[idx, feature_index].cpu().numpy()
        im = ax.imshow(patch_sum)
        ax.set_title(f'Cell {idx}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

def plot_stem_cell_expansion(adata, series_data, time_steps=None, figsize=(20, 4), 
                                     spot_size=5, alpha=1,  title=None):
    """
    Visualize the stem cell expansion series data overlaid on Visium spatial coordinates.
    
    Args:
        adata: AnnData object containing spatial coordinates
        series_data: List or tensor of states representing the expansion series
        time_steps: Number of time steps to show (default: all)
        figsize: Figure size tuple (width, height)
        spot_size: Size of spots in visualization
        alpha: Transparency of the overlay
        cmap: Colormap to use for visualization
        title: Optional title for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy if tensor
    if torch.is_tensor(series_data):
        series_data = series_data.cpu().numpy()
    
    # Select subset of timesteps if specified
    if time_steps is None:
        time_steps = len(series_data)
    
    n_steps = len(series_data)
    
    # Create subplot for each timestep
    fig, axes = plt.subplots(1, time_steps+1, figsize=figsize)
    if n_steps == 1:
        axes = [axes]
    
    # Get spatial coordinates
    spatial_coords = adata.obsm['spatial']
    
    for i, state in enumerate(series_data):
        # For multi-channel data, take argmax along channel dimension
        if i > time_steps:
            break
        if len(state.shape) > 1:
            state = np.argmax(state, axis=1) > 0

        adata.obs["stem_cells"] = state.flatten()
        
        sc.pl.spatial(adata=adata, color='stem_cells', size=5, spot_size=15, ax = axes[i], 
                      show = False, legend_loc = None, cmap = "viridis", palette = {"False" : "grey", "True" : "darkred"})

        
        axes[i].set_title(f'Step {i}', fontsize = 22)
        axes[i].axis('off')

    # Add colorbar
    #plt.colorbar(scatter, ax=axes.ravel().tolist())
    
    if title:
        fig.suptitle(title, fontsize=22)
    
    #plt.tight_layout()
    return fig

def plot_nca_results(
    model, 
    adata, 
    initial_state, 
    time_steps=10, 
    pathway_names=["WNT", "PI3K", "JAK-STAT", "MAPK"], 
    every_n_step=2, 
    sample_rule=False, 
    save_path=None, 
    device='cuda', 
    figsize=(4, 3), 
    seed=3
):
    """ 
    Visualize the NCA model results for progeny pathways alongside actual expression.
    
    All plots in the grid will have the same tissue size, and
    a dedicated colorbar column is added so the colorbar matches
    the height of each tissue plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import liana as li
    import torch
    import scanpy as sc

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set fontsize
    plt.rcParams.update({'font.size': 24})

    # -- Run simulation (as in your original code) --
    with torch.no_grad():
        states = []
        current_state = initial_state.to(device)
        states.append(current_state[:, :, 1, 1].squeeze().cpu().numpy())       
        for t in range(time_steps):
            new_states = model(current_state, 1, return_history=False, sample_rule=sample_rule).squeeze()
            states.append(new_states.detach().cpu().numpy())
            # generate_neighbour_patch presumably returns a new patch (as in your code)
            current_state = generate_neighbour_patch(new_states, adata, device=device)
    
    states = np.stack(states, axis=0)

    # Steps to plot
    plot_steps = range(0, len(states), every_n_step)
    n_plots = len(plot_steps)
    n_pathways = len(pathway_names)

    # Identify each pathway's index in the adata
    pathway_idx = [adata.var_names.tolist().index(pw) for pw in pathway_names]

    # Prepare an AnnData for adjacency, etc.
    temp_adata = adata.copy()
    li.ut.spatial_neighbors(
        temp_adata,
        bandwidth=100,
        cutoff=0.1,
        kernel='gaussian',
        set_diag=True,
        standardize=True
    )

    # Combine predicted + actual => to get global min/max
    all_values = np.concatenate([
        states[:, :, pathway_idx].flatten(),
        adata.X[:, pathway_idx].flatten()
    ])
    vmin, vmax = -1.5, 1.5  # or use np.percentile(all_values, [1, 99]) if desired

    # ------------------------------------------------------------------
    # 1) Create a gridspec with (n_plots+1) for the tissue columns
    #    plus 1 column for the colorbars => total (n_plots+2) columns
    # 2) The figure size in "inches" is controlled by (width x height).
    #    We'll multiply the base figsize by (n_plots+2) horizontally
    #    so each tissue plot stays the same size, and an extra column
    #    is reserved for the colorbar.
    # ------------------------------------------------------------------
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(
        figsize=(figsize[0] * (n_plots + 2), figsize[1] * n_pathways)
    )
    gs = GridSpec(
        nrows=n_pathways, 
        ncols=n_plots + 2,  
        width_ratios=[1]* (n_plots + 1) + [0.02],  # last column narrower for colorbar
        height_ratios=[1]* n_pathways,
        #wspace=0.05,  # smaller => less horizontal gap
        #hspace=0.05   # smaller => less vertical gap
    )

    # axes_main[i][j] will hold the tissue plot for pathway i, column j
    axes_main = []
    # axes_cb[i] will hold the colorbar for pathway i
    axes_cb = []

    for i in range(n_pathways):
        row_axes = []
        for j in range(n_plots + 1):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        # The colorbar axis is the last column
        cax = fig.add_subplot(gs[i, -1])
        axes_main.append(row_axes)
        axes_cb.append(cax)

    # ------------------------------------------------------------------
    # Plot the predicted values in columns [0..n_plots-1]
    # Then plot the actual values in column [n_plots]
    # Finally, colorbar goes in the axis at [-1].
    # ------------------------------------------------------------------

    # Plot predicted time steps
    for col_idx, step in enumerate(plot_steps):
        for row_idx, pw_idx in enumerate(pathway_idx):
            # Smooth or direct usage (your original code):
            temp_adata.obs[f'pathway_{pw_idx}'] = (
                temp_adata.obsp['spatial_connectivities'] @ states[step, :, pw_idx]
                if col_idx > 0
                else states[step, :, pw_idx]
            )

            # Plot with no colorbar
            sc.pl.spatial(
                temp_adata,
                color=f'pathway_{pw_idx}',
                title=f'{pathway_names[row_idx]}, Step {step}',
                size=5,
                spot_size=15,
                use_raw=False,
                vmin=vmin,
                vmax=vmax,
                cmap='RdBu_r',
                vcenter=0,
                colorbar_loc=None,   # IMPORTANT: turn off internal colorbar
                ax=axes_main[row_idx][col_idx],
                show=False
            )

    # Plot the actual values in the last tissue column
    for row_idx, pw_idx in enumerate(pathway_idx):
        temp_adata.obs[f'actual_{pw_idx}'] = adata.X[:, pw_idx]
        sc.pl.spatial(
            temp_adata,
            color=f'actual_{pw_idx}',
            title=f'Actual {pathway_names[row_idx]}',
            size=5,
            spot_size=15,
            use_raw=False,
            vmin=vmin,
            vmax=vmax,
            cmap='RdBu_r',
            vcenter=0,
            colorbar_loc=None,  # turn off internal colorbar
            ax=axes_main[row_idx][n_plots], 
            show=False
        )
        # Remove axis names and labels from tissue plots
    for row_axes in axes_main:
        for ax in row_axes:
            ax.set_xlabel("")  # Remove x-axis label
            ax.set_ylabel("")  # Remove y-axis label

    # ------------------------------------------------------------------
    # 2) Manually add a colorbar in each row, matching the height of
    #    the tissue plots. We can pick the first subplot's image or
    #    any from that row. If you prefer a single colorbar for all
    #    subplots, just do it once.
    # ------------------------------------------------------------------
    for row_idx in range(n_pathways):
        # Grab the first subplot's "collection" or "image" from sc.pl.spatial
        # Typically, it's something like axes_main[row_idx][0].collections[0]
        # or axes_main[row_idx][0].images[0], depending on the version.
        if axes_main[row_idx][0].collections:
            # For scatter-based plots
            artist = axes_main[row_idx][0].collections[0]
        else:
            # For image-based plots
            artist = axes_main[row_idx][0].images[0]

        cbar = fig.colorbar(
            artist,
            cax=axes_cb[row_idx],   # place colorbar in the dedicated axis
            orientation='vertical'
        )
        cbar.ax.set_title("")  # optional: remove any default colorbar title

    fig.tight_layout()

    # Optionally save
    if save_path:
        fig.savefig(
            f'{save_path}_comparison.png', 
            dpi=300, 
            bbox_inches='tight'
        )

    plt.show()

    return fig
