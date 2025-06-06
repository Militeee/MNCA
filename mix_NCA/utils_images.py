import torch
import time
import copy
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from PIL import Image
import matplotlib.pyplot as plt

import requests
import os

from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.MixtureNCANoise import MixtureNCANoise
from mix_NCA.NCA import NCA



def standard_update_net(n_channels_in, hidden_dims = 128, n_channels_out=None, device = "cuda"):
    if n_channels_out is None:
        n_channels_out = n_channels_in
    return nn.Sequential(
                      nn.Conv2d(n_channels_in, hidden_dims, 1),  # process perceptual inputs
                      nn.ReLU(),                              # nonlinearity
                      nn.Conv2d(hidden_dims, n_channels_out, 1)     # output a residual update
                    ).to(device)

def normalize_grads(model):  # makes training more stable, especially early on
  for p in model.parameters():
      p.grad = p.grad / (p.grad.norm() + 1e-8) if p.grad is not None else p.grad

def train_nca(model, data, device = "cuda", num_steps = (10,20), learning_rate=1e-3, decay=0, milestones=[], gamma=0.1, batch_size = 64,
               state_dim = 16, seed_loc = (10,10), pool_size = 1024, total_steps = 10000, print_every = 100, 
               return_history=False,temperature=None, min_temperature=0.1, anneal_rate=0.002, straight_through=True, init_black=False):
  
  model = model.to(device)  # put the model on GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
  scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

  target_rgba = torch.Tensor(data).to(device)
  if init_black:
    init_state = torch.ones(batch_size, state_dim, *target_rgba.shape[-2:]).to(device) #.half() 
    init_state[...,3:, seed_loc[0], seed_loc[1]] = 0  # initially, there is just one cell
  else:
    init_state = torch.zeros(batch_size, state_dim, *target_rgba.shape[-2:]).to(device)#.half()
    init_state[...,3:, seed_loc[0], seed_loc[1]] = 1  # initially, there is just one cell
  pool = init_state[:1].repeat(pool_size,1,1,1).cpu()
  
  results = {'loss':[], 'tprev': [time.time()]}
  for step in range(total_steps+1):

    # prepare batch and run forward pass
    if pool_size > 0:  # draw CAs from pool (if we have one)
      pool_ixs = np.random.randint(pool_size, size=[batch_size])
      input_states = pool[pool_ixs].to(device)
    else:
      input_states = init_state

    if temperature is None:
        states = model(input_states, np.random.randint(*num_steps), seed_loc, return_history=return_history)  # forward pass
    else:
        states = model(input_states, np.random.randint(*num_steps), seed_loc, return_history=return_history, temperature=temperature, straight_through=straight_through)  # forward pass

    if return_history:
        final_rgba = states[-1,:, :4]  # grab rgba channels of last frame
    else:
        final_rgba = states[:, :4]  # grab rgba channels of last frame

    # compute loss and run backward pass
    mses = (target_rgba.unsqueeze(0)-final_rgba).pow(2)
    batch_mses = mses.reshape(batch_size,-1).mean(-1)
    loss = batch_mses.mean() ; loss.backward() ; normalize_grads(model)
    optimizer.step() ; optimizer.zero_grad() ; scheduler.step()

    # update the pool (if we have one)
    if pool_size > 0:
      ixs_to_replace = torch.argsort(batch_mses, descending=True)[:int(.15*batch_size)]
      # ixs_to_replace = np.random.randint(args.batch_size, size=int(.15*args.batch_size))
      if return_history:
        final_states = states[-1].detach()
      else:
        final_states = states.detach()
      final_states[ixs_to_replace] = init_state[:1]
      pool[pool_ixs] = final_states.cpu()
      del batch_mses

    # bookkeeping and logging
    results['loss'].append(loss.item())

    if temperature is not None:
        temperature = max(min_temperature, temperature - anneal_rate * total_steps)

    if step % print_every == 0:
      print('step {}, dt {:.3f}s, loss {:.2e}, log10(loss) {:.2f}'\
          .format(step, time.time()-results['tprev'][-1], loss.item(), np.log10(loss.item())))
      results['tprev'].append(time.time())

  results['final_model'] = copy.deepcopy(model.cpu())
  return results



def load_emoji(path, target_size=40, padding=6):
    """
    Load and preprocess emoji image to RGBA tensor with padding
    Args:
        path: path to emoji image
        target_size: desired size of emoji (without padding)
        padding: number of pixels to pad on each side
    Returns:
        Padded tensor of shape (1, 4, H+2*padding, W+2*padding)
    """
    img = Image.open(path).convert('RGBA')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)) / 255.0
    
    # Convert to (1, 4, H, W) format
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Add padding with zeros
    padded_tensor = torch.nn.functional.pad(
        img_tensor, 
        (padding, padding, padding, padding), 
        mode='constant', 
        value=0
    )
    
    return padded_tensor.float()


def plot_automata_evolution(models, target_sizes, seed_location, device="cuda", n_steps=400, plot_every=20, figsize=None):
    """
    Plot the evolution of the automata starting from seed location
    
    Args:
        models: list of trained NCA models
        target_sizes: list of target sizes for different patterns
        seed_location: tuple (x, y) for seed placement
        device: computing device
        n_steps: total number of steps to simulate
        plot_every: plot every N steps
        figsize: optional figure size tuple (width, height)
    """

    # Calculate number of timesteps to plot
    timesteps = list(range(0, n_steps + 1, plot_every))
    n_timesteps = len(timesteps)
    n_patterns = len(target_sizes)
    
    # Set default figsize if not provided
    if figsize is None:
        figsize = (3 * n_timesteps, 3 * n_patterns)
    
    # Create figure
    fig, axes = plt.subplots(n_patterns, n_timesteps, figsize=figsize)
    if n_patterns == 1:
        axes = axes[np.newaxis, :]
    
    # Plot evolution for each pattern size
    for i, size in enumerate(target_sizes):
        # Initialize state with zeros and seed
        state = torch.zeros(1, models[i].state_dim, size, size).to(device)
        state[..., 3:, seed_location[0], seed_location[1]] = 1  # Set seed
        
        # Plot initial state
        rgba = state[0, :4].permute(1, 2, 0).cpu()
        axes[i, 0].imshow(rgba)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('t=0')
        
        # Simulate and plot evolution
        with torch.no_grad():
            for j, t in enumerate(timesteps[1:], 1):
                # Run simulation until next plotted timestep
                steps_to_take = plot_every if j > 1 else t
                # check if temperature is an attribute of the model
                if hasattr(models[i], 'temperature'):
                    state = models[i](state, steps_to_take, seed_location, sample_non_differentiable=False, straight_through=True)
                else:
                    state = models[i](state, steps_to_take, seed_location)
                
                # Plot RGBA channels
                rgba = torch.clip(state[0, :4], 0, 1).permute(1, 2, 0).cpu()
                rgba[3] = torch.clip(rgba[3], 0.1, 1)
                axes[i, j].imshow(rgba)
                axes[i, j].axis('off')
                axes[i, j].set_title(f't={t}')
    
    # Add pattern size labels as row labels
    for i, size in enumerate(target_sizes):
        axes[i, 0].set_ylabel(f'Size {size}', rotation=0, labelpad=40, va='center')
    
    plt.tight_layout()
    return fig





def download_emoji(emoji_code, save_dir='../data/raw/emojis/'):
    """
    Download a single emoji from Twemoji
    emoji_code: Unicode code point (e.g., '1F600' for grinning face)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Twemoji URL format
    url = f'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{emoji_code.lower()}.png'
    save_path = os.path.join(save_dir, f'{emoji_code}.png')
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download emoji: {response.status_code}")


def plot_emoji(tensor):
    """
    Plot a tensor containing RGBA image data
    tensor shape should be (1, 4, H, W) or (4, H, W)
    """
    if len(tensor.shape) == 4:
        # If batch dimension exists, take first image
        tensor = tensor[0]
    
    # Convert tensor to numpy and ensure it's on CPU
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor
    
    # Reshape to (H, W, 4)
    img = img.transpose(1, 2, 0)
    
    # Create figure with white background
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def load_model(model_path, model_class, device="cuda", n_channels=16, n_rules=6, hidden_dim=128, dropout=0.2):
    """
    Load a saved model from path
    """
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_CHANNELS = n_channels  
    DROPOUT = dropout
    HIDDEN_DIM = hidden_dim
    N_RULES = n_rules
      # Initialize models
    if model_class == NCA:

        model = NCA(update_net=standard_update_net(N_CHANNELS* 3, HIDDEN_DIM, N_CHANNELS, device=DEVICE), 
                state_dim=N_CHANNELS, 
                hidden_dim=HIDDEN_DIM, 
                dropout=DROPOUT, 
                device=DEVICE)
    elif model_class == MixtureNCA:
    
        model = MixtureNCA(update_nets=standard_update_net, 
                state_dim=N_CHANNELS, 
                num_rules = N_RULES,
                hidden_dim=HIDDEN_DIM, 
                dropout=DROPOUT, 
                device=DEVICE, temperature=1)
    else:
        model = MixtureNCANoise(update_nets=standard_update_net, 
                    state_dim=N_CHANNELS, 
                    num_rules = N_RULES,
                    hidden_dim=HIDDEN_DIM, 
                    dropout=DROPOUT, 
                    device=DEVICE, temperature=1)

    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    #model.eval()
    
    return model

def plot_emoji_rules(model, emojis_names, device="cuda"):
    """Create a grid visualization of emojis and their rule assignments"""
    # Get all emoji files
    emoji_files = [f'../data/emojis/{emoji}.png' for emoji in emojis_names]
    n_emojis = len(emoji_files)
    emojis = [load_emoji(emoji_file) for emoji_file in emoji_files]
    print(emojis[0].shape)
    # models
    model_files = [os.path.join(f"../results/experiment_{i}_robustness/mixture_model.pt") for i in emojis_names]
    models = [load_model(i, MixtureNCA, device) for i in model_files]
    
    # set font size
    plt.rcParams.update({'font.size': 24})

    n_rules = model.num_rules

    
    # Create figure
    fig, axes = plt.subplots(n_emojis, n_rules + 1, 
                            figsize=(3*(n_rules + 1), 3*n_emojis))
    
    # If only one emoji, wrap axes in list
    if n_emojis == 1:
        axes = [axes]
    
    for i  in range(n_emojis):
        # Load emoji
        emoji_tensor = emojis[i]
        emoji_tensor = emoji_tensor.to(device)

        #init_state_nca = mixture_model(init_state, 400, SEED_LOC, return_history=False).detach()
        # fill extra channel dimension with zeros to match standard nca
        emoji_tensor = torch.cat([emoji_tensor, torch.zeros(1, 16 - 4, *emoji_tensor.shape[-2:]).to(device)], dim=1)
        
        # Get rule probabilities
        with torch.no_grad():
            probs = model.get_rule_probabilities(emoji_tensor)
        
        # Plot original emoji
        axes[i][0].imshow(emojis[i][0].permute(1, 2, 0))
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')
        
        # Plot rule assignments
        for j in range(n_rules):
            im = axes[i][j+1].imshow(probs[0, j].cpu().numpy(), cmap='viridis')
            axes[i][j+1].set_title(f'Rule {j}')
            axes[i][j+1].axis('off')
            plt.colorbar(im, ax=axes[i][j+1], format='%.2f')
    
    plt.tight_layout()
    return fig