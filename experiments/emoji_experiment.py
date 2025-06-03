import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import json
from datetime import datetime
import sys
from celluloid import Camera
import pickle


sys.path.append('..')
from mix_NCA.NCA import NCA
from mix_NCA.utils_images import train_nca, standard_update_net
from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.RobustnessAnalysis import RobustnessAnalysis
from mix_NCA.MixtureNCANoise import MixtureNCANoise





def load_emoji(path, target_size=40, padding=6):
    img = Image.open(path).convert('RGBA')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    padded_tensor = torch.nn.functional.pad(
        img_tensor, 
        (padding, padding, padding, padding), 
        mode='constant', 
        value=0
    )
    return padded_tensor.float()

def download_emoji(emoji_code, save_dir='data/emojis/'):
    os.makedirs(save_dir, exist_ok=True)
    url = f'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{emoji_code.lower()}.png'
    save_path = os.path.join(save_dir, f'{emoji_code}.png')
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download emoji: {response.status_code}")




def run_experiment(emoji_code, output_dir):

    # Set up training parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_SIZE = 40
    N_CHANNELS = 16  
    BATCH_SIZE = 8
    POOL_SIZE = 1000 
    NUM_STEPS = [30, 50]
    SEED_LOC = (20,20)
    GAMMA = 0.2
    DECAY = 3e-5
    DROPOUT = 0.2
    HIDDEN_DIM = 128
    TOTAL_STEPS = 8000
    PRINT_EVERY = 200
    SEED = 3
    MILESTONES = [4000, 6000, 7000]
    LEARNING_RATE = 1e-3
    N_RULES = 6

    torch.manual_seed(SEED)
    np.random.seed(SEED)


    experiment_dir = os.path.join(output_dir, f"experiment_{emoji_code}_robustness")
    os.makedirs(experiment_dir, exist_ok=True)

    # Download and load emoji
    emoji_path = download_emoji(emoji_code)
    target = load_emoji(emoji_path, target_size=TARGET_SIZE).to(DEVICE)


    # Initialize models
    model = NCA(update_net=standard_update_net(N_CHANNELS* 3, HIDDEN_DIM, N_CHANNELS, device=DEVICE), 
            state_dim=N_CHANNELS, 
            hidden_dim=HIDDEN_DIM, 
            dropout=DROPOUT, 
            device=DEVICE)
    
    model_mix = MixtureNCA(update_nets=standard_update_net, 
            state_dim=N_CHANNELS, 
            num_rules = N_RULES,
            hidden_dim=HIDDEN_DIM, 
            dropout=DROPOUT, 
            device=DEVICE, temperature=1)
    
    model_gmix = MixtureNCANoise(update_nets=standard_update_net, 
                state_dim=N_CHANNELS, 
                num_rules = N_RULES,
                hidden_dim=HIDDEN_DIM, 
                dropout=DROPOUT, 
                device=DEVICE, temperature=1)


    # Train  models
    print("Training standard model...")
    loss_nca = train_nca(model, target, device=DEVICE, 
           num_steps=NUM_STEPS, 
           milestones=MILESTONES, 
           learning_rate=LEARNING_RATE, 
           gamma=GAMMA, 
           decay=DECAY, 
           total_steps=TOTAL_STEPS, 
           print_every=PRINT_EVERY, 
           batch_size = BATCH_SIZE, 
           state_dim = N_CHANNELS, 
           seed_loc = SEED_LOC, 
           pool_size = POOL_SIZE
           )

    print("Training mixture model...")
    loss_mix_nca = train_nca(model_mix, target, device=DEVICE, 
           num_steps=NUM_STEPS, 
           milestones=MILESTONES, 
           learning_rate=LEARNING_RATE, 
           gamma=GAMMA, 
           decay=DECAY, 
           total_steps=TOTAL_STEPS, 
           print_every=PRINT_EVERY, 
           batch_size = BATCH_SIZE, 
           state_dim = N_CHANNELS, 
           seed_loc = SEED_LOC, 
           pool_size = POOL_SIZE,
            temperature=1, min_temperature=1, anneal_rate=0.001,
                       straight_through = True

                       )
    # Train stochastic mixture model
    print("Training stochastic mixture model...")
    loss_gmix = train_nca(model_gmix, target, device=DEVICE, 
           num_steps=NUM_STEPS, 
           milestones=MILESTONES, 
           learning_rate=LEARNING_RATE, 
           gamma=GAMMA, 
           decay=DECAY, 
           total_steps=TOTAL_STEPS, 
           print_every=PRINT_EVERY, 
           batch_size = BATCH_SIZE, 
           state_dim = N_CHANNELS, 
           seed_loc = SEED_LOC, 
           pool_size = POOL_SIZE,
            temperature=1, min_temperature=1, anneal_rate=0.001,
                       straight_through = True

            
                       )

    # Save models
    torch.save(model.state_dict(), os.path.join(experiment_dir, f'standard_model.pt'))
    torch.save(model_mix.state_dict(), os.path.join(experiment_dir, f'mixture_model.pt'))
    torch.save(model_gmix.state_dict(), os.path.join(experiment_dir, f'mixture_model_noise.pt'))

    # Save loss history
    with open(os.path.join(experiment_dir, f'loss_history.json'), 'w') as f:
        json.dump({
            'standard': loss_nca['loss'],
            'mixture': loss_mix_nca['loss'],
            'mixture_noise': loss_gmix['loss']
        }, f)

    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(loss_nca['loss'], label='Standard NCA')
    plt.plot(loss_mix_nca['loss'], label='Mixture NCA')
    plt.plot(loss_gmix['loss'], label='Stochastic Mixture NCA')
    plt.legend()
    # plot y log scale
    plt.yscale('log')
    plt.title('Training Loss')
    plt.savefig(os.path.join(experiment_dir, 'training_loss.png'))
    plt.close()

        # Construct a video of our CA's growth
    #init_state = torch.zeros(1, N_CHANNELS, *target.shape[-2:]).to(DEVICE)
    init_state = target.to(DEVICE)
    init_state[...,3:, SEED_LOC[0], SEED_LOC[1]] = 1 # initially, there is just one cell

    #init_state_nca = mixture_model(init_state, 400, SEED_LOC, return_history=False).detach()
    # fill extra channel dimension with zeros to match standard nca
    init_state_nca = torch.cat([init_state, torch.zeros(1, N_CHANNELS - 4, *init_state.shape[-2:]).to(DEVICE)], dim=1)

    # Robustness analysis 
    robustness_analysis = RobustnessAnalysis(
    standard_nca=model.to(DEVICE)   ,
    mixture_nca=model_mix.to(DEVICE),
    stochastic_mixture_nca=model_gmix.to(DEVICE),
    device=DEVICE
    )

    # Test with different deletion sizes
    deletion_sizes = [5, 10]  # Test different levels of deletion
    deletion_results = {}

    print("\nTesting deletion robustness...")
    for size in deletion_sizes:
        deletion_results[size] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            'deletion',
            n_runs=50,
            size=size,
            steps=100,
            seed=SEED_LOC
        )
        with open(os.path.join(experiment_dir, f'deletion_{size}.pkl'), 'wb') as f:
            pickle.dump(deletion_results[size], f)

        # Plot results for each deletion size
        fig, ax = robustness_analysis.visualize_stored_results(
            results=deletion_results[size], 
            plot_type='error',
            figsize=(10, 5)
        )
        fig.savefig(os.path.join(experiment_dir, f'deletion_error_{size}.png'))
        plt.close()

        fig, ax = robustness_analysis.visualize_stored_results(
            results=deletion_results[size],
            plot_type='trajectories'
        )
        fig.savefig(os.path.join(experiment_dir, f'deletion_trajectories_{size}.png'))
        plt.close()

    # Test with different noise levels
    noise_levels = [0.1, 0.25]  # Test different levels of noise
    noise_results = {}

    print("\nTesting noise robustness...")
    for noise_level in noise_levels:
        noise_results[noise_level] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            'noise',
            n_runs=50,
            noise_level=noise_level,
            steps=100,
            seed=SEED_LOC
        )
        with open(os.path.join(experiment_dir, f'noise_{noise_level}.pkl'), 'wb') as f:
            pickle.dump(noise_results[noise_level], f)

        # Plot results for each noise level
        fig, ax = robustness_analysis.visualize_stored_results(
            results=noise_results[noise_level],
            plot_type='error',
            figsize=(10, 5)
        )
        fig.savefig(os.path.join(experiment_dir, f'noise_error_{noise_level}.png'))
        plt.close()

        fig, ax = robustness_analysis.visualize_stored_results(
            results=noise_results[noise_level],
            plot_type='trajectories'
        )
        fig.savefig(os.path.join(experiment_dir, f'noise_trajectories_{noise_level}.png'))
        plt.close()

    # Test with different numbers of masked pixels
    pixel_counts = [100, 500]
    results = {}
    
    print("\nTesting pixel removal robustness...")
    # Compute results
    for n_pixels in pixel_counts:
        results[n_pixels] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            perturbation_type='random_pixels',
            steps=100,
            seed=SEED_LOC,
            n_runs=50,
            n_pixels=n_pixels
        )
        with open(os.path.join(experiment_dir, f'pixel_removal_{n_pixels}.pkl'), 'wb') as f:
            pickle.dump(results[n_pixels], f)

    # Plot recovery trajectories for different pixel counts
    for i, n_pixels in enumerate(pixel_counts):
        fig, _ = robustness_analysis.visualize_stored_results(
            results[n_pixels], 
            plot_type='error', 
            figsize=(10, 5)
        )
        fig.savefig(os.path.join(experiment_dir, f'pixel_removal_error_{n_pixels}.png'))
        plt.close()

        fig, _ = robustness_analysis.visualize_stored_results(results=results[n_pixels], plot_type='trajectories')
        fig.savefig(os.path.join(experiment_dir, f'pixel_removal_trajectories_{n_pixels}.png'))
        plt.close()

    # Create a summary CSV with all replicates
    print("\nCreating detailed summary CSV...")
    
    # Initialize lists to store results for each experiment
    all_results = [
        *[(f'Deletion {size}', deletion_results[size]) for size in deletion_sizes],
        *[(f'Noise {level}', noise_results[level]) for level in noise_levels],
        *[(f'{n} Masked Pixels', results[n]) for n in pixel_counts]
    ]

    # Create CSV file
    csv_path = os.path.join(experiment_dir, 'detailed_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('Perturbation Type,Model Type,Replicate,Final Error\n')
        
        # Write data rows
        for perturb_name, result in all_results:
            for model_name, metrics in [
                ('Standard', result['standard_metrics']),
                ('Mixture', result['mixture_metrics']), 
                ('Stochastic', result['stochastic_metrics'])
            ]:
                for rep_idx, metric in enumerate(metrics):
                    row = [
                        perturb_name,
                        model_name,
                        str(rep_idx),
                        f"{metric['final_error']:.4f}"
                    ]
                    f.write(','.join(row) + '\n')

    print(f"Detailed metrics saved to {csv_path}")

    # Save final training losses
    final_losses = {
        'emoji_code': emoji_code,
        'standard_final_loss': loss_nca['loss'][-1],
        'mixture_final_loss': loss_mix_nca['loss'][-1],
        'stochastic_final_loss': loss_gmix['loss'][-1]
    }
    
    losses_path = os.path.join(experiment_dir, 'final_losses.csv')
    with open(losses_path, 'w') as f:
        f.write('emoji_code,standard_final_loss,mixture_final_loss,stochastic_final_loss\n')
        f.write(f"{emoji_code},{final_losses['standard_final_loss']:.6f},"
                f"{final_losses['mixture_final_loss']:.6f},"
                f"{final_losses['stochastic_final_loss']:.6f}\n")

    print(f"Final losses saved to {losses_path}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emoji_code', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    run_experiment(args.emoji_code, args.output_dir)