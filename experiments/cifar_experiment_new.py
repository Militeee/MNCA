import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import sys
import pickle
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

sys.path.append('..')
from mix_NCA.NCA import NCA
from mix_NCA.utils_images import train_nca, standard_update_net
from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.RobustnessAnalysis import RobustnessAnalysis
from mix_NCA.MixtureNCANoise import MixtureNCANoise

def load_samples(data_dir='../data', category_idx=0, seed=42):
    """Load samples from CIFAR-10
    
    Args:
        data_dir: Root directory for the dataset
        category_idx: Index of category to use
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Download and load the dataset
    dataset = datasets.CIFAR10(
        root=data_dir,
        download=True
    )
    
    # Get unique categories
    categories = sorted(dataset.classes)
    if category_idx >= len(categories):
        raise ValueError(f"Category index {category_idx} out of range (0-{len(categories)-1})")
    
    selected_category = categories[category_idx]
    
    # Find all samples for the selected category
    samples = []
    sample_indices = []  # Track which samples we've found
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if categories[label] == selected_category:
            # Convert to numpy arrays
            img_np = np.array(img) / 255.0
            
            # Create RGBA image
            rgba = np.zeros((*img_np.shape[:2], 4))
            rgba[..., :3] = img_np
            rgba[..., 3] = 1.
            
            samples.append(rgba)
            sample_indices.append(idx)
            
            if len(samples) == 1:  
                break
    
    print(f"Loaded {len(samples)} samples from category: {selected_category}")
    
    # Return samples with their individual indices as labels
    return samples, sample_indices, [selected_category] * len(samples)

def process_image(img_array, target_size=50, padding=6):
    """Convert numpy array to padded tensor"""
    # Convert to tensor and ensure correct shape
    img_tensor = torch.from_numpy(img_array).float()
    
    # Reshape to [B, C, H, W] format for interpolation
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
    
    # Resize if needed
    if img_tensor.shape[-2:] != (target_size, target_size):
        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    # Add padding
    padded_tensor = torch.nn.functional.pad(
        img_tensor,
        (padding, padding, padding, padding),
        mode='constant',
        value=0
    )
    
    return padded_tensor

def run_experiment(sample_idx, samples, labels, class_names, output_dir):
    # Set up training parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_SIZE = 32
    N_CHANNELS = 16
    BATCH_SIZE = 8
    POOL_SIZE = 1000
    NUM_STEPS = [30, 50]
    SEED_LOC = (20,20)
    GAMMA = 0.2
    DECAY = 3e-5 
    DROPOUT = 0.2
    HIDDEN_DIM = 64
    TOTAL_STEPS = 8000
    PRINT_EVERY = 200
    SEED = 3
    MILESTONES = [4000, 6000, 7000]
    LEARNING_RATE = 1e-3
    N_RULES = 6

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create experiment directory - use category name and sample index
    class_name = f"{class_names[0]}_{sample_idx}"  # class_names[0] is the category name
    experiment_dir = os.path.join(output_dir, f"experiment_{class_name}_robustness")
    experiment_dir_old = os.path.join("../results", f"experiment_{class_name}_robustness")
    os.makedirs(experiment_dir, exist_ok=True)

    # Process image
    target = process_image(samples[sample_idx], target_size=TARGET_SIZE).to(DEVICE)
    # Initialize models
    model = NCA(update_net=standard_update_net(N_CHANNELS* 3, HIDDEN_DIM, N_CHANNELS * 2, device=DEVICE), 
            state_dim=N_CHANNELS, 
            hidden_dim=HIDDEN_DIM, 
            dropout=DROPOUT, 
            device=DEVICE,
            distribution = "normal")
    
    # Mixture models: load from checkpoint
    model_mix = MixtureNCA(update_nets=standard_update_net, 
            state_dim=N_CHANNELS, 
            num_rules=N_RULES,
            hidden_dim=HIDDEN_DIM, 
            dropout=DROPOUT, 
            device=DEVICE)
    model_mix.load_state_dict(torch.load(os.path.join(experiment_dir_old, 'mixture_model.pt')))

    model_gmix = MixtureNCANoise(update_nets=standard_update_net, 
                state_dim=N_CHANNELS, 
                num_rules=N_RULES,
                hidden_dim=HIDDEN_DIM, 
                dropout=DROPOUT, 
                device=DEVICE)
    model_gmix.load_state_dict(torch.load(os.path.join(experiment_dir_old, 'mixture_model_noise.pt')))

    # Train models
    print(f"Training models for class: {class_name}")
    print("Training standard model...")
    loss_nca = train_nca(model, target, device=DEVICE, 
           num_steps=NUM_STEPS, 
           milestones=MILESTONES, 
           learning_rate=LEARNING_RATE, 
           gamma=GAMMA, 
           decay=DECAY, 
           total_steps=TOTAL_STEPS, 
           print_every=PRINT_EVERY, 
           batch_size=BATCH_SIZE, 
           state_dim=N_CHANNELS, 
           seed_loc=SEED_LOC, 
           pool_size=POOL_SIZE
           )

    # Save models
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'standard_model_with_noise.pt'))

    # Save loss history
    with open(os.path.join(experiment_dir, 'loss_history.json'), 'w') as f:
        json.dump({
            'standard': loss_nca['loss']
        }, f)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(loss_nca['loss'], label='Standard NCA')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Training Loss - {class_name}')
    plt.savefig(os.path.join(experiment_dir, 'training_loss.png'))
    plt.close()

    # Initialize state for robustness analysis
    init_state = target.to(DEVICE)
    init_state[...,3:, SEED_LOC[0], SEED_LOC[1]] = 1
    init_state_nca = torch.cat([init_state, torch.zeros(1, N_CHANNELS - 4, *init_state.shape[-2:]).to(DEVICE)], dim=1)

    # Robustness analysis
    robustness_analysis = RobustnessAnalysis(
        standard_nca=model.to(DEVICE),
        mixture_nca=model_mix.to(DEVICE),
        stochastic_mixture_nca=model_gmix.to(DEVICE),
        device=DEVICE
    )

    # Test different perturbation types
    print(f"\nRunning robustness analysis for {class_name}...")

    # Deletion tests
    deletion_sizes = [5, 10]
    deletion_results = {}
    for size in deletion_sizes:
        print(f"Testing deletion size {size}...")
        deletion_results[size] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            'deletion',
            n_runs=50,
            size=size,
            steps=200,
            seed=SEED_LOC
        )
        with open(os.path.join(experiment_dir, f'deletion_{size}.pkl'), 'wb') as f:
            pickle.dump(deletion_results[size], f)

        # Plot results
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

    # Noise tests
    noise_levels = [0.1, 0.25]
    noise_results = {}
    for noise_level in noise_levels:
        print(f"Testing noise level {noise_level}...")
        noise_results[noise_level] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            'noise',
            n_runs=50,
            noise_level=noise_level,
            steps=200,
            seed=SEED_LOC
        )
        with open(os.path.join(experiment_dir, f'noise_{noise_level}.pkl'), 'wb') as f:
            pickle.dump(noise_results[noise_level], f)

        # Plot results
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

    # Random pixel removal tests
    pixel_counts = [100, 500]
    results = {}
    for n_pixels in pixel_counts:
        print(f"Testing pixel removal count {n_pixels}...")
        results[n_pixels] = robustness_analysis.compute_robustness_metrics(
            init_state_nca,
            perturbation_type='random_pixels',
            steps=200,
            seed=SEED_LOC,
            n_runs=50,
            n_pixels=n_pixels
        )
        with open(os.path.join(experiment_dir, f'pixel_removal_{n_pixels}.pkl'), 'wb') as f:
            pickle.dump(results[n_pixels], f)

        # Plot results
        fig, ax = robustness_analysis.visualize_stored_results(
            results=results[n_pixels],
            plot_type='error',
            figsize=(10, 5)
        )
        fig.savefig(os.path.join(experiment_dir, f'pixel_removal_error_{n_pixels}.png'))
        plt.close()

        fig, ax = robustness_analysis.visualize_stored_results(
            results=results[n_pixels],
            plot_type='trajectories'
        )
        fig.savefig(os.path.join(experiment_dir, f'pixel_removal_trajectories_{n_pixels}.png'))
        plt.close()

    # Create summary CSV
    print("\nCreating summary CSV...")
    csv_path = os.path.join(experiment_dir, 'detailed_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('Perturbation Type,Model Type,Replicate,Final Error\n')
        
        # Combine all results
        all_results = [
            *[(f'Deletion {size}', deletion_results[size]) for size, u in deletion_results.items()],
            *[(f'Noise {level}', noise_results[level]) for level, u in noise_results.items()],
            *[(f'{n} Masked Pixels', results[n]) for n, u in results.items()]
        ]
        
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


    print(f"Experiment completed for {class_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load pet samples
    samples, labels, class_names = load_samples(args.data_dir, args.category)
    
    # Run experiment for each sample in category
    for i in range(len(samples)):
        run_experiment(i, samples, labels, class_names, args.output_dir) 