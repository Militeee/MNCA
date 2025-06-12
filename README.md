# Mixture of Neural Cellular Automata

This repository implements the Mixture of Neural Cellular Automata (MNCA) and the Mixture of Neural Cellular Automata with internal noise (MNCA-N). Two approaches to add stochasticity to the Neural Cellular Automata (NCA).

## Overview

The project implements several variants of Neural Cellular Automata:
- Standard NCA: Original NCA implementation of 
- Mixture NCA: Uses a mixture of multiple update rules
- Mixture NCA with internal noise: Adds an internal stochastic noise

## Features

- Multiple NCA architectures with configurable parameters
- Support for both image and patch-like inputs
- Flexible grid types (square and hexagonal)
- Various perception filters (Sobel, Laplacian)
- Robustness analysis tools
- Visualization utilities

## Project Structure

```
├── mix_NCA/               # Core NCAs implementations 
│   ├── NCA.py            # Base NCA implementation (eq. in the paper)
│   ├── MixtureNCA.py # Finite Mixture of NCAs (eq. in the paper)
│   ├── MixtureNCANoise.py #  Finite Mixture of NCAs with internal noise (eq. in the paper)
│   ├── utils_*.py        # Utility functions for fitting and plotting the data for each experiment
|   ├── TissueModel.py   # Class that implements the tissue simulation 
|   ├── AGB_ABC_model.py   # Implementation of the agent based model with ABC parameter inference
|   ├── BiologicalMetrics.py   # Implementation of the metrics used to evaluate the models on the tissue simulations
|   ├── RobustnessAnalysis.py   # Implementation of the perturbation analysis on image
|   
├── experiments/          # Experiment scripts for emoji experiments
|   ├── emoji_experiment.py # Experiment script for the emoji experiment
|   ├── cifar_experiment.py # Experiment script for the CIFAR-10 experiment
|   ├── emoji_experiment_new.py # Experiment script for the CIFAR-10 experiment for GCA
|   ├── cifar_experiment_new.py # Experiment script for the CIFAR-10 experiment for GCA
|   
├── notebooks/           # Jupyter notebooks for analysis of biological simuations and Visium
|   ├── tissue_simulation_MNCA.ipynb # Notebook for the tissue simulation with the mixture NCA
|   ├── tissue_simulation_other_models.ipynb # Notebook for the tissue simulation with the ABC models
|   ├── experiment_microscopy.ipynb # Notebook for to perform the microscopy experiment
|   ├── final_stats.ipynb # Notebook to generate the latex table with the final statistics
|   ├── lip_stability_analysis.ipynb # Notebook for the analysis of the Lipschitz upper bound and attractors after perturbation
|   ├── plots_paper.ipynb # Extra notebook to generate some of the plots in the paper
|   ├── histories.npy # Tissue simulation data used for the training
|
├── models/              # Saved model weights
├── results/             # Experiment results
|── figures/             # Figures
|── data/                # Data for the experiments
```

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic example of using the NCA models:

```python
from mix_NCA.NCA import NCA
from mix_NCA.utils_simulations import classification_update_net

# Create a standard NCA
model = NCA(
    update_net=classification_update_net,
    state_dim=16,
    hidden_dim=128,
    device="cuda"
)

# For mixture models
from mix_NCA.MixtureNCA import MixtureNCA
mixture_model = MixtureNCA(
    update_nets=classification_update_net,
    state_dim=16,
    num_rules=5,
    hidden_dim=128,
    device="cuda"
)
```

## Experiments

The repository includes several experiments:
- Emoji pattern generation
- Biological simulations
- Robustness analysis
- Pattern formation studies

## Documentation

Detailed documentation is available in the `docs/` directory. For more information about specific components:

- See `notebooks/` for example usage and analysis
- Check `experiments/` for running specific experiments
- Refer to individual module docstrings for API details

## License

MIT License

## Citation

If you use this code in your research, please cite:

XXX

