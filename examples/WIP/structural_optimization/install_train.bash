#!/bin/bash
# filepath: /home/michael/Code/projects/ml_dev/CSDmL/examples/WIP/structural_optimization/install.bash
# This script installs stuff for training the nets & visualizations.
# It assumes that conda is installed and accessible from your PATH.

# Usage: ./install.bash [env_name]
# If no argument is provided, the default environment name "train" is used.
ENV_NAME="${1:-train}"

# Exit immediately if a command exits with a non-zero status (except in the case below).
set -e

# 1. Create a conda environment with Python 3.9.10 (or try 3.9.7 if unavailable)
echo "Creating conda environment '$ENV_NAME' with Python 3.9.10..."
if ! conda create -y -n "$ENV_NAME" python=3.9.10; then
    echo "Python 3.9.10 not available, trying Python 3.9.7..."
    conda create -y -n "$ENV_NAME" python=3.9.7 || { echo "Failed to create environment with Python 3.9.7. Exiting."; exit 1; }
fi

# 2. Activate the conda environment
echo "Activating the conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# # 3. Install FEniCSx from conda-forge
# echo "Installing FEniCSx version 0.5.1 from conda-forge..."
# conda install -y -c conda-forge fenics-dolfinx=0.5.1

# 4. Install CSDL_alpha and ModOpt using pip
# echo "Installing CSDL_alpha..."
# pip install git+https://github.com/LSDOlab/CSDL_alpha.git
echo "Installing ModOpt..."
pip install git+https://github.com/LSDOlab/modopt.git

# 5. Install caddee_alpha using pip
# this should also install CSDL_alpha, lsdo_function_spaces
echo "Installing caddee_alpha..."
# git clone https://github.com/LSDOlab/CADDEE_alpha.git
# pip install ./CADDEE_alpha
pip install git+https://github.com/LSDOlab/CADDEE_alpha.git

# 5.5 Install specific branch of lsdo_function_spaces
echo "Installing lsdo_function_spaces..."
pip uninstall lsdo_function_spaces
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git@patch_parametric_maps


# # 6. Install femo_alpha as a user
# echo "Installing femo_alpha..."
# pip install -U git+https://github.com/LSDOlab/femo_alpha.git

# # 6. Install lsdo_function_spaces using pip
# echo "Installing lsdo_function_spaces..."
# pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git

# 7. Install jax using pip
echo "Installing jax..."
pip install -U jax

# 8. Install optax using pip
echo "Installing optax..."
pip install -U optax

# Optional: For a developer installation uncomment below
# echo "Cloning femo_alpha repo for developer installation..."
# git clone https://github.com/LSDOlab/femo_alpha.git
# cd femo_alpha
# pip install -e .
# cd ..

echo "Installation complete."
echo "Make sure to install CSDmL before running the examples."
echo "To activate the environment, run 'conda activate $ENV_NAME'."