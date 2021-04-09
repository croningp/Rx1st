# Rx1st
Reactivity first exploration of chemical space.

## Installation
1. Clone the repository
```shell
git clone https://github.com/croningp/Rx1st
```

2. Install dependencies using conda. We recommend using the [Miniforge] distribution.
```shell
# First, make sure both conda-forge and anaconda channels are enabled
conda config --add channels conda-forge
conda config --add channels anaconda
# Create and activate a conda environment with dependencies installed 
conda create -n Rx1st --file Rx1st/conda_list.txt
conda activate Rx1st
# Install kernel for Rx1st environment in Jupyter
python -m ipykernel install --user --name=Rx1st
```

3. Clone and install the [junction tree variational autoencoder][JTNN_VAE] package. You will also need to download and extract the relevant dataset to the correct location.
```shell
git clone https://github.com/croningp/JTNN_VAE
cd JTNN_VAE

# Download JTNN_VAE dataset; on Unix
wget "https://zenodo.org/record/4670997/files/jtnn_data.tar.xz?download=1"
# On Windows 10 using Powershell
Invoke-WebRequest -Uri "https://zenodo.org/record/4670997/files/jtnn_data.tar.xz?download=1" -OutFile jttn_data.tar.xz

# Extract JTNN dataset
tar xvf jtnn_data.tar.xz

pip install -e .
cd ..
```

4. Download and extract the Rx1st dataset.
```shell
cd Rx1st

# On Unix
wget "https://zenodo.org/record/4670997/files/exploration_data.tar.xz?download=1"
# On Windows 10 using Powershell
Invoke-WebRequest -Uri "https://zenodo.org/record/4670997/files/exploration_data.tar.xz?download=1" -OutFile exploration_data.tar.xz

# Extract Rx1st dataset
tar xvf exploration_data.tar.xz
```

5. Launch Jupyter lab, where you can open the included 

## Reproducing manuscript figures
The Jupyter notebook `Exploration.ipynb` contains the code to reproduce the figures in the manuscript linked below.

## Publications
- D. Caramelli, et al. _A Reactivity First Approach to Autonomous Discovery of New Chemistry_ ([preprint])

[JTNN_VAE]: https://github.com/croningp/JTNN_VAE
[Miniforge]: https://github.com/conda-forge/miniforge
[preprint]: https://chemrxiv.org/articles/preprint/An_Artificial_Intelligence_that_Discovers_Unpredictable_Chemical_Reactions/12924968