# Rx1st
Reactivity first exploration of chemical space.

## Installation
1. Clone the repository
```shell
git clone https://github.com/croningp/Rx1st
```

2. Prepare Python environment and dependencies using conda. We recommend using the [Miniforge] distribution.
```shell
# Create and activate a conda environment with dependencies installed 
# GPU
conda env create --file Rx1st/environment-gpu.yml
# or CPU
conda env create --file Rx1st/environment-cpu.yml

conda activate Rx1st
# Install kernel for Rx1st environment in Jupyter
python -m ipykernel install --user --name=Rx1st
```

3. Clone and install the [junction tree variational autoencoder][JTNN_VAE] package. You will also need to download and extract the relevant dataset to the correct location.
```shell
git clone https://github.com/croningp/JTNN_VAE
cd JTNN_VAE

# Download JTNN_VAE dataset; on Unix
curl -o jtnn_data.tar.xz "https://zenodo.org/record/4670997/files/jtnn_data.tar.xz?download=1"
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
curl -o exploration_data.tar.xz "https://zenodo.org/record/4670997/files/exploration_data.tar.xz?download=1"
# On Windows 10 using Powershell
Invoke-WebRequest -Uri "https://zenodo.org/record/4670997/files/exploration_data.tar.xz?download=1" -OutFile exploration_data.tar.xz

# Extract Rx1st dataset
tar xvf exploration_data.tar.xz
```

5. Launch Jupyter lab, where you can open the included Jupyter notebooks (see below).
```shell
jupyter lab
```

6. When opening a notebook make sure the correct kernel (Rx1st) is displayed at the top right corner of the Jupyter notebook as shown in the screenshot below. If not, click on the kernel name and select "Rx1st"
from the kernel list.

![Active kernel](kernel.png)

## Reproducing manuscript figures
- The Jupyter notebook `Rx1st.ipynb` contains the code to reproduce the figures in the manuscript linked below.
- `Novelty estimation.ipynb` and `Chemical space modelling.ipynb` contain the cheminformatic analysis the novelty and unpredictability of the trimer discovery, respectively.

## Publications
- D. Caramelli, et al. _A Reactivity First Approach to Autonomous Discovery of New Chemistry_ ([preprint])

[JTNN_VAE]: https://github.com/croningp/JTNN_VAE
[Miniforge]: https://github.com/conda-forge/miniforge
[preprint]: https://chemrxiv.org/articles/preprint/An_Artificial_Intelligence_that_Discovers_Unpredictable_Chemical_Reactions/12924968