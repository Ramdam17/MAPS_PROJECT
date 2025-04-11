# MAPS: A Metacognitive Architecture for Improved Social Learning

This repository contains the implementation of MAPS (Metacognitive Architecture for Improved Social Learning) as described in our paper published in the Proceedings of the International Workshop on Advancing AI Through Theory of Mind, 2025.

## Authors
- Juan David Vargas (Université de Montréal, MILA - Quebec AI Institute, CHU Sainte-Justine)
- Natalie Kastel (CHU Sainte-Justine, MILA - Quebec AI Institute, Université de Montréal)
- Antoine Pasquali (Université libre de Bruxelles)
- Axel Cleeremans (CrossLabs)
- Zahra Sheikhbahaee* (CHU Sainte-Justine, MILA - Quebec AI Institute)
- Guillaume Dumas* (CHU Sainte-Justine, MILA - Quebec AI Institute, Université de Montréal)

*Co-Senior Authors

## Overview

MAPS is a novel architecture that combines metacognitive components with advanced learning techniques. The key components include:

1. A secondary network (2nd-Net) with a comparator matrix connected to wagering units
2. A cascade model facilitating graded accumulation of activation

Our experiments demonstrate significant performance improvements across multiple domains including Blindsight, Artificial Grammar Learning (AGL), Single-Agent Reinforcement Learning (SARL), and Multi-Agent Reinforcement Learning (MARL).

## Methodology

- **Primary Network**: Main neural network with contrastive loss
- **Secondary Network**: Comparator matrix connected to two wagering units
- **Cascade Model**: Facilitates graded accumulation of activation (typically 50 iterations)
- **SARL Implementation**: DQN framework with convolutional layers, autoencoder, and replay buffer
- **MARL Implementation**: MAPPO framework with convolutional layers, sinusoidal-based relative positional encoding, and Gated Recurrent Unit (GRU)

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/yourusername/maps.git
cd maps
pip install -r requirements.txt
```

Additionall requirements for SARL:

```bash
cd SARL/MinAtar
pip install .
```

Additionall requirements for MARL:

```bash
cd MARL/MAPPO-ATTENTIOAN
pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir numpy 
pip install --no-index --no-cache-dir opencv-python
pip install --no-index --no-cache-dir ml-collections
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index wandb
pip install dm-env
pip install pygame
install DeepMind Lab2D https://github.com/deepmind/lab2d
wget https://files.pythonhosted.org/packages/4b/31/884879224de4627b5d45b307cec8f4cd1e60db9aa61871e4aa2518c6584b/dmlab2d-1.0.0_dev.10-cp310-cp310-manylinux_2_31_x86_64.whl -O dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl
setrpaths.sh --path dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
pip install dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
pip install --no-index libcst
git clone -b main https://github.com/deepmind/meltingpot
cd meltingpot
pip install --editable .[dev]
pip install  --no-index --no-cache-dir dm-acme
pip install -U pip
pip install "ray[cpp]" --no-index
```


## Experiments

This repository includes code for reproducing the experiments described in the paper:

### 1. Blindsight

```bash
cd BLINDSIGHT
python Blindsight_TMLR.py
```

### 2. Artificial Grammar Learning (AGL)

```bash
cd AGL
python AGL_TMLR.py
```

### 3. Single-Agent Reinforcement Learning (SARL)

The SARL experiments are conducted on MinAtar environments, specifically "Seaquest" and "Asterix". Instructions for running these experiments will be provided separately.

### 4. Multi-Agent Reinforcement Learning (MARL)

MARL experiments include Harvest Cleaner, Harvest Planter, Chemistry 3D, and Territory Inside Out environments. Instructions for running these experiments will be provided separately.

## Results Summary

Our results demonstrate significant improvements using the MAPS architecture:

1. **Blindsight**: Achieved 97% accuracy (significant improvement with Z-score: 8.6)
2. **AGL**: Achieved 62% accuracy (significant improvement with Z-score: 15.0)
3. **MinAtar SARL**: 
   - Seaquest: 6.15 rewards (Z-score: 2.97)
   - Asterix: 5.77 rewards (Z-score: 2.15)
4. **MARL**: Notable improvement in Territory Inside Out (Z-score: 2.59)

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{vargas2025maps,
  title={MAPS - A Metacognitive Architecture for Improved Social Learning},
  author={Vargas, Juan David and Kastel, Natalie and Pasquali, Antoine and Cleeremans, Axel and Sheikhbahaee, Zahra and Dumas, Guillaume},
  booktitle={Proceedings of the International Workshop on Advancing AI Through Theory of Mind},
  year={2025}
}
```
