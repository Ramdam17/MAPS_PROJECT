#!/bin/bash

# Fix METTA environment and rebuild mettagrid
set -e  # Exit on any error

echo "=== Updating conda packages ==="
conda update --all -y
conda install -c conda-forge libstdcxx-ng=12 -y

echo "=== Setting up environment variables for system libraries ==="
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo "=== Checking library versions ==="
echo "Conda libstdc++ version:"
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX | tail -3
echo "System libstdc++ version:"
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -3

echo "=== Cleaning existing installations ==="
# Remove problematic packages
pip uninstall pufferlib mettagrid -y || true

# Clean build artifacts from main directory
cd ~/MSc_CS/MAPS_PROJECT/METTA/metta
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -delete

echo "=== Cleaning mettagrid build ==="
cd ~/MSc_CS/MAPS_PROJECT/METTA/metta/mettagrid
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -delete

echo "=== Installing build dependencies ==="
pip install --upgrade pip setuptools wheel
pip install pybind11 cython

echo "=== Installing PufferLib separately ==="
pip install git+https://github.com/Metta-AI/PufferLib.git@d3cde3f4107daaf8e195bf87f55742d7cd026ddb --no-cache-dir

echo "=== Rebuilding mettagrid with system libraries ==="
cd ~/MSc_CS/MAPS_PROJECT/METTA/metta/mettagrid
pip install -e . --no-cache-dir --force-reinstall --verbose

echo "=== Installing main METTA project ==="
cd ~/MSc_CS/MAPS_PROJECT/METTA/metta
uv pip install -e . --no-cache-dir

echo "=== Testing the installation ==="
python -c "from mettagrid.mettagrid_c import MettaGrid; print('✓ mettagrid_c import successful')"

echo "=== Running training test ==="
python tools/train.py run=my_experiment +hardware=macbook wandb=off --help

echo "=== Setup complete! ==="
echo "You can now run: python tools/train.py run=my_experiment +hardware=macbook wandb=off"