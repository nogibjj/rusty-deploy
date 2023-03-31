#!/bin/bash
# Description: install pytorch in linux

#Install cmake
sudo apt-get update && sudo apt-get install -y cmake
# Set Torch_DIR and CMAKE_PREFIX_PATH
export Torch_DIR=/usr/local/lib/libtorch/share/cmake/Torch
export CMAKE_PREFIX_PATH=$Torch_DIR:$CMAKE_PREFIX_PATH

# Source ~/.bashrc to ensure the environment variable is available in the current session
source ~/.bashrc

# Installs PyTorch in Linux
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
mv libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Move libtorch to /usr/local/lib
if [ -d "/usr/local/lib/libtorch" ]; then
  echo "/usr/local/lib/libtorch already exists. Deleting it to proceed with the installation."
  sudo rm -rf /usr/local/lib/libtorch
fi
sudo mv libtorch /usr/local/lib

# Export LIBTORCH and LD_LIBRARY_PATH
export LIBTORCH=/usr/local/lib/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Check if ~/.bashrc has the following lines
# export LIBTORCH=/path/to/libtorch
# export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
if grep -q "export LIBTORCH" ~/.bashrc; then
  echo "LIBTORCH is already in ~/.bashrc"
else
  echo "export LIBTORCH=/usr/local/lib/libtorch" >> ~/.bashrc
fi
if grep -q "export LD_LIBRARY_PATH" ~/.bashrc; then
  echo "LD_LIBRARY_PATH is already in ~/.bashrc"
else
  echo 'export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Source the updated .bashrc
source ~/.bashrc

echo "Checking if libtorch is installed correctly..."
echo "Verifying that C++ PyTorch API is functional..."

cat > check_pytorch.cpp << EOL
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    return 0;
}
EOL

cmake -B build
cmake --build build --config Release
./build/check_pytorch

if [ $? -eq 0 ]; then
  echo "C++ PyTorch API is functional. Installation complete."
else
  echo "C++ PyTorch API is not functional. Please check the configuration and try again."
fi