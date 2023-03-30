#!/bin/bash
# Description: install pytorch in linux

# Installs PyTorch in Linux
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip
mv libtorch-shared-with-deps-2.0.0%2Bcpu.zip libtorch-shared-with-deps-2.0.0+cpu.zip
unzip libtorch-shared-with-deps-2.0.0+cpu.zip
rm libtorch-shared-with-deps-2.0.0+cpu.zip
# Move libtorch to /usr/local/lib
sudo mv libtorch /usr/local/lib
# Export LIBTORCH and LD_LIBRARY_PATH
export LIBTORCH=/usr/local/lib/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Check to see if ~/.bashrc has the following lines
# export LIBTORCH=/path/to/libtorch
# export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
if grep -q "LIBTORCH" ~/.bashrc; then
    echo "LIBTORCH is already in ~/.bashrc"
else
    echo "export LIBTORCH=/usr/local/lib/libtorch" >> ~/.bashrc
fi
if grep -q "LD_LIBRARY_PATH" ~/.bashrc; then
    echo "LD_LIBRARY_PATH is already in ~/.bashrc"
else
    echo 'export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Define check_libtorch alias if it doesn't exist in ~/.bashrc
if grep -q "check_libtorch" ~/.bashrc; then
    echo "check_libtorch alias is already in ~/.bashrc"
else
    echo 'alias check_libtorch="echo \"#include <torch/torch.h>\" | g++ -xc++ - -o /dev/null -v -L$LIBTORCH/lib -ltorch -lc10 -lstdc++ -Wl,-rpath=$LIBTORCH/lib 2>&1 | grep \"successfully\|not found\""' >> ~/.bashrc
fi

# Check if libtorch is installed correctly
echo "Checking if libtorch is installed correctly..."
check_libtorch
