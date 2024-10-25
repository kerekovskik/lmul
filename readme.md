# What is this 

This repo implements the algorithm from the paper titled 
[ADDITION IS ALL YOU NEED FOR ENERGY-EFFICIENT LANGUAGE MODELS](https://arxiv.org/abs/2410.00907)



## First, ensure you have Rust and Python development tools installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

## Create a virtual environment and source it 
deactivate ; python3.12 -m venv .venv && source ./.venv/bin/activate

## Install packages required for building the python package


### Build the package
maturin build --release


### Install the package in development mode
pip install -e .

### Or install the wheel directly
pip install dist/lmul-0.1.0-cp39-cp39-macosx_11_0_arm64.whl  # actual filename may vary

### Create repo file 
consolidator -g . --exclusions target,.venv