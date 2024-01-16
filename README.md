# wandb-demo
minimal example to set up wandb

## Setup
```bash
conda create -n wandb python=3.11
conda activate wandb
pip install wandb
pip install torch
```
create an account and get your WandB API key from <https://wandb.ai/home>
```bash
wandb login
```

## Simple Logging
```bash
python simple.py
```

## (Hyper-)Parameter Sweeps
