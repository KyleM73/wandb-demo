# wandb-demo
minimal example to set up wandb

## Setup
```bash
conda create -n wandb python=3.11
conda activate wandb
pip install wandb
pip install torch
```

if you intend to train on GPU, check that torch can find it:
```bash
python -c "print('device: ',torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('device: cpu')"
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
