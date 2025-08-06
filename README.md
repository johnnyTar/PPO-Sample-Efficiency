# Improving PPO Sample Efficiency on Sparse Reward Environments

## Setup Instructions

### TODO
**python train.py --track --tensorboard --total-timesteps 500000 --rollout-size 1024 --gym-id MiniGrid-DoorKey-6x6-v0**

### 1. Create and Activate Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
pip install -r requirements.txt
```bash
pip install -r requirements.txt
```

### 3. Run the Program

```bash
python train.py
```


### 4. Use w&b
```bash
wandb login
wandb login --relogin
```

### 5. Run with Custom Parameters
You can pass arguments like seed as follows:

```bash
python train.py --track --tensorboard
```