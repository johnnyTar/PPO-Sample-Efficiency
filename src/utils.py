import os
import torch
import numpy as np
import random
from datetime import datetime


def set_seed(seed):
    '''
    functions for setting up reproducible experiments and 
    maintaining consistent behavior across different
    components (PyTorch, NumPy, Python random, GPU).
    '''
    print('Seed in setSeed: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Force deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

