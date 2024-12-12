from IPython.display import FileLink
import numpy as np
import torch

# Import submodules
from . import plot

# Util functions
def year_to_t(x, reverse = False, start_year = 1992):
    # Transform years to t or
    # t to years (reverse = True)
    if reverse:
        return x + start_year
    else:
        return x - start_year
    
def unweight_adj(A):
    if isinstance(A, np.ndarray):
        A_unweighted = (A > 0).astype(int)
    elif isinstance(A,torch.Tensor):
        A_unweighted = (A > 0).long()
    else:
        print("Error: Array has to be either numpy array or torch tensor")
        A_unweighted = None
    return A_unweighted

def log_transform(arr):
    """
    log transform the data. However we cannot log transform 0
    so one solution is to perseve 0s but add eps to 1
    (we can also add eps to all values and not only 1)
    """
    arr = arr.copy() + 1 
    return np.log(arr)

def exp_transform(arr):
    arr = np.exp(arr) - 1
    return arr


def get_link(file):
    # Helps with downloading
    # on kaggle
    return FileLink(file)

