import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import ast
from torch.utils.data import Dataset
import scipy.sparse as sp
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as n
# Read control and perturbation data
import sys
import scanpy as sc
import pickle
import numpy as np
import torch.utils.data as Data
import scanpy as sc
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from anndata import AnnData
from cgi import test
from random import shuffle
import sys
import anndata
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from collections import defaultdict


def pearson_mean(data1, data2):
    """
    Compute the Pearson correlation for each sample and return the mean
    """
    sum_pearson = 0
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        sum_pearson += pearsonr_[0]
    
    return sum_pearson / data1.shape[0]


def r2_mean(data1, data2):
    """
    Compute R² for each sample and return the mean
    """
    sum_r2 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2 += r2_score_

    return sum_r2 / data1.shape[0]


def mse_mean(data1, data2):
    """
    Compute MSE for each sample and return the mean
    """
    sum_mse = 0
    for i in range(data1.shape[0]):
        mse_score_ = mean_squared_error(data1[i], data2[i])
        sum_mse += mse_score_

    return sum_mse / data1.shape[0]

def mmd_rbf(x, y, sigma=1.0):
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    k_xx = torch.exp(-xx / (2 * sigma))
    k_yy = torch.exp(-yy / (2 * sigma))
    k_xy = torch.exp(-xy / (2 * sigma))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def kl_loss(pred, target, eps=1e-8):
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    pred = pred / pred.sum(dim=1, keepdim=True)
    target = target / target.sum(dim=1, keepdim=True)

    return torch.nn.functional.kl_div(
        pred.log(), target, reduction="batchmean"
    )
