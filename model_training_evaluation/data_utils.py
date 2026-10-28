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
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
def fcfp4_embedding(adata):
    adata = adata.copy()
    # ============ 1. Read AnnData ============

    assert "canonical_smiles" in adata.obs.columns, "❌ canonical_smiles is missing from obs"

    smiles_all = adata.obs["canonical_smiles"].astype(str).values


    # ============ 2. Get unique SMILES ============
    unique_smiles = pd.unique(smiles_all)
    print(f"✅ Number of unique SMILES: {len(unique_smiles)}")

    # ============ 3. Compute FCFP4 (once per unique SMILES) ============
    num_bits = 1024
    smiles2fp = {}

    for smi in tqdm(unique_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            smiles2fp[smi] = np.zeros(num_bits, dtype=np.int8)
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            useFeatures=True,
            nBits=num_bits
        )
        smiles2fp[smi] = np.array(fp, dtype=np.int8)

    # ============ 4. Map fingerprints back to each cell ============
    fcfp_matrix = np.vstack([smiles2fp[smi] for smi in smiles_all])

    print(f"✅ Number of cells: {adata.n_obs}")
    print(f"✅ FCFP4 matrix shape: {fcfp_matrix.shape}")
    print(f"✅ Number of unique SMILES: {len(unique_smiles)}")
    # ============ 5. Store fingerprints in AnnData ============
    adata.obsm["X_fcfp4"] = fcfp_matrix
    return adata
def method_vectorized(non_zero_dose_data, zero_dose_data):
    # Get the paired_control_index column
    paired_control_index_non_zero = non_zero_dose_data.obs['paired_control_index'].values
    paired_control_index_zero = zero_dose_data.obs['paired_control_index'].values
    
    # Map paired_control_index values to zero_dose_data row indices
    # Build a dictionary containing only control_idx values present in paired_control_index_non_zero
    unique_controls = set(paired_control_index_non_zero)
    paired_zero_dose_data_dict = {idx: i for i, idx in enumerate(paired_control_index_zero) if idx in unique_controls}
    
    # Collect all row indices to retrieve
    control_indices = []
    valid_non_zero_indices = []
    
    for idx, control_idx in enumerate(paired_control_index_non_zero):
        if control_idx in paired_zero_dose_data_dict:
            control_indices.append(paired_zero_dose_data_dict[control_idx])
            valid_non_zero_indices.append(idx)
    
    # Retrieve all control samples in one operation
    control_group_adata = zero_dose_data[control_indices, :]
    
    return control_group_adata
def shuffle_adata(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata
def train_valid_test(adata: AnnData, split_key = 'cov_drug_dose_name_split'):


    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key]=='train'].index.tolist()
    valid_index = adata.obs[adata.obs[split_key]=='valid'].index.tolist()
    test_index = adata.obs[adata.obs[split_key]=='test'].index.tolist()
    control_index = adata.obs[adata.obs['dose'].astype(float)==0.0].index.tolist()

    if len(train_index)>0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index)>0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata=None
    if len(test_index)>0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata=None

    
    return train_adata, valid_adata, test_adata

