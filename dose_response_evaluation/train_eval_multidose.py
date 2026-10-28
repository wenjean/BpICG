import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import pandas as pd
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from scipy import sparse
import random
import os
import argparse
import warnings
import pickle
import sys
from pathlib import Path

from model import *
from data_utils import *
from evo_loss import *

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent


parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str, default="multidose_0to100um_unseen_drug_seed{seed}")
parser.add_argument("--drug_col", type=str, default="smiles")
parser.add_argument("--seed", type=int, default=11)

parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--test_ratio", type=float, default=0.1)

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument(
    "--preflight_only",
    action="store_true",
    help="Run preprocessing plus one forward/backward batch, then exit without full training",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    help="Skip training and evaluate the existing best checkpoint",
)
parser.add_argument("--filter_low_var_perturb", action="store_true", default=True)
parser.add_argument("--no_filter_low_var_perturb", action="store_false", dest="filter_low_var_perturb")
parser.add_argument("--min_perturb_expr_var", type=float, default=1e-4)
parser.add_argument("--use_bio_graph", action="store_true", default=True)
parser.add_argument("--no_use_bio_graph", action="store_false", dest="use_bio_graph")
parser.add_argument("--prior_min_importance", type=float, default=0.0)
parser.add_argument("--prior_topk", type=int, default=21, help="Per-target top-k for chromo/go/tftg graph priors; <=0 keeps all edges")
parser.add_argument("--kegg_min_importance", type=float, default=0.1)
parser.add_argument("--kegg_topk", type=int, default=21, help="Per-target top-k after threshold; <=0 keeps all edges")
parser.add_argument("--use_pathway_encoder", action="store_true", default=True)
parser.add_argument("--no_use_pathway_encoder", action="store_false", dest="use_pathway_encoder")
parser.add_argument("--pathway_dim", type=int, default=128)

parser.add_argument(
    "--kegg_graph_path",
    type=str,
    default="data/priors/con_uce_kegg_full.csv"
)

parser.add_argument(
    "--chromo_graph_path",
    type=str,
    default="data/priors/con_uce_chromo_full.csv"
)

parser.add_argument(
    "--go_graph_path",
    type=str,
    default="data/priors/con_uce_go_top21.csv"
)

parser.add_argument(
    "--tftg_graph_path",
    type=str,
    default="data/priors/con_uce_tftg_full.csv"
)

parser.add_argument(
    "--kegg_gene2pathway_path",
    type=str,
    default="data/processed/con_uce_gene2kegg.pkl"
)

parser.add_argument(
    "--control_path",
    type=str,
    default="data/processed/con_gene_aligned_12328_uce.h5ad"
)

parser.add_argument(
    "--perturb_path",
    type=str,
    default="data/processed/l1000_multidose_24h_0to100um_mean_aligned_12328.h5ad"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="results/seed_{seed}"
)

args = parser.parse_args()
args.save_dir = args.save_dir.replace("{seed}", str(args.seed))
args.run_name = args.run_name.replace("{seed}", str(args.seed))

for path_arg in (
    "kegg_graph_path", "chromo_graph_path", "go_graph_path", "tftg_graph_path",
    "kegg_gene2pathway_path", "control_path", "perturb_path", "save_dir",
):
    path_value = getattr(args, path_arg)
    if path_value and path_value.lower() != "none":
        setattr(args, path_arg, str(PROJECT_DIR / path_value))

assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-8
if args.preflight_only and args.eval_only:
    parser.error("--preflight_only and --eval_only cannot be used together")

DOSE_UNIT = "uM"
DOSE_TRANSFORM = "log1p"
DOSE_MIN_EXCLUSIVE = 0.0
DOSE_MAX_INCLUSIVE = 100.0
DOSE_BIN_EDGES = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
DOSE_BIN_LABELS = [
    "(0,0.01]",
    "(0.01,0.1]",
    "(0.1,1]",
    "(1,10]",
    "(10,100]",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✅ Random seed set to {seed}")


set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)


def to_dense_array(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_gene_symbols(adata):
    if "gene" in adata.var.columns:
        return adata.var["gene"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def build_sparse_graph_adj(
    graph_path,
    gene_symbols,
    graph_name="graph",
    min_importance=0.0,
    topk_per_target=0
):
    if graph_path is None or str(graph_path).lower() in ["", "none"]:
        print(f"⚠️ {graph_name}: graph path is empty, skip")
        return None

    if not os.path.exists(graph_path):
        print(f"⚠️ {graph_name}: graph file not found, skip: {graph_path}")
        return None

    edge_df = pd.read_csv(graph_path)
    required_cols = {"source", "target", "importance"}
    if not required_cols.issubset(edge_df.columns):
        raise ValueError(f"❌ {graph_name} is missing required columns {required_cols}: {graph_path}")

    edge_df = edge_df[edge_df["importance"].astype(float) >= float(min_importance)].copy()
    if topk_per_target is not None and int(topk_per_target) > 0:
        edge_df = (
            edge_df.groupby("target", group_keys=False)
            .apply(lambda x: x.nlargest(int(topk_per_target), ["importance"]))
            .reset_index(drop=True)
        )

    gene_to_idx = {str(g): i for i, g in enumerate(gene_symbols)}
    rows = []
    cols = []
    vals = []
    covered = set()
    self_loop_nodes = set()

    for source, target, weight in edge_df[["source", "target", "importance"]].itertuples(index=False):
        source = str(source)
        target = str(target)
        if source not in gene_to_idx or target not in gene_to_idx:
            continue

        source_idx = gene_to_idx[source]
        target_idx = gene_to_idx[target]
        rows.append(target_idx)
        cols.append(source_idx)
        vals.append(float(weight))
        covered.add(source_idx)
        covered.add(target_idx)
        if source_idx == target_idx:
            self_loop_nodes.add(source_idx)

    for idx in sorted(covered - self_loop_nodes):
        rows.append(idx)
        cols.append(idx)
        vals.append(1.0)

    if len(vals) == 0:
        print(f"⚠️ {graph_name}: no usable edges after matching current genes")
        return None

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(len(gene_symbols), len(gene_symbols))
    ).coalesce()

    idx = adj.indices()
    val = adj.values()
    row_sum = torch.zeros(len(gene_symbols), dtype=torch.float32)
    row_sum.index_add_(0, idx[0], val)
    norm_val = val / row_sum[idx[0]].clamp_min(1e-12)
    adj = torch.sparse_coo_tensor(idx, norm_val, adj.shape).coalesce()

    print(f"✅ {graph_name}: {graph_path}")
    print(f"  min_importance: {min_importance}")
    print(f"  topk_per_target: {topk_per_target}")
    print(f"  matched nodes: {len(covered)} / {len(gene_symbols)}")
    print(f"  sparse edges: {adj._nnz()}")
    print("  row-normalized: A[target, source]")

    return adj


def build_kegg_pathway_matrices(gene2pathway_path, gene_symbols):
    if gene2pathway_path is None or str(gene2pathway_path).lower() in ["", "none"]:
        print("⚠️ KEGG gene2pathway path is empty, skip pathway encoder")
        return None, None, 0, []

    if not os.path.exists(gene2pathway_path):
        print(f"⚠️ KEGG gene2pathway file not found, skip: {gene2pathway_path}")
        return None, None, 0, []

    with open(gene2pathway_path, "rb") as f:
        gene2pathway = pickle.load(f)

    gene_to_idx = {str(g): i for i, g in enumerate(gene_symbols)}
    matched = {
        str(g): set(paths)
        for g, paths in gene2pathway.items()
        if str(g) in gene_to_idx and len(paths) > 0
    }
    pathway_ids = sorted(set().union(*matched.values())) if matched else []

    if not pathway_ids:
        print("⚠️ No KEGG pathways matched current genes")
        return None, None, 0, []

    pathway_to_idx = {p: i for i, p in enumerate(pathway_ids)}
    pathway_sizes = np.zeros(len(pathway_ids), dtype=np.float32)
    gene_pathway_counts = np.zeros(len(gene_symbols), dtype=np.float32)

    memberships = []
    for gene, paths in matched.items():
        gene_idx = gene_to_idx[gene]
        for pathway in paths:
            pathway_idx = pathway_to_idx[pathway]
            memberships.append((gene_idx, pathway_idx))
            pathway_sizes[pathway_idx] += 1.0
            gene_pathway_counts[gene_idx] += 1.0

    g2p_rows = []
    g2p_cols = []
    g2p_vals = []
    p2g_rows = []
    p2g_cols = []
    p2g_vals = []

    for gene_idx, pathway_idx in memberships:
        g2p_rows.append(pathway_idx)
        g2p_cols.append(gene_idx)
        g2p_vals.append(1.0 / max(pathway_sizes[pathway_idx], 1.0))

        p2g_rows.append(gene_idx)
        p2g_cols.append(pathway_idx)
        p2g_vals.append(1.0 / max(gene_pathway_counts[gene_idx], 1.0))

    gene_to_pathway = torch.sparse_coo_tensor(
        torch.tensor([g2p_rows, g2p_cols], dtype=torch.long),
        torch.tensor(g2p_vals, dtype=torch.float32),
        size=(len(pathway_ids), len(gene_symbols))
    ).coalesce()
    pathway_to_gene = torch.sparse_coo_tensor(
        torch.tensor([p2g_rows, p2g_cols], dtype=torch.long),
        torch.tensor(p2g_vals, dtype=torch.float32),
        size=(len(gene_symbols), len(pathway_ids))
    ).coalesce()

    print(f"✅ KEGG pathway encoder: {gene2pathway_path}")
    print(f"  matched genes: {len(matched)} / {len(gene_symbols)}")
    print(f"  pathways: {len(pathway_ids)}")
    print(f"  memberships: {len(memberships)}")
    print("  gene_to_pathway: [P, G], pathway mean activity")
    print("  pathway_to_gene: [G, P], mean pathway delta back to genes")

    return gene_to_pathway, pathway_to_gene, len(pathway_ids), pathway_ids

def normalize_log1p(adata):
    import numpy as np
    from scipy import sparse

    X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    lib_size = X.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    X = X / lib_size * 1e6

    X = np.log1p(X)

    adata.X = X.astype(np.float32)
    return adata


def sanitize_adata_matrix(adata, name="adata"):
    X = adata.X

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        data = X.data.astype(np.float32, copy=False)

        print(f"{name} before sanitize:")
        print("  sparse data nan:", np.isnan(data).sum())
        print("  sparse data +inf:", np.isposinf(data).sum())
        print("  sparse data -inf:", np.isneginf(data).sum())
        print("  sparse data negative:", (data < 0).sum())

        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        data[data < 0] = 0.0

        X.data = data
        X.eliminate_zeros()
        adata.X = X

    else:
        X = np.asarray(X, dtype=np.float32)

        print(f"{name} before sanitize:")
        print("  dense nan:", np.isnan(X).sum())
        print("  dense +inf:", np.isposinf(X).sum())
        print("  dense -inf:", np.isneginf(X).sum())
        print("  dense negative:", (X < 0).sum())

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        X[X < 0] = 0.0
        adata.X = X

    return adata


def filter_low_variance_perturb_samples(
    adata,
    min_var=1e-4,
    save_dir=None,
    run_name="run"
):
    X = to_dense_array(adata.X).astype(np.float32, copy=False)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    row_mean = X.mean(axis=1, dtype=np.float64)
    row_var = X.var(axis=1, dtype=np.float64)
    row_std = np.sqrt(row_var)
    row_min = X.min(axis=1)
    row_max = X.max(axis=1)
    zero_fraction = 1.0 - (np.count_nonzero(X, axis=1) / X.shape[1])

    keep_mask = row_var > float(min_var)

    stats_df = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "sample_index": np.arange(adata.n_obs),
        "expr_mean": row_mean,
        "expr_var": row_var,
        "expr_std": row_std,
        "expr_min": row_min,
        "expr_max": row_max,
        "zero_fraction": zero_fraction,
        "keep": keep_mask,
    })

    for col in ["cell_type", "drug_name", "cmap_name", "target_gene", "smiles", "ctrl", "condition"]:
        if col in adata.obs.columns:
            stats_df[col] = adata.obs[col].astype(str).values

    removed_df = stats_df[~keep_mask].copy().sort_values("expr_var")

    print("\n🧹 Low-variance perturb sample filtering")
    print(f"  min_perturb_expr_var: {min_var}")
    print(f"  before samples: {adata.n_obs}")
    print(f"  removed samples: {removed_df.shape[0]}")
    print(f"  kept samples: {int(keep_mask.sum())}")

    if removed_df.shape[0] > 0:
        print("  removed cell_type counts:")
        print(removed_df["cell_type"].value_counts().to_string() if "cell_type" in removed_df.columns else "NA")
        print("  first removed samples:")
        show_cols = [
            c for c in [
                "obs_name", "cell_type", "drug_name", "condition",
                "expr_mean", "expr_var", "expr_max", "zero_fraction"
            ] if c in removed_df.columns
        ]
        print(removed_df[show_cols].head(20).to_string(index=False))

    if save_dir is not None:
        stats_path = os.path.join(save_dir, f"{run_name}_perturb_expression_variance_stats.csv")
        removed_path = os.path.join(save_dir, f"{run_name}_filtered_low_variance_perturb_samples.csv")
        stats_df.to_csv(stats_path, index=False)
        removed_df.to_csv(removed_path, index=False)
        print(f"  saved variance stats to: {stats_path}")
        print(f"  saved removed samples to: {removed_path}")

    if keep_mask.all():
        print("  all samples passed; reuse the existing expression matrix without copying")
        return adata, removed_df

    return adata[keep_mask].copy(), removed_df


def fcfp4_embedding(adata, smiles_col="smiles", n_bits=1024, radius=2):
    if smiles_col not in adata.obs.columns:
        raise ValueError(f"❌ adata.obs is missing the SMILES column: {smiles_col}")

    smiles_list = adata.obs[smiles_col].astype(str).tolist()
    unique_smiles = pd.unique(smiles_list)

    print(f"🧪 Unique input SMILES: {len(unique_smiles)}")

    smiles_to_fp = {}

    for smi in tqdm(unique_smiles, desc="Building FCFP4"):
        if smi.lower() in ["none", "nan", ""]:
            arr = np.zeros(n_bits, dtype=np.int8)
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                arr = np.zeros(n_bits, dtype=np.int8)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=radius,
                    nBits=n_bits,
                    useFeatures=True
                )
                arr = np.zeros((n_bits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)

        smiles_to_fp[smi] = arr

    X_fcfp4 = np.stack([smiles_to_fp[smi] for smi in smiles_list], axis=0)
    adata.obsm["X_fcfp4"] = X_fcfp4.astype(np.float32)

    return adata


def split_train_val_test_by_drug(
    perturb_adata,
    drug_col="smiles",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    drug_series = perturb_adata.obs[drug_col].astype(str)
    unique_drugs = drug_series.unique().tolist()

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_drugs)

    n_total = len(unique_drugs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n_total - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

    train_drugs = set(unique_drugs[:n_train])
    val_drugs = set(unique_drugs[n_train:n_train + n_val])
    test_drugs = set(unique_drugs[n_train + n_val:])

    train_mask = perturb_adata.obs[drug_col].astype(str).isin(train_drugs).values
    val_mask = perturb_adata.obs[drug_col].astype(str).isin(val_drugs).values
    test_mask = perturb_adata.obs[drug_col].astype(str).isin(test_drugs).values

    train_perturb = perturb_adata[train_mask]
    val_perturb = perturb_adata[val_mask]
    test_perturb = perturb_adata[test_mask]

    return train_perturb, val_perturb, test_perturb, train_drugs, val_drugs, test_drugs


def random_pair_control_by_cell_type(
    perturb,
    control,
    seed=42,
    cell_col="cell_type"
):
    rng = np.random.RandomState(seed)

    if "ctrl" in control.obs.columns:
        control_pool_adata = control[control.obs["ctrl"].astype(int).values == 1].copy()
    else:
        control_pool_adata = control.copy()

    print(f"Control samples: {control_pool_adata.n_obs}")
    print(f"Control cell types: {control_pool_adata.obs[cell_col].nunique()}")

    control_pool = {}
    control_cells = control_pool_adata.obs[cell_col].astype(str).to_numpy()
    for cell in np.unique(control_cells):
        control_pool[cell] = np.flatnonzero(control_cells == cell)

    perturb_cells = perturb.obs[cell_col].astype(str).to_numpy()
    paired_control_indices = np.full(perturb.n_obs, -1, dtype=np.int64)
    valid_mask = np.zeros(perturb.n_obs, dtype=bool)
    missing_cell_types = {}

    for cell in np.unique(perturb_cells):
        perturb_positions = np.flatnonzero(perturb_cells == cell)
        if cell not in control_pool or len(control_pool[cell]) == 0:
            missing_cell_types[cell] = int(perturb_positions.size)
            continue

        paired_control_indices[perturb_positions] = rng.choice(
            control_pool[cell],
            size=perturb_positions.size,
            replace=True,
        )
        valid_mask[perturb_positions] = True

    if not valid_mask.any():
        raise ValueError("❌ No perturbation sample can be matched to a control with the same cell_type")

    if len(missing_cell_types) > 0:
        print("⚠️ The following cell_type groups have no matching controls and will be discarded:")
        print(missing_cell_types)

    matched_perturb = perturb[valid_mask].copy()
    paired_control_indices = paired_control_indices[valid_mask]
    paired_control_names = control_pool_adata.obs_names[paired_control_indices].astype(str)

    matched_perturb.obs["paired_control_index"] = paired_control_indices
    matched_perturb.obs["paired_control_obs_name"] = paired_control_names

    print(f"Matched perturb samples: {matched_perturb.n_obs}")
    print(f"Paired through control bank rows: {np.unique(paired_control_indices).size}")

    return matched_perturb


def save_pair_table(perturb_adata, save_path, drug_col):
    pair_df = pd.DataFrame({
        "perturb_obs_name": perturb_adata.obs_names,
        "paired_control_obs_name": perturb_adata.obs["paired_control_obs_name"].values,
        "cell_type": perturb_adata.obs["cell_type"].values,
        "drug": perturb_adata.obs[drug_col].astype(str).values,
        "drug_name": perturb_adata.obs["drug_name"].astype(str).values
        if "drug_name" in perturb_adata.obs.columns else "NA",
        "dose": perturb_adata.obs["dose"].values,
    })

    pair_df.to_csv(save_path, index=False)


class PairedPerturbationDataset(Data.Dataset):

    def __init__(
        self,
        control_expression_bank,
        control_uce_bank,
        paired_control_indices,
        drug,
        dose,
        cell_type,
        target,
    ):
        self.control_expression_bank = control_expression_bank
        self.control_uce_bank = control_uce_bank
        self.paired_control_indices = paired_control_indices
        self.drug = drug
        self.dose = dose
        self.cell_type = cell_type
        self.target = target

        n = self.target.shape[0]
        assert n == self.paired_control_indices.shape[0]
        assert n == self.drug.shape[0] == self.dose.shape[0] == self.cell_type.shape[0]

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        control_index = self.paired_control_indices[index]
        return (
            self.control_expression_bank[control_index],
            self.control_uce_bank[control_index],
            self.drug[index],
            self.dose[index],
            self.cell_type[index],
            self.target[index],
        )


def build_control_bank(control):
    if "X_uce" not in control.obsm:
        raise ValueError("❌ control.obsm is missing X_uce")

    control_expression = np.ascontiguousarray(
        to_dense_array(control.X), dtype=np.float32
    )
    control_uce = np.ascontiguousarray(
        np.asarray(control.obsm["X_uce"]), dtype=np.float32
    )
    if control_expression.shape[0] != control_uce.shape[0]:
        raise ValueError("❌ The control expression matrix and X_uce have different row counts")
    if not np.isfinite(control_expression).all() or not np.isfinite(control_uce).all():
        raise ValueError("❌ The control expression matrix or X_uce contains NaN/Inf")

    expression_bank = torch.from_numpy(control_expression)
    uce_bank = torch.from_numpy(control_uce)
    print("control expression bank shape:", expression_bank.shape)
    print("control UCE bank shape:", uce_bank.shape)
    return expression_bank, uce_bank


def build_dataset(control_expression_bank, control_uce_bank, perturb):
    if "X_fcfp4" not in perturb.obsm:
        raise ValueError("❌ perturb.obsm is missing X_fcfp4")
    if "paired_control_index" not in perturb.obs.columns:
        raise ValueError("❌ perturb.obs is missing paired_control_index")

    paired_control_indices = torch.from_numpy(
        perturb.obs["paired_control_index"].astype(np.int64).to_numpy()
    )
    if paired_control_indices.min() < 0 or paired_control_indices.max() >= control_expression_bank.shape[0]:
        raise ValueError("❌ paired_control_index is outside the control bank range")

    drug = torch.from_numpy(np.ascontiguousarray(
        np.asarray(perturb.obsm["X_fcfp4"]), dtype=np.float32
    ))
    dose = torch.from_numpy(np.ascontiguousarray(
        perturb.obs["dose"].astype(np.float32).to_numpy()
    )).view(-1, 1)
    cell_type = torch.from_numpy(np.ascontiguousarray(
        perturb.obs["cell_type_id"].astype(np.int64).to_numpy()
    ))
    target = torch.from_numpy(np.ascontiguousarray(
        to_dense_array(perturb.X), dtype=np.float32
    ))

    print("paired control index shape:", paired_control_indices.shape)
    print("drug shape:", drug.shape)
    print("dose shape:", dose.shape)
    print(
        "dose raw/model range:",
        (float(dose.min()), float(dose.max())),
        (float(torch.log1p(dose).min()), float(torch.log1p(dose).max()))
    )
    print("target shape:", target.shape)

    print("drug nan/inf:", torch.isnan(drug).any().item(), torch.isinf(drug).any().item())
    print("dose nan/inf:", torch.isnan(dose).any().item(), torch.isinf(dose).any().item())
    print("target nan/inf:", torch.isnan(target).any().item(), torch.isinf(target).any().item())

    return PairedPerturbationDataset(
        control_expression_bank=control_expression_bank,
        control_uce_bank=control_uce_bank,
        paired_control_indices=paired_control_indices,
        drug=drug,
        dose=dose,
        cell_type=cell_type,
        target=target,
    )


def transform_dose_for_model(raw_dose):
    return torch.log1p(raw_dose)


def calculate_sample_metrics(y_true, y_pred):
    diff = y_true - y_pred
    mse = np.mean(diff * diff, axis=1, dtype=np.float64)

    true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)

    numerator = np.sum(true_centered * pred_centered, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=1, dtype=np.float64)
        * np.sum(pred_centered * pred_centered, axis=1, dtype=np.float64)
    )
    pearson = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )

    ss_res = np.sum(diff * diff, axis=1, dtype=np.float64)
    ss_tot = np.sum(true_centered * true_centered, axis=1, dtype=np.float64)
    r2 = np.divide(
        ss_res,
        ss_tot,
        out=np.ones_like(ss_res, dtype=np.float64),
        where=ss_tot > 0,
    )
    r2 = 1.0 - r2

    return pearson, r2, mse


def evaluate_model(dataloader, model, return_details=False):
    model.eval()

    pearson_sum = 0.0
    r2_sum = 0.0
    mse_sum = 0.0
    n = 0
    skipped = 0
    sample_offset = 0
    detail_parts = []

    with torch.no_grad():
        for batch_id, data in enumerate(dataloader):
            batch_con, batch_uce, batch_drug_embeddings, batch_doses, batch_cell_type, batch_y = data

            batch_size = batch_y.shape[0]
            batch_indices = np.arange(sample_offset, sample_offset + batch_size)
            sample_offset += batch_size
            raw_doses = batch_doses.detach().cpu().numpy().reshape(-1)

            batch_con = batch_con.to(device)
            batch_uce = batch_uce.to(device)
            batch_drug_embeddings = batch_drug_embeddings.to(device)
            batch_doses = batch_doses.to(device)
            batch_y = batch_y.to(device)

            outputs, _, _ = model(
                batch_con,
                batch_uce,
                batch_drug_embeddings,
                transform_dose_for_model(batch_doses)
            )

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"⚠️ batch {batch_id} outputs contain NaN/Inf; skipping this batch")
                skipped += outputs.size(0)
                continue

            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                print(f"⚠️ batch {batch_id} batch_y contains NaN/Inf; skipping this batch")
                skipped += batch_y.size(0)
                continue

            y_true_batch = batch_y.detach().cpu().numpy()
            y_pred_batch = outputs.detach().cpu().numpy()

            valid_mask = (
                np.isfinite(y_true_batch).all(axis=1)
                & np.isfinite(y_pred_batch).all(axis=1)
            )

            if not valid_mask.all():
                skipped += int((~valid_mask).sum())

            y_true_batch = y_true_batch[valid_mask]
            y_pred_batch = y_pred_batch[valid_mask]
            valid_indices = batch_indices[valid_mask]
            valid_doses = raw_doses[valid_mask]

            if y_true_batch.shape[0] == 0:
                continue

            pearson, r2, mse = calculate_sample_metrics(y_true_batch, y_pred_batch)
            pearson_sum += float(pearson.sum())
            r2_sum += float(r2.sum())
            mse_sum += float(mse.sum())
            n += int(y_true_batch.shape[0])

            if return_details:
                detail_parts.append(pd.DataFrame({
                    "sample_index": valid_indices,
                    "dose_um": valid_doses.astype(np.float64),
                    "pearson": pearson,
                    "r2": r2,
                    "mse": mse,
                }))

    print(f"evaluate valid samples: {n}, skipped: {skipped}")

    if n == 0:
        if return_details:
            return 0.0, 0.0, 0.0, pd.DataFrame(
                columns=["sample_index", "dose_um", "pearson", "r2", "mse"]
            )
        return 0.0, 0.0, 0.0

    mean_pearson = pearson_sum / n
    mean_r2 = r2_sum / n
    mean_mse = mse_sum / n

    if return_details:
        details = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
        return mean_pearson, mean_r2, mean_mse, details

    return mean_pearson, mean_r2, mean_mse


def add_evaluation_metadata(metrics_df, perturb_adata, drug_col):
    if metrics_df.empty:
        raise ValueError("❌ Evaluation produced no valid sample metrics")

    sample_indices = metrics_df["sample_index"].astype(int).to_numpy()
    obs = perturb_adata.obs.iloc[sample_indices]
    reported_doses = obs["dose"].astype(float).to_numpy()
    metric_doses = metrics_df["dose_um"].astype(float).to_numpy()

    expected_model_doses = reported_doses.astype(np.float32).astype(np.float64)
    same_model_dose = np.array_equal(expected_model_doses, metric_doses)
    same_source_dose = np.array_equal(reported_doses, metric_doses)
    if not (same_model_dose or same_source_dose):
        max_abs_diff = float(np.max(np.abs(reported_doses - metric_doses)))
        raise ValueError(
            "❌ Doses in the evaluation metrics do not match perturb.obs; "
            f"max_abs_diff={max_abs_diff:.6g}"
        )
    metrics_df = metrics_df.copy()
    metrics_df["dose_um"] = reported_doses

    metadata = pd.DataFrame({
        "perturb_obs_name": perturb_adata.obs_names[sample_indices].astype(str),
        "cell_type": obs["cell_type"].astype(str).to_numpy(),
        "drug": obs[drug_col].astype(str).to_numpy(),
        "drug_name": obs["drug_name"].astype(str).to_numpy()
        if "drug_name" in obs.columns else "NA",
        "condition": obs["condition"].astype(str).to_numpy(),
    })
    return pd.concat(
        [metadata.reset_index(drop=True), metrics_df.reset_index(drop=True)],
        axis=1,
    )


def summarize_metric_groups(metrics_df, group_col):
    return (
        metrics_df.groupby(group_col, observed=True, sort=True)
        .agg(
            n_samples=("pearson", "size"),
            mean_pearson=("pearson", "mean"),
            mean_r2=("r2", "mean"),
            mean_mse=("mse", "mean"),
        )
        .reset_index()
    )


def save_dose_evaluation_reports(metrics_df, perturb_adata, split_name):
    metrics_df = add_evaluation_metadata(metrics_df, perturb_adata, args.drug_col)
    sample_path = os.path.join(
        args.save_dir, f"{args.run_name}_{split_name}_sample_metrics.csv"
    )
    metrics_df.to_csv(sample_path, index=False)

    exact_dose_df = summarize_metric_groups(metrics_df, "dose_um")
    exact_dose_path = os.path.join(
        args.save_dir, f"{args.run_name}_{split_name}_dose_metrics.csv"
    )
    exact_dose_df.to_csv(exact_dose_path, index=False)

    metrics_df["dose_bin_um"] = pd.cut(
        metrics_df["dose_um"],
        bins=DOSE_BIN_EDGES,
        labels=DOSE_BIN_LABELS,
        include_lowest=False,
        right=True,
    )
    if metrics_df["dose_bin_um"].isna().any():
        raise ValueError(f"❌ {split_name} contains doses outside the predefined intervals")

    dose_bin_df = summarize_metric_groups(metrics_df, "dose_bin_um")
    dose_bin_path = os.path.join(
        args.save_dir, f"{args.run_name}_{split_name}_dose_bin_metrics.csv"
    )
    dose_bin_df.to_csv(dose_bin_path, index=False)

    print(f"✅ Saved {split_name} sample metrics: {sample_path}")
    print(f"✅ Saved {split_name} exact-dose metrics: {exact_dose_path}")
    print(f"✅ Saved {split_name} dose-bin metrics: {dose_bin_path}")

    return metrics_df, exact_dose_df, dose_bin_df


print("📦 Loading data...")

control_adata = sc.read_h5ad(args.control_path)
perturb_adata = sc.read_h5ad(args.perturb_path)

print("control_adata:", control_adata)
print("perturb_adata:", perturb_adata)

required_cols = ["cell_type", args.drug_col, "ctrl", "condition", "dose"]

for col in required_cols:
    if col not in perturb_adata.obs.columns:
        raise ValueError(f"❌ perturb_adata.obs is missing required column {col}.")

for col in ["cell_type", "ctrl", "condition"]:
    if col not in control_adata.obs.columns:
        raise ValueError(f"❌ control_adata.obs is missing required column {col}.")

if "X_uce" not in control_adata.obsm:
    raise ValueError("❌ control_adata.obsm is missing X_uce; ensure the control H5AD contains UCE features")


perturb_mask = perturb_adata.obs["ctrl"].astype(int).values == 0
control_mask = control_adata.obs["ctrl"].astype(int).values == 1

if not perturb_mask.all():
    perturb_adata = perturb_adata[perturb_mask].copy()
else:
    print("✅ All perturb input rows have ctrl=0; skipped redundant matrix copy")

if not control_mask.all():
    control_adata = control_adata[control_mask].copy()
else:
    print("✅ All control input rows have ctrl=1; skipped redundant matrix copy")

print("After filter:")
print("perturb_adata:", perturb_adata.shape)
print("control_adata:", control_adata.shape)


dose_values = pd.to_numeric(perturb_adata.obs["dose"], errors="coerce").to_numpy(dtype=np.float64)
if not np.isfinite(dose_values).all():
    raise ValueError("❌ perturb_adata.obs['dose'] contains missing values or NaN/Inf")
if np.any(dose_values <= DOSE_MIN_EXCLUSIVE) or np.any(dose_values > DOSE_MAX_INCLUSIVE):
    raise ValueError(
        f"❌ Perturbation doses must satisfy {DOSE_MIN_EXCLUSIVE} < dose <= {DOSE_MAX_INCLUSIVE} {DOSE_UNIT}"
    )
perturb_adata.obs["dose"] = dose_values
control_adata.obs["dose"] = 0.0

print(
    f"✅ Preserved perturb doses: {dose_values.min():.6g} to "
    f"{dose_values.max():.6g} {DOSE_UNIT}, {np.unique(dose_values).size} unique doses"
)
print(f"✅ Model dose transform: {DOSE_TRANSFORM}")
print(f"✅ Set control dose = 0.0 {DOSE_UNIT}")

unique_smiles = perturb_adata.obs[args.drug_col].astype(str).unique()
invalid_smiles = [smi for smi in unique_smiles if Chem.MolFromSmiles(smi) is None]
if invalid_smiles:
    raise ValueError(f"❌ Detected {len(invalid_smiles)} invalid SMILES, for example: {invalid_smiles[:5]}")
print(f"✅ Validated {len(unique_smiles)} unique SMILES")


if not np.array_equal(perturb_adata.var_names, control_adata.var_names):
    print("⚠️ perturb and control have different var_names; aligning their intersection")

    common_genes = perturb_adata.var_names.intersection(control_adata.var_names)
    print("common genes:", len(common_genes))

    if len(common_genes) == 0:
        raise ValueError("❌ perturb and control have no genes in common")

    perturb_adata = perturb_adata[:, common_genes].copy()
    control_adata = control_adata[:, common_genes].copy()

else:
    print("✅ perturb and control have identical var_names")


print("🔬 Applying log1p + CPM normalization...")



if args.filter_low_var_perturb:
    perturb_adata, filtered_low_var_perturb_df = filter_low_variance_perturb_samples(
        perturb_adata,
        min_var=args.min_perturb_expr_var,
        save_dir=args.save_dir,
        run_name=args.run_name
    )

    if perturb_adata.n_obs == 0:
        raise ValueError("❌ No perturbation samples remain after low-variance filtering; lower --min_perturb_expr_var")
else:
    print("⚠️ Skip low-variance perturb sample filtering")

all_cell_types = sorted(
    set(perturb_adata.obs["cell_type"].astype(str).unique())
    | set(control_adata.obs["cell_type"].astype(str).unique())
)

cell_type_id = {ct: idx for idx, ct in enumerate(all_cell_types)}

perturb_adata.obs["cell_type_id"] = (
    perturb_adata.obs["cell_type"].astype(str).map(cell_type_id).astype(int)
)

control_adata.obs["cell_type_id"] = (
    control_adata.obs["cell_type"].astype(str).map(cell_type_id).astype(int)
)

print(f"✅ Number of cell types: {len(cell_type_id)}")


print("\n" + "=" * 80)
print("🚀 Train / Val / Test split by unseen drugs")
print("=" * 80)

train_perturb, val_perturb, test_perturb, train_drugs, val_drugs, test_drugs = split_train_val_test_by_drug(
    perturb_adata=perturb_adata,
    drug_col=args.drug_col,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    seed=args.seed
)

print(f"Total perturbation samples: {perturb_adata.n_obs}")
print(f"Train perturbation samples: {train_perturb.n_obs}")
print(f"Val perturbation samples: {val_perturb.n_obs}")
print(f"Test perturbation samples: {test_perturb.n_obs}")
print(f"Train drugs: {len(train_drugs)}")
print(f"Val drugs: {len(val_drugs)}")
print(f"Test drugs: {len(test_drugs)}")

split_overlaps = {
    "train_val": train_drugs & val_drugs,
    "train_test": train_drugs & test_drugs,
    "val_test": val_drugs & test_drugs,
}
if any(split_overlaps.values()):
    raise ValueError(f"❌ Drug leakage detected across train/val/test: {split_overlaps}")
if (train_drugs | val_drugs | test_drugs) != set(perturb_adata.obs[args.drug_col].astype(str)):
    raise ValueError("❌ The train/val/test drug sets do not cover the full input data")
print("✅ Verified disjoint train/val/test SMILES sets")

split_summary_df = pd.DataFrame([
    {
        "split": split_name,
        "n_samples": split_adata.n_obs,
        "n_unique_smiles": split_adata.obs[args.drug_col].astype(str).nunique(),
        "n_unique_drug_names": split_adata.obs["drug_name"].astype(str).nunique()
        if "drug_name" in split_adata.obs.columns else np.nan,
        "n_cell_types": split_adata.obs["cell_type"].astype(str).nunique(),
        "n_unique_doses": split_adata.obs["dose"].nunique(),
        "min_dose_um": float(split_adata.obs["dose"].min()),
        "max_dose_um": float(split_adata.obs["dose"].max()),
    }
    for split_name, split_adata in [
        ("train", train_perturb),
        ("val", val_perturb),
        ("test", test_perturb),
    ]
])
split_summary_df.to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_split_summary.csv"),
    index=False,
)

pd.DataFrame({"train_drugs": sorted(list(train_drugs))}).to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_train_drugs.csv"),
    index=False
)

pd.DataFrame({"val_drugs": sorted(list(val_drugs))}).to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_val_drugs.csv"),
    index=False
)

pd.DataFrame({"test_drugs": sorted(list(test_drugs))}).to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_test_drugs.csv"),
    index=False
)


print("\n🔄 Pairing controls by same cell_type...")

train_perturb = random_pair_control_by_cell_type(
    perturb=train_perturb,
    control=control_adata,
    seed=args.seed,
    cell_col="cell_type"
)

val_perturb = random_pair_control_by_cell_type(
    perturb=val_perturb,
    control=control_adata,
    seed=args.seed + 1,
    cell_col="cell_type"
)

test_perturb = random_pair_control_by_cell_type(
    perturb=test_perturb,
    control=control_adata,
    seed=args.seed + 2,
    cell_col="cell_type"
)

save_pair_table(
    train_perturb,
    os.path.join(args.save_dir, f"{args.run_name}_train_pairs.csv"),
    args.drug_col
)

save_pair_table(
    val_perturb,
    os.path.join(args.save_dir, f"{args.run_name}_val_pairs.csv"),
    args.drug_col
)

save_pair_table(
    test_perturb,
    os.path.join(args.save_dir, f"{args.run_name}_test_pairs.csv"),
    args.drug_col
)

print("✅ Pair tables saved.")


print("\n🧬 Building drug fingerprints...")

train_perturb = fcfp4_embedding(train_perturb, smiles_col=args.drug_col)
val_perturb = fcfp4_embedding(val_perturb, smiles_col=args.drug_col)
test_perturb = fcfp4_embedding(test_perturb, smiles_col=args.drug_col)


print("\n🧼 Sanitizing data...")

control_adata = sanitize_adata_matrix(control_adata, "control_bank")

train_perturb = sanitize_adata_matrix(train_perturb, "train_perturb")
val_perturb = sanitize_adata_matrix(val_perturb, "val_perturb")
test_perturb = sanitize_adata_matrix(test_perturb, "test_perturb")


print("\n🧱 Building tensors...")

control_expression_bank, control_uce_bank = build_control_bank(control_adata)
train_data = build_dataset(control_expression_bank, control_uce_bank, train_perturb)
val_data = build_dataset(control_expression_bank, control_uce_bank, val_perturb)
test_data = build_dataset(control_expression_bank, control_uce_bank, test_perturb)

train_iter = Data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True
)

val_iter = Data.DataLoader(
    val_data,
    batch_size=args.batch_size,
    shuffle=False
)

test_iter = Data.DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False
)


print("\n🧠 Initializing model...")
gene_dim = train_data.target.shape[1]
gene_symbols = get_gene_symbols(control_adata)
if len(gene_symbols) != gene_dim:
    raise ValueError(f"❌ Number of gene_symbols {len(gene_symbols)} != gene_dim {gene_dim}")

if args.use_bio_graph:
    print("\n🧬 Building sparse graph priors...")
    chromo_adj = build_sparse_graph_adj(
        args.chromo_graph_path,
        gene_symbols,
        graph_name="Chromo",
        min_importance=args.prior_min_importance,
        topk_per_target=args.prior_topk
    )
    go_adj = build_sparse_graph_adj(
        args.go_graph_path,
        gene_symbols,
        graph_name="GO",
        min_importance=args.prior_min_importance,
        topk_per_target=args.prior_topk
    )
    kegg_adj = build_sparse_graph_adj(
        args.kegg_graph_path,
        gene_symbols,
        graph_name="KEGG",
        min_importance=args.kegg_min_importance,
        topk_per_target=args.kegg_topk
    )
    tftg_adj = build_sparse_graph_adj(
        args.tftg_graph_path,
        gene_symbols,
        graph_name="TF-TG",
        min_importance=args.prior_min_importance,
        topk_per_target=args.prior_topk
    )
else:
    print("⚠️ Skip sparse graph priors")
    chromo_adj = None
    go_adj = None
    kegg_adj = None
    tftg_adj = None

if args.use_pathway_encoder:
    print("\n🧬 Building KEGG pathway encoder matrices...")
    gene_to_pathway, pathway_to_gene, num_pathways, pathway_ids = build_kegg_pathway_matrices(
        args.kegg_gene2pathway_path,
        gene_symbols
    )
    if gene_to_pathway is None or pathway_to_gene is None:
        args.use_pathway_encoder = False
        num_pathways = 0
        pathway_ids = []
else:
    print("⚠️ Skip KEGG pathway encoder")
    gene_to_pathway = None
    pathway_to_gene = None
    num_pathways = 0
    pathway_ids = []

model = DrugPerturbationModel_embedding_v2(
    gene_size=gene_dim,
    num_cell_types=len(cell_type_id),
    use_bio_graph=args.use_bio_graph,
    chromo_adj=chromo_adj,
    go_adj=go_adj,
    kegg_adj=kegg_adj,
    tftg_adj=tftg_adj,
    use_pathway_encoder=args.use_pathway_encoder,
    num_pathways=num_pathways,
    gene_to_pathway=gene_to_pathway,
    pathway_to_gene=pathway_to_gene,
    pathway_dim=args.pathway_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()

best_pearson = -1e9
best_r2 = -1e9
best_mse = 1e9
best_epoch = -1

save_path = os.path.join(args.save_dir, f"best_model_{args.run_name}.pth")


if args.preflight_only:
    print("\n🧪 Running one-batch preflight...")
    model.train()
    preflight_data = next(iter(train_iter))
    preflight_con, preflight_uce, preflight_drug, preflight_dose, preflight_cell, preflight_y = [
        x.to(device) for x in preflight_data
    ]
    preflight_outputs, preflight_cell_logits, _ = model(
        preflight_con,
        preflight_uce,
        preflight_drug,
        transform_dose_for_model(preflight_dose),
    )
    preflight_loss_expr = criterion(
        preflight_outputs - preflight_con,
        preflight_y - preflight_con,
    )
    preflight_loss_cls = criterion_cls(preflight_cell_logits, preflight_cell)
    preflight_loss = preflight_loss_expr + 0.1 * preflight_loss_cls
    if not torch.isfinite(preflight_outputs).all() or not torch.isfinite(preflight_loss):
        raise ValueError("❌ preflight forward outputs or loss contain NaN/Inf")
    optimizer.zero_grad()
    preflight_loss.backward()
    print(
        f"✅ PREFLIGHT PASSED | batch={preflight_y.shape[0]} | "
        f"loss={preflight_loss.item():.6f} | UCE dim={preflight_uce.shape[1]} | "
        f"raw dose range=({preflight_dose.min().item():.6g}, {preflight_dose.max().item():.6g}) {DOSE_UNIT}"
    )
    sys.exit(0)


print("\n🏋️ Start training...")

if args.eval_only:
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"❌ --eval_only could not find the best model checkpoint: {save_path}")
    existing_ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    if existing_ckpt.get("run_name") != args.run_name:
        raise ValueError(
            f"❌ checkpoint run_name={existing_ckpt.get('run_name')} "
            f"does not match the current run_name={args.run_name}."
        )
    best_epoch = int(existing_ckpt["epoch"])
    best_pearson = float(existing_ckpt["best_pearson"])
    best_r2 = float(existing_ckpt["best_r2"])
    best_mse = float(existing_ckpt["best_mse"])
    del existing_ckpt
    training_epochs = 0
    print(f"⏭️ --eval_only: reuse checkpoint {save_path} (best epoch {best_epoch})")
else:
    training_epochs = args.epochs

for epoch in range(training_epochs):
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for data in train_iter:
        train_con, train_uce, train_drug, train_dose, train_cell, train_y = [
            x.to(device) for x in data
        ]

        outputs, cell_logits, fusion_output2 = model(
            train_con,
            train_uce,
            train_drug,
            transform_dose_for_model(train_dose)
        )

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("❌ outputs contain NaN/Inf; skipping this batch")
            continue

        if torch.isnan(train_y).any() or torch.isinf(train_y).any():
            print("❌ train_y contains NaN/Inf; skipping this batch")
            continue

        loss_expr = criterion(outputs - train_con, train_y - train_con)
        loss_cls = criterion_cls(cell_logits, train_cell)
        loss = loss_expr + 0.1 * loss_cls

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ loss is NaN/Inf; skipping this batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        valid_batches += 1

    val_pearson, val_r2, val_mse = evaluate_model(val_iter, model)

    train_loss = running_loss / max(valid_batches, 1)

    print(
        f"[{args.run_name}] "
        f"Epoch {epoch + 1:03d} | "
        f"Val Pearson {val_pearson:.4f} | "
        f"Val Sample R2 {val_r2:.4f} | "
        f"Val MSE {val_mse:.4f} | "
        f"Train Loss {train_loss:.4f}"
    )

    if val_pearson > best_pearson:
        best_pearson = val_pearson
        best_r2 = val_r2
        best_mse = val_mse
        best_epoch = epoch + 1

        torch.save({
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pearson": best_pearson,
            "best_r2": best_r2,
            "best_mse": best_mse,
            "run_name": args.run_name,
            "seed": args.seed,
            "drug_col": args.drug_col,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "n_train_drugs": len(train_drugs),
            "n_val_drugs": len(val_drugs),
            "n_test_drugs": len(test_drugs),
            "cell_type_id": cell_type_id,
            "dose_unit": DOSE_UNIT,
            "dose_transform": DOSE_TRANSFORM,
            "perturb_dose_min_um": float(perturb_adata.obs["dose"].min()),
            "perturb_dose_max_um": float(perturb_adata.obs["dose"].max()),
            "n_unique_doses": int(perturb_adata.obs["dose"].nunique()),
            "control_path": args.control_path,
            "perturb_path": args.perturb_path,
            "uce_used": True,
            "uce_dim": int(control_adata.obsm["X_uce"].shape[1]),
            "use_bio_graph": args.use_bio_graph,
            "chromo_graph_path": args.chromo_graph_path,
            "go_graph_path": args.go_graph_path,
            "kegg_graph_path": args.kegg_graph_path,
            "tftg_graph_path": args.tftg_graph_path,
            "prior_min_importance": args.prior_min_importance,
            "prior_topk": args.prior_topk,
            "kegg_min_importance": args.kegg_min_importance,
            "kegg_topk": args.kegg_topk,
            "use_pathway_encoder": args.use_pathway_encoder,
            "kegg_gene2pathway_path": args.kegg_gene2pathway_path,
            "num_pathways": num_pathways,
            "pathway_dim": args.pathway_dim,
        }, save_path)

        print(f"💾 Saved best checkpoint to: {save_path}")


print("\n📌 Evaluating best model on validation and test sets...")

best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()

final_val_pearson, final_val_r2, final_val_mse, val_sample_metrics = evaluate_model(
    val_iter, model, return_details=True
)
test_pearson, test_r2, test_mse, test_sample_metrics = evaluate_model(
    test_iter, model, return_details=True
)

save_dose_evaluation_reports(val_sample_metrics, val_perturb, "val")
save_dose_evaluation_reports(test_sample_metrics, test_perturb, "test")

print("\n" + "=" * 80)
print("✅ Final model training finished")
print("=" * 80)
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Pearson: {best_pearson:.6f}")
print(f"Best Val Sample R2: {best_r2:.6f}")
print(f"Best Val MSE: {best_mse:.6f}")
print(f"Reloaded Val Pearson: {final_val_pearson:.6f}")
print(f"Reloaded Val Sample R2: {final_val_r2:.6f}")
print(f"Reloaded Val MSE: {final_val_mse:.6f}")
print(f"Test Pearson: {test_pearson:.6f}")
print(f"Test Sample R2: {test_r2:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Best model saved to: {save_path}")

pd.DataFrame([{
    "best_epoch": best_epoch,
    "best_val_pearson": best_pearson,
    "best_val_r2": best_r2,
    "best_val_mse": best_mse,
    "reloaded_val_pearson": final_val_pearson,
    "reloaded_val_r2": final_val_r2,
    "reloaded_val_mse": final_val_mse,
    "test_pearson": test_pearson,
    "test_r2": test_r2,
    "test_mse": test_mse,
    "n_train_drugs": len(train_drugs),
    "n_val_drugs": len(val_drugs),
    "n_test_drugs": len(test_drugs),
    "n_train_samples": train_perturb.n_obs,
    "n_val_samples": val_perturb.n_obs,
    "n_test_samples": test_perturb.n_obs,
    "dose_unit": DOSE_UNIT,
    "dose_transform": DOSE_TRANSFORM,
    "dose_min_um": float(perturb_adata.obs["dose"].min()),
    "dose_max_um": float(perturb_adata.obs["dose"].max()),
    "n_unique_doses": int(perturb_adata.obs["dose"].nunique()),
    "split_by": args.drug_col,
    "uce_used": True,
}]).to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_final_metrics.csv"),
    index=False
)
