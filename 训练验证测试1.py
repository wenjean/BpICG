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

from model import *
from data_utils import *
from evo_loss import *

warnings.filterwarnings("ignore")


# ===============================
# 参数
# ===============================
parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str, default="final_10uM_celltype_pair")
parser.add_argument("--drug_col", type=str, default="smiles")
parser.add_argument("--seed", type=int, default=11)

parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--test_ratio", type=float, default=0.1)

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
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
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_kegg_full.csv"
)

parser.add_argument(
    "--chromo_graph_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_chromo_full.csv"
)

parser.add_argument(
    "--go_graph_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_go_top21.csv"
)

parser.add_argument(
    "--tftg_graph_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_tftg_full.csv"
)

parser.add_argument(
    "--kegg_gene2pathway_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/con_uce_gene2kegg.pkl"
)

parser.add_argument(
    "--control_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/ctl_vehicle_default164_ctrl1.h5ad"
)

parser.add_argument(
    "--perturb_path",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/10uM-24h_default164_ctrl0.h5ad"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图全部扰动/训练验证测试结果/finalmodel3"
)

args = parser.parse_args()

assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-8


# ===============================
# 固定随机种子
# ===============================
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


# ===============================
# 工具函数
# ===============================
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
        raise ValueError(f"❌ {graph_name} 缺少列 {required_cols}: {graph_path}")

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
        # A[target, source], so x @ A.T aggregates source neighbors into target genes.
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
        # [P, G], pathway activity = mean expression of member genes.
        g2p_rows.append(pathway_idx)
        g2p_cols.append(gene_idx)
        g2p_vals.append(1.0 / max(pathway_sizes[pathway_idx], 1.0))

        # [G, P], gene delta bias = mean pathway delta of pathways containing the gene.
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

    # 转 dense（你当前模型是 dense tensor）
    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    # 1. library size normalization (CPM)
    lib_size = X.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    X = X / lib_size * 1e6

    # 2. log1p transform
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

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
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

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X[X < 0] = 0.0
        adata.X = X

    return adata


def filter_low_variance_perturb_samples(
    adata,
    min_var=1e-4,
    save_dir=None,
    run_name="run"
):
    """
    过滤表达方差极低的扰动样本。
    逐样本 R2 = 1 - MSE / var(y_true)，如果样本自身方差接近 0，
    即使 MSE 正常也会产生巨大的负 R2。
    """
    X = to_dense_array(adata.X).astype(np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    row_mean = X.mean(axis=1)
    row_var = X.var(axis=1)
    row_std = np.sqrt(row_var)
    row_min = X.min(axis=1)
    row_max = X.max(axis=1)
    zero_fraction = (X == 0).mean(axis=1)

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

    return adata[keep_mask].copy(), removed_df


def fcfp4_embedding(adata, smiles_col="smiles", n_bits=1024, radius=2):
    if smiles_col not in adata.obs.columns:
        raise ValueError(f"❌ adata.obs 中缺少 smiles 列: {smiles_col}")

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
    """
    当前 perturb_adata 已经全部是 10uM 扰动组。
    按 drug_col 划分 train/val/test，保证同一个药物不会同时出现在不同集合。
    """
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

    train_perturb = perturb_adata[train_mask].copy()
    val_perturb = perturb_adata[val_mask].copy()
    test_perturb = perturb_adata[test_mask].copy()

    return train_perturb, val_perturb, test_perturb, train_drugs, val_drugs, test_drugs


def random_pair_control_by_cell_type(
    perturb,
    control,
    seed=42,
    cell_col="cell_type"
):
    """
    按 cell_type 给每个扰动样本随机匹配一个同细胞系控制组。
    你的 control 里 ctrl==1，且一般每个 cell_type 只有一个控制样本。
    """
    rng = np.random.RandomState(seed)

    if "ctrl" in control.obs.columns:
        control_pool_adata = control[control.obs["ctrl"].astype(int).values == 1].copy()
    else:
        control_pool_adata = control.copy()

    print(f"Control samples: {control_pool_adata.n_obs}")
    print(f"Control cell types: {control_pool_adata.obs[cell_col].nunique()}")

    control_pool = {}
    for cell, sub_obs in control_pool_adata.obs.groupby(cell_col):
        control_pool[str(cell)] = sub_obs.index.values

    matched_control_indices = []
    valid_perturb_indices = []
    missing_cell_types = {}

    for obs_name, row in perturb.obs.iterrows():
        cell = str(row[cell_col])

        if cell not in control_pool or len(control_pool[cell]) == 0:
            missing_cell_types[cell] = missing_cell_types.get(cell, 0) + 1
            continue

        chosen_ctl = rng.choice(control_pool[cell])

        matched_control_indices.append(chosen_ctl)
        valid_perturb_indices.append(obs_name)

    if len(valid_perturb_indices) == 0:
        raise ValueError("❌ 没有任何 perturb 样本能匹配到相同 cell_type 的 control")

    if len(missing_cell_types) > 0:
        print("⚠️ 以下 cell_type 在 control 中没有匹配，将被丢弃：")
        print(missing_cell_types)

    matched_perturb = perturb[valid_perturb_indices].copy()
    matched_control = control_pool_adata[matched_control_indices].copy()

    matched_perturb.obs["paired_control_obs_name"] = matched_control.obs_names.values
    matched_control.obs["paired_perturb_obs_name"] = matched_perturb.obs_names.values

    print(f"Matched perturb samples: {matched_perturb.n_obs}")
    print(f"Matched control samples: {matched_control.n_obs}")

    return matched_control, matched_perturb


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


def build_tensor(control, perturb):
    X = torch.tensor(to_dense_array(control.X), dtype=torch.float32)

    if "X_uce" not in control.obsm:
        raise ValueError("❌ control.obsm 中缺少 X_uce")

    if "X_fcfp4" not in perturb.obsm:
        raise ValueError("❌ perturb.obsm 中缺少 X_fcfp4")

    uce = torch.tensor(np.asarray(control.obsm["X_uce"]), dtype=torch.float32)
    drug = torch.tensor(np.asarray(perturb.obsm["X_fcfp4"]), dtype=torch.float32)

    dose = torch.tensor(
        perturb.obs["dose"].astype(float).values,
        dtype=torch.float32
    ).view(-1, 1)

    cell_type = torch.tensor(
        perturb.obs["cell_type_id"].astype(int).values,
        dtype=torch.long
    )

    Y = torch.tensor(to_dense_array(perturb.X), dtype=torch.float32)

    assert X.shape[0] == Y.shape[0] == uce.shape[0] == drug.shape[0] == dose.shape[0]

    print("X shape:", X.shape)
    print("uce shape:", uce.shape)
    print("drug shape:", drug.shape)
    print("dose shape:", dose.shape)
    print("Y shape:", Y.shape)

    print("X nan/inf:", torch.isnan(X).any().item(), torch.isinf(X).any().item())
    print("uce nan/inf:", torch.isnan(uce).any().item(), torch.isinf(uce).any().item())
    print("drug nan/inf:", torch.isnan(drug).any().item(), torch.isinf(drug).any().item())
    print("dose nan/inf:", torch.isnan(dose).any().item(), torch.isinf(dose).any().item())
    print("Y nan/inf:", torch.isnan(Y).any().item(), torch.isinf(Y).any().item())

    return X, uce, drug, dose, cell_type, Y


def evaluate_model(dataloader, model):
    model.eval()

    pearson_sum = 0.0
    r2_sum = 0.0
    mse_sum = 0.0
    n = 0
    skipped = 0

    with torch.no_grad():
        for batch_id, data in enumerate(dataloader):
            batch_con, batch_uce, batch_drug_embeddings, batch_doses, batch_cell_type, batch_y = data

            batch_con = batch_con.to(device)
            batch_uce = batch_uce.to(device)
            batch_drug_embeddings = batch_drug_embeddings.to(device)
            batch_doses = batch_doses.to(device)
            batch_y = batch_y.to(device)

            outputs, _, _ = model(
                batch_con,
                batch_uce,
                batch_drug_embeddings,
                batch_doses
            )

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"⚠️ batch {batch_id} outputs 含 NaN/Inf，跳过该 batch")
                skipped += outputs.size(0)
                continue

            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                print(f"⚠️ batch {batch_id} batch_y 含 NaN/Inf，跳过该 batch")
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

            if y_true_batch.shape[0] == 0:
                continue

            for i in range(y_true_batch.shape[0]):
                y_true = y_true_batch[i]
                y_pred = y_pred_batch[i]

                try:
                    p = pearsonr(y_true, y_pred)[0]
                    if np.isnan(p):
                        p = 0.0
                except Exception:
                    p = 0.0

                try:
                    r2 = r2_score(y_true, y_pred)
                    if np.isnan(r2):
                        r2 = 0.0
                except Exception:
                    r2 = 0.0

                try:
                    mse = mean_squared_error(y_true, y_pred)
                    if np.isnan(mse):
                        mse = 0.0
                except Exception:
                    mse = 0.0

                pearson_sum += p
                r2_sum += r2
                mse_sum += mse
                n += 1

    print(f"evaluate valid samples: {n}, skipped: {skipped}")

    if n == 0:
        return 0.0, 0.0, 0.0

    mean_pearson = pearson_sum / n
    mean_r2 = r2_sum / n
    mean_mse = mse_sum / n

    return mean_pearson, mean_r2, mean_mse


# ===============================
# 读取数据
# ===============================
print("📦 Loading data...")

control_adata = sc.read_h5ad(args.control_path)
perturb_adata = sc.read_h5ad(args.perturb_path)

print("control_adata:", control_adata)
print("perturb_adata:", perturb_adata)

required_cols = ["cell_type", args.drug_col, "ctrl", "condition"]

for col in required_cols:
    if col not in perturb_adata.obs.columns:
        raise ValueError(f"❌ perturb_adata.obs 中缺少 {col} 列")

for col in ["cell_type", "ctrl", "condition"]:
    if col not in control_adata.obs.columns:
        raise ValueError(f"❌ control_adata.obs 中缺少 {col} 列")

if "X_uce" not in control_adata.obsm:
    raise ValueError("❌ control_adata.obsm 中没有 X_uce，请确认控制组 h5ad 已经包含 UCE 特征")


# ===============================
# 只保留扰动组和控制组
# ===============================
perturb_adata = perturb_adata[perturb_adata.obs["ctrl"].astype(int).values == 0].copy()
control_adata = control_adata[control_adata.obs["ctrl"].astype(int).values == 1].copy()

print("After filter:")
print("perturb_adata:", perturb_adata.shape)
print("control_adata:", control_adata.shape)


# ===============================
# 当前扰动组全部是 10uM
# ===============================
perturb_adata.obs["dose"] = 10.0
control_adata.obs["dose"] = 0.0

print("✅ Set perturb dose = 10.0 uM")
print("✅ Set control dose = 0.0")


# ===============================
# gene 对齐
# ===============================
if not np.array_equal(perturb_adata.var_names, control_adata.var_names):
    print("⚠️ perturb 和 control 的 var_names 不完全一致，开始取交集并对齐")

    common_genes = perturb_adata.var_names.intersection(control_adata.var_names)
    print("common genes:", len(common_genes))

    if len(common_genes) == 0:
        raise ValueError("❌ perturb 和 control 没有共同基因")

    perturb_adata = perturb_adata[:, common_genes].copy()
    control_adata = control_adata[:, common_genes].copy()

else:
    print("✅ perturb 和 control 的 var_names 完全一致")


print("🔬 Applying log1p + CPM normalization...")

# control_adata = normalize_log1p(control_adata)
# perturb_adata = normalize_log1p(perturb_adata)


# ===============================
# 过滤低方差异常扰动样本
# ===============================
if args.filter_low_var_perturb:
    perturb_adata, filtered_low_var_perturb_df = filter_low_variance_perturb_samples(
        perturb_adata,
        min_var=args.min_perturb_expr_var,
        save_dir=args.save_dir,
        run_name=args.run_name
    )

    if perturb_adata.n_obs == 0:
        raise ValueError("❌ 低方差过滤后 perturb_adata 没有剩余样本，请调低 --min_perturb_expr_var")
else:
    print("⚠️ Skip low-variance perturb sample filtering")

# ===============================
# cell_type id 映射
# ===============================
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

print(f"✅ cell_type 数量: {len(cell_type_id)}")


# ===============================
# train / val / test 按药物划分
# ===============================
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


# ===============================
# 按 cell_type 匹配控制组
# ===============================
print("\n🔄 Pairing controls by same cell_type...")

train_control, train_perturb = random_pair_control_by_cell_type(
    perturb=train_perturb,
    control=control_adata,
    seed=args.seed,
    cell_col="cell_type"
)

val_control, val_perturb = random_pair_control_by_cell_type(
    perturb=val_perturb,
    control=control_adata,
    seed=args.seed + 1,
    cell_col="cell_type"
)

test_control, test_perturb = random_pair_control_by_cell_type(
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


# ===============================
# 构建药物指纹
# ===============================
print("\n🧬 Building drug fingerprints...")

train_perturb = fcfp4_embedding(train_perturb, smiles_col=args.drug_col)
val_perturb = fcfp4_embedding(val_perturb, smiles_col=args.drug_col)
test_perturb = fcfp4_embedding(test_perturb, smiles_col=args.drug_col)


# ===============================
# 数据清洗
# ===============================
print("\n🧼 Sanitizing data...")

train_control = sanitize_adata_matrix(train_control, "train_control")
val_control = sanitize_adata_matrix(val_control, "val_control")
test_control = sanitize_adata_matrix(test_control, "test_control")

train_perturb = sanitize_adata_matrix(train_perturb, "train_perturb")
val_perturb = sanitize_adata_matrix(val_perturb, "val_perturb")
test_perturb = sanitize_adata_matrix(test_perturb, "test_perturb")


# ===============================
# Tensor / DataLoader
# ===============================
print("\n🧱 Building tensors...")

train_data = build_tensor(train_control, train_perturb)
val_data = build_tensor(val_control, val_perturb)
test_data = build_tensor(test_control, test_perturb)

train_iter = Data.DataLoader(
    Data.TensorDataset(*train_data),
    batch_size=args.batch_size,
    shuffle=True
)

val_iter = Data.DataLoader(
    Data.TensorDataset(*val_data),
    batch_size=args.batch_size,
    shuffle=False
)

test_iter = Data.DataLoader(
    Data.TensorDataset(*test_data),
    batch_size=args.batch_size,
    shuffle=False
)


# ===============================
# 初始化模型
# ===============================
print("\n🧠 Initializing model...")
gene_dim = train_data[5].shape[1]
gene_symbols = get_gene_symbols(control_adata)
if len(gene_symbols) != gene_dim:
    raise ValueError(f"❌ gene_symbols 数量 {len(gene_symbols)} != gene_dim {gene_dim}")

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


# ===============================
# 训练
# ===============================
print("\n🏋️ Start training...")

for epoch in range(args.epochs):
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
            train_dose
        )

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("❌ outputs 出现 NaN/Inf，跳过该 batch")
            continue

        if torch.isnan(train_y).any() or torch.isinf(train_y).any():
            print("❌ train_y 出现 NaN/Inf，跳过该 batch")
            continue

        loss_expr = criterion(outputs - train_con, train_y - train_con)
        loss_cls = criterion_cls(cell_logits, train_cell)
        loss = loss_expr + 0.1 * loss_cls

        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ loss 出现 NaN/Inf，跳过该 batch")
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
            "dose_value": 10.0,
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


# ===============================
# 测试集评估
# ===============================
print("\n📌 Evaluating best model on test set...")

best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()

test_pearson, test_r2, test_mse = evaluate_model(test_iter, model)

print("\n" + "=" * 80)
print("✅ Final model training finished")
print("=" * 80)
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Pearson: {best_pearson:.6f}")
print(f"Best Val Sample R2: {best_r2:.6f}")
print(f"Best Val MSE: {best_mse:.6f}")
print(f"Test Pearson: {test_pearson:.6f}")
print(f"Test Sample R2: {test_r2:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Best model saved to: {save_path}")

pd.DataFrame([{
    "best_epoch": best_epoch,
    "best_val_pearson": best_pearson,
    "best_val_r2": best_r2,
    "best_val_mse": best_mse,
    "test_pearson": test_pearson,
    "test_r2": test_r2,
    "test_mse": test_mse,
    "n_train_drugs": len(train_drugs),
    "n_val_drugs": len(val_drugs),
    "n_test_drugs": len(test_drugs),
    "n_train_samples": train_perturb.n_obs,
    "n_val_samples": val_perturb.n_obs,
    "n_test_samples": test_perturb.n_obs,
    "dose": 10.0,
}]).to_csv(
    os.path.join(args.save_dir, f"{args.run_name}_final_metrics.csv"),
    index=False
)
