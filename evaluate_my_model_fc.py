import argparse
import gc
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.utils.data as Data
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse
from scipy.stats import rankdata


MODEL_CODE_DIR = "/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图评估其他模型"
sys.path.insert(0, MODEL_CODE_DIR)
from model import DrugPerturbationModel_embedding_v2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Load the biological-graph model and compute CPA-style extra metrics.")
    parser.add_argument("--run_name", type=str, default="final_10uM_celltype_pair")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图评估其他模型/训练验证测试结果_随机/lmodelseed3/best_model_final_10uM_celltype_pair.pth",
    )
    parser.add_argument(
        "--train_result_dir",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图评估其他模型/训练验证测试结果_随机/lmodelseed3",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/con_uce.h5ad",
    )
    parser.add_argument(
        "--perturb_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/10um_24h_mean_perturbation_varnames_gene.h5ad",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/评估数据/我的结果/训练验证测试结果_随机/lfcseed3",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--min_expression",
        type=float,
        default=0.1,
        help="Only genes with max(mean_treated, mean_control) >= this value can enter log2FC top-k selection.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_dense_array(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def sanitize_adata_matrix(adata):
    x = to_dense_array(adata.X).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0.0
    adata.X = x.astype(np.float32)
    return adata


def get_gene_symbols(adata):
    if "gene" in adata.var.columns:
        return adata.var["gene"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def build_sparse_graph_adj(graph_path, gene_symbols, min_importance=0.0, topk_per_target=0):
    if graph_path is None or str(graph_path).lower() in {"", "none"} or not os.path.exists(graph_path):
        return None

    edge_df = pd.read_csv(graph_path)
    edge_df = edge_df[edge_df["importance"].astype(float) >= float(min_importance)].copy()
    if topk_per_target is not None and int(topk_per_target) > 0:
        edge_df = (
            edge_df.groupby("target", group_keys=False)
            .apply(lambda x: x.nlargest(int(topk_per_target), ["importance"]))
            .reset_index(drop=True)
        )

    gene_to_idx = {str(g): i for i, g in enumerate(gene_symbols)}
    rows, cols, vals = [], [], []
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

    if not vals:
        return None

    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, size=(len(gene_symbols), len(gene_symbols))).coalesce()
    idx = adj.indices()
    val = adj.values()
    row_sum = torch.zeros(len(gene_symbols), dtype=torch.float32)
    row_sum.index_add_(0, idx[0], val)
    norm_val = val / row_sum[idx[0]].clamp_min(1e-12)
    return torch.sparse_coo_tensor(idx, norm_val, adj.shape).coalesce()


def build_kegg_pathway_matrices(gene2pathway_path, gene_symbols):
    if gene2pathway_path is None or str(gene2pathway_path).lower() in {"", "none"} or not os.path.exists(gene2pathway_path):
        return None, None, 0

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
        return None, None, 0

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

    g2p_rows, g2p_cols, g2p_vals = [], [], []
    p2g_rows, p2g_cols, p2g_vals = [], [], []
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
        size=(len(pathway_ids), len(gene_symbols)),
    ).coalesce()
    pathway_to_gene = torch.sparse_coo_tensor(
        torch.tensor([p2g_rows, p2g_cols], dtype=torch.long),
        torch.tensor(p2g_vals, dtype=torch.float32),
        size=(len(gene_symbols), len(pathway_ids)),
    ).coalesce()
    return gene_to_pathway, pathway_to_gene, len(pathway_ids)


def smiles_to_fcfp4(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useFeatures=True)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_fcfp4(adata, smiles_col):
    smiles = adata.obs[smiles_col].astype(str).values
    cache = {smi: smiles_to_fcfp4(smi) for smi in sorted(set(smiles))}
    adata.obsm["X_fcfp4"] = np.stack([cache[smi] for smi in smiles], axis=0).astype(np.float32)
    return adata


def build_split_from_pairs(control_adata, perturb_adata, pair_path, drug_col, cell_type_id):
    pairs = pd.read_csv(pair_path)
    perturb = perturb_adata[pairs["perturb_obs_name"].astype(str).tolist()].copy()
    control = control_adata[pairs["paired_control_obs_name"].astype(str).tolist()].copy()

    perturb.obs["dose"] = pairs["dose"].astype(float).values
    control.obs["dose"] = 0.0
    perturb.obs["cell_type_id"] = perturb.obs["cell_type"].astype(str).map(cell_type_id).astype(int)
    control.obs["cell_type_id"] = control.obs["cell_type"].astype(str).map(cell_type_id).astype(int)
    perturb.obs["paired_control_obs_name"] = pairs["paired_control_obs_name"].astype(str).values
    perturb = add_fcfp4(perturb, drug_col)
    perturb = sanitize_adata_matrix(perturb)
    control = sanitize_adata_matrix(control)
    return control, perturb, pairs


def build_tensor(control, perturb):
    x_control = torch.tensor(to_dense_array(control.X), dtype=torch.float32)
    x_uce = torch.tensor(np.asarray(control.obsm["X_uce"]), dtype=torch.float32)
    drug = torch.tensor(np.asarray(perturb.obsm["X_fcfp4"]), dtype=torch.float32)
    dose = torch.tensor(perturb.obs["dose"].astype(float).values, dtype=torch.float32).view(-1, 1)
    cell_type = torch.tensor(perturb.obs["cell_type_id"].astype(int).values, dtype=torch.long)
    y_true = torch.tensor(to_dense_array(perturb.X), dtype=torch.float32)
    return x_control, x_uce, drug, dose, cell_type, y_true


def row_pearson(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=1)
    denominator = np.sqrt(np.sum(true_centered ** 2, axis=1) * np.sum(pred_centered ** 2, axis=1))
    out = numerator / np.maximum(denominator, eps)
    out[~np.isfinite(out)] = 0.0
    return out


def row_spearman(y_true, y_pred):
    true_rank = np.apply_along_axis(rankdata, 1, y_true)
    pred_rank = np.apply_along_axis(rankdata, 1, y_pred)
    return row_pearson(true_rank, pred_rank)


def row_r2(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    out = 1.0 - ss_res / np.maximum(ss_tot, eps)
    out[~np.isfinite(out)] = 0.0
    return out


def row_mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=1)


def topk_by_abs_logfc(log2fc, candidate_mask, k):
    candidate_idx = np.where(candidate_mask & np.isfinite(log2fc))[0]
    if candidate_idx.size == 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), candidate_idx.size)
    candidate_scores = np.abs(log2fc[candidate_idx])
    idx_unsorted = candidate_idx[np.argpartition(candidate_scores, -k)[-k:]]
    order = np.argsort(-np.abs(log2fc[idx_unsorted]))
    return idx_unsorted[order]


def gather_topk(values, top_idx):
    row_idx = np.arange(values.shape[0])[:, None]
    return values[row_idx, top_idx]


def compute_chunk_metrics(y_true, y_pred, x_control, top_idx):
    delta_true = y_true - x_control
    delta_pred = y_pred - x_control
    metrics = {
        "pcc": row_pearson(y_true, y_pred),
        "scc": row_spearman(y_true, y_pred),
        "r2": row_r2(y_true, y_pred),
        "mse": row_mse(y_true, y_pred),
        "delta_pcc": row_pearson(delta_true, delta_pred),
        "delta_scc": row_spearman(delta_true, delta_pred),
        "delta_r2": row_r2(delta_true, delta_pred),
        "delta_mse": row_mse(delta_true, delta_pred),
    }
    y_true_top = gather_topk(y_true, top_idx)
    y_pred_top = gather_topk(y_pred, top_idx)
    delta_true_top = gather_topk(delta_true, top_idx)
    delta_pred_top = gather_topk(delta_pred, top_idx)
    metrics.update({
        "top100_pcc": row_pearson(y_true_top, y_pred_top),
        "top100_scc": row_spearman(y_true_top, y_pred_top),
        "top100_r2": row_r2(y_true_top, y_pred_top),
        "top100_mse": row_mse(y_true_top, y_pred_top),
        "top100_delta_pcc": row_pearson(delta_true_top, delta_pred_top),
        "top100_delta_scc": row_spearman(delta_true_top, delta_pred_top),
        "top100_delta_r2": row_r2(delta_true_top, delta_pred_top),
        "top100_delta_mse": row_mse(delta_true_top, delta_pred_top),
    })
    return metrics


def build_celltype_drug_keys(metadata):
    if "condition" in metadata.columns:
        return metadata["condition"].astype(str).to_numpy()
    if "cell_type" not in metadata.columns:
        raise ValueError("Cannot build celltype-drug keys: missing cell_type in metadata.")
    cell_type = metadata["cell_type"].astype(str).to_numpy()
    for col in ("drug_name", "drug", "cmap_name", "smiles", "SMILES"):
        if col in metadata.columns:
            drug = metadata[col].astype(str).to_numpy()
            break
    else:
        raise ValueError("Cannot build celltype-drug keys: missing drug identifier in metadata.")
    return np.char.add(np.char.add(cell_type.astype(str), "__"), drug.astype(str))


def compute_logfc_top_indices(y_true, x_control, condition_keys, gene_symbols, topk, eps, min_expression):
    top_indices_by_key = {}
    rows = []
    genes = np.asarray(gene_symbols, dtype=str)
    for key in pd.unique(pd.Series(condition_keys)):
        sample_idx = np.where(condition_keys == key)[0]
        if sample_idx.size == 0:
            continue

        mean_treated = y_true[sample_idx].mean(axis=0)
        mean_control = x_control[sample_idx].mean(axis=0)
        log2fc = np.log2(mean_treated + eps) - np.log2(mean_control + eps)
        candidate_mask = np.maximum(mean_treated, mean_control) >= min_expression
        top_idx = topk_by_abs_logfc(log2fc, candidate_mask, topk)
        top_indices_by_key[key] = top_idx

        for rank, gene_idx in enumerate(top_idx, start=1):
            rows.append({
                "condition_key": key,
                "rank": rank,
                "gene_index": int(gene_idx),
                "gene": genes[gene_idx],
                "n_samples": int(sample_idx.size),
                "mean_treated": float(mean_treated[gene_idx]),
                "mean_control": float(mean_control[gene_idx]),
                "log2fc": float(log2fc[gene_idx]),
                "abs_log2fc": float(abs(log2fc[gene_idx])),
                "min_expression": float(min_expression),
                "n_candidate_genes": int(candidate_mask.sum()),
            })

    return top_indices_by_key, pd.DataFrame(rows)


def expand_top_indices(condition_keys, top_indices_by_key, topk):
    rows = []
    for key in condition_keys:
        top_idx = top_indices_by_key[key]
        if top_idx.size == 0:
            raise ValueError(f"No genes passed log2FC top-k filtering for condition {key!r}.")
        if top_idx.size != topk:
            padded = np.resize(top_idx, topk)
            top_idx = padded.astype(np.int64, copy=False)
        rows.append(top_idx)
    return np.vstack(rows)


def build_logfc_top_indices_for_split(tensor_data, metadata, gene_symbols, save_dir, run_name, split_name, topk, eps, min_expression):
    x_control = tensor_data[0].cpu().numpy().astype(np.float32, copy=False)
    y_true = tensor_data[5].cpu().numpy().astype(np.float32, copy=False)
    condition_keys = build_celltype_drug_keys(metadata)
    print(f"{split_name}: computing celltype-drug log2FC top{topk} for {len(pd.unique(pd.Series(condition_keys)))} conditions", flush=True)
    top_indices_by_key, top_genes_df = compute_logfc_top_indices(
        y_true,
        x_control,
        condition_keys,
        gene_symbols,
        topk,
        eps,
        min_expression,
    )
    top_genes_path = os.path.join(save_dir, f"{run_name}_{split_name}_logfc_top{topk}_genes.csv")
    top_genes_df.to_csv(top_genes_path, index=False)
    print(f"Saved logFC top genes: {top_genes_path}")
    top_idx_all = expand_top_indices(condition_keys, top_indices_by_key, min(int(topk), y_true.shape[1]))
    return condition_keys, top_idx_all, len(top_indices_by_key)


def predict_split(model, split_name, tensor_data, metadata, save_dir, run_name, batch_size, topk, top_idx_all, condition_keys, n_logfc_conditions, min_expression, device):
    loader = Data.DataLoader(Data.TensorDataset(*tensor_data), batch_size=batch_size, shuffle=False)
    metric_rows = []
    sample_start = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_con, batch_uce, batch_drug, batch_dose, _, batch_y = [x.to(device) for x in batch]
            outputs, _, _ = model(batch_con, batch_uce, batch_drug, batch_dose)
            y_pred = torch.nan_to_num(outputs).cpu().numpy().astype(np.float32)
            y_true = batch_y.cpu().numpy().astype(np.float32)
            x_control = batch_con.cpu().numpy().astype(np.float32)

            end = sample_start + y_true.shape[0]
            print(f"{split_name}: {sample_start}:{end} / {len(metadata)}", flush=True)
            chunk_metrics = compute_chunk_metrics(y_true, y_pred, x_control, top_idx_all[sample_start:end])
            chunk_df = pd.DataFrame(chunk_metrics)
            chunk_df["logfc_condition_key"] = condition_keys[sample_start:end]
            chunk_df["sample_index"] = np.arange(sample_start, end)
            metric_rows.append(chunk_df)

            sample_start = end

    metrics_df = pd.concat(metric_rows, axis=0, ignore_index=True)
    out_df = pd.concat([metadata.reset_index(drop=True), metrics_df], axis=1)
    metric_cols = [c for c in metrics_df.columns if c not in {"sample_index", "logfc_condition_key"}]
    summary_df = pd.DataFrame([{
        "split": split_name,
        "n_samples": int(metrics_df.shape[0]),
        "topk": int(topk),
        "top100_method": "celltype_drug_abs_log2fc",
        "min_expression": float(min_expression),
        "n_logfc_conditions": int(n_logfc_conditions),
        **{c: float(out_df[c].mean()) for c in metric_cols},
    }])

    sample_path = os.path.join(save_dir, f"{run_name}_{split_name}_extra_sample_metrics.csv")
    summary_path = os.path.join(save_dir, f"{run_name}_{split_name}_extra_metrics_summary.csv")
    out_df.to_csv(sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved sample metrics: {sample_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df


def evaluate_split_from_pairs(
    model,
    split_name,
    pair_path,
    control_adata,
    perturb_adata,
    drug_col,
    cell_type_id,
    gene_symbols,
    args,
    device,
):
    control, perturb, pairs = build_split_from_pairs(control_adata, perturb_adata, pair_path, drug_col, cell_type_id)
    tensor_data = build_tensor(control, perturb)
    condition_keys, top_idx_all, n_logfc_conditions = build_logfc_top_indices_for_split(
        tensor_data,
        pairs,
        gene_symbols,
        args.save_dir,
        args.run_name,
        split_name,
        args.topk,
        args.eps,
        args.min_expression,
    )
    summary = predict_split(
        model,
        split_name,
        tensor_data,
        pairs.copy(),
        args.save_dir,
        args.run_name,
        args.batch_size,
        args.topk,
        top_idx_all,
        condition_keys,
        n_logfc_conditions,
        args.min_expression,
        device,
    )
    del control, perturb, pairs, tensor_data, condition_keys, top_idx_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    set_seed(int(ckpt.get("seed", 11)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint epoch:", ckpt.get("epoch"))

    control_adata = sc.read_h5ad(args.control_path)
    perturb_adata = sc.read_h5ad(args.perturb_path)
    perturb_adata = perturb_adata[perturb_adata.obs["ctrl"].astype(int).values == 0].copy()
    control_adata = control_adata[control_adata.obs["ctrl"].astype(int).values == 1].copy()
    perturb_adata.obs["dose"] = float(ckpt.get("dose_value", 10.0))
    control_adata.obs["dose"] = 0.0

    if not np.array_equal(perturb_adata.var_names, control_adata.var_names):
        common_genes = perturb_adata.var_names.intersection(control_adata.var_names)
        perturb_adata = perturb_adata[:, common_genes].copy()
        control_adata = control_adata[:, common_genes].copy()

    gene_symbols = get_gene_symbols(control_adata)
    val_pair_path = os.path.join(args.train_result_dir, f"{args.run_name}_val_pairs.csv")
    test_pair_path = os.path.join(args.train_result_dir, f"{args.run_name}_test_pairs.csv")
    cell_type_id = ckpt["cell_type_id"]
    drug_col = ckpt.get("drug_col", "smiles")
    gene_dim = control_adata.n_vars

    chromo_adj = go_adj = kegg_adj = tftg_adj = None
    if ckpt.get("use_bio_graph", False):
        chromo_adj = build_sparse_graph_adj(ckpt.get("chromo_graph_path"), gene_symbols, ckpt.get("prior_min_importance", 0.0), ckpt.get("prior_topk", 0))
        go_adj = build_sparse_graph_adj(ckpt.get("go_graph_path"), gene_symbols, ckpt.get("prior_min_importance", 0.0), ckpt.get("prior_topk", 0))
        kegg_adj = build_sparse_graph_adj(ckpt.get("kegg_graph_path"), gene_symbols, ckpt.get("kegg_min_importance", 0.0), ckpt.get("kegg_topk", 0))
        tftg_adj = build_sparse_graph_adj(ckpt.get("tftg_graph_path"), gene_symbols, ckpt.get("prior_min_importance", 0.0), ckpt.get("prior_topk", 0))

    gene_to_pathway = pathway_to_gene = None
    num_pathways = 0
    use_pathway_encoder = ckpt.get("use_pathway_encoder", False)
    if use_pathway_encoder:
        gene_to_pathway, pathway_to_gene, num_pathways = build_kegg_pathway_matrices(ckpt.get("kegg_gene2pathway_path"), gene_symbols)
        if gene_to_pathway is None or pathway_to_gene is None:
            use_pathway_encoder = False

    model = DrugPerturbationModel_embedding_v2(
        gene_size=gene_dim,
        num_cell_types=len(cell_type_id),
        use_bio_graph=ckpt.get("use_bio_graph", False),
        chromo_adj=chromo_adj,
        go_adj=go_adj,
        kegg_adj=kegg_adj,
        tftg_adj=tftg_adj,
        use_pathway_encoder=use_pathway_encoder,
        num_pathways=num_pathways,
        gene_to_pathway=gene_to_pathway,
        pathway_to_gene=pathway_to_gene,
        pathway_dim=ckpt.get("pathway_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    summaries = [
        evaluate_split_from_pairs(model, "val", val_pair_path, control_adata, perturb_adata, drug_col, cell_type_id, gene_symbols, args, device),
        evaluate_split_from_pairs(model, "test", test_pair_path, control_adata, perturb_adata, drug_col, cell_type_id, gene_symbols, args, device),
    ]
    all_summary = pd.concat(summaries, axis=0, ignore_index=True)
    all_path = os.path.join(args.save_dir, f"{args.run_name}_extra_metrics_summary.csv")
    all_summary.to_csv(all_path, index=False)
    print(f"Saved combined summary: {all_path}")


if __name__ == "__main__":
    main()
