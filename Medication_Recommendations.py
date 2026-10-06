import os
import argparse
import warnings
import random
import pickle
import numpy as np
import pandas as pd
import torch
import scanpy as sc

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse
from tqdm import tqdm

from model import DrugPerturbationModel_embedding_v2

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ===============================
# 基础工具
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


def to_dense_array(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def sanitize_array(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0.0
    return x


def sanitize_adata_matrix(adata, name="adata"):
    X = adata.X

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        data = X.data.astype(np.float32, copy=False)

        print(f"{name}:")
        print("  nan:", np.isnan(data).sum())
        print("  +inf:", np.isposinf(data).sum())
        print("  -inf:", np.isneginf(data).sum())
        print("  negative:", (data < 0).sum())

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[data < 0] = 0.0

        X.data = data
        X.eliminate_zeros()
        adata.X = X

    else:
        X = np.asarray(X, dtype=np.float32)

        print(f"{name}:")
        print("  nan:", np.isnan(X).sum())
        print("  +inf:", np.isposinf(X).sum())
        print("  -inf:", np.isneginf(X).sum())
        print("  negative:", (X < 0).sum())

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X[X < 0] = 0.0
        adata.X = X

    return adata


def limit_control_cells(adata, cell_col, max_cells, seed, label):
    if max_cells is None or int(max_cells) <= 0:
        return adata

    max_cells = int(max_cells)
    rng = np.random.default_rng(seed)
    obs_cell = adata.obs[cell_col].astype(str)
    keep_positions = []

    for cell_type in sorted(obs_cell.unique()):
        positions = np.flatnonzero(obs_cell.values == cell_type)
        original_count = len(positions)

        if original_count > max_cells:
            positions = rng.choice(positions, size=max_cells, replace=False)
            positions = np.sort(positions)

        keep_positions.extend(positions.tolist())
        print(f"{label} {cell_type}: using {len(positions)} / {original_count} control cells")

    keep_positions = np.array(sorted(keep_positions), dtype=int)
    return adata[keep_positions].copy()


def get_gene_symbols(adata):
    if "gene" in adata.var.columns:
        return adata.var["gene"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def load_control_gene_symbols(control_path: str):
    adata = sc.read_h5ad(control_path, backed="r")
    gene_symbols = get_gene_symbols(adata)
    adata.file.close()
    return gene_symbols


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

    return gene_to_pathway, pathway_to_gene, len(pathway_ids), pathway_ids


# ===============================
# SMILES 处理
# ===============================
def detect_smiles_column(df: pd.DataFrame, preferred=None) -> str:
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "SMILES", "smiles", "canonical_smiles", "CanonicalSMILES",
        "unique_smiles", "mol", "molecule", "Smiles"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        f"❌ 无法自动识别 SMILES 列。当前列为: {list(df.columns)}。"
        f"请手动指定 --smiles_col"
    )


def detect_source_column(df: pd.DataFrame):
    candidates = ["source", "supplier_id", "supplier", "type", "reference"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_smiles_table(
    smiles_df: pd.DataFrame,
    smiles_col: str,
    source_col: str = None,
    default_source: str = None
):
    table = smiles_df.copy()
    table = table.dropna(subset=[smiles_col]).copy()
    table[smiles_col] = table[smiles_col].astype(str).str.strip()

    table = table[
        ~table[smiles_col].str.lower().isin(["", "none", "nan"])
    ].copy()

    if "hit_id" not in table.columns:
        table["hit_id"] = table.index.astype(str)

    if "source" not in table.columns:
        if source_col and source_col in table.columns:
            table["source"] = table[source_col].astype(str)
        elif default_source is not None:
            table["source"] = str(default_source)
        else:
            table["source"] = "unknown"

    table = table.reset_index(drop=True)
    table["candidate_id"] = [f"candidate_{i}" for i in range(len(table))]

    return table


def build_fcfp4_embeddings(smiles_list, candidate_ids=None, n_bits: int = 1024):
    if candidate_ids is None:
        candidate_ids = [str(i) for i in range(len(smiles_list))]
    if len(candidate_ids) != len(smiles_list):
        raise ValueError("candidate_ids 与 smiles_list 长度不一致")

    fps = []
    valid_smiles = []
    valid_ids = []
    invalid_smiles = []
    invalid_ids = []

    for candidate_id, s in tqdm(list(zip(candidate_ids, smiles_list)), desc="Building FCFP4"):
        mol = Chem.MolFromSmiles(str(s))

        if mol is None:
            invalid_smiles.append(s)
            invalid_ids.append(candidate_id)
            continue

        try:
            bitvect = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=n_bits,
                useFeatures=True
            )

            arr = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bitvect, arr)

            fps.append(arr)
            valid_smiles.append(s)
            valid_ids.append(candidate_id)

        except Exception:
            invalid_smiles.append(s)
            invalid_ids.append(candidate_id)

    if not fps:
        raise ValueError("❌ 输入文件里没有有效 SMILES")

    return np.stack(fps, axis=0), valid_smiles, valid_ids, invalid_smiles, invalid_ids


# ===============================
# 控制组准备：适配新数据结构
# ===============================
def prepare_controls_by_cell_type(
    control_path: str,
    cell_type: str,
    cell_col: str = "cell_type",
    ctrl_col: str = "ctrl",
    max_control_cells: int = 0,
    seed: int = 42,
):
    adata = sc.read_h5ad(control_path)

    print("Loaded control adata:", adata)

    if "X_uce" not in adata.obsm:
        raise ValueError("❌ control h5ad 中没有 obsm['X_uce']")

    if cell_col not in adata.obs.columns:
        raise ValueError(f"❌ control adata.obs 中缺少 {cell_col}")

    if ctrl_col not in adata.obs.columns:
        raise ValueError(f"❌ control adata.obs 中缺少 {ctrl_col}")

    mask = (
        (adata.obs[ctrl_col].astype(int).values == 1) &
        (adata.obs[cell_col].astype(str).values == str(cell_type))
    )

    ctrl = adata[mask].copy()

    if ctrl.n_obs == 0:
        available = sorted(adata.obs[cell_col].astype(str).unique().tolist())
        raise ValueError(
            f"❌ 没有找到 cell_type={cell_type} 的控制组。\n"
            f"可用 cell_type 示例: {available[:30]}"
        )

    ctrl = limit_control_cells(
        ctrl,
        cell_col=cell_col,
        max_cells=max_control_cells,
        seed=seed,
        label="Selected"
    )

    ctrl = sanitize_adata_matrix(ctrl, name=f"{cell_type}_control_raw")

    gene_names = ctrl.var_names.astype(str).tolist()

    ctrl_raw = sanitize_array(to_dense_array(ctrl.X))

    ctrl_for_model = ctrl.copy()

    # 注意：
    # 这里不做 normalize_total / log1p
    # 因为你的训练代码也是直接使用当前 X
    ctrl_for_model = sanitize_adata_matrix(
        ctrl_for_model,
        name=f"{cell_type}_control_for_model"
    )

    ctrl_norm = sanitize_array(to_dense_array(ctrl_for_model.X))
    ctrl_uce = sanitize_array(np.asarray(ctrl.obsm["X_uce"], dtype=np.float32))

    return ctrl_norm, ctrl_raw, ctrl_uce, gene_names, ctrl.obs.copy()


def prepare_controls_by_cell_types_mean(
    control_path: str,
    cell_types,
    cell_col: str = "cell_type",
    ctrl_col: str = "ctrl",
    max_control_cells_per_cell_type: int = 0,
    seed: int = 42,
):
    adata = sc.read_h5ad(control_path)

    print("Loaded control adata:", adata)

    if "X_uce" not in adata.obsm:
        raise ValueError("❌ control h5ad 中没有 obsm['X_uce']")

    if cell_col not in adata.obs.columns:
        raise ValueError(f"❌ control adata.obs 中缺少 {cell_col}")

    if ctrl_col not in adata.obs.columns:
        raise ValueError(f"❌ control adata.obs 中缺少 {ctrl_col}")

    cell_types = [str(ct) for ct in cell_types]
    mask = (
        (adata.obs[ctrl_col].astype(int).values == 1) &
        (adata.obs[cell_col].astype(str).isin(cell_types))
    )

    ctrl = adata[mask].copy()

    if ctrl.n_obs == 0:
        available = sorted(adata.obs[cell_col].astype(str).unique().tolist())
        raise ValueError(
            f"❌ 没有找到 cell_type in {cell_types} 的控制组。\n"
            f"可用 cell_type 示例: {available[:30]}"
        )

    ctrl = limit_control_cells(
        ctrl,
        cell_col=cell_col,
        max_cells=max_control_cells_per_cell_type,
        seed=seed,
        label="Selected merged"
    )

    ctrl = sanitize_adata_matrix(ctrl, name="merged_control_raw")

    gene_names = ctrl.var_names.astype(str).tolist()

    ctrl_raw_all = sanitize_array(to_dense_array(ctrl.X))

    ctrl_for_model = ctrl.copy()
    ctrl_for_model = sanitize_adata_matrix(
        ctrl_for_model,
        name="merged_control_for_model"
    )

    ctrl_norm_all = sanitize_array(to_dense_array(ctrl_for_model.X))
    ctrl_uce_all = sanitize_array(np.asarray(ctrl.obsm["X_uce"], dtype=np.float32))

    ctrl_raw_mean = ctrl_raw_all.mean(axis=0, keepdims=True)
    ctrl_norm_mean = ctrl_norm_all.mean(axis=0, keepdims=True)
    ctrl_uce_mean = ctrl_uce_all.mean(axis=0, keepdims=True)

    counts = (
        ctrl.obs[cell_col].astype(str)
        .value_counts()
        .reindex(cell_types, fill_value=0)
        .to_dict()
    )

    return ctrl_norm_mean, ctrl_raw_mean, ctrl_uce_mean, gene_names, counts


# ===============================
# 模型加载
# ===============================
def infer_model_kwargs_from_state_dict(state_dict):
    gene_size = None
    uce_dim = 1280
    drug_dim = 1024
    hidden = 1024
    num_cell_types = 165
    use_bio_graph = any(k.startswith("graph_encoder.") for k in state_dict)
    use_pathway_encoder = any(k.startswith("pathway_encoder.") for k in state_dict)

    if "decoder.5.weight" in state_dict:
        gene_size = int(state_dict["decoder.5.weight"].shape[0])
    elif "decoder.5.bias" in state_dict:
        gene_size = int(state_dict["decoder.5.bias"].shape[0])

    if "input_proj.0.weight" in state_dict:
        hidden = int(state_dict["input_proj.0.weight"].shape[0])
        if gene_size is not None:
            input_dim = int(state_dict["input_proj.0.weight"].shape[1])
            graph_context_dim = hidden if use_bio_graph else 0
            uce_dim = int(input_dim - gene_size - graph_context_dim)

    if "dose_encoder.2.weight" in state_dict:
        drug_dim = int(state_dict["dose_encoder.2.weight"].shape[0])

    if "cell_type_head.3.weight" in state_dict:
        num_cell_types = int(state_dict["cell_type_head.3.weight"].shape[0])

    if gene_size is None:
        raise ValueError("❌ 无法从 checkpoint state_dict 推断 gene_size")

    model_kwargs = {
        "gene_size": gene_size,
        "uce_dim": uce_dim,
        "drug_dim": drug_dim,
        "hidden": hidden,
        "num_cell_types": num_cell_types,
        "use_bio_graph": use_bio_graph,
        "use_pathway_encoder": use_pathway_encoder,
    }

    if use_pathway_encoder and "pathway_encoder.pathway_emb.weight" in state_dict:
        model_kwargs["num_pathways"] = int(state_dict["pathway_encoder.pathway_emb.weight"].shape[0])
        model_kwargs["pathway_dim"] = int(state_dict["pathway_encoder.pathway_emb.weight"].shape[1])

    return model_kwargs


def get_ckpt_or_default(ckpt, key, default):
    value = ckpt.get(key, default)
    if value is None:
        return default
    return value


def build_model_priors_from_checkpoint(ckpt, model_kwargs, gene_symbols):
    prior_kwargs = {}

    if model_kwargs.get("use_bio_graph", False):
        print("\n🧬 Rebuilding sparse graph priors for recommendation...")
        prior_min_importance = float(get_ckpt_or_default(ckpt, "prior_min_importance", 0.0))
        prior_topk = int(get_ckpt_or_default(ckpt, "prior_topk", 21))
        kegg_min_importance = float(get_ckpt_or_default(ckpt, "kegg_min_importance", 0.1))
        kegg_topk = int(get_ckpt_or_default(ckpt, "kegg_topk", 21))

        prior_kwargs["chromo_adj"] = build_sparse_graph_adj(
            get_ckpt_or_default(
                ckpt,
                "chromo_graph_path",
                "/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_chromo_full.csv",
            ),
            gene_symbols,
            graph_name="Chromo",
            min_importance=prior_min_importance,
            topk_per_target=prior_topk,
        )
        prior_kwargs["go_adj"] = build_sparse_graph_adj(
            get_ckpt_or_default(
                ckpt,
                "go_graph_path",
                "/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_go_top21.csv",
            ),
            gene_symbols,
            graph_name="GO",
            min_importance=prior_min_importance,
            topk_per_target=prior_topk,
        )
        prior_kwargs["kegg_adj"] = build_sparse_graph_adj(
            get_ckpt_or_default(
                ckpt,
                "kegg_graph_path",
                "/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_kegg_full.csv",
            ),
            gene_symbols,
            graph_name="KEGG",
            min_importance=kegg_min_importance,
            topk_per_target=kegg_topk,
        )
        prior_kwargs["tftg_adj"] = build_sparse_graph_adj(
            get_ckpt_or_default(
                ckpt,
                "tftg_graph_path",
                "/data/home/wenjian/药物扰动模型/宇峰师兄数据/先验知识图/con_uce_tftg_full.csv",
            ),
            gene_symbols,
            graph_name="TF-TG",
            min_importance=prior_min_importance,
            topk_per_target=prior_topk,
        )

        missing = [name for name in ["chromo_adj", "go_adj", "kegg_adj", "tftg_adj"] if prior_kwargs[name] is None]
        if missing:
            raise ValueError(f"❌ checkpoint 需要生物图模块，但以下图未成功构建: {missing}")

    if model_kwargs.get("use_pathway_encoder", False):
        print("\n🧬 Rebuilding KEGG pathway encoder matrices for recommendation...")
        gene_to_pathway, pathway_to_gene, num_pathways, _ = build_kegg_pathway_matrices(
            get_ckpt_or_default(
                ckpt,
                "kegg_gene2pathway_path",
                "/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/con_uce_gene2kegg.pkl",
            ),
            gene_symbols,
        )
        if gene_to_pathway is None or pathway_to_gene is None:
            raise ValueError("❌ checkpoint 需要 pathway_encoder，但 KEGG pathway 矩阵未成功构建")

        expected_num_pathways = model_kwargs.get("num_pathways")
        if expected_num_pathways is not None and int(expected_num_pathways) != int(num_pathways):
            raise ValueError(
                f"❌ pathway 数量与 checkpoint 不一致: "
                f"rebuilt={num_pathways}, checkpoint={expected_num_pathways}"
            )

        prior_kwargs["gene_to_pathway"] = gene_to_pathway
        prior_kwargs["pathway_to_gene"] = pathway_to_gene
        prior_kwargs["num_pathways"] = num_pathways

    return prior_kwargs


def load_model(checkpoint_path: str, device: torch.device, gene_symbols):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_kwargs = infer_model_kwargs_from_state_dict(ckpt["model_state_dict"])
    prior_kwargs = build_model_priors_from_checkpoint(ckpt, model_kwargs, gene_symbols)
    model_kwargs.update(prior_kwargs)
    printable_kwargs = {}
    for key, value in model_kwargs.items():
        if torch.is_tensor(value):
            printable_kwargs[key] = {
                "shape": tuple(value.shape),
                "nnz": value._nnz() if value.layout == torch.sparse_coo else None,
            }
        else:
            printable_kwargs[key] = value
    print("Model kwargs inferred from checkpoint:", printable_kwargs)

    model = DrugPerturbationModel_embedding_v2(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


# ===============================
# 基因集处理
# ===============================
def load_gene_list(path: str):
    genes = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)

    return genes


def load_up_down_gene_sets(up_path: str, down_path: str):
    up_raw = load_gene_list(up_path)
    down_raw = load_gene_list(down_path)

    up_set = set(up_raw)
    down_set = set(down_raw)

    overlap = up_set & down_set

    up_clean = sorted(list(up_set - overlap))
    down_clean = sorted(list(down_set - overlap))

    summary = {
        "up_raw_n": len(up_raw),
        "down_raw_n": len(down_raw),
        "overlap_n": len(overlap),
        "up_clean_n": len(up_clean),
        "down_clean_n": len(down_clean),
    }

    return up_clean, down_clean, sorted(list(overlap)), summary


def filter_signature_to_genes(up_genes, down_genes, gene_set):
    up_f = [g for g in up_genes if g in gene_set]
    down_f = [g for g in down_genes if g in gene_set]

    return up_f, down_f


# ===============================
# reverse score
# ===============================
def ranklist(DT):
    ranks = DT.rank(ascending=False, method="first")
    return ranks


def compute_a_score(query_genes, ranked_matrix):
    p = len(query_genes)

    if p == 0:
        raise ValueError("query_genes for a-score is empty")

    n = ranked_matrix.shape[0]

    sub = ranked_matrix.loc[query_genes, :].to_numpy()
    r_sorted = np.sort(sub, axis=0)

    m = np.arange(1, p + 1, dtype=float).reshape(-1, 1)

    a_matrix = m / p - r_sorted / n

    return a_matrix.max(axis=0)


def compute_b_score(query_genes, ranked_matrix):
    q = len(query_genes)

    if q == 0:
        raise ValueError("query_genes for b-score is empty")

    n = ranked_matrix.shape[0]

    sub = ranked_matrix.loc[query_genes, :].to_numpy()
    r_sorted = np.sort(sub, axis=0)

    m = np.arange(1, q + 1, dtype=float).reshape(-1, 1)

    b_inner = r_sorted / n - (m - 1) / q

    return b_inner.max(axis=0)


def computecs(qup, qdown, expression):
    """
    reverse score:
    - melanoma 上调基因，希望药物压低
    - melanoma 下调基因，希望药物拉高

    score 越小，逆转效果越强。
    """
    ranked_matrix = ranklist(expression)

    if not qup or not qdown:
        raise ValueError("Both qup and qdown are required.")

    a_up = compute_a_score(qup, ranked_matrix)
    b_up = compute_b_score(qup, ranked_matrix)
    a_down = compute_a_score(qdown, ranked_matrix)
    b_down = compute_b_score(qdown, ranked_matrix)

    score_up = a_up - b_up
    score_down = a_down - b_down
    score = score_up - score_down

    return pd.DataFrame(score, index=expression.columns, columns=["score"])


def score_predicted_drugs_from_fc(
    fc_df: pd.DataFrame,
    up_genes,
    down_genes,
    score_col_name="reverse_score"
):
    expression = fc_df.T.copy()
    gene_set = set(expression.index)

    up_f, down_f = filter_signature_to_genes(up_genes, down_genes, gene_set)

    summary = pd.DataFrame([{
        "up_input_n": len(up_genes),
        "down_input_n": len(down_genes),
        "up_overlap_with_expression_n": len(up_f),
        "down_overlap_with_expression_n": len(down_f),
        "overlap_total_n": len(up_f) + len(down_f),
    }])

    if len(up_f) < 5 or len(down_f) < 5:
        raise ValueError(
            f"❌ 与表达矩阵重叠基因太少: up={len(up_f)}, down={len(down_f)}"
        )

    cs = computecs(up_f, down_f, expression)
    cs = cs.rename(columns={"score": score_col_name})
    ranked = cs.sort_values(score_col_name, ascending=True)
    ranked.index.name = fc_df.index.name

    return ranked, summary, up_f, down_f


# ===============================
# 预测

def predict_mean_profiles(
    model,
    controls_norm: np.ndarray,
    controls_raw: np.ndarray,
    controls_uce: np.ndarray,
    smiles_embeddings: np.ndarray,
    valid_ids,
    dose_um: float,
    gene_names,
    batch_size: int,
    device: torch.device,
):
    """
    加速版：
    - 如果某个 cell_type 只有 1 个 control，则一次预测很多药物
    - 避免候选药物逐个 forward
    """
    eps = 1e-6
    n_ctrl = controls_norm.shape[0]
    n_drug = smiles_embeddings.shape[0]

    mean_perturbed_rows = []
    mean_fc_rows = []

    model.eval()

    with torch.no_grad():

        # ===============================
        # 情况1：只有一个 control，最快
        # ===============================
        if n_ctrl == 1:
            ctrl_norm = controls_norm[0:1]
            ctrl_raw = controls_raw[0:1]
            ctrl_uce = controls_uce[0:1]

            for start in tqdm(range(0, n_drug, batch_size), desc="Predicting drugs fast"):
                end = min(start + batch_size, n_drug)
                cur_bs = end - start

                x_con = torch.tensor(
                    np.repeat(ctrl_norm, cur_bs, axis=0),
                    dtype=torch.float32,
                    device=device
                )

                x_uce = torch.tensor(
                    np.repeat(ctrl_uce, cur_bs, axis=0),
                    dtype=torch.float32,
                    device=device
                )

                x_drug = torch.tensor(
                    smiles_embeddings[start:end],
                    dtype=torch.float32,
                    device=device
                )

                x_dose = torch.full(
                    (cur_bs, 1),
                    float(dose_um),
                    dtype=torch.float32,
                    device=device
                )

                outputs, _, _ = model(x_con, x_uce, x_drug, x_dose)

                pred_expr = outputs.detach().cpu().numpy()
                pred_expr = sanitize_array(pred_expr)

                pred_logfc = np.log2((pred_expr + eps) / (ctrl_raw + eps))

                mean_perturbed_rows.append(pred_expr)
                mean_fc_rows.append(pred_logfc)

            perturbed_arr = np.concatenate(mean_perturbed_rows, axis=0)
            fc_arr = np.concatenate(mean_fc_rows, axis=0)

        # ===============================
        # 情况2：多个 control，仍然 batch 化
        # ===============================
        else:
            all_perturbed = []
            all_fc = []

            for start in tqdm(range(0, n_drug, batch_size), desc="Predicting drugs fast"):
                end = min(start + batch_size, n_drug)
                cur_drugs = smiles_embeddings[start:end]
                cur_bs = end - start

                batch_preds = []

                for c in range(n_ctrl):
                    x_con = torch.tensor(
                        np.repeat(controls_norm[c:c+1], cur_bs, axis=0),
                        dtype=torch.float32,
                        device=device
                    )

                    x_uce = torch.tensor(
                        np.repeat(controls_uce[c:c+1], cur_bs, axis=0),
                        dtype=torch.float32,
                        device=device
                    )

                    x_drug = torch.tensor(
                        cur_drugs,
                        dtype=torch.float32,
                        device=device
                    )

                    x_dose = torch.full(
                        (cur_bs, 1),
                        float(dose_um),
                        dtype=torch.float32,
                        device=device
                    )

                    outputs, _, _ = model(x_con, x_uce, x_drug, x_dose)
                    pred = outputs.detach().cpu().numpy()
                    pred = sanitize_array(pred)

                    batch_preds.append(pred)

                batch_preds = np.stack(batch_preds, axis=0)
                mean_pred = batch_preds.mean(axis=0)

                fc_each_ctrl = []
                for c in range(n_ctrl):
                    fc = np.log2((batch_preds[c] + eps) / (controls_raw[c:c+1] + eps))
                    fc_each_ctrl.append(fc)

                mean_fc = np.stack(fc_each_ctrl, axis=0).mean(axis=0)

                all_perturbed.append(mean_pred)
                all_fc.append(mean_fc)

            perturbed_arr = np.concatenate(all_perturbed, axis=0)
            fc_arr = np.concatenate(all_fc, axis=0)

    perturbed_df = pd.DataFrame(
        perturbed_arr,
        index=valid_ids,
        columns=gene_names
    )

    fc_df = pd.DataFrame(
        fc_arr,
        index=valid_ids,
        columns=gene_names
    )

    perturbed_df.index.name = "candidate_id"
    fc_df.index.name = "candidate_id"

    return perturbed_df, fc_df

# ===============================
# 单个 cell_type 推荐流程
# ===============================
def run_single_cell_type(
    model,
    control_path,
    cell_type,
    smiles_embeddings,
    valid_ids,
    valid_meta_df,
    dose_um,
    batch_size,
    device,
    up_genes,
    down_genes,
    out_dir,
    max_control_cells,
    seed,
):
    print("\n" + "=" * 80)
    print(f"Processing cell_type: {cell_type}")
    print("=" * 80)

    controls_norm, controls_raw, controls_uce, gene_names, control_obs = prepare_controls_by_cell_type(
        control_path=control_path,
        cell_type=cell_type,
        cell_col="cell_type",
        ctrl_col="ctrl",
        max_control_cells=max_control_cells,
        seed=seed,
    )

    print(f"Control samples for {cell_type}: {controls_norm.shape[0]}")
    print(f"Number of genes: {controls_norm.shape[1]}")
    print(f"X_uce dim: {controls_uce.shape[1]}")

    if controls_norm.shape[1] != model.gene_size:
        raise ValueError(
            f"❌ control gene 数与模型不一致: "
            f"control={controls_norm.shape[1]}, model={model.gene_size}。"
            f"请确认 --control_path 与训练 checkpoint 时使用的基因集合和顺序一致。"
        )

    if controls_uce.shape[1] != model.uce_dim:
        raise ValueError(
            f"❌ control X_uce 维度与模型不一致: "
            f"control={controls_uce.shape[1]}, model={model.uce_dim}"
        )

    if smiles_embeddings.shape[1] != model.drug_dim:
        raise ValueError(
            f"❌ 药物指纹维度与模型不一致: "
            f"fingerprint={smiles_embeddings.shape[1]}, model={model.drug_dim}。"
            f"请确认 --fp_bits 与训练时一致。"
        )

    control_raw_df = pd.DataFrame(
        controls_raw,
        columns=gene_names,
        index=control_obs.index.astype(str)
    )
    control_raw_df.index.name = "sample_id"

    control_raw_path = os.path.join(
        out_dir,
        f"{cell_type}_control_raw_expr.csv"
    )
    control_raw_df.to_csv(control_raw_path)

    control_norm_df = pd.DataFrame(
        controls_norm,
        columns=gene_names,
        index=control_obs.index.astype(str)
    )
    control_norm_df.index.name = "sample_id"

    control_norm_path = os.path.join(
        out_dir,
        f"{cell_type}_control_for_model_expr.csv"
    )
    control_norm_df.to_csv(control_norm_path)

    perturbed_df, fc_df = predict_mean_profiles(
        model=model,
        controls_norm=controls_norm,
        controls_raw=controls_raw,
        controls_uce=controls_uce,
        smiles_embeddings=smiles_embeddings,
        valid_ids=valid_ids,
        dose_um=dose_um,
        gene_names=gene_names,
        batch_size=batch_size,
        device=device,
    )

    dose_str = str(dose_um).replace(".0", "")

    # pert_path = os.path.join(
    #     out_dir,
    #     f"{cell_type}_predicted_perturbed_expr_{dose_str}uM.csv"
    # )

    fc_path = os.path.join(
        out_dir,
        f"{cell_type}_predicted_log2FC_{dose_str}uM.csv"
    )

    # perturbed_df.to_csv(pert_path)
    fc_df.to_csv(fc_path)

    score_col = f"{cell_type}_reverse_score"

    ranked_scores, score_summary_df, up_filtered, down_filtered = score_predicted_drugs_from_fc(
        fc_df=fc_df,
        up_genes=up_genes,
        down_genes=down_genes,
        score_col_name=score_col,
    )

    pd.DataFrame({"gene": up_filtered}).to_csv(
        os.path.join(out_dir, f"{cell_type}_filtered_up_genes_used.csv"),
        index=False
    )

    pd.DataFrame({"gene": down_filtered}).to_csv(
        os.path.join(out_dir, f"{cell_type}_filtered_down_genes_used.csv"),
        index=False
    )

    score_summary_path = os.path.join(
        out_dir,
        f"{cell_type}_reverse_score_gene_summary_{dose_str}uM.csv"
    )
    score_summary_df.to_csv(score_summary_path, index=False)

    per_cell_result = valid_meta_df.join(ranked_scores, how="inner")
    per_cell_result = per_cell_result.sort_values(score_col, ascending=True)

    per_cell_result_path = os.path.join(
        out_dir,
        f"{cell_type}_predicted_reverse_scores_{dose_str}uM.csv"
    )

    per_cell_result.to_csv(per_cell_result_path, index=False)

    print(f"Saved {cell_type} reverse-score ranking to:")
    print(per_cell_result_path)

    return per_cell_result[[score_col]].copy()


def run_merged_cell_types(
    model,
    control_path,
    cell_types,
    smiles_embeddings,
    valid_ids,
    valid_meta_df,
    dose_um,
    batch_size,
    device,
    up_genes,
    down_genes,
    out_dir,
    max_control_cells_per_cell_type,
    seed,
):
    label = "_".join(cell_types) + "_merged"

    print("\n" + "=" * 80)
    print(f"Processing merged cell_types: {cell_types}")
    print("=" * 80)

    controls_norm, controls_raw, controls_uce, gene_names, counts = prepare_controls_by_cell_types_mean(
        control_path=control_path,
        cell_types=cell_types,
        cell_col="cell_type",
        ctrl_col="ctrl",
        max_control_cells_per_cell_type=max_control_cells_per_cell_type,
        seed=seed,
    )

    print("Merged control counts:", counts)
    print(f"Merged controls (mean): {controls_norm.shape[0]}")
    print(f"Number of genes: {controls_norm.shape[1]}")
    print(f"X_uce dim: {controls_uce.shape[1]}")

    if controls_norm.shape[1] != model.gene_size:
        raise ValueError(
            f"❌ control gene 数与模型不一致: "
            f"control={controls_norm.shape[1]}, model={model.gene_size}。"
            f"请确认 --control_path 与训练 checkpoint 时使用的基因集合和顺序一致。"
        )

    if controls_uce.shape[1] != model.uce_dim:
        raise ValueError(
            f"❌ control X_uce 维度与模型不一致: "
            f"control={controls_uce.shape[1]}, model={model.uce_dim}"
        )

    if smiles_embeddings.shape[1] != model.drug_dim:
        raise ValueError(
            f"❌ 药物指纹维度与模型不一致: "
            f"fingerprint={smiles_embeddings.shape[1]}, model={model.drug_dim}。"
            f"请确认 --fp_bits 与训练时一致。"
        )

    control_raw_df = pd.DataFrame(
        controls_raw,
        columns=gene_names,
        index=["merged_mean"]
    )
    control_raw_df.index.name = "sample_id"

    control_raw_path = os.path.join(
        out_dir,
        f"{label}_control_raw_expr_mean.csv"
    )
    control_raw_df.to_csv(control_raw_path)

    control_norm_df = pd.DataFrame(
        controls_norm,
        columns=gene_names,
        index=["merged_mean"]
    )
    control_norm_df.index.name = "sample_id"

    control_norm_path = os.path.join(
        out_dir,
        f"{label}_control_for_model_expr_mean.csv"
    )
    control_norm_df.to_csv(control_norm_path)

    perturbed_df, fc_df = predict_mean_profiles(
        model=model,
        controls_norm=controls_norm,
        controls_raw=controls_raw,
        controls_uce=controls_uce,
        smiles_embeddings=smiles_embeddings,
        valid_ids=valid_ids,
        dose_um=dose_um,
        gene_names=gene_names,
        batch_size=batch_size,
        device=device,
    )

    dose_str = str(dose_um).replace(".0", "")

    pert_path = os.path.join(
        out_dir,
        f"{label}_predicted_perturbed_expr_{dose_str}uM.csv"
    )

    fc_path = os.path.join(
        out_dir,
        f"{label}_predicted_log2FC_{dose_str}uM.csv"
    )

    perturbed_df.to_csv(pert_path)
    fc_df.to_csv(fc_path)

    score_col = f"{label}_reverse_score"

    ranked_scores, score_summary_df, up_filtered, down_filtered = score_predicted_drugs_from_fc(
        fc_df=fc_df,
        up_genes=up_genes,
        down_genes=down_genes,
        score_col_name=score_col,
    )

    pd.DataFrame({"gene": up_filtered}).to_csv(
        os.path.join(out_dir, f"{label}_filtered_up_genes_used.csv"),
        index=False
    )

    pd.DataFrame({"gene": down_filtered}).to_csv(
        os.path.join(out_dir, f"{label}_filtered_down_genes_used.csv"),
        index=False
    )

    score_summary_path = os.path.join(
        out_dir,
        f"{label}_reverse_score_gene_summary_{dose_str}uM.csv"
    )
    score_summary_df.to_csv(score_summary_path, index=False)

    merged_result = valid_meta_df.join(ranked_scores, how="inner")
    merged_result = merged_result.sort_values(score_col, ascending=True)

    merged_result_path = os.path.join(
        out_dir,
        f"{label}_predicted_reverse_scores_{dose_str}uM.csv"
    )

    merged_result.to_csv(merged_result_path, index=False)

    print("Saved merged reverse-score ranking to:")
    print(merged_result_path)

    return merged_result[[score_col]].copy(), label


# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图全部扰动/训练验证结果/finalmodel_seed3/best_model_final_10uM_celltype_pair.pth"
    )

    parser.add_argument(
        "--control_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/processed/ctl_vehicle_default164_ctrl1.h5ad"
    )

    parser.add_argument(
        "--smiles_csv",
        type=str,
        default="/data/home/wenjian/药物扰动模型/小分子库/topscience/t001/t001.csv"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/home/wenjian/药物扰动模型/宇峰师兄数据/生物图全部扰动/药物推荐结果/finalmodel_seed3_结构"
    )

    parser.add_argument(
        "--cell_types",
        nargs="+",
        default=["A375", "SKMEL5"]
    )

    parser.add_argument(
        "--merge_cell_types_first",
        action="store_true",
        help="先合并多个 cell_type 的 control 表达取均值，再进行药物推荐"
    )

    parser.add_argument(
        "--max_control_cells_per_cell_type",
        type=int,
        default=100,
        help="每个指定 cell_type 最多使用的控制组细胞数量；0 表示使用全部"
    )

    parser.add_argument("--dose_um", type=float, default=10.0)
    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--fp_bits", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--up_genes_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/我的数据/melanoma_cancer_up_genes.txt"
    )

    parser.add_argument(
        "--down_genes_path",
        type=str,
        default="/data/home/wenjian/药物扰动模型/我的数据/melanoma_cancer_down_genes.txt"
    )

    args = parser.parse_args()

    if args.max_control_cells_per_cell_type < 0:
        raise ValueError("❌ --max_control_cells_per_cell_type 必须 >= 0")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print("Loading trained model...")
    gene_symbols_for_model = load_control_gene_symbols(args.control_path)
    model, ckpt = load_model(args.checkpoint_path, device, gene_symbols_for_model)

    if len(gene_symbols_for_model) != model.gene_size:
        raise ValueError(
            f"❌ control gene 数与模型不一致: "
            f"control={len(gene_symbols_for_model)}, model={model.gene_size}。"
            f"请确认 --control_path 与训练 checkpoint 时使用的基因集合和顺序一致。"
        )

    print(f"Loaded checkpoint from epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Best val pearson: {ckpt.get('best_pearson', 'unknown')}")
    print(f"Checkpoint run_name: {ckpt.get('run_name', 'unknown')}")
    print(f"Checkpoint dose: {ckpt.get('dose_value', 'unknown')} uM")
    print(f"Dose used for prediction: {args.dose_um} uM")

    ckpt_dose = ckpt.get("dose_value")
    if ckpt_dose is not None and abs(float(ckpt_dose) - float(args.dose_um)) > 1e-8:
        raise ValueError(
            f"❌ 预测 dose 与训练 checkpoint 不一致: "
            f"checkpoint dose={ckpt_dose}, prediction dose={args.dose_um}"
        )

    # 2. 加载候选药物
    print("\nLoading candidate SMILES...")
    smiles_df = pd.read_csv(args.smiles_csv)

    smiles_col = detect_smiles_column(smiles_df, args.smiles_col)
    source_col = detect_source_column(smiles_df)

    smiles_table = prepare_smiles_table(
        smiles_df=smiles_df,
        smiles_col=smiles_col,
        source_col=source_col,
        default_source=os.path.splitext(os.path.basename(args.smiles_csv))[0],
    )

    smiles_list = smiles_table[smiles_col].tolist()
    candidate_ids = smiles_table["candidate_id"].tolist()

    print(f"SMILES column: {smiles_col}")
    print(f"Source column: {source_col if source_col else '<derived>'}")
    print(f"Input candidate rows: {len(smiles_list)}")
    print(f"Unique input SMILES: {pd.Series(smiles_list).nunique()}")

    # 3. 构建 FCFP4
    smiles_embeddings, valid_smiles, valid_ids, invalid_smiles, invalid_ids = build_fcfp4_embeddings(
        smiles_list,
        candidate_ids=candidate_ids,
        n_bits=args.fp_bits
    )

    print(f"Valid SMILES: {len(valid_smiles)}")
    print(f"Invalid SMILES: {len(invalid_smiles)}")

    valid_meta_df = smiles_table.set_index("candidate_id").loc[valid_ids].copy()
    valid_meta_df["candidate_id"] = valid_meta_df.index
    if smiles_col != "SMILES":
        valid_meta_df = valid_meta_df.rename(columns={smiles_col: "SMILES"})
    valid_meta_df.index.name = "candidate_id"

    bad_path = os.path.join(args.out_dir, "invalid_smiles.csv")
    if invalid_smiles:
        pd.DataFrame({
            "candidate_id": invalid_ids,
            "invalid_smiles": invalid_smiles,
        }).to_csv(
            bad_path,
            index=False
        )

    # 4. 加载 melanoma 上下调基因集
    print("\nLoading disease up/down gene sets...")

    up_genes, down_genes, overlap_genes, sig_summary = load_up_down_gene_sets(
        args.up_genes_path,
        args.down_genes_path
    )

    overlap_path = os.path.join(
        args.out_dir,
        "up_down_overlap_removed_genes.csv"
    )

    pd.DataFrame({"overlap_gene": overlap_genes}).to_csv(
        overlap_path,
        index=False
    )

    sig_summary_df = pd.DataFrame([sig_summary])
    sig_summary_df["dose_um"] = args.dose_um

    sig_summary_path = os.path.join(
        args.out_dir,
        "reverse_score_gene_summary_all_cell_types.csv"
    )

    sig_summary_df.to_csv(sig_summary_path, index=False)

    dose_str = str(args.dose_um).replace(".0", "")

    if args.merge_cell_types_first and len(args.cell_types) > 1:
        merged_scores, merged_label = run_merged_cell_types(
            model=model,
            control_path=args.control_path,
            cell_types=args.cell_types,
            smiles_embeddings=smiles_embeddings,
            valid_ids=valid_ids,
            valid_meta_df=valid_meta_df,
            dose_um=args.dose_um,
            batch_size=args.batch_size,
            device=device,
            up_genes=up_genes,
            down_genes=down_genes,
            out_dir=args.out_dir,
            max_control_cells_per_cell_type=args.max_control_cells_per_cell_type,
            seed=args.seed,
        )

        final_results = valid_meta_df.join(merged_scores, how="inner")
        score_col = f"{merged_label}_reverse_score"
        final_results = final_results.sort_values(score_col, ascending=True).reset_index(drop=True)

        ranked_result_path = os.path.join(
            args.out_dir,
            f"{merged_label}_predicted_reverse_scores_{dose_str}uM.csv"
        )

        final_results.to_csv(ranked_result_path, index=False)

        print("\nDone.")
        print(f"Saved merged reverse-score ranking to: {ranked_result_path}")
        print(f"Saved overlap genes to: {overlap_path}")
        print(f"Saved gene summary to: {sig_summary_path}")

        if invalid_smiles:
            print(f"Saved invalid SMILES to: {bad_path}")

        return

    # 5. 每个 cell_type 分别推荐
    all_score_tables = []

    for cell_type in args.cell_types:
        per_cell_scores = run_single_cell_type(
            model=model,
            control_path=args.control_path,
            cell_type=cell_type,
            smiles_embeddings=smiles_embeddings,
            valid_ids=valid_ids,
            valid_meta_df=valid_meta_df,
            dose_um=args.dose_um,
            batch_size=args.batch_size,
            device=device,
            up_genes=up_genes,
            down_genes=down_genes,
            out_dir=args.out_dir,
            max_control_cells=args.max_control_cells_per_cell_type,
            seed=args.seed,
        )

        all_score_tables.append(per_cell_scores)

    # 6. 合并多个 cell_type 的 reverse score
    final_results = valid_meta_df.copy()

    for score_df in all_score_tables:
        final_results = final_results.join(score_df, how="inner")

    score_cols = [f"{ct}_reverse_score" for ct in args.cell_types]

    final_results["mean_reverse_score"] = final_results[score_cols].mean(axis=1)

    final_results = final_results.sort_values(
        "mean_reverse_score",
        ascending=True
    ).reset_index(drop=True)

    ranked_result_path = os.path.join(
        args.out_dir,
        f'{"_".join(args.cell_types)}_predicted_reverse_scores_{dose_str}uM.csv'
    )

    final_results.to_csv(ranked_result_path, index=False)

    print("\nDone.")
    print(f"Saved combined reverse-score ranking to: {ranked_result_path}")
    print(f"Saved overlap genes to: {overlap_path}")
    print(f"Saved gene summary to: {sig_summary_path}")

    if invalid_smiles:
        print(f"Saved invalid SMILES to: {bad_path}")


if __name__ == "__main__":
    main()
