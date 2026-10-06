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
# 读取控制组和扰动组数据
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


class GraphPriorEncoder(nn.Module):
    def __init__(
        self,
        gene_size,
        hidden,
        chromo_adj=None,
        go_adj=None,
        kegg_adj=None,
        tftg_adj=None,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.gene_size = gene_size
        self.hidden = hidden

        graph_adjs = {
            "chromo": chromo_adj,
            "go": go_adj,
            "kegg": kegg_adj,
            "tftg": tftg_adj,
        }
        self.graph_names = [name for name, adj in graph_adjs.items() if adj is not None]

        for name, adj in graph_adjs.items():
            self.register_buffer(
                f"{name}_adj",
                self._prepare_adj(adj),
                persistent=False,
            )

        self.gene_embedding = nn.Embedding(gene_size, hidden)
        self.graph_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden, hidden),
            )
            for name in self.graph_names
        })
        self.graph_norms = nn.ModuleDict({
            name: nn.LayerNorm(hidden)
            for name in self.graph_names
        })
        self.graph_gate_logits = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(-2.0))
            for name in self.graph_names
        })
        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def _prepare_adj(adj):
        if adj is None:
            return None
        if adj.layout != torch.sparse_coo:
            adj = adj.to_sparse_coo()
        return adj.coalesce().float()

    def _propagate(self, x, adj):
        if adj is None or adj._nnz() == 0:
            return torch.zeros_like(x)
        return torch.sparse.mm(adj, x)

    def forward(self, x):
        gene_ids = torch.arange(self.gene_size, device=x.device)
        base_nodes = self.gene_embedding(gene_ids)

        fused_nodes = torch.zeros_like(base_nodes)
        for name in self.graph_names:
            adj = getattr(self, f"{name}_adj")
            graph_nodes = self._propagate(base_nodes, adj)
            graph_nodes = self.graph_mlps[name](graph_nodes)
            graph_nodes = self.graph_norms[name](graph_nodes)
            gate = torch.sigmoid(self.graph_gate_logits[name])
            fused_nodes = fused_nodes + gate * graph_nodes

        fused_nodes = self.dropout(self.output_norm(fused_nodes))

        expr_weight = torch.clamp(x, min=0.0)
        expr_weight = expr_weight / expr_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.matmul(expr_weight, fused_nodes)


class PathwayPriorEncoder(nn.Module):
    def __init__(
        self,
        gene_size,
        drug_dim,
        num_pathways,
        gene_to_pathway=None,
        pathway_to_gene=None,
        pathway_dim=128,
        dropout_rate=0.1
    ):
        super().__init__()
        self.gene_size = gene_size
        self.num_pathways = num_pathways
        self.pathway_dim = pathway_dim

        self.register_buffer(
            "gene_to_pathway",
            self._prepare_adj(gene_to_pathway),
            persistent=True
        )
        self.register_buffer(
            "pathway_to_gene",
            self._prepare_adj(pathway_to_gene),
            persistent=True
        )

        self.activity_proj = nn.Sequential(
            nn.Linear(1, pathway_dim),
            nn.ReLU(),
            nn.LayerNorm(pathway_dim)
        )
        self.pathway_emb = nn.Embedding(num_pathways, pathway_dim)
        self.drug_query = nn.Linear(drug_dim, pathway_dim)
        self.pathway_delta = nn.Sequential(
            nn.Linear(pathway_dim, pathway_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pathway_dim, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def _prepare_adj(adj):
        if adj is None:
            return None
        if adj.layout != torch.sparse_coo:
            adj = adj.to_sparse_coo()
        return adj.coalesce().float()

    def forward(self, x_expr, drug_emb):
        if (
            self.gene_to_pathway is None
            or self.pathway_to_gene is None
            or self.num_pathways == 0
        ):
            return torch.zeros_like(x_expr)

        # [B, G] -> [B, P], pathway activity is the mean expression of member genes.
        pathway_activity = torch.sparse.mm(
            self.gene_to_pathway,
            x_expr.transpose(0, 1)
        ).transpose(0, 1)

        pathway_ids = torch.arange(self.num_pathways, device=x_expr.device)
        pathway_tokens = (
            self.activity_proj(pathway_activity.unsqueeze(-1))
            + self.pathway_emb(pathway_ids).unsqueeze(0)
        )
        pathway_tokens = self.dropout(pathway_tokens)

        query = self.drug_query(drug_emb).unsqueeze(1)
        attn = torch.softmax(
            torch.matmul(query, pathway_tokens.transpose(1, 2))
            / (self.pathway_dim ** 0.5),
            dim=-1
        ).squeeze(1)

        pathway_delta = self.pathway_delta(pathway_tokens).squeeze(-1)
        pathway_delta = pathway_delta * attn

        # [B, P] -> [B, G], distribute drug-attended pathway deltas to genes.
        gene_delta_bias = torch.sparse.mm(
            self.pathway_to_gene,
            pathway_delta.transpose(0, 1)
        ).transpose(0, 1)

        return gene_delta_bias

class DrugPerturbationModel_embedding_v2(nn.Module):
    def __init__(
        self,
        gene_size=12328,
        uce_dim=1280,
        drug_dim=1024,
        hidden=1024,
        num_cell_types=165,
        dropout_rate=0.1,
        use_bio_graph=False,
        chromo_adj=None,
        go_adj=None,
        kegg_adj=None,
        tftg_adj=None,
        use_pathway_encoder=False,
        num_pathways=0,
        gene_to_pathway=None,
        pathway_to_gene=None,
        pathway_dim=128
    ):
        super().__init__()

        # allow dynamic gene size to match dataset after filtering
        self.gene_size = gene_size
        self.uce_dim = uce_dim
        self.drug_dim = drug_dim
        self.hidden = hidden
        self.num_cell_types = num_cell_types
        self.dropout_rate = dropout_rate
        self.use_bio_graph = use_bio_graph
        self.use_pathway_encoder = use_pathway_encoder

        if self.use_bio_graph:
            self.graph_encoder = GraphPriorEncoder(
                gene_size=self.gene_size,
                hidden=self.hidden,
                chromo_adj=chromo_adj,
                go_adj=go_adj,
                kegg_adj=kegg_adj,
                tftg_adj=tftg_adj,
                dropout_rate=self.dropout_rate
            )
            graph_context_dim = self.hidden
        else:
            self.graph_encoder = None
            graph_context_dim = 0

        if self.use_pathway_encoder:
            self.pathway_encoder = PathwayPriorEncoder(
                gene_size=self.gene_size,
                drug_dim=self.drug_dim,
                num_pathways=num_pathways,
                gene_to_pathway=gene_to_pathway,
                pathway_to_gene=pathway_to_gene,
                pathway_dim=pathway_dim,
                dropout_rate=self.dropout_rate
            )
            self.pathway_gate_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.pathway_encoder = None
            self.pathway_gate_logit = None

        # ===============================
        # Dose Encoder（重要升级）
        # ===============================
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.drug_dim)
        )

        # ===============================
        # 输入融合（control + UCE）
        # ===============================
        # input dimension equals (uce_dim + gene_size)
        self.input_proj = nn.Sequential(
            nn.Linear(self.uce_dim + self.gene_size + graph_context_dim, self.hidden),
            nn.ReLU(),
            nn.LayerNorm(self.hidden),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.hidden)
        )

        # ===============================
        # Multi-token Attention（核心升级）
        # ===============================
        self.num_tokens = 8
        self.token_dim = self.hidden // self.num_tokens  # 128

        self.query_map = nn.Linear(self.token_dim, self.token_dim)
        self.key_map = nn.Linear(self.token_dim, self.token_dim)
        self.value_map = nn.Linear(self.token_dim, self.token_dim)

        self.attn_dropout = nn.Dropout(self.dropout_rate)

        # ===============================
        # Fusion MLP
        # ===============================
        self.fusion_decoder = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.LayerNorm(self.hidden),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.hidden)
        )

        # ===============================
        # Cell type head
        # ===============================
        self.cell_type_head = nn.Sequential(
            nn.Linear(self.hidden, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_cell_types)
        )

        # ===============================
        # Decoder（更深）
        # ===============================
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.gene_size)
        )

    def forward(self, x_pert, x_uce, drug_emb, drug_dose):

        # ===============================
        # Dose embedding（替代原乘法）
        # ===============================
        # drug_dose = drug_dose.unsqueeze(1)
        dose_emb = self.dose_encoder(drug_dose)
        drug_emb = drug_emb + dose_emb

        # ===============================
        # 输入融合
        # ===============================
        x_expr = x_pert.squeeze(1)
        if self.use_bio_graph:
            graph_context = self.graph_encoder(x_expr)
            x = torch.cat((x_uce, x_expr, graph_context), dim=1)
        else:
            x = torch.cat((x_uce, x_expr), dim=1)
        x = self.input_proj(x)

        # ===============================
        # reshape成多token（关键）
        # ===============================
        B = x.shape[0]

        x_tokens = x.view(B, self.num_tokens, self.token_dim)
        drug_tokens = drug_emb.view(B, self.num_tokens, self.token_dim)

        # ===============================
        # Cross Attention
        # ===============================
        Q = self.query_map(x_tokens)
        K = self.key_map(drug_tokens)
        V = self.value_map(drug_tokens)

        attn = torch.softmax(
            torch.matmul(Q, K.transpose(1, 2)) / (self.token_dim ** 0.5),
            dim=-1
        )

        attn = self.attn_dropout(attn)

        fusion = torch.matmul(attn, V)

        # flatten
        fusion = fusion.reshape(B, -1)

        # ===============================
        # Fusion MLP
        # ===============================
        fusion = self.fusion_decoder(fusion)

        # ===============================
        # Cell type prediction
        # ===============================
        cell_type_logits = self.cell_type_head(fusion)

        # ===============================
        # 拼接 + Decoder
        # ===============================
        # fusion2 = torch.cat((fusion, x_pert.squeeze(1)), dim=1)

        delta = self.decoder(fusion)

        if self.use_pathway_encoder:
            pathway_delta_bias = self.pathway_encoder(x_expr, drug_emb)
            delta = delta + torch.sigmoid(self.pathway_gate_logit) * pathway_delta_bias

        # ===============================
        # Residual（非常关键）
        # ===============================
        output = x_expr + delta

        return output, cell_type_logits, fusion
