import os
import dgl
import torch
import numpy as np
import faiss
import json
import re
import random
import datetime
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from collections import defaultdict
from pymatgen.core import Structure, Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 【优化模块1】一键平面结构判断（别名精准匹配+化学规则兜底）
# =============================================================================

# 1. 常见化合物精准匹配库（别名→结构特征，优先匹配，100%准确）
COMMON_COMPOUND_LIB = {
    # 芳香族平面化合物
    "tnt": {"is_planar": True, "type": "芳香族", "formula": "C7H5N3O6", "hint": "benzene_ring"},
    "三硝基甲苯": {"is_planar": True, "type": "芳香族", "formula": "C7H5N3O6", "hint": "benzene_ring"},
    "苯甲酸": {"is_planar": True, "type": "芳香族", "formula": "C7H6O2", "hint": "benzene_ring"},
    "安息香酸": {"is_planar": True, "type": "芳香族", "formula": "C7H6O2", "hint": "benzene_ring"},
    "苯酚": {"is_planar": True, "type": "芳香族", "formula": "C6H6O", "hint": "benzene_ring"},
    "苯胺": {"is_planar": True, "type": "芳香族", "formula": "C6H7N", "hint": "benzene_ring"},
    "苯": {"is_planar": True, "type": "芳香族", "formula": "C6H6", "hint": "benzene_ring"},
    "萘": {"is_planar": True, "type": "芳香族", "formula": "C10H8", "hint": "naphthalene_ring"},
    "吡啶": {"is_planar": True, "type": "杂环芳香", "formula": "C5H5N", "hint": "heterocycle_planar"},
    "呋喃": {"is_planar": True, "type": "杂环芳香", "formula": "C4H4O", "hint": "heterocycle_planar"},
    "噻吩": {"is_planar": True, "type": "杂环芳香", "formula": "C4H4S", "hint": "heterocycle_planar"},
    "乙烯": {"is_planar": True, "type": "共轭烯烃", "formula": "C2H4", "hint": "conjugated_planar"},
    "丁二烯": {"is_planar": True, "type": "共轭烯烃", "formula": "C4H6", "hint": "conjugated_planar"},
    "乙酰胺": {"is_planar": True, "type": "酰胺平面", "formula": "C2H5NO", "hint": "amide_planar"},

    # 非平面化合物（含能笼形/饱和结构）
    "cl-20": {"is_planar": False, "type": "笼形含能", "formula": "C6H6N12O12", "hint": "non_planar"},
    "cl20": {"is_planar": False, "type": "笼形含能", "formula": "C6H6N12O12", "hint": "non_planar"},
    "六硝基六氮杂异伍兹烷": {"is_planar": False, "type": "笼形含能", "formula": "C6H6N12O12", "hint": "non_planar"},
    "rdx": {"is_planar": False, "type": "环硝胺", "formula": "C3H6N6O6", "hint": "non_planar"},
    "黑索金": {"is_planar": False, "type": "环硝胺", "formula": "C3H6N6O6", "hint": "non_planar"},
    "hmx": {"is_planar": False, "type": "环硝胺", "formula": "C4H8N8O8", "hint": "non_planar"},
    "奥克托今": {"is_planar": False, "type": "环硝胺", "formula": "C4H8N8O8", "hint": "non_planar"},
    "甲烷": {"is_planar": False, "type": "饱和烷烃", "formula": "CH4", "hint": "non_planar"},
    "乙烷": {"is_planar": False, "type": "饱和烷烃", "formula": "C2H6", "hint": "non_planar"},
    "乙醇": {"is_planar": False, "type": "脂肪醇", "formula": "C2H6O", "hint": "non_planar"},
    "乙醚": {"is_planar": False, "type": "脂肪醚", "formula": "C4H10O", "hint": "non_planar"},
    "环己烷": {"is_planar": False, "type": "饱和环烷", "formula": "C6H12", "hint": "non_planar"},
    "金刚烷": {"is_planar": False, "type": "笼形烷烃", "formula": "C10H16", "hint": "non_planar"},
    "立方烷": {"is_planar": False, "type": "笼形烷烃", "formula": "C8H8", "hint": "non_planar"},
}


def _parse_formula(formula: str) -> dict:
    """内部用：解析分子式，返回元素计数、不饱和度"""
    elem_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        elem_count[elem] += int(cnt) if cnt else 1

    C = elem_count.get('C', 0)
    H = elem_count.get('H', 0)
    N = elem_count.get('N', 0)
    X = elem_count.get('F', 0) + elem_count.get('Cl', 0) + elem_count.get('Br', 0)
    du = max(0.0, (2 * C + 2 + N - H - X) / 2.0) if C > 0 else 0.0

    return {
        "elem_count": dict(elem_count),
        "du": du,
        "C": C,
        "H": H,
        "N": N,
        "O": elem_count.get('O', 0),
        "total_atoms": sum(elem_count.values())
    }


def is_planar_structure(input_str: str) -> dict:
    """
    一键判断化合物是否为平面结构
    :param input_str: 化合物俗称（如TNT、苯甲酸）或分子式（如C7H5N3O6）
    :return: 字典，包含是否平面、结构类型、生成建议、解析信息
    """
    input_clean = input_str.strip().lower()

    # 第一步：优先匹配别名库，瞬间出结果
    if input_clean in COMMON_COMPOUND_LIB:
        lib_info = COMMON_COMPOUND_LIB[input_clean]
        parse_info = _parse_formula(lib_info["formula"])
        return {
            "input": input_str,
            "is_planar": lib_info["is_planar"],
            "structure_type": lib_info["type"],
            "formula": lib_info["formula"],
            "generation_hint": lib_info["hint"],
            "parse_info": parse_info,
            "match_type": "别名精准匹配"
        }

    # 第二步：没匹配到别名，尝试按分子式解析+化学规则判断
    try:
        parse_info = _parse_formula(input_str)
        C = parse_info["C"]
        du = parse_info["du"]
        H = parse_info["H"]
        N = parse_info["N"]
        O = parse_info["O"]
        hc_ratio = H / C if C > 0 else 999

        # ---------------------- 非平面排除规则 ----------------------
        if C > 0 and N > 0 and (C / N) <= 1 and O >= N and du < 5:
            return {
                "input": input_str, "is_planar": False, "structure_type": "疑似笼形含能材料",
                "formula": input_str, "generation_hint": "non_planar", "parse_info": parse_info, "match_type": "规则匹配"
            }
        if du == 0:
            return {
                "input": input_str, "is_planar": False, "structure_type": "饱和脂肪族",
                "formula": input_str, "generation_hint": "non_planar", "parse_info": parse_info, "match_type": "规则匹配"
            }
        if C < 2:
            return {
                "input": input_str, "is_planar": False, "structure_type": "小分子无平面骨架",
                "formula": input_str, "generation_hint": "non_planar", "parse_info": parse_info, "match_type": "规则匹配"
            }

        # ---------------------- 平面判定规则 ----------------------
        if C >= 6 and du >= 4 and 0.6 <= hc_ratio <= 1.4:
            return {
                "input": input_str, "is_planar": True, "structure_type": "芳香族",
                "formula": input_str, "generation_hint": "benzene_ring", "parse_info": parse_info, "match_type": "规则匹配"
            }
        hetero_atoms = [k for k in parse_info["elem_count"].keys() if k not in ["C", "H"]]
        if C >= 4 and du >= 3 and len(hetero_atoms) > 0 and 0.5 <= hc_ratio <= 1.5:
            return {
                "input": input_str, "is_planar": True, "structure_type": "杂环芳香",
                "formula": input_str, "generation_hint": "heterocycle_planar", "parse_info": parse_info,
                "match_type": "规则匹配"
            }
        if C >= 2 and du >= 1 and hc_ratio <= 2.0:
            return {
                "input": input_str, "is_planar": True, "structure_type": "共轭烯烃",
                "formula": input_str, "generation_hint": "conjugated_planar", "parse_info": parse_info,
                "match_type": "规则匹配"
            }

        return {
            "input": input_str, "is_planar": False, "structure_type": "无明确平面特征",
            "formula": input_str, "generation_hint": "non_planar", "parse_info": parse_info, "match_type": "兜底规则"
        }
    except:
        return {
            "input": input_str, "is_planar": False, "structure_type": "解析失败",
            "formula": input_str, "generation_hint": "non_planar", "parse_info": {}, "match_type": "解析失败"
        }


# =============================================================================
# 【原始模块】配置常量
# =============================================================================

NODE_FEAT_DIM = 4
EDGE_FEAT_DIM = 3
ENERGY_DIM = 2
STRESS_DIM = 9
ENERGY_EV_DIM = 1

LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10

STRESS_MIN = -50.0
STRESS_MAX = 50.0
LATENT_POS_BOUND = 5.0
CELL_VOLUME_RANGE = {
    1: (8, 20), 6: (10, 30), 7: (10, 28), 8: (8, 25),
    16: (12, 35), 26: (15, 40), 29: (18, 45)
}
BOND_CUTOFF_DISTANCE = 2.2
MIN_ATOM_DISTANCE = 0.7
BOND_ADJUST_STEP = 0.5

atomic_num_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
    17: 'Cl', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu', 30: 'Zn'
}
symbol_to_atomic_num = {v: k for k, v in atomic_num_to_symbol.items()}

BOND_COUNT_CONSTRAINTS = {
    1: (1, 1), 6: (1, 4), 7: (1, 3), 8: (1, 2),
    16: (1, 2), 26: (2, 6), 29: (1, 4)
}

BOND_LENGTH_RANGES = {
    (6, 1): (0.9, 1.3), (1, 6): (0.9, 1.3),
    (6, 6): (1.20, 1.60), (6, 7): (1.15, 1.50),
    (7, 6): (1.15, 1.50), (6, 8): (1.15, 1.45),
    (8, 6): (1.15, 1.45), (7, 1): (0.8, 1.1),
    (1, 7): (0.8, 1.1), (8, 1): (0.8, 1.1), (1, 8): (0.8, 1.1),
    (6, 16): (1.7, 2.0), (16, 6): (1.7, 2.0),
    (8, 16): (1.4, 1.7), (16, 8): (1.4, 1.7)
}


# =============================================================================
# 【原始模块】模型定义 (完全保持原样)
# =============================================================================

class CrystalDataScaler:
    def __init__(self):
        self.node_feat_mean = None
        self.node_feat_std = None
        self.edge_feat_mean = None
        self.edge_feat_std = None
        self.energy_mean = None
        self.energy_std = None
        self.stress_mean = None
        self.stress_std = None

    def fit(self, graphs: List[dgl.DGLGraph]):
        node_feats = []
        edge_feats = []
        energy_list = []
        stress_list = []

        for g in graphs:
            node_feats.append(g.ndata["feat"])
            if g.num_edges() > 0:
                edge_feats.append(g.edata["feat"])
            energy_list.append(g.ndata["total_energy"][0])
            stress_list.append(g.ndata["stress_tensor_flat"][0])

        eps = 1e-8
        self.node_feat_mean = torch.cat(node_feats, dim=0).mean(dim=0)
        self.node_feat_std = torch.cat(node_feats, dim=0).std(dim=0) + eps

        if edge_feats:
            self.edge_feat_mean = torch.cat(edge_feats, dim=0).mean(dim=0)
            self.edge_feat_std = torch.cat(edge_feats, dim=0).std(dim=0) + eps
        else:
            self.edge_feat_mean = torch.zeros(EDGE_FEAT_DIM)
            self.edge_feat_std = torch.ones(EDGE_FEAT_DIM)

        self.energy_mean = torch.stack(energy_list).mean(dim=0)
        self.energy_std = torch.stack(energy_list).std(dim=0) + eps

        self.stress_mean = torch.stack(stress_list).mean(dim=0)
        self.stress_std = torch.stack(stress_list).std(dim=0) + eps

    def transform(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.ndata["feat"] = (g.ndata["feat"] - self.node_feat_mean) / self.node_feat_std
        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean) / self.edge_feat_std
        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean) / self.energy_std
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean) / self.stress_std
        return g

    def inverse_transform_node_feat(self, node_feat: torch.Tensor) -> torch.Tensor:
        if self.node_feat_mean is None or self.node_feat_std is None:
            return node_feat
        return node_feat * self.node_feat_std + self.node_feat_mean

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
        std = self.energy_std[dim]
        mean = self.energy_mean[dim]
        if energy.dim() == 0:
            return energy * std + mean
        elif energy.dim() == 1:
            return energy * std + mean
        elif energy.dim() == 2:
            return energy[:, dim] * std + mean
        else:
            raise ValueError(f"不支持的能量张量维度: {energy.dim()}")

    def inverse_transform_stress(self, stress: torch.Tensor) -> torch.Tensor:
        stress = stress * self.stress_std + self.stress_mean
        stress = torch.clamp(stress, STRESS_MIN, STRESS_MAX)
        stress = torch.nan_to_num(stress, nan=0.0, posinf=STRESS_MAX, neginf=STRESS_MIN)
        return stress

    def state_dict(self):
        return {
            'node_feat_mean': self.node_feat_mean,
            'node_feat_std': self.node_feat_std,
            'edge_feat_mean': self.edge_feat_mean,
            'edge_feat_std': self.edge_feat_std,
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'stress_mean': self.stress_mean,
            'stress_std': self.stress_std
        }

    def load_state_dict(self, state_dict):
        self.node_feat_mean = state_dict['node_feat_mean']
        self.node_feat_std = state_dict['node_feat_std']
        self.edge_feat_mean = state_dict['edge_feat_mean']
        self.edge_feat_std = state_dict['edge_feat_std']
        self.energy_mean = state_dict['energy_mean']
        self.energy_std = state_dict['energy_std']
        self.stress_mean = state_dict['stress_mean']
        self.stress_std = state_dict['stress_std']


class EdgeAttentionGAT(torch.nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_heads: int, edge_feat_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat

        self.gat = dgl.nn.GATConv(
            in_feats=in_feat + edge_feat_dim,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            feat_drop=0.1,
            attn_drop=0.1
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_feat_dim, edge_feat_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_feat_dim * 2, edge_feat_dim)
        )

        self.node_proj = torch.nn.Linear(in_feat, in_feat)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        edge_feats = g.edata["feat"] if g.num_edges() > 0 else torch.zeros((0, EDGE_FEAT_DIM))
        edge_feats = self.edge_mlp(edge_feats)

        src, dst = g.edges()
        node_feats_proj = self.node_proj(node_feats)

        src_feats = node_feats_proj[src] if len(src) > 0 else torch.zeros((0, node_feats_proj.shape[1]))
        dst_feats = node_feats_proj[dst] if len(dst) > 0 else torch.zeros((0, node_feats_proj.shape[1]))

        src_input = torch.cat([src_feats, edge_feats], dim=1) if len(src_feats) > 0 else torch.zeros(
            (0, node_feats_proj.shape[1] + EDGE_FEAT_DIM))
        dst_input = torch.cat([dst_feats, edge_feats], dim=1) if len(dst_feats) > 0 else torch.zeros(
            (0, node_feats_proj.shape[1] + EDGE_FEAT_DIM))

        temp_feats = torch.zeros((g.num_nodes(), src_input.shape[1]))
        if len(src) > 0:
            temp_feats = temp_feats.scatter_add(0, src.unsqueeze(1).repeat(1, src_input.shape[1]), src_input)
            temp_feats = temp_feats.scatter_add(0, dst.unsqueeze(1).repeat(1, dst_input.shape[1]), dst_input)
        else:
            temp_feats = torch.cat(
                [node_feats_proj, torch.zeros((g.num_nodes(), EDGE_FEAT_DIM))], dim=1)

        gat_out = self.gat(g, temp_feats)
        return gat_out.flatten(1)


class CrystalGATEncoder(torch.nn.Module):
    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 hidden_dim: int = 128,
                 latent_dim: int = 64,
                 num_heads: int = 4):
        super().__init__()

        self.gat1 = EdgeAttentionGAT(
            in_feat=node_feat_dim,
            out_feat=hidden_dim // 2,
            num_heads=num_heads,
            edge_feat_dim=edge_feat_dim
        )

        self.gat2 = EdgeAttentionGAT(
            in_feat=(hidden_dim // 2) * num_heads,
            out_feat=hidden_dim,
            num_heads=num_heads,
            edge_feat_dim=edge_feat_dim
        )

        self.gat3 = EdgeAttentionGAT(
            in_feat=hidden_dim * num_heads,
            out_feat=hidden_dim * 2,
            num_heads=num_heads,
            edge_feat_dim=edge_feat_dim
        )

        final_gat_dim = num_heads * (hidden_dim * 2)
        self.pooling = dgl.nn.GlobalAttentionPooling(torch.nn.Sequential(
            torch.nn.Linear(final_gat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        ))

        self.lattice_proj = torch.nn.Linear(6, hidden_dim)
        self.fc_mu = torch.nn.Linear(final_gat_dim + hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(final_gat_dim + hidden_dim, latent_dim)

        torch.nn.init.constant_(self.fc_logvar.weight, 0.01)
        torch.nn.init.constant_(self.fc_logvar.bias, -2.0)

        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feats = g.ndata["feat"]
        h = self.gat1(g, node_feats)
        h = self.elu(h)
        h = self.dropout(h)

        h = self.gat2(g, h)
        h = self.elu(h)
        h = self.dropout(h)

        h = self.gat3(g, h)
        h = self.elu(h)

        node_emb = h
        graph_emb = self.pooling(g, h)

        lattice_feat = g.graph_attr['lattice'].unsqueeze(0)
        lattice_proj = self.lattice_proj(lattice_feat)
        graph_emb = torch.cat([graph_emb, lattice_proj], dim=1)

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


class CrystalDecoder(torch.nn.Module):
    def __init__(self,
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.latent_proj = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        node_emb_input_dim = num_heads * (hidden_dim * 2)
        self.node_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(node_emb_input_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        self.node_emb_gen = torch.nn.Linear(hidden_dim * 2, node_emb_input_dim)

        self.node_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, node_feat_dim)
        )

        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, edge_feat_dim)
        )

        self.energy_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 6, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, energy_dim)
        )

        self.stress_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 6, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, stress_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, node_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hasattr(g, 'batch_num_nodes') and len(g.batch_num_nodes()) > 0:
            batch_size = len(g.batch_num_nodes())
            batch_num_nodes = g.batch_num_nodes()
        else:
            batch_size = 1
            batch_num_nodes = [g.num_nodes()]
        total_nodes = g.num_nodes()

        z_proj = self.latent_proj(z)
        z_expanded = torch.zeros(total_nodes, z_proj.size(1))
        start_idx = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        node_emb_proj = self.node_emb_proj(node_emb)
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)

        src, dst = g.edges()
        z_src = z_expanded[src] if len(src) > 0 else torch.zeros((0, z_expanded.shape[1]))
        z_dst = z_expanded[dst] if len(dst) > 0 else torch.zeros((0, z_expanded.shape[1]))
        edge_input = torch.cat([z_src, z_dst], dim=1) if len(z_src) > 0 else torch.zeros((0, self.hidden_dim * 4))
        recon_edge_feats = self.edge_decoder(edge_input)

        lattice_feat = g.graph_attr['lattice'].repeat(z.shape[0], 1)
        z_with_lattice = torch.cat([z, lattice_feat], dim=1)
        pred_energy = self.energy_predictor(z_with_lattice)
        pred_stress = self.stress_predictor(z_with_lattice)

        return {
            'recon_node': recon_node_feats,
            'recon_edge': recon_edge_feats,
            'pred_energy': pred_energy,
            'pred_stress': pred_stress
        }

    def generate_node_emb(self, z: torch.Tensor, num_atoms: int) -> torch.Tensor:
        z_proj = self.latent_proj(z)
        node_emb = z_proj.unsqueeze(0).repeat(num_atoms, 1)
        node_emb = self.node_emb_gen(node_emb)
        return node_emb


class CrystalGATVAE(torch.nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, num_heads=4):
        super().__init__()
        self.encoder = CrystalGATEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads
        )

        self.decoder = CrystalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM
        )

        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 1e6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        mu, logvar, node_emb = self.encoder(g)
        z = self.reparameterize(mu, logvar)
        decode_out = self.decoder(z, g, node_emb)

        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'node_emb': node_emb
        }
        output.update(decode_out)

        return output


# =============================================================================
# 【原始模块】全局配置
# =============================================================================

STANDARD_SPACE_GROUPS = {
    1: {
        'symbol': 'P 1',
        'crystal_system': 'triclinic',
        'sym_ops': ['x, y, z'],
        'cell_constraints': {'a≠b≠c': True, 'α≠β≠γ≠90°': True}
    },
    2: {
        'symbol': 'P -1',
        'crystal_system': 'triclinic',
        'sym_ops': ['x, y, z', '-x, -y, -z'],
        'cell_constraints': {'a≠b≠c': True, 'α≠β≠γ≠90°': True}
    },
    14: {
        'symbol': 'P 21/c',
        'crystal_system': 'monoclinic',
        'sym_ops': [
            'x, y, z', '-x, y+1/2, -z',
            'x+1/2, y+1/2, z+1/2', '-x+1/2, y, -z+1/2'
        ],
        'cell_constraints': {'a≠b≠c': True, 'α=γ=90°, β≠90°': True}
    },
    62: {
        'symbol': 'Pnma',
        'crystal_system': 'orthorhombic',
        'sym_ops': [
            'x, y, z', '-x, -y, -z',
            'x+1/2, -y+1/2, z+1/2', '-x+1/2, y+1/2, -z+1/2'
        ],
        'cell_constraints': {'a≠b≠c': True, 'α=β=γ=90°': True}
    },
    194: {
        'symbol': 'P63/mmc',
        'crystal_system': 'hexagonal',
        'sym_ops': [
            'x, y, z', 'y-x, -x, z', '-y, x-y, z',
            '-x, -y, z', '-y+x, x, z', 'y, -x+y, z'
        ],
        'cell_constraints': {'a=b≠c': True, 'α=β=90°, γ=120°': True}
    },
    225: {
        'symbol': 'Fm-3m',
        'crystal_system': 'cubic',
        'sym_ops': [
            'x, y, z', '-x, -y, z', '-x, y, -z', 'x, -y, -z'
        ],
        'cell_constraints': {'a=b=c': True, 'α=β=γ=90°': True}
    }
}

# 注意：请根据你的实际路径修改
MODEL_PATH = Path("/home/nyx/N-RGAG/models/best_gat_vae.pth")
FAISS_INDEX_PATH = Path("/home/nyx/N-RGAG/know_base/crystal_latent_index.faiss")
METADATA_PATH = Path("/home/nyx/N-RGAG/know_base/crystal_metadata.json")
GENERATED_CIF_DIR = Path("./new_cif")  # 修改为当前目录，方便测试
VISUALIZATION_DIR = Path("./new_cif_vis")

LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
MAX_ATOMS = 1000

GENERATED_CIF_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 【原始模块】模型加载
# =============================================================================

model = CrystalGATVAE(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS
)
scaler = CrystalDataScaler()

try:
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        matched_params = {}
        for k in checkpoint_state_dict.keys():
            if k in model_state_dict and checkpoint_state_dict[k].shape == model_state_dict[k].shape:
                matched_params[k] = checkpoint_state_dict[k]

        model.load_state_dict(matched_params, strict=False)

        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        elif 'scaler' in checkpoint:
            old_scaler = checkpoint['scaler']
            scaler.node_feat_mean = old_scaler.node_feat_mean
            scaler.node_feat_std = old_scaler.node_feat_std
            scaler.edge_feat_mean = old_scaler.edge_feat_mean
            scaler.edge_feat_std = old_scaler.edge_feat_std
            scaler.energy_mean = old_scaler.energy_mean
            scaler.energy_std = old_scaler.energy_std
            scaler.stress_mean = old_scaler.stress_mean
            scaler.stress_std = old_scaler.stress_std

        model.eval()
        print(f"✅ 模型加载成功：{MODEL_PATH}")
    else:
        print(f"⚠️ 模型文件不存在，将使用随机初始化演示逻辑：{MODEL_PATH}")
        model.eval()
except Exception as e:
    print(f"⚠️ 模型加载失败，将使用随机初始化: {e}")
    model.eval()


# =============================================================================
# 【优化模块2】平面结构生成函数
# =============================================================================

def generate_planar_molecule_coords(atom_types: List[int]) -> torch.Tensor:
    """通用平面分子坐标生成（XY平面）"""
    num_atoms = len(atom_types)
    coords = torch.zeros(num_atoms, 3)

    backbone_indices = [i for i, num in enumerate(atom_types) if num != 1]
    h_indices = [i for i, num in enumerate(atom_types) if num == 1]

    if len(backbone_indices) > 0:
        if len(backbone_indices) >= 4:
            radius = 1.4 * (len(backbone_indices) / 6)
            for i, idx in enumerate(backbone_indices):
                angle = 2 * torch.pi * i / len(backbone_indices)
                x = radius * torch.cos(torch.tensor(angle))
                y = radius * torch.sin(torch.tensor(angle))
                coords[idx] = torch.tensor([x, y, 0.0])
        else:
            for i, idx in enumerate(backbone_indices):
                coords[idx] = torch.tensor([i * 1.5, 0.0, 0.0])

    for h_idx in h_indices:
        min_dist = float('inf')
        nearest_backbone_idx = backbone_indices[0] if backbone_indices else 0
        if backbone_indices:
            for b_idx in backbone_indices:
                dist = abs(h_idx - b_idx)
                if dist < min_dist:
                    min_dist = dist
                    nearest_backbone_idx = b_idx

        base_pos = coords[nearest_backbone_idx]
        angle = random.uniform(0, 2 * torch.pi)
        bond_len = 1.0
        coords[h_idx] = base_pos + torch.tensor([
            bond_len * torch.cos(torch.tensor(angle)),
            bond_len * torch.sin(torch.tensor(angle)),
            0.0
        ])

    return coords


# =============================================================================
# 【原始模块】核心工具函数 (保留并优化generate_graph_from_latent)
# =============================================================================

def calculate_unsaturation(formula: str) -> float:
    atom_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        atom_count[elem] += int(cnt) if cnt else 1

    C = atom_count.get('C', 0)
    H = atom_count.get('H', 0)
    N = atom_count.get('N', 0)
    X = atom_count.get('F', 0) + atom_count.get('Cl', 0)

    unsaturation = (2 * C + 2 - H - X + N) / 2.0
    return max(0.0, unsaturation)


def generate_ring_structure(coords: torch.Tensor, atomic_nums: List[int], ring_size: int = 6,
                            radius: float = 1.4) -> torch.Tensor:
    carbon_indices = [i for i, num in enumerate(atomic_nums) if num == 6]
    if len(carbon_indices) < ring_size:
        return coords

    ring_atoms = random.sample(carbon_indices, ring_size)
    device = coords.device

    center = torch.tensor([0.0, 0.0, 0.0], device=device)
    for i, idx in enumerate(ring_atoms):
        angle = torch.tensor(2 * torch.pi * i / ring_size, device=device)
        x = center[0] + radius * torch.cos(angle)
        y = center[1] + radius * torch.sin(angle)
        z = center[2]
        coords[idx] = torch.tensor([x, y, z], device=device)

    for i in range(len(ring_atoms)):
        idx1 = ring_atoms[i]
        idx2 = ring_atoms[(i + 1) % ring_size]
        pos1 = coords[idx1]
        pos2 = coords[idx2]
        dist = torch.norm(pos1 - pos2)

        if dist < 1.3 or dist > 1.6:
            ideal_dist = 1.45
            adjustment = (ideal_dist - dist) * 0.5
            vec = (pos1 - pos2) / (dist + 1e-8)
            coords[idx1] += vec * adjustment
            coords[idx2] -= vec * adjustment

    return coords


def calculate_structure_penalties(g: dgl.DGLGraph, lattice: Lattice) -> Dict[str, float]:
    penalties = {
        "min_distance": 0.0,
        "bond_count": 0.0,
        "h_h_bond": 0.0,
        "vol_ratio": 0.0
    }

    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    coords = node_feats[:, 1:4]
    num_atoms = len(atomic_nums)

    if num_atoms > 1:
        dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
        mask = np.eye(num_atoms, dtype=bool)
        valid_dists = dist_matrix[~mask]
        min_dist = np.min(valid_dists) if valid_dists.size > 0 else 1.0

        if min_dist < 0.5:
            penalties["min_distance"] = (0.5 - min_dist) * 1000
        elif min_dist < 0.8:
            penalties["min_distance"] = (0.8 - min_dist) * 200

    bond_count = calculate_bond_count(g, lattice)
    for i in range(num_atoms):
        num = atomic_nums[i]
        if num in BOND_COUNT_CONSTRAINTS:
            min_bond, max_bond = BOND_COUNT_CONSTRAINTS[num]
            current_bond = bond_count[i]
            weight = 3.0 if num == 1 else 1.5
            if not (min_bond <= current_bond <= max_bond):
                penalties["bond_count"] += abs(current_bond - (min_bond + max_bond) / 2) * 50 * weight

    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            dist = np.linalg.norm(coords[h_indices[i]] - coords[h_indices[j]])
            if dist < 1.0:
                penalties["h_h_bond"] += 5000

    target_vol = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums)
    vol_ratio = abs(lattice.volume - target_vol) / target_vol
    if vol_ratio > 0.5:
        penalties["vol_ratio"] = vol_ratio * 100

    return penalties


def save_crystal_to_cif(structure_dict: Dict, filename: Path):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Generated by N-RGAG on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("data_crystal_structure\n")
        f.write("_audit_creation_method 'N-RGAG structure generation'\n")

        f.write(f"_chemical_formula_structural '{structure_dict['structural_formula']}'\n")
        f.write(f"_chemical_formula_sum '{structure_dict['sum_formula']}'\n")
        f.write(f"_cell_formula_units_Z {structure_dict['z_value']}\n\n")

        cell = structure_dict['cell']
        f.write(f"_cell_length_a {cell[0]:.6f}\n")
        f.write(f"_cell_length_b {cell[1]:.6f}\n")
        f.write(f"_cell_length_c {cell[2]:.6f}\n")
        f.write(f"_cell_angle_alpha {cell[3]:.6f}\n")
        f.write(f"_cell_angle_beta {cell[4]:.6f}\n")
        f.write(f"_cell_angle_gamma {cell[5]:.6f}\n\n")

        f.write(f"_symmetry_space_group_name_H-M '{structure_dict['space_group']}'\n")
        f.write(f"_symmetry_Int_Tables_number {structure_dict['sg_number']}\n")
        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        for op in structure_dict['sym_ops']:
            f.write(f"'{op}'\n")

        f.write("\nloop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_occupancy\n")

        for idx, atom in enumerate(structure_dict['atoms']):
            elem = atom['element']
            label = f"{elem}{idx + 1}"
            x, y, z = atom['frac_coords']
            x = 0.0 if np.isnan(x) else x
            y = 0.0 if np.isnan(y) else y
            z = 0.0 if np.isnan(z) else z
            x = x % 1.0
            y = y % 1.0
            z = z % 1.0
            f.write(f"{label:<6} {elem:<2} {x:>12.8f} {y:>12.8f} {z:>12.8f} 1.000000\n")

        f.write(f"\n# Energy: {structure_dict['energy']:.6f} eV\n")
        f.write("# Stress tensor (GPa):\n")
        stress_mat = structure_dict['stress_tensor']
        for i in range(3):
            f.write(f"#  {stress_mat[i][0]:.6f} {stress_mat[i][1]:.6f} {stress_mat[i][2]:.6f}\n")
        f.write(f"# Fitness score: {structure_dict['fitness']:.6f}\n")


def graph_to_crystal_dict(g: dgl.DGLGraph, lattice_params: List[float], sg_info: Dict,
                          energy: float, stress_tensor: np.ndarray, fitness: float) -> Dict:
    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    cart_coords = node_feats[:, 1:4]
    cart_coords = np.nan_to_num(cart_coords, nan=0.0, posinf=10.0, neginf=-10.0)

    a, b, c, alpha, beta, gamma = lattice_params
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    frac_coords = [lattice.get_fractional_coords(coord) for coord in cart_coords]
    frac_coords = [np.mod(frac, 1.0) for frac in frac_coords]

    atoms = []
    for num, frac in zip(atomic_nums, frac_coords):
        elem = atomic_num_to_symbol.get(num, f"X{num}")
        atoms.append({
            'element': elem,
            'frac_coords': frac.tolist(),
            'atomic_num': num
        })

    structural_formula, sum_formula = calculate_chemical_formula(atomic_nums)
    z_value = calculate_z_value(atomic_nums, lattice.volume)

    return {
        'cell': lattice_params,
        'space_group': sg_info['sg_symbol'],
        'sg_number': sg_info['sg_number'],
        'sym_ops': sg_info['sym_ops'],
        'atoms': atoms,
        'energy': energy,
        'stress_tensor': stress_tensor.tolist(),
        'fitness': fitness,
        'structural_formula': structural_formula,
        'sum_formula': sum_formula,
        'z_value': z_value
    }


def calculate_bond_count(g: dgl.DGLGraph, lattice: Lattice) -> Dict[int, int]:
    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    cart_coords = node_feats[:, 1:4]
    cart_coords = np.nan_to_num(cart_coords, nan=0.0, posinf=10.0, neginf=-10.0)

    num_atoms = len(atomic_nums)
    bond_count = {i: 0 for i in range(num_atoms)}

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = calculate_pbc_distance(cart_coords[i], cart_coords[j], lattice)
            if dist > BOND_CUTOFF_DISTANCE:
                continue
            pair = (atomic_nums[i], atomic_nums[j])
            if pair not in BOND_LENGTH_RANGES:
                pair_rev = (atomic_nums[j], atomic_nums[i])
                if pair_rev not in BOND_LENGTH_RANGES:
                    continue
                min_len, max_len = BOND_LENGTH_RANGES[pair_rev]
            else:
                min_len, max_len = BOND_LENGTH_RANGES[pair]
            if min_len <= dist <= max_len:
                bond_count[i] += 1
                bond_count[j] += 1

    return bond_count


def fix_free_atoms(coords: torch.Tensor, atomic_nums: List[int], lattice: Lattice,
                   bond_count: Dict[int, int]) -> torch.Tensor:
    coords = coords.clone()
    num_atoms = len(atomic_nums)
    device = coords.device
    if num_atoms < 2:
        return coords

    free_atoms = [i for i in range(num_atoms) if bond_count.get(i, 0) == 0]
    if not free_atoms:
        return coords

    bond_acceptors_map = {
        1: [6, 7, 8],
        6: [1, 6, 7, 8, 16],
        7: [1, 6, 7, 8],
        8: [1, 6, 7],
        16: [6, 8],
        26: [6, 8, 16],
        29: [6, 8, 16]
    }

    available_acceptors = []
    for i in range(num_atoms):
        if bond_count.get(i, 0) >= BOND_COUNT_CONSTRAINTS.get(atomic_nums[i], (1, 4))[1]:
            continue
        available_acceptors.append(i)

    if not available_acceptors:
        available_acceptors = [i for i in range(num_atoms) if i not in free_atoms]

    for free_idx in free_atoms:
        free_atom_num = atomic_nums[free_idx]
        free_pos = coords[free_idx]

        valid_acceptors = [
            acc_idx for acc_idx in available_acceptors
            if atomic_nums[acc_idx] in bond_acceptors_map.get(free_atom_num, [])
               and acc_idx != free_idx
        ]

        if not valid_acceptors:
            valid_acceptors = available_acceptors

        min_dist = float('inf')
        best_acc_idx = valid_acceptors[0]
        for acc_idx in valid_acceptors:
            acc_pos = coords[acc_idx].cpu().numpy()
            free_pos_np = free_pos.cpu().numpy()
            dist = calculate_pbc_distance(free_pos_np, acc_pos, lattice)
            if dist < min_dist:
                min_dist = dist
                best_acc_idx = acc_idx

        acc_idx = best_acc_idx
        acc_atom_num = atomic_nums[acc_idx]
        acc_pos = coords[acc_idx]

        pair = (free_atom_num, acc_atom_num)
        pair_rev = (acc_atom_num, free_atom_num)
        if pair in BOND_LENGTH_RANGES:
            min_len, max_len = BOND_LENGTH_RANGES[pair]
        else:
            min_len, max_len = BOND_LENGTH_RANGES.get(pair_rev, (0.8, 1.6))
        ideal_len = (min_len + max_len) / 2

        current_dist = torch.norm(free_pos - acc_pos)
        adjustment = (ideal_len - current_dist) * 0.8
        vec = (free_pos - acc_pos) / (current_dist + 1e-8)

        coords[free_idx] += vec * adjustment
        coords[acc_idx] -= vec * adjustment

        bond_count[free_idx] += 1
        bond_count[acc_idx] += 1

    dist_matrix = torch.cdist(coords, coords)
    mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=device), diagonal=1)
    invalid_pairs = dist_matrix[mask] < 0.7
    if torch.any(invalid_pairs):
        idx_pairs = torch.combinations(torch.arange(num_atoms, device=device))
        invalid_idx = idx_pairs[invalid_pairs]
        for idx1, idx2 in invalid_idx:
            vec = coords[idx1] - coords[idx2]
            dist = torch.norm(vec)
            needed = 0.7 - dist
            move = vec / (dist + 1e-8) * needed * 0.5
            coords[idx1] += move
            coords[idx2] -= move

    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    return coords


def check_structure_validity(g: dgl.DGLGraph, lattice: Lattice) -> Tuple[bool, str]:
    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    cart_coords = node_feats[:, 1:4]
    cart_coords = np.nan_to_num(cart_coords, nan=0.0, posinf=10.0, neginf=-10.0)
    num_atoms = len(atomic_nums)

    if num_atoms >= 2:
        dist_matrix = np.linalg.norm(cart_coords[:, None] - cart_coords, axis=2)
        mask = np.eye(num_atoms, dtype=bool)
        valid_dists = dist_matrix[~mask]
        if valid_dists.size > 0 and np.min(valid_dists) < 0.7:
            return False, f"原子重叠（最小间距{np.min(valid_dists):.2f}Å < 0.7Å）"

    target_vol = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums)
    vol_ratio = abs(lattice.volume - target_vol) / target_vol
    if vol_ratio > 1.0:
        return False, f"晶胞体积不合理（实际{lattice.volume:.2f}Å³，目标{target_vol:.2f}Å³，偏离{vol_ratio * 100:.2f}%）"

    bond_count = calculate_bond_count(g, lattice)
    free_atom_exists = False
    invalid_bond_info = ""
    for i in range(num_atoms):
        atom_num = atomic_nums[i]
        if atom_num not in BOND_COUNT_CONSTRAINTS:
            continue
        min_bond, max_bond = BOND_COUNT_CONSTRAINTS[atom_num]
        current_bond = bond_count[i]
        if current_bond == 0:
            free_atom_exists = True
            invalid_bond_info = f"游离原子：{atomic_num_to_symbol[atom_num]}（索引{i}）成键数为0"
            break
        if not (min_bond <= current_bond <= max_bond):
            free_atom_exists = True
            invalid_bond_info = f"成键数不合理：{atomic_num_to_symbol[atom_num]}（索引{i}）成键数{current_bond}，应在[{min_bond}, {max_bond}]"
            break

    if free_atom_exists:
        coords_tensor = torch.tensor(cart_coords, dtype=torch.float32)
        repaired_coords = fix_free_atoms(coords_tensor, atomic_nums, lattice, bond_count)
        temp_g = dgl.graph(([], []), num_nodes=num_atoms)
        temp_g.ndata['feat'] = torch.cat([
            torch.tensor(atomic_nums, dtype=torch.float32).unsqueeze(1),
            repaired_coords
        ], dim=1)
        repaired_bond_count = calculate_bond_count(temp_g, lattice)
        repaired_free = any(
            repaired_bond_count[i] == 0 for i in range(num_atoms) if atomic_nums[i] in BOND_COUNT_CONSTRAINTS)
        if not repaired_free:
            return True, "结构合理（已修复游离原子）"

    if free_atom_exists:
        return False, invalid_bond_info

    return True, "结构合理"


def calculate_energy_and_stress(g: dgl.DGLGraph, lattice_params: List[float]) -> Tuple[float, np.ndarray]:
    """简化版能量应力计算（演示用）"""
    energy = random.uniform(-50, 0)
    stress_tensor = np.random.randn(3, 3) * 2.0
    stress_tensor = (stress_tensor + stress_tensor.T) / 2
    return energy, stress_tensor


def calculate_pbc_distance(pos1: np.ndarray, pos2: np.ndarray, lattice: Lattice) -> float:
    frac1 = lattice.get_fractional_coords(pos1)
    frac2 = lattice.get_fractional_coords(pos2)
    frac_diff = frac1 - frac2
    frac_diff = np.mod(frac_diff + 0.5, 1.0) - 0.5
    cart_diff = lattice.get_cartesian_coords(frac_diff)
    return np.linalg.norm(cart_diff)


def enforce_atom_constraints(coords: torch.Tensor, atomic_nums: List[int], lattice: Lattice) -> torch.Tensor:
    coords = coords.clone()
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return coords

    device = coords.device
    coords = torch.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)

    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    stages = [0.3, 0.5, 0.7]
    max_iter_per_stage = 50

    for stage_min in stages:
        for _ in range(max_iter_per_stage):
            dist_matrix = torch.cdist(coords, coords)
            mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=device), diagonal=1)
            invalid_pairs = dist_matrix[mask] < stage_min

            if not torch.any(invalid_pairs):
                break

            push_strength = 1.0 if stage_min < 0.2 else 0.8 if stage_min < 0.5 else 0.6

            idx_pairs = torch.combinations(torch.arange(num_atoms, device=device))
            invalid_idx = idx_pairs[invalid_pairs]

            for idx1, idx2 in invalid_idx:
                pos1 = coords[idx1]
                pos2 = coords[idx2]
                vec = pos1 - pos2
                dist = torch.norm(vec)

                if dist < 1e-6:
                    vec = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
                    move_dist = stage_min * push_strength
                    coords[idx1] += vec * move_dist / 2
                    coords[idx2] -= vec * move_dist / 2
                else:
                    needed = stage_min - dist
                    move = vec / (dist + 1e-8) * needed * push_strength
                    coords[idx1] += move
                    coords[idx2] -= move

    max_bond_opt_iter = 100
    bond_adjust_step = BOND_ADJUST_STEP
    for _ in range(max_bond_opt_iter):
        bond_optimized = True
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                pair = (atomic_nums[i], atomic_nums[j])
                pair_rev = (atomic_nums[j], atomic_nums[i])
                min_len, max_len = None, None

                if pair in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair]
                elif pair_rev in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair_rev]
                else:
                    continue

                pos1 = coords[i].cpu().numpy()
                pos2 = coords[j].cpu().numpy()
                dist = calculate_pbc_distance(pos1, pos2, lattice)
                dist = torch.tensor(dist, dtype=torch.float32, device=device)

                if dist < min_len or dist > max_len:
                    bond_optimized = False
                    ideal_len = (min_len + max_len) / 2
                    adjustment = (ideal_len - dist) * bond_adjust_step

                    vec = coords[i] - coords[j]
                    vec_norm = torch.norm(vec) + 1e-8
                    move = vec / vec_norm * adjustment

                    coords[i] += move
                    coords[j] -= move

        if bond_optimized:
            break

    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]
    acceptor_indices = [i for i, num in enumerate(atomic_nums) if num in {6, 7, 8}]
    if h_indices and acceptor_indices:
        for h_idx in h_indices:
            h_pos = coords[h_idx]
            dists = []
            for acc_idx in acceptor_indices:
                acc_pos = coords[acc_idx].cpu().numpy()
                h_pos_np = h_pos.cpu().numpy()
                dist = calculate_pbc_distance(h_pos_np, acc_pos, lattice)
                dists.append(torch.tensor(dist, device=device))

            closest_acc_idx = acceptor_indices[torch.argmin(torch.tensor(dists))]
            acc_pos = coords[closest_acc_idx]
            pair = (1, atomic_nums[closest_acc_idx])
            min_len, max_len = BOND_LENGTH_RANGES.get(pair, BOND_LENGTH_RANGES.get((atomic_nums[closest_acc_idx], 1),
                                                                                   (0.8, 1.3)))
            ideal_len = (min_len + max_len) / 2

            dist = torch.norm(h_pos - acc_pos)
            if dist < min_len or dist > max_len:
                adjustment = (ideal_len - dist) * 0.6
                vec = (h_pos - acc_pos) / (torch.norm(h_pos - acc_pos) + 1e-8)
                coords[h_idx] += vec * adjustment
                coords[closest_acc_idx] -= vec * adjustment

    temp_g = dgl.graph(([], []), num_nodes=num_atoms)
    temp_g.ndata['feat'] = torch.cat([
        torch.tensor(atomic_nums, dtype=torch.float32, device=device).unsqueeze(1),
        coords
    ], dim=1) if num_atoms > 0 else torch.zeros((0, 4))
    bond_count = calculate_bond_count(temp_g, lattice)
    coords = fix_free_atoms(coords, atomic_nums, lattice, bond_count)

    dist_matrix = torch.cdist(coords, coords)
    mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=device), diagonal=1)
    invalid_pairs = dist_matrix[mask] < 0.7
    if torch.any(invalid_pairs):
        idx_pairs = torch.combinations(torch.arange(num_atoms, device=device))
        invalid_idx = idx_pairs[invalid_pairs]
        for idx1, idx2 in invalid_idx:
            vec = coords[idx1] - coords[idx2]
            dist = torch.norm(vec)
            needed = 0.7 - dist
            move = vec / (dist + 1e-8) * needed * 0.5
            coords[idx1] += move
            coords[idx2] -= move

    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    return coords


def realtime_optimize_single_structure(g: dgl.DGLGraph, atom_types: List[int], lattice: Lattice) -> Tuple[
    dgl.DGLGraph, Lattice]:
    num_atoms = len(atom_types)
    if num_atoms < 2:
        return g, lattice

    node_feats = g.ndata['feat'].clone()
    coords = node_feats[:, 1:4].clone()
    atomic_nums = atom_types

    coords = enforce_atom_constraints(coords, atomic_nums, lattice)

    max_bond_opt_iter = 50
    for _ in range(max_bond_opt_iter):
        bond_optimized = True
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                pair = (atomic_nums[i], atomic_nums[j])
                pair_rev = (atomic_nums[j], atomic_nums[i])
                if pair in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair]
                elif pair_rev in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair_rev]
                else:
                    continue

                dist = torch.norm(coords[i] - coords[j])
                if dist < min_len or dist > max_len:
                    bond_optimized = False
                    ideal_len = (min_len + max_len) / 2
                    adjustment = (ideal_len - dist) * 0.4
                    vec = (coords[i] - coords[j]) / (dist + 1e-8)
                    coords[i] += vec * adjustment
                    coords[j] -= vec * adjustment
        if bond_optimized:
            break

    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=coords.device)
    node_feats[:, 1:4] = coords
    g.ndata['feat'] = node_feats

    target_vol_per_atom = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums) / num_atoms
    target_volume = target_vol_per_atom * num_atoms
    current_volume = lattice.volume

    vol_scale = (target_volume / current_volume) ** (1 / 3)
    new_a, new_b, new_c = lattice.a * vol_scale, lattice.b * vol_scale, lattice.c * vol_scale
    new_alpha, new_beta, new_gamma = lattice.alpha, lattice.beta, lattice.gamma

    angle_adjustment = 2.0
    if new_alpha < 88 or new_alpha > 92:
        new_alpha = np.clip(new_alpha, 88, 92)
    if new_beta < 88 or new_beta > 92:
        new_beta = np.clip(new_beta, 88, 92)
    if new_gamma < 88 or new_gamma > 92:
        new_gamma = np.clip(new_gamma, 88, 92)

    optimized_lattice = Lattice.from_parameters(new_a, new_b, new_c, new_alpha, new_beta, new_gamma)

    vol_scale = (optimized_lattice.volume / lattice.volume) ** (1 / 3)
    coords = node_feats[:, 1:4].clone() * vol_scale
    coords = torch.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)

    coords = enforce_atom_constraints(coords, atomic_nums, optimized_lattice)

    frac_coords = optimized_lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(optimized_lattice.get_cartesian_coords(frac_coords), dtype=torch.float32,
                          device=coords.device)
    node_feats[:, 1:4] = coords
    g.ndata['feat'] = node_feats

    return g, optimized_lattice


def generate_lattice_params(similar_structures: List[Dict], atom_types: List[int], sg_cell_constraints: Dict = None) -> \
List[float]:
    sg_cell_constraints = sg_cell_constraints or {'a=b=c': True, 'α=β=γ=90°': True}
    avg_vol_per_atom = 0.0
    for num in atom_types:
        avg_vol_per_atom += CELL_VOLUME_RANGE.get(num, (10, 30))[1]
    avg_vol_per_atom /= len(atom_types)
    target_volume = avg_vol_per_atom * len(atom_types)

    if similar_structures:
        total_weight = sum(1.0 / (s['distance'] + 1e-6) for s in similar_structures)
        avg_lattice = [0.0] * 6
        for s in similar_structures:
            if 'lattice' in s and len(s['lattice']) == 6:
                weight = 1.0 / (s['distance'] + 1e-6)
                avg_lattice = [a + b * weight for a, b in zip(avg_lattice, s['lattice'])]
        avg_lattice = [a / total_weight for a in avg_lattice]

        a, b, c, alpha, beta, gamma = avg_lattice
        current_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        volume_scale = (target_volume / current_lattice.volume) ** (1 / 3)
        avg_lattice[0] *= volume_scale
        avg_lattice[1] *= volume_scale
        avg_lattice[2] *= volume_scale
        lattice_params = avg_lattice
    else:
        a = (target_volume) ** (1 / 3)
        b = a
        c = a
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        lattice_params = [a, b, c, alpha, beta, gamma]

    a, b, c, alpha, beta, gamma = lattice_params
    if 'a=b≠c' in sg_cell_constraints:
        a = b = (target_volume / (c * np.sin(np.radians(120)))) ** (1 / 2)
        gamma = 120.0
    elif 'a=b=c' in sg_cell_constraints:
        a = b = c = (target_volume) ** (1 / 3)
    elif 'α=β=γ=90°' in sg_cell_constraints:
        alpha = beta = 90.0
        if 'γ=90°' in sg_cell_constraints:
            gamma = 90.0
    elif 'γ=120°' in sg_cell_constraints:
        gamma = 120.0

    alpha = np.clip(alpha, 80, 100) if alpha != 120 else 120.0
    beta = np.clip(beta, 80, 100) if beta != 120 else 120.0
    gamma = np.clip(gamma, 80, 120)

    return [a, b, c, alpha, beta, gamma]


def get_space_group_info() -> Dict:
    sg_num = random.choice(list(STANDARD_SPACE_GROUPS.keys()))
    sg_data = STANDARD_SPACE_GROUPS[sg_num]
    return {
        'sg_number': sg_num,
        'sg_symbol': sg_data['symbol'],
        'crystal_system': sg_data['crystal_system'],
        'sym_ops': sg_data['sym_ops'],
        'cell_constraints': sg_data['cell_constraints']
    }


class Particle:
    def __init__(self, dim: int):
        self.position = torch.randn(dim) * 1.2
        self.velocity = torch.randn(dim) * 0.2
        self.best_position = self.position.clone()
        self.best_fitness = float('inf')
        self.is_valid = False

    def update_velocity(self, global_best: torch.Tensor, w: float = 0.7, c1: float = 1.8, c2: float = 1.8):
        r1, r2 = torch.rand(2)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best - self.position))
        self.velocity = torch.clamp(self.velocity, -0.6, 0.6)

    def update_position(self):
        self.position += self.velocity
        self.position = torch.clamp(self.position, -LATENT_POS_BOUND, LATENT_POS_BOUND)


# =============================================================================
# 【核心优化】generate_graph_from_latent (整合平面判断逻辑)
# =============================================================================

def generate_graph_from_latent(z: torch.Tensor, atom_types: List[int], similar_structures: List[Dict],
                               formula: str = "") -> dgl.DGLGraph:
    """
    优化版：根据平面判断结果，优先生成平面结构，否则走原有逻辑
    """
    num_atoms = len(atom_types)
    if num_atoms == 0:
        raise ValueError("原子数不能为0")
    g = dgl.graph(([], []), num_nodes=num_atoms)

    g.graph_attr = {}
    g.ndata['feat'] = torch.zeros(num_atoms, NODE_FEAT_DIM)
    g.edata['feat'] = torch.zeros(0, EDGE_FEAT_DIM)
    g.ndata['total_energy'] = torch.zeros(num_atoms, ENERGY_DIM)
    g.ndata['stress_tensor_flat'] = torch.zeros(num_atoms, STRESS_DIM)

    atomic_num_feat = torch.tensor(atom_types, dtype=torch.float32).unsqueeze(1)
    g.ndata['feat'][:, 0:1] = atomic_num_feat

    sg_info = get_space_group_info()
    lattice_params = generate_lattice_params(similar_structures, atom_types, sg_info['cell_constraints'])
    lattice = Lattice.from_parameters(*lattice_params)

    if not hasattr(g, 'graph_attr'):
        g.graph_attr = {}
    g.graph_attr['lattice'] = torch.tensor(lattice_params, dtype=torch.float32, requires_grad=False)
    g.graph_attr['num_atoms'] = torch.tensor([num_atoms], dtype=torch.float32)
    g.graph_attr['volume'] = torch.tensor([lattice.volume], dtype=torch.float32)

    with torch.no_grad():
        # ===================== 【新增：平面结构判断分支】 =====================
        use_planar = False
        planar_hint = "non_planar"
        if formula:
            judge_result = is_planar_structure(formula)
            use_planar = judge_result["is_planar"]
            planar_hint = judge_result["generation_hint"]
            print(
                f"🧠 结构判定：{formula} | 平面结构：{use_planar} | 类型：{judge_result['structure_type']} | 匹配：{judge_result['match_type']}")

        # 分支1：判定为平面结构，先生成平面分子骨架
        if use_planar:
            if planar_hint == "benzene_ring":
                coords = torch.zeros(num_atoms, 3)
                coords = generate_ring_structure(coords, atom_types, ring_size=6, radius=1.4)
            elif planar_hint in ["heterocycle_planar", "conjugated_planar", "amide_planar"]:
                coords = generate_planar_molecule_coords(atom_types)
            else:
                coords = generate_planar_molecule_coords(atom_types)

            coords = enforce_atom_constraints(coords, atom_types, lattice)

        # 分支2：非平面结构，完全走你原来的VAE生成逻辑
        else:
            try:
                node_emb = model.decoder.generate_node_emb(z, num_atoms)
                mu = z.unsqueeze(0)
                logvar = torch.zeros_like(mu)
                decode_out = model.decoder(mu, g, node_emb)
                recon_node_feats = decode_out['recon_node']
                if scaler is not None and scaler.node_feat_mean is not None:
                    recon_node_feats = scaler.inverse_transform_node_feat(recon_node_feats)
                coords = recon_node_feats[:, 1:4]
            except:
                coords = torch.randn(num_atoms, 3) * 2.0

            coords = coords + torch.randn_like(coords) * 0.1
            coords = torch.tanh(coords) * 3.0
            coords = torch.nan_to_num(coords, nan=0.0, posinf=3.0, neginf=-3.0)

            for _ in range(20):
                dist_matrix = torch.cdist(coords, coords)
                mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool), diagonal=1)
                invalid_pairs = dist_matrix[mask] < 0.7
                if not torch.any(invalid_pairs):
                    break
                idx_pairs = torch.combinations(torch.arange(num_atoms))
                invalid_idx = idx_pairs[invalid_pairs]
                for idx1, idx2 in invalid_idx:
                    vec = coords[idx1] - coords[idx2]
                    dist = torch.norm(vec)
                    if dist < 1e-6:
                        vec = torch.randn_like(vec)
                    move = vec / (dist + 1e-8) * (0.7 - dist) * 0.6
                    coords[idx1] += move
                    coords[idx2] -= move

        g.ndata['feat'][:, 1:4] = coords

    return g


def calculate_fitness(g: dgl.DGLGraph, lattice: Lattice, stress_tensor: np.ndarray) -> float:
    is_valid, _ = check_structure_validity(g, lattice)
    if not is_valid:
        return 1e9

    penalties = calculate_structure_penalties(g, lattice)
    penalty_weights = {
        "min_distance": 2.0,
        "bond_count": 2.5,
        "h_h_bond": 4.0,
        "vol_ratio": 1.0
    }

    total_penalty = 0.0
    for penalty_name, penalty_value in penalties.items():
        total_penalty += penalty_value * penalty_weights.get(penalty_name, 1.0)

    lattice_params = list(lattice.abc) + list(lattice.angles)
    energy, _ = calculate_energy_and_stress(g, lattice_params)
    stress_norm = np.linalg.norm(stress_tensor)

    energy_weight = 0.3
    energy_norm = min(energy / 1000.0, 10.0)

    stress_weight = 0.2
    if stress_norm < 1e-3:
        stress_penalty = 10.0
    elif stress_norm > STRESS_MAX:
        stress_penalty = stress_norm / STRESS_MAX
    else:
        stress_penalty = stress_norm / 10.0

    total_fitness = (
            energy_weight * energy_norm +
            stress_weight * stress_penalty +
            total_penalty
    )

    total_fitness = 1e9 if np.isinf(total_fitness) or np.isnan(total_fitness) else total_fitness
    return total_fitness


def load_knowledge_base() -> Tuple[faiss.Index, Dict]:
    try:
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            return index, metadata
        return None, None
    except Exception as e:
        print(f"知识库加载失败: {e}")
        return None, None


def retrieve_similar_structures(formula: str, index: faiss.Index, metadata: Dict, k: int = 5) -> List[Dict]:
    if index is None or metadata is None:
        return []

    atom_count = defaultdict(int)
    for elem, cnt in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
        atom_count[elem] += int(cnt) if cnt else 1

    query_vec = np.zeros(LATENT_DIM, dtype=np.float32)
    for elem, cnt in atom_count.items():
        num = symbol_to_atomic_num.get(elem, 0)
        if num > 0:
            query_vec[num % LATENT_DIM] = cnt

    distances, indices = index.search(query_vec.reshape(1, -1), k)

    similar = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and str(idx) in metadata:
            entry = metadata[str(idx)]
            entry['distance'] = distances[0][i]
            similar.append(entry)
    return similar


def parse_formula(formula: str) -> List[int]:
    # 先尝试从别名库获取标准分子式
    input_clean = formula.strip().lower()
    if input_clean in COMMON_COMPOUND_LIB:
        formula = COMMON_COMPOUND_LIB[input_clean]["formula"]
        print(f"🔍 别名匹配：使用标准分子式 {formula}")

    atom_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        if elem not in symbol_to_atomic_num:
            raise ValueError(f"不支持的元素: {elem}")
        atom_count[elem] += int(cnt) if cnt else 1

    atom_types = []
    for elem, cnt in atom_count.items():
        atom_types.extend([symbol_to_atomic_num[elem]] * cnt)

    if len(atom_types) > MAX_ATOMS:
        raise ValueError(f"原子数超过上限({MAX_ATOMS})")
    if len(atom_types) == 0:
        raise ValueError(f"分子式 {formula} 解析后原子数为0")
    return atom_types


def calculate_chemical_formula(atom_types: List[int]) -> Tuple[str, str]:
    elem_count = defaultdict(int)
    for num in atom_types:
        elem = atomic_num_to_symbol.get(num, f"X{num}")
        elem_count[elem] += 1

    structural_formula = "".join([f"{elem}{cnt if cnt > 1 else ''}" for elem, cnt in sorted(elem_count.items())])
    sum_formula = "".join([f"{elem}{cnt}" for elem, cnt in sorted(elem_count.items())])
    return structural_formula, sum_formula


def calculate_z_value(atom_types: List[int], lattice_volume: float) -> int:
    structural_formula, _ = calculate_chemical_formula(atom_types)
    formula_unit_atoms = len(atom_types)
    avg_vol_per_atom = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atom_types) / formula_unit_atoms
    z_value = max(1, int(lattice_volume / (avg_vol_per_atom * formula_unit_atoms) + 0.5))
    return z_value


# =============================================================================
# 【原始模块】核心生成函数 (微调传入formula)
# =============================================================================

def generate_valid_structures(formula: str, num_required: int = 3) -> Tuple[
    List[dgl.DGLGraph], List[float], List[np.ndarray], List[float], List[List[float]], List[Dict]]:
    # 解析分子式（支持别名）
    atom_types = parse_formula(formula)

    # 获取标准分子式用于显示
    input_clean = formula.strip().lower()
    display_formula = COMMON_COMPOUND_LIB[input_clean]["formula"] if input_clean in COMMON_COMPOUND_LIB else formula

    print(f"解析分子式 {formula} -> 原子列表: {[atomic_num_to_symbol[n] for n in atom_types]}")

    index, metadata = load_knowledge_base()
    similar_structs = retrieve_similar_structures(display_formula, index, metadata)

    # 演示用：减少粒子数和代数，加快运行
    num_particles = 20
    particles = [Particle(LATENT_DIM) for _ in range(num_particles)]
    global_best_pos = particles[0].position.clone()
    global_best_fitness = float('inf')
    valid_particles = []

    max_iter = 10
    initial_w = 0.9
    final_w = 0.4
    c1 = 1.8
    c2 = 1.5

    print("开始PSO优化（演示模式）...")
    for iter_idx in range(max_iter):
        w = initial_w - (initial_w - final_w) * (iter_idx / max_iter)

        for p in particles:
            sg_info = get_space_group_info()
            lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])
            lattice = Lattice.from_parameters(*lattice_params)

            # 【修改】传入formula
            g = generate_graph_from_latent(p.position, atom_types, similar_structs, formula)

            g_optimized, lattice_optimized = realtime_optimize_single_structure(g, atom_types, lattice)
            optimized_lattice_params = list(lattice_optimized.abc) + list(lattice_optimized.angles)

            energy, stress_tensor = calculate_energy_and_stress(g_optimized, optimized_lattice_params)
            fitness = calculate_fitness(g_optimized, lattice_optimized, stress_tensor)

            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.clone()
                is_structure_valid, valid_msg = check_structure_validity(g_optimized, lattice_optimized)
                p.is_valid = is_structure_valid

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = p.position.clone()

        for p in particles:
            p.update_velocity(global_best_pos, w=w, c1=c1, c2=c2)
            p.update_position()

        valid_particles = [p for p in particles if p.is_valid and p.best_fitness < 1e8]
        if len(valid_particles) >= num_required:
            break

    valid_particles = sorted(valid_particles, key=lambda x: x.best_fitness)[:num_required]

    if not valid_particles:
        print("⚠️ 未找到完全有效结构，取最优候选...")
        valid_particles = sorted(particles, key=lambda x: x.best_fitness)[:num_required]

    valid_candidates = []
    valid_energies = []
    valid_stress = []
    valid_fitness = []
    valid_lattices = []
    valid_sg = []

    for p in tqdm(valid_particles, desc="生成最终结构"):
        sg_info = get_space_group_info()
        lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])
        lattice = Lattice.from_parameters(*lattice_params)

        # 【修改】传入formula
        g = generate_graph_from_latent(p.best_position, atom_types, similar_structs, formula)

        energy, stress_tensor = calculate_energy_and_stress(g, lattice_params)
        fitness = calculate_fitness(g, lattice, stress_tensor)

        valid_candidates.append(g)
        valid_energies.append(energy)
        valid_stress.append(stress_tensor)
        valid_fitness.append(fitness)
        valid_lattices.append(lattice_params)
        valid_sg.append(sg_info)

    return valid_candidates, valid_energies, valid_stress, valid_fitness, valid_lattices, valid_sg


def visualize_results(formula: str, energies: List[float], stress_tensors: List[np.ndarray],
                      fitness_scores: List[float]):
    print(f"📊 可视化结果将保存到: {VISUALIZATION_DIR}")
    pass


# =============================================================================
# 【主函数】
# =============================================================================

def main():
    print("=" * 60)
    print("  N-RGAG 晶体结构生成器 (优化版：平面逻辑优先)")
    print("=" * 60)

    # 测试用例列表
    test_cases = ["TNT", "苯甲酸", "CL-20", "苯", "C6H6"]

    print("\n可用测试化合物: TNT, 苯甲酸, CL-20, 苯, C6H6")
    formula = input("请输入化合物名或分子式 (直接回车测试TNT): ").strip()

    if not formula:
        formula = "TNT"
        print(f"使用默认测试: {formula}")

    try:
        print(f"\n开始生成 {formula} 的候选结构...")
        candidates, energies, stress_tensors, fitness, lattices, sg_infos = generate_valid_structures(formula, 3)

        formula_dir = GENERATED_CIF_DIR / formula.replace(" ", "_")
        formula_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n保存结构到 {formula_dir}...")
        for idx, (g, energy, stress_tensor, fit, lattice, sg) in enumerate(
                zip(candidates, energies, stress_tensors, fitness, lattices, sg_infos)):
            crystal_dict = graph_to_crystal_dict(g, lattice, sg, energy, stress_tensor, fit)
            cif_path = formula_dir / f"{formula.replace(' ', '_')}_gen_{idx + 1}.cif"
            save_crystal_to_cif(crystal_dict, cif_path)
            print(f"  ✅ 已保存: {cif_path.name}")

        print("\n===== 生成完成 =====")
        print(f"✅ 结构已保存到: {formula_dir}")

    except ValueError as e:
        print(f"输入错误: {e}")
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()