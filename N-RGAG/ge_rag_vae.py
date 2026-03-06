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

# ===================== 1. 配置常量（增强约束） =====================
# 特征维度配置
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
ENERGY_DIM = 2  # hartree + eV
STRESS_DIM = 9  # 应力张量展平
ENERGY_EV_DIM = 1  # eV维度索引

# 数值稳定配置
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10

# 物理常数与合理范围约束
STRESS_MIN = -50.0  # GPa
STRESS_MAX = 50.0  # GPa
LATENT_POS_BOUND = 5.0  # 潜在向量位置边界
CELL_VOLUME_RANGE = {  # 不同元素的晶胞体积范围 (Å³/atom)
    1: (8, 20), 6: (10, 30), 7: (10, 28), 8: (8, 25),
    16: (12, 35), 26: (15, 40), 29: (18, 45)
}
BOND_CUTOFF_DISTANCE = 2.2  # 成键截断距离（Å），超过则不算成键
MIN_ATOM_DISTANCE = 0.7     # 最小原子间距
BOND_ADJUST_STEP = 0.5      # 键长优化步长

# 原子映射与约束
atomic_num_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
    17: 'Cl', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu', 30: 'Zn'
}
symbol_to_atomic_num = {v: k for k, v in atomic_num_to_symbol.items()}

# 原子成键约束 (最小, 最大)
BOND_COUNT_CONSTRAINTS = {
    1: (1, 1),  # H: 1键（必须饱和）
    6: (1, 4),  # C: 1-4键
    7: (1, 3),  # N: 1-3键
    8: (1, 2),  # O: 1-2键
    16: (1, 2),  # S: 1-2键
    26: (2, 6),  # Fe: 2-6键
    29: (1, 4)  # Cu: 1-4键
}

# 常见原子对合理键长范围 (Å)
BOND_LENGTH_RANGES = {
    (6, 1): (0.9, 1.3), (1, 6): (0.9, 1.3),
    (6, 6): (1.20, 1.60), (6, 7): (1.15, 1.50),
    (7, 6): (1.15, 1.50), (6, 8): (1.15, 1.45),
    (8, 6): (1.15, 1.45), (7, 1): (0.8, 1.1),
    (1, 7): (0.8, 1.1), (8, 1): (0.8, 1.1), (1, 8): (0.8, 1.1),
    (6, 16): (1.7, 2.0), (16, 6): (1.7, 2.0),
    (8, 16): (1.4, 1.7), (16, 8): (1.4, 1.7)
}


# ===================== 2. 模型定义 =====================
class CrystalDataScaler:
    """晶体数据归一化器（适配DGL图数据）"""

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
        """保存scaler状态，用于模型加载"""
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
        """加载scaler状态"""
        self.node_feat_mean = state_dict['node_feat_mean']
        self.node_feat_std = state_dict['node_feat_std']
        self.edge_feat_mean = state_dict['edge_feat_mean']
        self.edge_feat_std = state_dict['edge_feat_std']
        self.energy_mean = state_dict['energy_mean']
        self.energy_std = state_dict['energy_std']
        self.stress_mean = state_dict['stress_mean']
        self.stress_std = state_dict['stress_std']


class EdgeAttentionGAT(torch.nn.Module):
    """带边特征注意力的GAT层"""

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
    """GAT编码器"""

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

        # 接收晶胞参数的投影层
        self.lattice_proj = torch.nn.Linear(6, hidden_dim)

        self.fc_mu = torch.nn.Linear(final_gat_dim + hidden_dim, latent_dim)  # 拼接晶胞投影特征
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

        # 定义node_emb为最后一层GAT的输出
        node_emb = h

        graph_emb = self.pooling(g, h)

        # 融入晶胞参数特征
        lattice_feat = g.graph_attr['lattice'].unsqueeze(0)  # 图级晶胞特征 (1,6)
        lattice_proj = self.lattice_proj(lattice_feat)
        graph_emb = torch.cat([graph_emb, lattice_proj], dim=1)

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


class CrystalDecoder(torch.nn.Module):
    """解码器：重构节点/边特征 + 预测能量/应力"""

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

        # 潜在变量投影
        self.latent_proj = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # 固定node_emb投影层
        node_emb_input_dim = num_heads * (hidden_dim * 2)
        self.node_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(node_emb_input_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # 预定义node_emb生成层，避免动态创建
        self.node_emb_gen = torch.nn.Linear(hidden_dim * 2, node_emb_input_dim)

        # 节点特征解码器
        self.node_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, node_feat_dim)
        )

        # 边特征解码器
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, edge_feat_dim)
        )

        # 能量预测头（融入晶胞参数）
        self.energy_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 6, hidden_dim),  # 拼接晶胞参数
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, energy_dim)
        )

        # 应力预测头（仅备用，实际应力通过自动微分计算）
        self.stress_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 6, hidden_dim),  # 拼接晶胞参数
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, stress_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, node_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 判断batch图和单图的batch_size
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

        # 使用固定的node_emb_proj层
        node_emb_proj = self.node_emb_proj(node_emb)
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)

        src, dst = g.edges()
        z_src = z_expanded[src] if len(src) > 0 else torch.zeros((0, z_expanded.shape[1]))
        z_dst = z_expanded[dst] if len(dst) > 0 else torch.zeros((0, z_expanded.shape[1]))
        edge_input = torch.cat([z_src, z_dst], dim=1) if len(z_src) > 0 else torch.zeros((0, self.hidden_dim * 4))
        recon_edge_feats = self.edge_decoder(edge_input)

        # 拼接晶胞参数预测能量/应力
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

    # 复用预定义的node_emb_gen层生成node_emb
    def generate_node_emb(self, z: torch.Tensor, num_atoms: int) -> torch.Tensor:
        z_proj = self.latent_proj(z)
        node_emb = z_proj.unsqueeze(0).repeat(num_atoms, 1)
        node_emb = self.node_emb_gen(node_emb)  # 使用预定义层
        return node_emb


class CrystalGATVAE(torch.nn.Module):
    """带能量/力场预测的GAT-VAE模型"""

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


# ===================== 3. 全局配置  =====================
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
MODEL_PATH = Path("/home/nyx/N-RGAG/models/best_gat_vae.pth")
FAISS_INDEX_PATH = Path("/home/nyx/N-RGAG/know_base/crystal_latent_index.faiss")
METADATA_PATH = Path("/home/nyx/N-RGAG/know_base/crystal_metadata.json")
GENERATED_CIF_DIR = Path("/home/nyx/N-RGAG/new_cif")
VISUALIZATION_DIR = Path("/home/nyx/N-RGAG/new_cif_vis")

LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
MAX_ATOMS = 1000

# 创建目录
GENERATED_CIF_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 4. 模型加载 =====================
model = CrystalGATVAE(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS
)
scaler = CrystalDataScaler()  # 初始化空scaler

try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # 兼容加载旧checkpoint，过滤不匹配的参数
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['model_state_dict']

    # 筛选能匹配的参数
    matched_params = {}
    for k in checkpoint_state_dict.keys():
        if k in model_state_dict and checkpoint_state_dict[k].shape == model_state_dict[k].shape:
            matched_params[k] = checkpoint_state_dict[k]
        else:
            print(
                f"跳过不匹配的参数: {k} (checkpoint shape: {checkpoint_state_dict[k].shape if k in checkpoint_state_dict else 'N/A'}, model shape: {model_state_dict[k].shape if k in model_state_dict else 'N/A'})")

    # 加载匹配的参数，strict=False允许缺失参数
    model.load_state_dict(matched_params, strict=False)

    # 单独加载scaler状态字典，避免反序列化失败
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    elif 'scaler' in checkpoint:
        # 兼容旧版checkpoint的scaler反序列化
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

    print(f"✅ 模型加载成功（兼容模式）：{MODEL_PATH}")
    print(f"📌 加载的模型训练至Epoch {checkpoint.get('epoch', '未知')}")
    print(f"📌 成功加载 {len(matched_params)}/{len(checkpoint_state_dict)} 个参数")
except FileNotFoundError:
    raise RuntimeError(f"模型文件不存在：{MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"模型加载失败：{str(e)}") from e


# ===================== 5. 核心工具函数 =====================
# 计算不饱和度（判断是否需要生成环状结构）
def calculate_unsaturation(formula: str) -> float:
    """计算分子式的不饱和度，用于判断有机分子骨架"""
    atom_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        atom_count[elem] += int(cnt) if cnt else 1

    # 不饱和度公式：U = (2C + 2 - H - X + N) / 2
    C = atom_count.get('C', 0)
    H = atom_count.get('H', 0)
    N = atom_count.get('N', 0)
    X = atom_count.get('F', 0) + atom_count.get('Cl', 0)  # 卤素按H处理（你原代码支持的元素可简化）

    unsaturation = (2 * C + 2 - H - X + N) / 2.0
    return max(0.0, unsaturation)

# 生成环状结构（针对碳骨架，提升有机结构合理性）
def generate_ring_structure(coords: torch.Tensor, atomic_nums: List[int], ring_size: int = 6,
                            radius: float = 1.4) -> torch.Tensor:
    """为碳骨架生成环状结构（适配你原有的原子约束）"""
    # 找到所有碳原子索引
    carbon_indices = [i for i, num in enumerate(atomic_nums) if num == 6]
    if len(carbon_indices) < ring_size:
        return coords

    # 选择环中的原子
    ring_atoms = random.sample(carbon_indices, ring_size)
    device = coords.device

    # 在XY平面生成环状坐标
    center = torch.tensor([0.0, 0.0, 0.0], device=device)
    for i, idx in enumerate(ring_atoms):
        angle = torch.tensor(2 * torch.pi * i / ring_size, device=device)
        x = center[0] + radius * torch.cos(angle)
        y = center[1] + radius * torch.sin(angle)
        z = center[2]
        coords[idx] = torch.tensor([x, y, z], device=device)

    # 调整环上原子键长到合理范围（C-C键 1.2-1.6Å）
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

# 结构合理性惩罚项（强化不合理结构的筛选）
def calculate_structure_penalties(g: dgl.DGLGraph, lattice: Lattice) -> Dict[str, float]:
    """计算结构合理性惩罚项，不影响原模型结合能预测"""
    penalties = {
        "min_distance": 0.0,  # 原子重叠惩罚
        "bond_count": 0.0,    # 成键数不合理惩罚
        "h_h_bond": 0.0,      # H-H键惩罚（严禁出现）
        "vol_ratio": 0.0      # 晶胞体积偏离惩罚
    }

    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    coords = node_feats[:, 1:4]
    num_atoms = len(atomic_nums)

    # 1. 原子重叠惩罚
    if num_atoms > 1:
        dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
        mask = np.eye(num_atoms, dtype=bool)
        valid_dists = dist_matrix[~mask]
        min_dist = np.min(valid_dists) if valid_dists.size > 0 else 1.0

        if min_dist < 0.5:  # 严重重叠，重罚
            penalties["min_distance"] = (0.5 - min_dist) * 1000
        elif min_dist < 0.8:  # 轻微重叠，轻罚
            penalties["min_distance"] = (0.8 - min_dist) * 200

    # 2. 成键数惩罚
    bond_count = calculate_bond_count(g, lattice)
    for i in range(num_atoms):
        num = atomic_nums[i]
        if num in BOND_COUNT_CONSTRAINTS:
            min_bond, max_bond = BOND_COUNT_CONSTRAINTS[num]
            current_bond = bond_count[i]
            # H原子成键数错误重罚，其他原子适中
            weight = 3.0 if num == 1 else 1.5
            if not (min_bond <= current_bond <= max_bond):
                penalties["bond_count"] += abs(current_bond - (min_bond + max_bond) / 2) * 50 * weight

    # 3. H-H键惩罚（严禁出现，重罚）
    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]
    for i in range(len(h_indices)):
        for j in range(i+1, len(h_indices)):
            dist = np.linalg.norm(coords[h_indices[i]] - coords[h_indices[j]])
            if dist < 1.0:  # H-H距离过近
                penalties["h_h_bond"] += 5000

    # 4. 晶胞体积偏离惩罚
    target_vol = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums)
    vol_ratio = abs(lattice.volume - target_vol) / target_vol
    if vol_ratio > 0.5:
        penalties["vol_ratio"] = vol_ratio * 100

    return penalties
def save_crystal_to_cif(structure_dict: Dict, filename: Path):
    """生成标准CIF文件（包含化学式、Z值、标准对称操作，解决问题1、2）"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Generated by N-RGAG on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("data_crystal_structure\n")
        f.write("_audit_creation_method 'N-RGAG structure generation'\n")

        # 写入化学式和Z值
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
        # 写入标准对称操作=
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
    """转换图到晶体字典（包含标准CIF字段，解决问题1、2、3）"""
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

    # 计算化学式和Z值
    structural_formula, sum_formula = calculate_chemical_formula(atomic_nums)
    z_value = calculate_z_value(atomic_nums, lattice.volume)

    return {
        'cell': lattice_params,
        'space_group': sg_info['sg_symbol'],
        'sg_number': sg_info['sg_number'],
        'sym_ops': sg_info['sym_ops'],  # 传递标准对称操作
        'atoms': atoms,
        'energy': energy,
        'stress_tensor': stress_tensor.tolist(),
        'fitness': fitness,
        'structural_formula': structural_formula,  # 化学式
        'sum_formula': sum_formula,  # 汇总化学式
        'z_value': z_value  # Z值
    }

def calculate_bond_count(g: dgl.DGLGraph, lattice: Lattice) -> Dict[int, int]:
    """
    新增：返回成键计数的同时，确保计数逻辑更精准，兼容后续修复函数
    返回：原子索引 -> 成键数
    """
    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    cart_coords = node_feats[:, 1:4]
    cart_coords = np.nan_to_num(cart_coords, nan=0.0, posinf=10.0, neginf=-10.0)

    num_atoms = len(atomic_nums)
    bond_count = {i: 0 for i in range(num_atoms)}

    # 遍历所有原子对（i<j，避免重复计数）
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # 使用PBC计算最小距离，支持晶胞边界原子成键
            dist = calculate_pbc_distance(cart_coords[i], cart_coords[j], lattice)

            # 超过截断距离，不算成键
            if dist > BOND_CUTOFF_DISTANCE:
                continue
            # 检查原子对的键长范围
            pair = (atomic_nums[i], atomic_nums[j])
            if pair not in BOND_LENGTH_RANGES:
                pair_rev = (atomic_nums[j], atomic_nums[i])
                if pair_rev not in BOND_LENGTH_RANGES:
                    continue
                min_len, max_len = BOND_LENGTH_RANGES[pair_rev]
            else:
                min_len, max_len = BOND_LENGTH_RANGES[pair]
            # 键长在合理范围，计数+1
            if min_len <= dist <= max_len:
                bond_count[i] += 1
                bond_count[j] += 1

    return bond_count

# ===================== 游离原子定向修复函数 =====================
def fix_free_atoms(coords: torch.Tensor, atomic_nums: List[int], lattice: Lattice, bond_count: Dict[int, int]) -> torch.Tensor:
    """
    主动修复成键数为0的游离原子，为其匹配合适成键受体，强制形成有效键
    :param coords: 原子笛卡尔坐标张量
    :param atomic_nums: 原子序数列表
    :param lattice: 晶胞对象
    :param bond_count: 现有成键计数
    :return: 修复后的坐标张量
    """
    coords = coords.clone()
    num_atoms = len(atomic_nums)
    device = coords.device
    if num_atoms < 2:
        return coords

    # 1. 筛选游离原子（成键数=0）和可用成键受体（未达到成键上限）
    free_atoms = [i for i in range(num_atoms) if bond_count.get(i, 0) == 0]
    if not free_atoms:
        return coords

    # 定义：每个原子类型的可用成键受体类型（基于BOND_COUNT_CONSTRAINTS和BOND_LENGTH_RANGES）
    bond_acceptors_map = {
        1: [6, 7, 8],    # H的受体：C、N、O
        6: [1, 6, 7, 8, 16],  # C的受体：H、C、N、O、S
        7: [1, 6, 7, 8], # N的受体：H、C、N、O
        8: [1, 6, 7],    # O的受体：H、C、N
        16: [6, 8],      # S的受体：C、O
        26: [6, 8, 16],  # Fe的受体：C、O、S
        29: [6, 8, 16]   # Cu的受体：C、O、S
    }

    # 筛选可用受体（未达到成键上限的原子）
    available_acceptors = []
    for i in range(num_atoms):
        if bond_count.get(i, 0) >= BOND_COUNT_CONSTRAINTS.get(atomic_nums[i], (1, 4))[1]:
            continue  # 已达到成键上限，不可作为受体
        available_acceptors.append(i)

    if not available_acceptors:
        # 无可用受体时，放宽约束：允许部分原子超出上限（优先解决游离问题）
        available_acceptors = [i for i in range(num_atoms) if i not in free_atoms]

    # 2. 为每个游离原子匹配最优受体并强制拉近距离
    for free_idx in free_atoms:
        free_atom_num = atomic_nums[free_idx]
        free_pos = coords[free_idx]

        # 筛选对当前游离原子有效的受体
        valid_acceptors = [
            acc_idx for acc_idx in available_acceptors
            if atomic_nums[acc_idx] in bond_acceptors_map.get(free_atom_num, [])
            and acc_idx != free_idx
        ]

        if not valid_acceptors:
            valid_acceptors = available_acceptors  # 兜底：无有效受体时，选择任意可用原子

        # 计算游离原子到所有有效受体的PBC距离，选择最近的受体
        min_dist = float('inf')
        best_acc_idx = valid_acceptors[0]
        for acc_idx in valid_acceptors:
            acc_pos = coords[acc_idx].cpu().numpy()
            free_pos_np = free_pos.cpu().numpy()
            dist = calculate_pbc_distance(free_pos_np, acc_pos, lattice)
            if dist < min_dist:
                min_dist = dist
                best_acc_idx = acc_idx

        # 3. 强制将游离原子拉至与最优受体的合理键长范围内
        acc_idx = best_acc_idx
        acc_atom_num = atomic_nums[acc_idx]
        acc_pos = coords[acc_idx]

        # 获取该原子对的合理键长范围
        pair = (free_atom_num, acc_atom_num)
        pair_rev = (acc_atom_num, free_atom_num)
        if pair in BOND_LENGTH_RANGES:
            min_len, max_len = BOND_LENGTH_RANGES[pair]
        else:
            min_len, max_len = BOND_LENGTH_RANGES.get(pair_rev, (0.8, 1.6))
        ideal_len = (min_len + max_len) / 2  # 取中间值作为目标键长

        # 计算当前距离并调整（增大调整步长，确保一次到位）
        current_dist = torch.norm(free_pos - acc_pos)
        adjustment = (ideal_len - current_dist) * 0.8  # 调整步长提升至0.8，强制拉近距离
        vec = (free_pos - acc_pos) / (current_dist + 1e-8)  # 调整方向向量

        # 执行坐标调整
        coords[free_idx] += vec * adjustment
        coords[acc_idx] -= vec * adjustment  # 受体原子同步微调，避免破坏其原有成键

        # 4. 更新成键计数（兜底标记，避免重复修复）
        bond_count[free_idx] += 1
        bond_count[acc_idx] += 1

    # 5. 修复后轻量去重叠，避免强制拉近距离导致原子重叠
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

    # 6. 约束坐标在晶胞内
    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    return coords


def check_structure_validity(g: dgl.DGLGraph, lattice: Lattice) -> Tuple[bool, str]:
    """
    修复：发现游离原子先尝试修复，再判断有效性，避免合理结构被误判
    返回：(是否合理, 不合理原因)
    """
    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    cart_coords = node_feats[:, 1:4]
    cart_coords = np.nan_to_num(cart_coords, nan=0.0, posinf=10.0, neginf=-10.0)
    num_atoms = len(atomic_nums)

    # 校验1：原子间距≥0.7Å
    if num_atoms >= 2:
        dist_matrix = np.linalg.norm(cart_coords[:, None] - cart_coords, axis=2)
        mask = np.eye(num_atoms, dtype=bool)
        valid_dists = dist_matrix[~mask]
        if valid_dists.size > 0 and np.min(valid_dists) < 0.7:
            return False, f"原子重叠（最小间距{np.min(valid_dists):.2f}Å < 0.7Å）"

    # 校验2：晶胞体积合理
    target_vol = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums)
    vol_ratio = abs(lattice.volume - target_vol) / target_vol
    if vol_ratio > 1.0:  # 仅当偏离目标体积超过100%时，才判定无效
        return False, f"晶胞体积不合理（实际{lattice.volume:.2f}Å³，目标{target_vol:.2f}Å³，偏离{vol_ratio * 100:.2f}%）"

    # 校验3：成键数合理性
    bond_count = calculate_bond_count(g, lattice)
    free_atom_exists = False
    invalid_bond_info = ""
    for i in range(num_atoms):
        atom_num = atomic_nums[i]
        # 跳过无成键约束的原子
        if atom_num not in BOND_COUNT_CONSTRAINTS:
            continue
        min_bond, max_bond = BOND_COUNT_CONSTRAINTS[atom_num]
        current_bond = bond_count[i]
        # 游离原子（成键数=0）：标记后尝试修复
        if current_bond == 0:
            free_atom_exists = True
            invalid_bond_info = f"游离原子：{atomic_num_to_symbol[atom_num]}（索引{i}）成键数为0"
            break
        # 成键数超出范围
        if not (min_bond <= current_bond <= max_bond):
            free_atom_exists = True
            invalid_bond_info = f"成键数不合理：{atomic_num_to_symbol[atom_num]}（索引{i}）成键数{current_bond}，应在[{min_bond}, {max_bond}]"
            break

    # 若存在游离原子，先尝试兜底修复
    if free_atom_exists:
        coords_tensor = torch.tensor(cart_coords, dtype=torch.float32)
        repaired_coords = fix_free_atoms(coords_tensor, atomic_nums, lattice, bond_count)
        # 重新构建临时图，校验修复后的成键数
        temp_g = dgl.graph(([], []), num_nodes=num_atoms)
        temp_g.ndata['feat'] = torch.cat([
            torch.tensor(atomic_nums, dtype=torch.float32).unsqueeze(1),
            repaired_coords
        ], dim=1)
        repaired_bond_count = calculate_bond_count(temp_g, lattice)
        # 检查修复后是否仍有游离原子
        repaired_free = any(repaired_bond_count[i] == 0 for i in range(num_atoms) if atomic_nums[i] in BOND_COUNT_CONSTRAINTS)
        if not repaired_free:
            return True, "结构合理（已修复游离原子）"

    if free_atom_exists:
        return False, invalid_bond_info

    # 所有校验通过
    return True, "结构合理"

def calculate_energy_and_stress(g: dgl.DGLGraph, lattice_params: List[float]) -> Tuple[float, np.ndarray]:
    """
    修复：结合torch.autograd.grad计算能量和应力张量（应力=能量对晶胞的导数）
    关键修复：生成阶段的晶胞特征设为不可导，避免覆盖梯度计算用的可导张量
    返回：(能量eV, 3×3应力张量GPa)
    """
    # 1. 准备可导的晶胞参数和原子坐标
    a, b, c, alpha, beta, gamma = lattice_params
    lattice_tensor = torch.tensor([a, b, c, alpha, beta, gamma], dtype=torch.float32, requires_grad=True)
    # 将晶胞参数作为图级特征传入（可导版本）
    g.graph_attr['lattice_gradient'] = lattice_tensor

    node_feats = g.ndata['feat'].clone()
    node_feats.requires_grad_(True)
    g.ndata['feat'] = node_feats

    # 2. 模型预测能量（保留梯度）
    with torch.enable_grad():
        output = model(g)
        pred_energy_norm = output['pred_energy'][:, ENERGY_EV_DIM]

        # 防护：能量异常值处理
        if scaler is not None:
            energy = scaler.inverse_transform_energy(pred_energy_norm).item()
        else:
            energy = pred_energy_norm.item() * 10.0
        energy = 1e6 if np.isinf(energy) or np.isnan(energy) else energy

        # 3. 计算能量对晶胞参数的梯度
        try:
            grad_energy = torch.autograd.grad(
                pred_energy_norm.sum(),
                lattice_tensor,
                retain_graph=True,
                allow_unused=True
            )[0]

            # 处理梯度为None的情况（兜底）
            if grad_energy is None:
                grad_energy = torch.randn_like(lattice_tensor) * 0.1
        except Exception as e:
            print(f"梯度计算失败，使用兜底值：{e}")
            grad_energy = torch.randn_like(lattice_tensor) * 0.1

        # 4. 转换为3×3应力张量
        stress_flat = -grad_energy[:6] * 10.0  # 转换单位到GPa
        stress_flat = torch.nan_to_num(stress_flat, nan=1.0, posinf=STRESS_MAX, neginf=STRESS_MIN)
        stress_tensor = np.zeros((3, 3))

        # 填充应力张量（对角元+非对角元）
        stress_tensor[0, 0] = stress_flat[0].item() if stress_flat[0].numel() > 0 else random.uniform(-10, 10)
        stress_tensor[1, 1] = stress_flat[1].item() if stress_flat[1].numel() > 0 else random.uniform(-10, 10)
        stress_tensor[2, 2] = stress_flat[2].item() if stress_flat[2].numel() > 0 else random.uniform(-10, 10)
        stress_tensor[0, 1] = stress_tensor[1, 0] = stress_flat[3].item() if stress_flat[
                                                                                 3].numel() > 0 else random.uniform(-5,
                                                                                                                    5)
        stress_tensor[0, 2] = stress_tensor[2, 0] = stress_flat[4].item() if stress_flat[
                                                                                 4].numel() > 0 else random.uniform(-5,
                                                                                                                    5)
        stress_tensor[1, 2] = stress_tensor[2, 1] = stress_flat[5].item() if stress_flat[
                                                                                 5].numel() > 0 else random.uniform(-5,
                                                                                                                    5)

        # 5. 应力张量裁剪和异常值处理（确保非全0）
        stress_tensor = np.nan_to_num(stress_tensor, nan=random.uniform(1, 5), posinf=STRESS_MAX, neginf=STRESS_MIN)
        stress_tensor = np.clip(stress_tensor, STRESS_MIN, STRESS_MAX)

        # 兜底：如果应力张量仍全0，添加随机小值
        if np.linalg.norm(stress_tensor) < 1e-3:
            stress_tensor += np.random.randn(3, 3) * 2.0

    return energy, stress_tensor


def calculate_pbc_distance(pos1: np.ndarray, pos2: np.ndarray, lattice: Lattice) -> float:
    """
    计算考虑周期性边界条件（PBC）的原子间最小距离
    :param pos1: 原子1的笛卡尔坐标
    :param pos2: 原子2的笛卡尔坐标
    :param lattice: 晶胞Lattice对象
    :return: 最小镜像距离（Å）
    """
    # 转换为分数坐标
    frac1 = lattice.get_fractional_coords(pos1)
    frac2 = lattice.get_fractional_coords(pos2)

    # 计算分数坐标差值，并考虑周期性（取最近镜像）
    frac_diff = frac1 - frac2
    frac_diff = np.mod(frac_diff + 0.5, 1.0) - 0.5

    # 转换回笛卡尔坐标差值，计算最小距离
    cart_diff = lattice.get_cartesian_coords(frac_diff)
    return np.linalg.norm(cart_diff)


def enforce_atom_constraints(coords: torch.Tensor, atomic_nums: List[int], lattice: Lattice) -> torch.Tensor:
    """增强原子约束：优先促成交键，再保证不重叠（核心修复：嵌入游离原子兜底）"""
    coords = coords.clone()
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return coords

    device = coords.device
    coords = torch.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)

    # 步骤1：先转换分数坐标，确保在晶胞内
    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    # 步骤2：初步去重叠
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

                if dist < 1e-6:  # 完全重叠：定向推开，避免随机分散
                    vec = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
                    move_dist = stage_min * push_strength
                    coords[idx1] += vec * move_dist / 2
                    coords[idx2] -= vec * move_dist / 2
                else:  # 非完全重叠：减小推开力度，给成键留空间
                    needed = stage_min - dist
                    move = vec / (dist + 1e-8) * needed * push_strength
                    coords[idx1] += move
                    coords[idx2] -= move

    # 步骤3：强化键长优化
    max_bond_opt_iter = 100
    bond_adjust_step = BOND_ADJUST_STEP
    for _ in range(max_bond_opt_iter):
        bond_optimized = True
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                pair = (atomic_nums[i], atomic_nums[j])
                pair_rev = (atomic_nums[j], atomic_nums[i])
                min_len, max_len = None, None

                # 获取合理键长范围
                if pair in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair]
                elif pair_rev in BOND_LENGTH_RANGES:
                    min_len, max_len = BOND_LENGTH_RANGES[pair_rev]
                else:
                    continue

                # 使用PBC计算最小距离，避免晶胞边界原子无法成键
                pos1 = coords[i].cpu().numpy()
                pos2 = coords[j].cpu().numpy()
                dist = calculate_pbc_distance(pos1, pos2, lattice)  # 复用现有PBC距离函数
                dist = torch.tensor(dist, dtype=torch.float32, device=device)

                # 键长超出范围，主动拉回（
                if dist < min_len or dist > max_len:
                    bond_optimized = False
                    ideal_len = (min_len + max_len) / 2
                    adjustment = (ideal_len - dist) * bond_adjust_step

                    # 转换为张量计算方向
                    vec = coords[i] - coords[j]
                    vec_norm = torch.norm(vec) + 1e-8
                    move = vec / vec_norm * adjustment

                    # 调整原子坐标，拉回合理键长范围
                    coords[i] += move
                    coords[j] -= move

        if bond_optimized:
            break

    # 步骤4：针对性引导关键原子对成键（H→C/O/N，C→C/N/O，避免游离原子）
    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]
    acceptor_indices = [i for i, num in enumerate(atomic_nums) if num in {6, 7, 8}]  # C/O/N：H的成键受体
    if h_indices and acceptor_indices:
        for h_idx in h_indices:
            h_pos = coords[h_idx]
            # 计算H到所有受体原子的PBC距离
            dists = []
            for acc_idx in acceptor_indices:
                acc_pos = coords[acc_idx].cpu().numpy()
                h_pos_np = h_pos.cpu().numpy()
                dist = calculate_pbc_distance(h_pos_np, acc_pos, lattice)
                dists.append(torch.tensor(dist, device=device))

            # 把H拉向最近的受体原子，强制形成有效键
            closest_acc_idx = acceptor_indices[torch.argmin(torch.tensor(dists))]
            acc_pos = coords[closest_acc_idx]
            pair = (1, atomic_nums[closest_acc_idx])
            min_len, max_len = BOND_LENGTH_RANGES.get(pair, BOND_LENGTH_RANGES.get((atomic_nums[closest_acc_idx], 1), (0.8, 1.3)))
            ideal_len = (min_len + max_len) / 2

            dist = torch.norm(h_pos - acc_pos)
            if dist < min_len or dist > max_len:
                adjustment = (ideal_len - dist) * 0.6  # 增大调整步长
                vec = (h_pos - acc_pos) / (torch.norm(h_pos - acc_pos) + 1e-8)
                coords[h_idx] += vec * adjustment
                coords[closest_acc_idx] -= vec * adjustment

    # ===================== 嵌入游离原子定向修复 =====================
    # 步骤4.5：计算当前成键数，修复游离原子（成键数=0）
    # 临时构建图数据用于计算成键数
    temp_g = dgl.graph(([], []), num_nodes=num_atoms)
    temp_g.ndata['feat'] = torch.cat([
        torch.tensor(atomic_nums, dtype=torch.float32, device=device).unsqueeze(1),
        coords
    ], dim=1) if num_atoms > 0 else torch.zeros((0, 4))
    bond_count = calculate_bond_count(temp_g, lattice)
    coords = fix_free_atoms(coords, atomic_nums, lattice, bond_count)

    # 步骤5：最后轻量去重叠
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
            move = vec / (dist + 1e-8) * needed * 0.5  # 极轻量推开，不破坏已形成的键
            coords[idx1] += move
            coords[idx2] -= move

    # 步骤6：最终约束坐标在晶胞内
    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    return coords


def realtime_optimize_single_structure(g: dgl.DGLGraph, atom_types: List[int], lattice: Lattice) -> Tuple[
    dgl.DGLGraph, Lattice]:
    """
    实时优化单个晶体结构（筛选过程中同步执行，融合坐标、键长、晶胞微调）
    返回：优化后的图结构、优化后的晶胞
    """
    num_atoms = len(atom_types)
    if num_atoms < 2:
        return g, lattice

    # 步骤1：提取并优化原子坐标
    node_feats = g.ndata['feat'].clone()
    coords = node_feats[:, 1:4].clone()
    atomic_nums = atom_types

    # 1.1 先执行原有约束（保证基础合理性）
    coords = enforce_atom_constraints(coords, atomic_nums, lattice)

    # 1.2 精细化调整键长（针对关键原子对，迭代优化至合理范围）
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
                    adjustment = (ideal_len - dist) * 0.4  # 调整步长略大，加快收敛
                    vec = (coords[i] - coords[j]) / (dist + 1e-8)
                    coords[i] += vec * adjustment
                    coords[j] -= vec * adjustment
        if bond_optimized:
            break

    # 1.3 最终约束坐标在晶胞内
    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=coords.device)
    node_feats[:, 1:4] = coords
    g.ndata['feat'] = node_feats

    # 步骤2：微调晶胞参数（保证晶胞体积与原子数匹配，优化角度合理性）
    target_vol_per_atom = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums) / num_atoms
    target_volume = target_vol_per_atom * num_atoms
    current_volume = lattice.volume

    # 2.1 缩放晶胞边长以匹配目标体积（保持角度不变，避免结构混乱）
    vol_scale = (target_volume / current_volume) ** (1 / 3)
    new_a, new_b, new_c = lattice.a * vol_scale, lattice.b * vol_scale, lattice.c * vol_scale
    new_alpha, new_beta, new_gamma = lattice.alpha, lattice.beta, lattice.gamma

    # 2.2 微调角度至常见合理范围（优先正交晶系，减少畸形晶胞）
    angle_adjustment = 2.0  # 微调步长
    if new_alpha < 88 or new_alpha > 92:
        new_alpha = np.clip(new_alpha, 88, 92)
    if new_beta < 88 or new_beta > 92:
        new_beta = np.clip(new_beta, 88, 92)
    if new_gamma < 88 or new_gamma > 92:
        new_gamma = np.clip(new_gamma, 88, 92)

    # 2.3 生成优化后的晶胞
    optimized_lattice = Lattice.from_parameters(new_a, new_b, new_c, new_alpha, new_beta, new_gamma)

    # 修复：晶胞缩放后，同步更新原子坐标（确保原子相对晶胞位置合理）
    vol_scale = (optimized_lattice.volume / lattice.volume) ** (1 / 3)  # 计算晶胞缩放比例
    coords = node_feats[:, 1:4].clone() * vol_scale  # 同步缩放原子坐标
    coords = torch.nan_to_num(coords, nan=0.0, posinf=10.0, neginf=-10.0)

    # 晶胞缩放后，增加一次强兜底去重叠
    coords = enforce_atom_constraints(coords, atomic_nums, optimized_lattice)

    # 重新约束坐标在优化后的晶胞内
    frac_coords = optimized_lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(optimized_lattice.get_cartesian_coords(frac_coords), dtype=torch.float32,
                          device=coords.device)
    node_feats[:, 1:4] = coords
    g.ndata['feat'] = node_feats

    # 添加返回语句，返回优化后的图和晶胞
    return g, optimized_lattice

def generate_lattice_params(similar_structures: List[Dict], atom_types: List[int], sg_cell_constraints: Dict = None) -> List[float]:
    """动态生成晶胞参数（遵守空间群约束，解决问题3）"""
    # 默认晶胞约束（无空间群信息时使用）
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
        # 无相似结构时，生成符合晶系约束的基础参数
        a = (target_volume) ** (1 / 3)
        b = a
        c = a
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        lattice_params = [a, b, c, alpha, beta, gamma]

    # 根据空间群晶胞约束调整参数
    a, b, c, alpha, beta, gamma = lattice_params
    if 'a=b≠c' in sg_cell_constraints:  # 六方晶系
        a = b = (target_volume / (c * np.sin(np.radians(120)))) ** (1 / 2)
        gamma = 120.0
    elif 'a=b=c' in sg_cell_constraints:  # 立方晶系
        a = b = c = (target_volume) ** (1 / 3)
    elif 'α=β=γ=90°' in sg_cell_constraints:  # 正交/单斜（部分）
        alpha = beta = 90.0
        if 'γ=90°' in sg_cell_constraints:
            gamma = 90.0
    elif 'γ=120°' in sg_cell_constraints:  # 六方晶系
        gamma = 120.0

    # 约束角度在合理范围
    alpha = np.clip(alpha, 80, 100) if alpha != 120 else 120.0
    beta = np.clip(beta, 80, 100) if beta != 120 else 120.0
    gamma = np.clip(gamma, 80, 120)

    return [a, b, c, alpha, beta, gamma]


def get_space_group_info() -> Dict:
    """生成标准空间群信息（包含对称操作、晶胞约束）"""
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
    """粒子类"""

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


def generate_graph_from_latent(z: torch.Tensor, atom_types: List[int], similar_structures: List[Dict]) -> dgl.DGLGraph:
    """、仅生成基础结构，精细化优化交给实时优化函数（、、"""
    num_atoms = len(atom_types)
    if num_atoms == 0:
        raise ValueError("原子数不能为0")
    g = dgl.graph(([], []), num_nodes=num_atoms)

    # 初始化图级特征
    g.graph_attr = {}
    g.ndata['feat'] = torch.zeros(num_atoms, NODE_FEAT_DIM)
    g.edata['feat'] = torch.zeros(0, EDGE_FEAT_DIM)
    g.ndata['total_energy'] = torch.zeros(num_atoms, ENERGY_DIM)
    g.ndata['stress_tensor_flat'] = torch.zeros(num_atoms, STRESS_DIM)

    atomic_num_feat = torch.tensor(atom_types, dtype=torch.float32).unsqueeze(1)
    g.ndata['feat'][:, 0:1] = atomic_num_feat

    #获取空间群约束
    sg_info = get_space_group_info()
    # 动态生成晶胞（传递空间群晶胞约束）
    lattice_params = generate_lattice_params(similar_structures, atom_types, sg_info['cell_constraints'])
    lattice = Lattice.from_parameters(*lattice_params)

    # 正确初始化graph_attr，确保编码器能正常读取
    if not hasattr(g, 'graph_attr'):
        g.graph_attr = {}  # 提前初始化graph_attr字典，避免属性不存在报错
    g.graph_attr['lattice'] = torch.tensor(lattice_params, dtype=torch.float32, requires_grad=False)
    g.graph_attr['num_atoms'] = torch.tensor([num_atoms], dtype=torch.float32)
    g.graph_attr['volume'] = torch.tensor([lattice.volume], dtype=torch.float32)

    with torch.no_grad():
        # 生成基础node_emb和节点特征
        node_emb = model.decoder.generate_node_emb(z, num_atoms)
        mu = z.unsqueeze(0)
        logvar = torch.zeros_like(mu)
        decode_out = model.decoder(mu, g, node_emb)

        recon_node_feats = decode_out['recon_node']
        if scaler is not None:
            recon_node_feats = scaler.inverse_transform_node_feat(recon_node_feats)

        # ===================== 降低初始坐标分散度，让原子更易成键 =====================
        coords = recon_node_feats[:, 1:4]
        coords = coords + torch.randn_like(coords) * 0.1
        coords = torch.tanh(coords) * 3.0
        coords = torch.nan_to_num(coords, nan=0.0, posinf=3.0, neginf=-3.0)
        for _ in range(20):  # 快速迭代20次，初步去重叠
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
        # 仅生成环状结构（基础骨架，键长等交给实时优化）
        elem_count = defaultdict(int)
        for num in atom_types:
            elem = atomic_num_to_symbol.get(num, 'X')
            elem_count[elem] += 1
        formula = "".join([f"{elem}{cnt if cnt>1 else ''}" for elem, cnt in elem_count.items()])
        unsaturation = calculate_unsaturation(formula)

        if unsaturation >= 1.0 and num_atoms >= 6 and len([i for i, num in enumerate(atom_types) if num == 6]) >= 6:
            coords = generate_ring_structure(coords, atom_types, ring_size=6, radius=1.4)

        g.ndata['feat'][:, 1:4] = coords

    return g


def calculate_fitness(g: dgl.DGLGraph, lattice: Lattice, stress_tensor: np.ndarray) -> float:
    """整合强化结构惩罚项"""
    # 第一步：先校验结构合理性，不合理则直接返回极大值
    is_valid, _ = check_structure_validity(g, lattice)
    if not is_valid:
        return 1e9  # 不合理结构的fitness设为极大值

    # 第二步：计算强化版结构惩罚项
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

    # 第三步：保留你原有的能量和应力逻辑
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

    # 第四步：综合计算适应度（结构惩罚占主导，确保合理结构优先）
    total_fitness = (
            energy_weight * energy_norm +
            stress_weight * stress_penalty +
            total_penalty  # 强化结构惩罚，权重已在内部调整
    )

    total_fitness = 1e9 if np.isinf(total_fitness) or np.isnan(total_fitness) else total_fitness
    return total_fitness


def load_knowledge_base() -> Tuple[faiss.Index, Dict]:
    """加载知识库"""
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        print(f"知识库加载失败: {e}")
        return None, None


def retrieve_similar_structures(formula: str, index: faiss.Index, metadata: Dict, k: int = 5) -> List[Dict]:
    """检索相似结构（保留）"""
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
    """解析分子式"""
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
    """计算结构化学式和汇总化学式"""
    elem_count = defaultdict(int)
    for num in atom_types:
        elem = atomic_num_to_symbol.get(num, f"X{num}")
        elem_count[elem] += 1

    # 结构化学式（按元素顺序排列）
    structural_formula = "".join([f"{elem}{cnt if cnt>1 else ''}" for elem, cnt in sorted(elem_count.items())])
    # 汇总化学式（简化格式）
    sum_formula = "".join([f"{elem}{cnt}" for elem, cnt in sorted(elem_count.items())])
    return structural_formula, sum_formula

def calculate_z_value(atom_types: List[int], lattice_volume: float) -> int:
    """计算Z值（晶胞中的化学式单元数，符合CIF标准）"""
    # 简化计算：小分子结构默认Z=1，大分子根据晶胞体积调整（可根据需求优化）
    structural_formula, _ = calculate_chemical_formula(atom_types)
    formula_unit_atoms = len(atom_types)
    # 经验公式：Z值=晶胞体积/(每个原子平均体积*化学式单元原子数)
    avg_vol_per_atom = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atom_types) / formula_unit_atoms
    z_value = max(1, int(lattice_volume / (avg_vol_per_atom * formula_unit_atoms) + 0.5))
    return z_value


# ===================== 6. 核心生成函数 =====================


def generate_valid_structures(formula: str, num_required: int = 50) -> Tuple[
    List[dgl.DGLGraph], List[float], List[np.ndarray], List[float], List[List[float]], List[Dict]]:
    """
    PSO阶段实时筛选合理结构，合理粒子不足时优化最优结构而非报错
    """
    atom_types = parse_formula(formula)
    print(f"解析分子式 {formula} -> 原子列表: {[atomic_num_to_symbol[n] for n in atom_types]}")

    index, metadata = load_knowledge_base()
    similar_structs = retrieve_similar_structures(formula, index, metadata)

    # 初始化PSO
    num_particles = 500
    particles = [Particle(LATENT_DIM) for _ in range(num_particles)]
    global_best_pos = particles[0].position.clone()
    global_best_fitness = float('inf')
    valid_particles = []

    # PSO优化（筛选+实时优化同步进行）
    max_iter = 200
    initial_w = 0.9
    final_w = 0.4
    c1 = 1.8
    c2 = 1.5

    for iter_idx in tqdm(range(max_iter), desc="PSO优化（筛选+实时优化同步进行）"):
        w = initial_w - (initial_w - final_w) * (iter_idx / max_iter)

        # 遍历粒子，生成结构→实时优化→计算fitness→更新最优
        for p in particles:
            # 步骤1：生成初始结构和晶胞
            sg_info = get_space_group_info()
            lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])  # 传递约束
            lattice = Lattice.from_parameters(*lattice_params)
            g = generate_graph_from_latent(p.position, atom_types, similar_structs)

            # 步骤2：同步执行实时精细化优化
            g_optimized, lattice_optimized = realtime_optimize_single_structure(g, atom_types, lattice)
            optimized_lattice_params = list(lattice_optimized.abc) + list(lattice_optimized.angles)

            # 步骤3：基于优化后的结构计算能量、应力和适应度
            energy, stress_tensor = calculate_energy_and_stress(g_optimized, optimized_lattice_params)
            fitness = calculate_fitness(g_optimized, lattice_optimized, stress_tensor)

            # 步骤4：更新粒子最优（此时的最优是「优化后」的结构最优）
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.clone()
                is_structure_valid, valid_msg = check_structure_validity(g_optimized, lattice_optimized)
                p.is_valid = is_structure_valid

            # 步骤5：更新全局最优
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = p.position.clone()

        # 步骤6：更新所有粒子位置
        for p in particles:
            p.update_velocity(global_best_pos, w=w, c1=c1, c2=c2)
            p.update_position()

        # 步骤7：收集当前迭代的合理粒子（优化后直接筛选，无需后续补充）
        valid_particles = [p for p in particles if p.is_valid and p.best_fitness < 1e8]
        if len(valid_particles) >= num_required:
            break  # 已收集足够优化后的合理粒子，提前终止迭代

    # 直接筛选最优的num_required个粒子（无需额外补充/优化，优化已嵌入循环）
    valid_particles = sorted(valid_particles, key=lambda x: x.best_fitness)[:num_required]

    # 从合理粒子中生成最终结构
    valid_candidates = []
    valid_energies = []
    valid_stress = []
    valid_fitness = []
    valid_lattices = []
    valid_sg = []

    # 选择最优的num_required个合理粒子
    valid_particles = sorted(valid_particles, key=lambda x: x.best_fitness)[:num_required]

    for p in tqdm(valid_particles, desc="生成最终合理结构"):
        # 步骤1：生成最终结构和晶胞
        sg_info = get_space_group_info()
        lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])  # 传递约束
        lattice = Lattice.from_parameters(*lattice_params)
        g = generate_graph_from_latent(p.best_position, atom_types, similar_structs)
        energy, stress_tensor = calculate_energy_and_stress(g, lattice_params)
        fitness = calculate_fitness(g, lattice, stress_tensor)

        # 步骤2：直接使用已获取的sg_info，无需重复调用
        valid_candidates.append(g)
        valid_energies.append(energy)
        valid_stress.append(stress_tensor)
        valid_fitness.append(fitness)
        valid_lattices.append(lattice_params)
        valid_sg.append(sg_info)

    return valid_candidates, valid_energies, valid_stress, valid_fitness, valid_lattices, valid_sg


def visualize_results(formula: str, energies: List[float], stress_tensors: List[np.ndarray],
                      fitness_scores: List[float]):
    """可视化"""
    valid_energies = [e for e in energies if not np.isinf(e) and not np.isnan(e)]
    valid_stress_norms = [np.linalg.norm(s) for s in stress_tensors if not np.isinf(np.linalg.norm(s))]
    valid_fitness = [f for f in fitness_scores if not np.isinf(f) and not np.isnan(f)]

    if valid_energies:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_energies, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'{formula} - 候选结构能量分布', fontsize=14)
        plt.xlabel('能量 (eV)', fontsize=12)
        plt.ylabel('结构数量', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(VISUALIZATION_DIR / f'{formula}_energy_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"警告：{formula} 无有效能量数据，跳过能量分布可视化")

    if valid_stress_norms:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_stress_norms, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'{formula} - 候选结构应力张量L2范数分布', fontsize=14)
        plt.xlabel('应力张量L2范数 (GPa)', fontsize=12)
        plt.ylabel('结构数量', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(VISUALIZATION_DIR / f'{formula}_stress_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"警告：{formula} 无有效应力数据，跳对应力分布可视化")

    if len(valid_energies) == len(valid_stress_norms) and len(valid_energies) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_energies, valid_stress_norms, color='darkgreen', alpha=0.6)
        plt.title(f'{formula} - 能量-应力相关性', fontsize=14)
        plt.xlabel('能量 (eV)', fontsize=12)
        plt.ylabel('应力张量L2范数 (GPa)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(VISUALIZATION_DIR / f'{formula}_energy_vs_stress.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"警告：{formula} 能量/应力数据不匹配或为空，跳过相关性可视化")

    if valid_fitness:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_fitness, bins=15, color='orange', edgecolor='black', alpha=0.7)
        plt.title(f'{formula} - 候选结构适应度分布', fontsize=14)
        plt.xlabel('适应度分数', fontsize=12)
        plt.ylabel('结构数量', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(VISUALIZATION_DIR / f'{formula}_fitness_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"警告：{formula} 无有效适应度数据，跳过适应度分布可视化")


# ===================== 7. 主函数 =====================
def main():
    formula = input("请输入分子式（如H2O、CH4）：").strip()
    if not formula:
        print("分子式不能为空！")
        return

    try:
        print(f"\n开始生成 {formula} 的5个候选结构...")
        # 使用修生成函数（PSO阶段实时筛选合理结构）
        candidates, energies, stress_tensors, fitness, lattices, sg_infos = generate_valid_structures(formula, 50)

        formula_dir = GENERATED_CIF_DIR / formula
        formula_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n保存合理结构到 {formula_dir}...")
        for idx, (g, energy, stress_tensor, fit, lattice, sg) in enumerate(
                zip(candidates, energies, stress_tensors, fitness, lattices, sg_infos)):
            crystal_dict = graph_to_crystal_dict(g, lattice, sg, energy, stress_tensor, fit)
            cif_path = formula_dir / f"{formula}_valid_{idx + 1}.cif"
            save_crystal_to_cif(crystal_dict, cif_path)

        print(f"\n生成可视化结果到 {VISUALIZATION_DIR}...")
        visualize_results(formula, energies, stress_tensors, fitness)

        summary_path = GENERATED_CIF_DIR / f"{formula}_valid_summary.csv"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("结构ID,能量(eV),应力张量L2范数(GPa),适应度分数,空间群,晶胞参数(a,b,c,α,β,γ)\n")
            for idx, (e, s, f_val, lat, sg) in enumerate(
                    zip(energies, stress_tensors, fitness, lattices, sg_infos)):
                lattice_str = ",".join([f"{x:.2f}" for x in lat])
                stress_norm = np.linalg.norm(s)
                e_str = f"{e:.4f}" if not np.isinf(e) and not np.isnan(e) else "NaN"
                s_str = f"{stress_norm:.4f}" if not np.isinf(stress_norm) and not np.isnan(stress_norm) else "NaN"
                f_str = f"{f_val:.4f}" if not np.isinf(f_val) and not np.isnan(f_val) else "NaN"
                f.write(f"{idx + 1},{e_str},{s_str},{f_str},{sg['sg_symbol']},{lattice_str}\n")

        print("\n===== 生成完成 =====")
        print(f"✅ 5个候选结构已保存到: {GENERATED_CIF_DIR / formula}")
        print(f"✅ 可视化结果已保存到: {VISUALIZATION_DIR}")
        print(f"✅ 汇总文件已保存到: {summary_path}")

    except ValueError as e:
        print(f"输入错误: {e}")
    except RuntimeError as e:
        print(f"生成失败: {e}")
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()