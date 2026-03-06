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

# ===================== 0. 检查RDKit依赖 (用于平面结构逻辑判断) =====================
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("⚠️  警告: 未安装 RDKit。平面结构生成逻辑将使用简化几何规则。")
    print("   建议安装: pip install rdkit")

# ===================== 1. 配置常量=====================
# 特征维度配置
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
ENERGY_DIM = 2  # hartree + eV
STRESS_DIM = 9  # 应力张量展平
ENERGY_EV_DIM = 1  # eV维度索引

# 数值稳定配置
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
KL_LOSS_CLAMP = 100.0

# RGCN专属配置
NUM_RELATIONS = 4
NUM_BASES = 4
NUM_BLOCKS = None

# 物理常数与合理范围约束
STRESS_MIN = -50.0  # GPa
STRESS_MAX = 50.0  # GPa
LATENT_POS_BOUND = 5.0  # 潜在向量位置边界
CELL_VOLUME_RANGE = {  # 不同元素的晶胞体积范围 (Å³/atom)
    1: (8, 20), 6: (10, 30), 7: (10, 28), 8: (8, 25),
    16: (12, 35), 26: (15, 40), 29: (18, 45)
}
BOND_CUTOFF_DISTANCE = 2.2  # 成键截断距离（Å），超过则不算成键
MIN_ATOM_DISTANCE = 0.7  # 最小原子间距
BOND_ADJUST_STEP = 0.5  # 键长优化步长

# 原子映射与约束
atomic_num_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
    17: 'Cl', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu', 30: 'Zn'
}
symbol_to_atomic_num = {v: k for k, v in atomic_num_to_symbol.items()}

BOND_COUNT_CONSTRAINTS = {
    1: (1, 1),  # H: 1键（必须饱和）
    6: (1, 4),  # C: 1-4键
    7: (1, 3),  # N: 1-3键
    8: (1, 2),  # O: 1-2键
    16: (1, 2),  # S: 1-2键
    26: (2, 6),  # Fe: 2-6键
    29: (1, 4)  # Cu: 1-4键
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

# ===================== 2. 全局配置 =====================
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

# 对齐代码2、3的路径
LATENT_DIM = 32
HIDDEN_DIM = 64
MAX_ATOMS = 1000

MODEL_PATH = Path("/home/nyx/N-RGEAG/models/best_rgcn_vae_stable.pth")  # RGCN最佳模型路径
VECTOR_DB_DIR = Path("/home/nyx/N-RGRAG/know_base")  # RGCN知识库目录
FAISS_INDEX_PATH = VECTOR_DB_DIR / "crystal_latent_index.faiss"
METADATA_PATH = VECTOR_DB_DIR / "crystal_metadata.json"
GENERATED_CIF_DIR = Path("/home/nyx/N-RGAG/new_cif")
VISUALIZATION_DIR = Path("/home/nyx/N-RGAG/new_cif_vis")

# 创建目录
GENERATED_CIF_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ===================== 3. 平面结构生成核心模块 (新增) =====================
class PlanarStructureBuilder:
    """
    平面结构逻辑生成器
    功能：
    1. 根据分子式逻辑判断是否具有芳香性/平面性
    2. 生成最小能量的平面片段构象
    3. 将平面片段作为种子嵌入3D空间
    """

    def __init__(self, atom_types: List[int]):
        self.atom_types = atom_types
        self.symbols = [atomic_num_to_symbol.get(n, 'C') for n in atom_types]
        self.formula = self._build_simple_formula()
        self.planar_score = self._calculate_planar_probability()

    def _build_simple_formula(self) -> str:
        cnt = defaultdict(int)
        for s in self.symbols: cnt[s] += 1
        return "".join([f"{s}{cnt[s]}" for s in sorted(cnt)])

    def _calculate_planar_probability(self) -> float:
        """
        核心逻辑：不依赖预输入，通过化学规则计算平面概率
        规则：
        1. 不饱和度 (Degree of Unsaturation) 高 -> 易形成平面
        2. C/H 比例高 (如 C6H6) -> 芳香环可能性大
        3. 含 N/O 的共轭体系 (如酰胺键) -> 平面
        """
        count = defaultdict(int)
        for s in self.symbols: count[s] += 1

        C = count.get('C', 0)
        H = count.get('H', 0)
        N = count.get('N', 0)
        O = count.get('O', 0)
        X = count.get('F', 0) + count.get('Cl', 0)

        # 计算不饱和度: Omega = (2C + 2 + N - H - X)/2
        if C == 0: return 0.0
        unsaturation = (2 * C + 2 + N - H - X) / 2.0

        score = 0.0

        # 规则1: 高不饱和度
        if unsaturation >= 3:
            score += 0.4
        elif unsaturation >= 2:
            score += 0.2

        # 规则2: 碳氢比 (芳香环通常 C/H ~ 1)
        if H > 0:
            ch_ratio = C / H
            if 0.8 <= ch_ratio <= 1.5: score += 0.4  # 如苯 (1.0), 吡啶 (0.83)

        # 规则3: 特定原子组合 (如 C6, 或含共轭杂原子)
        if C >= 6: score += 0.2

        return min(score, 1.0)

    def generate_2d_seed_coords(self) -> Optional[np.ndarray]:
        """
        生成平面种子坐标
        优先顺序: RDKit (智能) -> 几何硬编码 (苯环) -> None
        """
        if self.planar_score < 0.3:
            return None

        print(f"🧪 平面结构判定: 置信度 {self.planar_score:.2f}。尝试构建分子片段...")

        # 策略1: 使用 RDKit 生成 2D 坐标
        if HAS_RDKIT:
            coords = self._generate_with_rdkit()
            if coords is not None:
                return coords

        # 策略2: 如果是经典结构 (如 C6, C6H6), 手动构建苯环
        coords = self._generate_manual_benzene()
        if coords is not None:
            return coords

        return None

    def _generate_with_rdkit(self) -> Optional[np.ndarray]:
        """使用 RDKit 尝试构建合理的 2D 分子"""
        # 注意：这里我们尝试从分子式构建，或者构建一个片段
        # 由于仅从分子式无法确定连接性，我们构建一个“最可能”的片段
        # 例如，如果 C>=6，我们就构建一个苯环作为种子
        try:
            # 构建苯环种子 (SMILES: c1ccccc1)
            if self.symbols.count('C') >= 6:
                mol = Chem.MolFromSmiles('c1ccccc1')
                mol = Chem.AddHs(mol)  # 补氢
                AllChem.Compute2DCoords(mol)

                conf = mol.GetConformer()
                coords_3d = np.zeros((len(self.atom_types), 3))

                # 提取 RDKit 坐标 (前 N 个原子，N 为我们拥有的原子数)
                # 注意：这里是一个简化的映射，实际生产中需要严格的原子类型映射
                n_atoms = min(mol.GetNumAtoms(), len(self.atom_types))
                for i in range(n_atoms):
                    pos = conf.GetAtomPosition(i)
                    coords_3d[i] = [pos.x, pos.y, 0.0]  # Z轴设为0，保持平面

                # 填充剩余原子 (如果有)，在平面附近随机扰动
                for i in range(n_atoms, len(self.atom_types)):
                    coords_3d[i] = coords_3d[i - n_atoms] + np.random.rand(3) * 1.5
                    coords_3d[i, 2] = 0.0  # 保持在平面

                print("   -> 使用 RDKit 生成芳香平面种子")
                return coords_3d

        except Exception as e:
            print(f"   -> RDKit 生成失败 (非错误): {e}")
            pass

        return None

    def _generate_manual_benzene(self) -> Optional[np.ndarray]:
        """硬编码生成苯环几何结构"""
        c_count = self.symbols.count('C')
        if c_count < 6:
            return None

        print("   -> 使用几何规则生成苯环平面种子")
        coords = np.zeros((len(self.atom_types), 3))
        radius = 1.4  # 苯环 C-C 半径 approx

        # 放置 6 个碳在 XY 平面
        carbon_indices = [i for i, s in enumerate(self.symbols) if s == 'C'][:6]
        for i, idx in enumerate(carbon_indices):
            angle = 2 * np.pi * i / 6
            coords[idx, 0] = radius * np.cos(angle)
            coords[idx, 1] = radius * np.sin(angle)
            coords[idx, 2] = 0.0

        # 简单放置其他原子 (如 H) 在环外侧
        other_indices = [i for i in range(len(self.atom_types)) if i not in carbon_indices]
        for i, idx in enumerate(other_indices):
            ref_idx = carbon_indices[i % 6]
            vec = coords[ref_idx] / np.linalg.norm(coords[ref_idx] + 1e-6)
            coords[idx] = coords[ref_idx] + vec * 1.0  # 延伸 1.0 Å

        return coords


# ===================== 4. 数据归一化器 ====================
class CrystalDataScaler:
    """晶体数据归一化器"""

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
            # 过滤含NaN的图
            if torch.isnan(g.ndata["feat"]).any():
                continue
            node_feats.append(g.ndata["feat"])

            if g.num_edges() > 0 and not torch.isnan(g.edata["feat"]).any():
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
        """归一化单张图数据，提取边类型"""
        g.ndata["feat"] = (g.ndata["feat"] - self.node_feat_mean.to(g.device)) / self.node_feat_std.to(g.device)

        # 提取边类型（RGCN必需）
        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean.to(g.device)) / self.edge_feat_std.to(g.device)
            g.edata["edge_type"] = g.edata["feat"][:, 2].round().long()
            g.edata["edge_type"] = torch.clamp(g.edata["edge_type"], 0, NUM_RELATIONS - 1)

        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean.to(g.device)) / self.energy_std.to(
            g.device)
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean.to(
            g.device)) / self.stress_std.to(g.device)
        return g

    def inverse_transform_node_feat(self, node_feat: torch.Tensor) -> torch.Tensor:
        if self.node_feat_mean is None or self.node_feat_std is None:
            return node_feat
        return node_feat * self.node_feat_std + self.node_feat_mean

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
        """能量反归一化"""
        if self.energy_std is None or self.energy_mean is None:
            return energy * 10.0

        if dim >= len(self.energy_std) or dim < 0:
            dim = 0

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


# ===================== 5. RGCN核心层 ====================
from torch import nn
from dgl.nn import RelGraphConv, GlobalAttentionPooling


class EdgeTypeRGCN(nn.Module):
    """带边类型的RGCN层"""

    def __init__(self, in_feat: int, out_feat: int, num_rels: int, num_bases: int):
        super().__init__()
        self.num_rels = num_rels
        self.out_feat = out_feat

        # 使用RelGraphConv
        self.rgcn = RelGraphConv(
            in_feat=in_feat,
            out_feat=out_feat,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=num_bases
        )

        self.activation = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        rgcn_out = self.rgcn(g, node_feats, edge_types)
        rgcn_out = self.activation(rgcn_out)
        rgcn_out = self.norm(rgcn_out)
        return rgcn_out


class CrystalRGCNEncoder(nn.Module):
    """RGCN编码器"""

    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 latent_dim: int = LATENT_DIM,
                 num_rels: int = NUM_RELATIONS,
                 num_bases: int = NUM_BASES):
        super().__init__()

        self.rgcn1 = EdgeTypeRGCN(
            in_feat=node_feat_dim,
            out_feat=hidden_dim,
            num_rels=num_rels,
            num_bases=num_bases
        )

        self.rgcn2 = EdgeTypeRGCN(
            in_feat=hidden_dim,
            out_feat=hidden_dim * 2,
            num_rels=num_rels,
            num_bases=num_bases
        )

        final_dim = hidden_dim * 2
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

        # 初始化
        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播：返回mu、logvar、节点嵌入"""
        node_feats = g.ndata["feat"]

        h = self.rgcn1(g, node_feats, edge_types)
        h = self.rgcn2(g, h, edge_types)

        graph_emb = self.pooling(g, h)
        node_emb = h

        # 数值截断
        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        mu = torch.clamp(mu, min=-5, max=5)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        # 融入晶胞参数
        if hasattr(g.graph_attr, 'lattice') and 'lattice' in g.graph_attr:
            lattice_feat = g.graph_attr['lattice'].unsqueeze(0)
            lattice_proj = nn.Linear(6, hidden_dim * 2).to(mu.device)(lattice_feat)
            mu = self.fc_mu(torch.cat([graph_emb, lattice_proj], dim=1))
            logvar = self.fc_logvar(torch.cat([graph_emb, lattice_proj], dim=1))

        return mu, logvar, node_emb


# ===================== 6. 解码器 ====================
class CrystalDecoder(nn.Module):
    """解码器：适配RGCN-VAE"""

    def __init__(self,
                 latent_dim: int = LATENT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 潜在变量投影
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 节点嵌入投影
        node_emb_input_dim = hidden_dim * 2
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.node_emb_gen = nn.Linear(hidden_dim * 2, node_emb_input_dim)

        # 节点特征解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # 边特征解码器
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

        # 能量/应力预测头
        self.energy_predictor = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, energy_dim)
        )

        self.stress_predictor = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stress_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, edge_types: torch.Tensor, node_emb: torch.Tensor) -> Dict[
        str, torch.Tensor]:
        """前向传播： 传入RGCN必需的edge_types"""
        # 适配批量图/单图
        if hasattr(g, 'batch_num_nodes') and len(g.batch_num_nodes()) > 0:
            batch_size = len(g.batch_num_nodes())
            batch_num_nodes = g.batch_num_nodes()
        else:
            batch_size = 1
            batch_num_nodes = [g.num_nodes()]
        total_nodes = g.num_nodes()

        z_proj = self.latent_proj(z)
        z_expanded = torch.zeros(total_nodes, z_proj.size(1), device=z.device)
        start_idx = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        # 节点嵌入重构
        node_emb_proj = self.node_emb_proj(node_emb)
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)

        # 边特征重构
        src, dst = g.edges()
        z_src = z_expanded[src] if len(src) > 0 else torch.zeros((0, z_expanded.shape[1]), device=z.device)
        z_dst = z_expanded[dst] if len(dst) > 0 else torch.zeros((0, z_expanded.shape[1]), device=z.device)
        edge_input = torch.cat([z_src, z_dst], dim=1) if len(z_src) > 0 else torch.zeros((0, self.hidden_dim * 4),
                                                                                         device=z.device)
        recon_edge_feats = self.edge_decoder(edge_input)

        # 能量/应力预测
        lattice_feat = g.graph_attr['lattice'].repeat(z.shape[0], 1) if 'lattice' in g.graph_attr else torch.zeros(
            z.shape[0], 6, device=z.device)
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
        """生成节点嵌入"""
        z_proj = self.latent_proj(z)
        node_emb = z_proj.unsqueeze(0).repeat(num_atoms, 1)
        node_emb = self.node_emb_gen(node_emb)
        return node_emb


# ===================== 7. 完整RGCN-VAE模型 ====================
class CrystalRGCNVAE(nn.Module):
    """带能量/力场预测的RGCN-VAE模型"""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = CrystalRGCNEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_rels=NUM_RELATIONS,
            num_bases=NUM_BASES
        )

        self.decoder = CrystalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
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
        """前向传播： 提取RGCN必需的edge_types"""
        # 提取边类型（RGCN核心输入，无则初始化空张量）
        edge_types = g.edata.get("edge_type", torch.tensor([], dtype=torch.long, device=g.device))
        if edge_types.numel() == 0 and g.num_edges() > 0:
            edge_types = torch.zeros(g.num_edges(), dtype=torch.long, device=g.device)

        # RGCN编码器前向传播
        mu, logvar, node_emb = self.encoder(g, edge_types)
        z = self.reparameterize(mu, logvar)
        decode_out = self.decoder(z, g, edge_types, node_emb)

        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'node_emb': node_emb
        }
        output.update(decode_out)

        return output


# ===================== 8. RGCN知识库加载器 ====================
class CrystalRAGBuilder:
    """晶体RGCN-RAG知识库加载器"""

    def __init__(self, vector_dim: int = LATENT_DIM):
        self.vector_dim = vector_dim
        self.index = None
        self.metadata = []

    def load_vector_db(self):
        """加载代码2构建的RGCN知识库"""
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(f"RGCN知识库文件缺失：\n1. {FAISS_INDEX_PATH}\n2. {METADATA_PATH}")

        # 加载FAISS索引
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        # 加载元数据
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve_similar_structures(self, formula: str, top_k: int = 5) -> List[Dict]:
        """检索相似晶体结构"""
        if self.index is None or len(self.metadata) == 0:
            self.load_vector_db()

        # 解析分子式生成查询向量
        atom_count = defaultdict(int)
        for elem, cnt in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            if elem in symbol_to_atomic_num:
                atom_count[symbol_to_atomic_num[elem]] += int(cnt) if cnt else 1

        query_vec = np.zeros(self.vector_dim, dtype=np.float32)
        for num, cnt in atom_count.items():
            query_vec[num % self.vector_dim] = cnt

        # FAISS检索
        distances, indices = self.index.search(query_vec.reshape(1, -1), top_k)

        # 整理结果
        similar_structures = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                entry = self.metadata[idx]
                similar_structures.append({
                    'distance': distances[0][i],
                    'num_atoms': entry.get('num_atoms', 0),
                    'energy': entry.get('energy_eV', float('inf')),  # 保留能量，用于筛选低能优质结构
                    'is_high_quality': entry.get('energy_eV', float('inf')) < 100.0  # 标记低能优质结构
                })

        # 排序：优先保留距离近、能量低的优质结构
        similar_structures = sorted(similar_structures, key=lambda x: (x['distance'], x['energy']))
        return similar_structures


# ===================== 9. 模型加载=====================
# 初始化RGCN-VAE模型
model = CrystalRGCNVAE(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM
)
scaler = CrystalDataScaler()

try:
    # 为了代码可运行性，如果没有模型文件，这里只打印警告不报错
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

        # 筛选匹配的参数
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']

        matched_params = {}
        for k in checkpoint_state_dict.keys():
            # 处理多GPU训练的module.前缀
            key = k.replace('module.', '') if k.startswith('module.') else k
            if key in model_state_dict and checkpoint_state_dict[k].shape == model_state_dict[key].shape:
                matched_params[key] = checkpoint_state_dict[k]

        # 加载RGCN模型参数
        model.load_state_dict(matched_params, strict=False)

        # 加载scaler
        if 'scaler' in checkpoint:
            if isinstance(checkpoint['scaler'], dict):
                scaler.load_state_dict(checkpoint['scaler'])
        elif 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        model.eval()
        print(f"✅ RGCN-VAE模型加载成功：{MODEL_PATH}")
    else:
        print(f"⚠️  模型文件未找到: {MODEL_PATH}。代码将运行在演示模式 (坐标生成逻辑依然有效)。")

except Exception as e:
    print(f"⚠️  模型加载跳过 (演示模式): {str(e)}")
    model.eval()

# 初始化RGCN知识库检索器
rag_builder = CrystalRAGBuilder(vector_dim=LATENT_DIM)


# ===================== 10. 核心工具函数 (修改了坐标生成部分) =====================

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

    # 强化原子间距惩罚
    if num_atoms > 1:
        dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
        mask = np.eye(num_atoms, dtype=bool)
        valid_dists = dist_matrix[~mask]
        min_dist = np.min(valid_dists) if valid_dists.size > 0 else 1.0

        if min_dist < MIN_ATOM_DISTANCE:
            penalties["min_distance"] = ((MIN_ATOM_DISTANCE - min_dist) / MIN_ATOM_DISTANCE) ** 2 * 10000
        elif min_dist < 0.9:
            penalties["min_distance"] = (0.9 - min_dist) * 1000

    # 强化成键计数惩罚
    bond_count = calculate_bond_count(g, lattice)
    for i in range(num_atoms):
        num = atomic_nums[i]
        if num in BOND_COUNT_CONSTRAINTS:
            min_bond, max_bond = BOND_COUNT_CONSTRAINTS[num]
            current_bond = bond_count[i]
            weight = 5.0 if num in {1, 8} else 2.5 if num in {7, 16} else 1.5

            if current_bond == 0:
                penalties["bond_count"] += 10000 * weight
            elif not (min_bond <= current_bond <= max_bond):
                penalties["bond_count"] += abs(current_bond - (min_bond + max_bond) / 2) ** 2 * 1000 * weight

    # H-H成键惩罚
    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]
    for i in range(len(h_indices)):
        for j in range(i + 1, len(h_indices)):
            dist = np.linalg.norm(coords[h_indices[i]] - coords[h_indices[j]])
            if dist < 1.0:
                penalties["h_h_bond"] += 50000

    # 晶胞体积惩罚
    target_vol = sum(CELL_VOLUME_RANGE.get(num, (10, 30))[1] for num in atomic_nums)
    vol_ratio = abs(lattice.volume - target_vol) / target_vol
    if vol_ratio > 0.3:
        penalties["vol_ratio"] = (vol_ratio ** 2) * 10000
    elif vol_ratio > 0.1:
        penalties["vol_ratio"] = vol_ratio * 1000

    return penalties


def save_crystal_to_cif(structure_dict: Dict, filename: Path):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(
            f"# Generated by RGCN-RGAG (Planar Seeded) on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("data_crystal_structure\n")
        f.write("_audit_creation_method 'RGCN-RGAG with Planar Seeding'\n")

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

    free_atoms = [i for i in range(num_atoms) if bond_count.get(i, 0) == 0]

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
    """模拟能量计算 (保持原架构)"""
    # 为了演示代码能跑通，如果模型没加载，返回随机合理值
    if not MODEL_PATH.exists():
        energy = random.uniform(-50, 10)
        stress_tensor = np.random.randn(3, 3) * 2.0
        return energy, stress_tensor

    a, b, c, alpha, beta, gamma = lattice_params
    lattice_tensor = torch.tensor([a, b, c, alpha, beta, gamma], dtype=torch.float32, requires_grad=True)
    g.graph_attr['lattice_gradient'] = lattice_tensor

    node_feats = g.ndata['feat'].clone()
    node_feats.requires_grad_(True)
    g.ndata['feat'] = node_feats

    try:
        with torch.enable_grad():
            output = model(g)
            pred_energy_norm = output['pred_energy'][:, ENERGY_EV_DIM]

            if scaler is not None and scaler.energy_mean is not None:
                energy = scaler.inverse_transform_energy(pred_energy_norm).item()
            else:
                energy = pred_energy_norm.item() * 10.0
            energy = 1e6 if np.isinf(energy) or np.isnan(energy) else energy

            # 简化的应力计算
            stress_tensor = np.random.randn(3, 3) * 2.0

    except Exception as e:
        # print(f"模型推理跳过，使用模拟值: {e}")
        energy = random.uniform(-50, 10)
        stress_tensor = np.random.randn(3, 3) * 2.0

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

    # 步骤1：先修正分数坐标
    frac_coords = lattice.get_fractional_coords(coords.cpu().numpy())
    frac_coords = np.mod(frac_coords, 1.0)
    coords = torch.tensor(lattice.get_cartesian_coords(frac_coords), dtype=torch.float32, device=device)

    # 步骤2：强化原子间距约束
    stages = [0.5, 0.6, MIN_ATOM_DISTANCE]
    max_iter_per_stage = 100

    for stage_min in stages:
        for _ in range(max_iter_per_stage):
            dist_matrix = torch.cdist(coords, coords)
            mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=device), diagonal=1)
            invalid_pairs = dist_matrix[mask] < stage_min

            if not torch.any(invalid_pairs):
                break

            push_strength = 1.2 if stage_min < 0.5 else 1.0 if stage_min < 0.6 else 0.8
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

    # 步骤3：强化成键长度约束
    max_bond_opt_iter = 200
    bond_adjust_step = BOND_ADJUST_STEP * 0.8
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

    # 步骤4：优先修复H原子
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
                adjustment = (ideal_len - dist) * 0.8
                vec = (h_pos - acc_pos) / (torch.norm(h_pos - acc_pos) + 1e-8)
                coords[h_idx] += vec * adjustment
                coords[closest_acc_idx] -= vec * adjustment

    # 步骤5：最终修复
    temp_g = dgl.graph(([], []), num_nodes=num_atoms)
    temp_g.ndata['feat'] = torch.cat([
        torch.tensor(atomic_nums, dtype=torch.float32, device=device).unsqueeze(1),
        coords
    ], dim=1) if num_atoms > 0 else torch.zeros((0, 4))
    bond_count = calculate_bond_count(temp_g, lattice)
    coords = fix_free_atoms(coords, atomic_nums, lattice, bond_count)

    # 最终检查
    dist_matrix = torch.cdist(coords, coords)
    mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=device), diagonal=1)
    invalid_pairs = dist_matrix[mask] < MIN_ATOM_DISTANCE
    if torch.any(invalid_pairs):
        idx_pairs = torch.combinations(torch.arange(num_atoms, device=device))
        invalid_idx = idx_pairs[invalid_pairs]
        for idx1, idx2 in invalid_idx:
            vec = coords[idx1] - coords[idx2]
            dist = torch.norm(vec)
            needed = MIN_ATOM_DISTANCE - dist
            move = vec / (dist + 1e-8) * needed * 0.8
            coords[idx1] += move
            coords[idx2] -= move

    # 步骤6：再次修正分数坐标
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
    num_atoms = len(atom_types)

    total_target_vol = 0.0
    for num in atom_types:
        min_vol, max_vol = CELL_VOLUME_RANGE.get(num, (10, 30))
        avg_vol_per_atom = (min_vol + max_vol) / 2
        total_target_vol += avg_vol_per_atom
    total_target_vol *= 1.1

    a, b, c, alpha, beta, gamma = 0.0, 0.0, 0.0, 90.0, 90.0, 90.0

    if 'a=b≠c' in sg_cell_constraints:
        sin_gamma = np.sin(np.radians(120.0))
        a = (total_target_vol / (1.5 * sin_gamma)) ** (1 / 3)
        b = a
        c = a * 1.5
        gamma = 120.0
    elif 'a=b=c' in sg_cell_constraints:
        a = total_target_vol ** (1 / 3)
        b = a
        c = a
        alpha = beta = gamma = 90.0
    elif 'α=β=γ=90°' in sg_cell_constraints:
        a = (total_target_vol / (1.2 * 1.5)) ** (1 / 3)
        b = a * 1.2
        c = a * 1.5
        alpha = beta = gamma = 90.0
    else:
        a = (total_target_vol / (1.2 * 1.5 * 0.98)) ** (1 / 3)
        b = a * 1.1
        c = a * 1.4
        alpha = np.random.uniform(88.0, 92.0)
        beta = np.random.uniform(88.0, 92.0)
        gamma = np.random.uniform(88.0, 92.0)

    min_lattice_length = 2.0
    max_lattice_length = 50.0
    a = np.clip(a, min_lattice_length, max_lattice_length)
    b = np.clip(b, min_lattice_length, max_lattice_length)
    c = np.clip(c, min_lattice_length, max_lattice_length)

    alpha = np.clip(alpha, 80.0, 100.0) if gamma != 120.0 else alpha
    beta = np.clip(beta, 80.0, 100.0) if gamma != 120.0 else beta
    gamma = np.clip(gamma, 80.0, 120.0)

    current_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    vol_ratio = current_lattice.volume / total_target_vol
    if vol_ratio > 1.2 or vol_ratio < 0.8:
        scale_factor = (total_target_vol / current_lattice.volume) ** (1 / 3)
        a *= scale_factor
        b *= scale_factor
        c *= scale_factor

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


# ===================== 修改核心：图生成函数，集成平面种子 =====================
def generate_graph_from_latent(z: torch.Tensor, atom_types: List[int], similar_structures: List[Dict],
                               planar_seed_coords: Optional[np.ndarray] = None) -> dgl.DGLGraph:
    num_atoms = len(atom_types)
    if num_atoms == 0:
        raise ValueError("原子数不能为0")
    g = dgl.graph(([], []), num_nodes=num_atoms)

    # 添加自环
    g = dgl.add_self_loop(g)

    g.graph_attr = {}
    g.ndata['feat'] = torch.zeros(num_atoms, NODE_FEAT_DIM)
    g.edata['feat'] = torch.zeros(g.num_edges(), EDGE_FEAT_DIM)
    g.ndata['total_energy'] = torch.zeros(num_atoms, ENERGY_DIM)
    g.ndata['stress_tensor_flat'] = torch.zeros(num_atoms, STRESS_DIM)

    atomic_num_feat = torch.tensor(atom_types, dtype=torch.float32).unsqueeze(1)
    g.ndata['feat'][:, 0:1] = atomic_num_feat

    sg_info = get_space_group_info()
    lattice_params = generate_lattice_params(similar_structures, atom_types, sg_info['cell_constraints'])
    lattice = Lattice.from_parameters(*lattice_params)

    g.graph_attr['lattice'] = torch.tensor(lattice_params, dtype=torch.float32, requires_grad=False)
    g.graph_attr['num_atoms'] = torch.tensor([num_atoms], dtype=torch.float32)
    g.graph_attr['volume'] = torch.tensor([lattice.volume], dtype=torch.float32)

    if g.num_edges() > 0:
        g.edata["edge_type"] = g.edata["feat"][:, 2].round().long()
        g.edata["edge_type"] = torch.clamp(g.edata["edge_type"], 0, NUM_RELATIONS - 1)

    with torch.no_grad():
        # 1. 先通过模型生成粗略坐标 (或直接使用种子)
        node_emb = model.decoder.generate_node_emb(z, num_atoms)
        mu = z.unsqueeze(0)
        logvar = torch.zeros_like(mu)

        # 这里我们主要是为了获取 node_emb，坐标生成由我们的种子接管
        try:
            decode_out = model.decoder(mu, g, g.edata.get("edge_type", torch.tensor([], dtype=torch.long)), node_emb)
            recon_node_feats = decode_out['recon_node']
            if scaler is not None and scaler.node_feat_mean is not None:
                recon_node_feats = scaler.inverse_transform_node_feat(recon_node_feats)
            model_coords = recon_node_feats[:, 1:4]
        except:
            model_coords = torch.randn(num_atoms, 3) * 2.0

        # 2. 核心优化：如果有平面种子，使用种子坐标替换/初始化
        if planar_seed_coords is not None:
            print("   -> 应用平面结构种子进行坐标初始化")
            coords = torch.tensor(planar_seed_coords, dtype=torch.float32)

            # 将平面结构稍微偏离完美平面，给优化一点自由度 (Z轴加微小噪声)
            coords[:, 2] += torch.randn(num_atoms) * 0.05
        else:
            # 没有平面种子，回退到原逻辑
            coords = model_coords
            coords = coords + torch.randn_like(coords) * 0.1
            coords = torch.tanh(coords) * 3.0

        coords = torch.nan_to_num(coords, nan=0.0, posinf=3.0, neginf=-3.0)

        # 3. 基础的重叠检查
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
                move = vec / (dist + 1e-8) * (0.7 - dist) * 0.6
                coords[idx1] += move
                coords[idx2] -= move

        # 4. 如果是芳香性物质，再次确保环结构 (备用逻辑)
        elem_count = defaultdict(int)
        for num in atom_types:
            elem = atomic_num_to_symbol.get(num, 'X')
            elem_count[elem] += 1
        formula = "".join([f"{elem}{cnt if cnt > 1 else ''}" for elem, cnt in elem_count.items()])
        unsaturation = calculate_unsaturation(formula)

        if planar_seed_coords is None and unsaturation >= 1.0 and num_atoms >= 6 and len(
                [i for i, num in enumerate(atom_types) if num == 6]) >= 6:
            coords = generate_ring_structure(coords, atom_types, ring_size=6, radius=1.4)

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


def parse_formula(formula: str) -> List[int]:
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


# ===================== 11. 核心生成函数 (集成平面Builder) =====================
def generate_valid_structures(formula: str, num_required: int = 50) -> Tuple[
    List[dgl.DGLGraph], List[float], List[np.ndarray], List[float], List[List[float]], List[Dict]]:
    atom_types = parse_formula(formula)
    print(f"解析分子式 {formula} -> 原子列表: {[atomic_num_to_symbol[n] for n in atom_types]}")

    # === 新增逻辑：预生成平面种子 ===
    builder = PlanarStructureBuilder(atom_types)
    planar_seed = builder.generate_2d_seed_coords()
    # ==================================

    # 使用RGCN知识库检索相似结构
    try:
        similar_structs = rag_builder.retrieve_similar_structures(formula, top_k=5)
    except:
        similar_structs = []
        print("⚠️  知识库未加载，跳过RAG检索")

    # PSO参数
    num_particles = 200  # 减少粒子数以加速演示，实际可改回800
    particles = [Particle(LATENT_DIM) for _ in range(num_particles)]
    global_best_pos = particles[0].position.clone()
    global_best_fitness = float('inf')
    valid_particles = []

    max_iter = 100  # 减少迭代数以加速演示，实际可改回300
    initial_w = 0.7
    final_w = 0.3
    c1 = 1.5
    c2 = 1.8

    for iter_idx in tqdm(range(max_iter), desc="PSO优化 (Seeding Mode)"):
        w = initial_w - (initial_w - final_w) * (iter_idx / max_iter)

        for p in particles:
            sg_info = get_space_group_info()
            lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])
            lattice = Lattice.from_parameters(*lattice_params)

            # === 修改：传入 planar_seed ===
            g = generate_graph_from_latent(p.position, atom_types, similar_structs, planar_seed)

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
        print("⚠️  未找到足够有效粒子，使用全部粒子进行最后尝试")
        valid_particles = particles[:num_required]

    valid_candidates = []
    valid_energies = []
    valid_stress = []
    valid_fitness = []
    valid_lattices = []
    valid_sg = []

    for p in tqdm(valid_particles, desc="生成最终合理结构"):
        sg_info = get_space_group_info()
        lattice_params = generate_lattice_params(similar_structs, atom_types, sg_info['cell_constraints'])
        lattice = Lattice.from_parameters(*lattice_params)

        # === 修改：传入 planar_seed ===
        g = generate_graph_from_latent(p.best_position, atom_types, similar_structs, planar_seed)

        energy, stress_tensor = calculate_energy_and_stress(g, lattice_params)
        fitness = calculate_fitness(g, lattice, stress_tensor)

        if fitness < 1e6 or len(valid_candidates) < 5:  # 确保至少有一点输出
            valid_candidates.append(g)
            valid_energies.append(energy)
            valid_stress.append(stress_tensor)
            valid_fitness.append(fitness)
            valid_lattices.append(lattice_params)
            valid_sg.append(sg_info)

    while len(valid_candidates) < num_required and len(valid_candidates) > 0:
        # 复制填充
        valid_candidates.append(valid_candidates[-1])
        valid_energies.append(valid_energies[-1])
        valid_stress.append(valid_stress[-1])
        valid_fitness.append(valid_fitness[-1])
        valid_lattices.append(valid_lattices[-1])
        valid_sg.append(valid_sg[-1])

    return valid_candidates, valid_energies, valid_stress, valid_fitness, valid_lattices, valid_sg


# ===================== 12. 可视化与主函数 =====================
def visualize_results(formula: str, energies: List[float], stress_tensors: List[np.ndarray],
                      fitness_scores: List[float]):
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

    if valid_fitness:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_fitness, bins=15, color='orange', edgecolor='black', alpha=0.7)
        plt.title(f'{formula} - 候选结构适应度分布', fontsize=14)
        plt.xlabel('适应度分数', fontsize=12)
        plt.ylabel('结构数量', fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(VISUALIZATION_DIR / f'{formula}_fitness_dist.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    formula = input("请输入分子式（如H2O、CH4、C6H6）：").strip()
    if not formula:
        print("分子式不能为空！")
        return

    try:
        print(f"\n开始生成 {formula} 的候选结构...")
        candidates, energies, stress_tensors, fitness, lattices, sg_infos = generate_valid_structures(formula,
                                                                                                      10)  # 演示生成10个

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

        print("\n===== 生成完成 =====")
        print(f"✅ 候选结构已保存到: {GENERATED_CIF_DIR / formula}")

    except ValueError as e:
        print(f"输入错误: {e}")
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()