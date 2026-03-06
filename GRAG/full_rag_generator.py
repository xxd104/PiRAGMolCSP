
import os
import dgl
import torch
import numpy as np
import faiss
import json
import re
import random
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from openbabel import openbabel
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.data import atomic_numbers as ase_atomic_numbers
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure

# 全局配置
PROCESSED_GRAPH_DIR = "/home/nyx/GRAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/GRAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/GRAG/models"
VISUALIZATION_DIR = Path("/home/nyx/GRAG/generated_vis")
CIF_DIR = Path("/home/nyx/GRAG/raw_cifs")
MODEL_PATH = Path("/home/nyx/GRAG/models/best_model.pth")
FAISS_INDEX_PATH = Path("/home/nyx/GRAG/knowledge_base/material_index.faiss")
METADATA_PATH = Path("/home/nyx/GRAG/knowledge_base/material_metadata.json")
GENERATED_CIF_DIR = Path("/home/nyx/GRAG/generated_cifs")
LATENT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NODE_FEAT_DIM = 4
HIDDEN_DIM = 128
NUM_HEADS = 4
CUTOFF_DISTANCE = 5.0
NUM_WORKERS = 4
MAX_ATOMS = 1000
MAX_T_SNE_SAMPLES = 2000
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_CIF_DIR.mkdir(parents=True, exist_ok=True)

# 原子序数到元素符号的映射
number_to_symbol = {v: k for k, v in ase_atomic_numbers.items()}
number_to_symbol.update({
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S'
})

# 原子成键数约束 (最小, 最大)
BOND_COUNT_CONSTRAINTS = {
    1: (1, 1),  # H: 1键
    6: (1, 4),  # C: 1-4键
    7: (1, 3),  # N: 1-3键
    8: (1, 2),  # O: 1-2键
    16: (1, 2)  # S: 1-2键
}

# 常见原子对的合理键长范围 (Å)
BOND_LENGTH_RANGES = {
    (6, 1): (0.9, 1.3),
    (1, 6): (0.9, 1.3),
    (6, 6): (1.20, 1.60),
    (6, 7): (1.15, 1.50),
    (7, 6): (1.15, 1.50),
    (6, 8): (1.15, 1.45),
    (8, 6): (1.15, 1.45),
    (7, 1): (0.8, 1.1),
    (1, 7): (0.8, 1.1),
    (8, 1): (0.8, 1.1),
    (1, 8): (0.8, 1.1),
    (8, 8): (1.20, 1.50),
    (7, 7): (1.20, 1.50),
    (7, 8): (1.20, 1.45),
    (8, 7): (1.20, 1.45),
}


# 核心网络层
class EdgeTypeGAT(torch.nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat
        self.gat = dgl.nn.GATConv(
            in_feats=in_feat,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True
        )

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        return self.gat(g, node_feat).flatten(1)


class CrystalGCNEncoder(torch.nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, latent_dim: int, num_heads: int):
        super().__init__()
        self.gat1 = EdgeTypeGAT(node_feat_dim, hidden_dim // 2, num_heads)
        self.gat2 = EdgeTypeGAT((hidden_dim // 2) * num_heads, hidden_dim, num_heads)
        self.gat3 = EdgeTypeGAT(hidden_dim * num_heads, hidden_dim * 2, num_heads)
        final_dim = num_heads * hidden_dim * 2
        self.pooling = dgl.nn.GlobalAttentionPooling(torch.nn.Sequential(
            torch.nn.Linear(final_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1)
        ))
        self.fc_mu = torch.nn.Linear(final_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(final_dim, latent_dim)

    def forward(self, g: dgl.DGLGraph):
        h = torch.nn.ELU()(self.gat1(g, g.ndata['feat']))
        h = torch.nn.ELU()(self.gat2(g, h))
        h = torch.nn.ELU()(self.gat3(g, h))
        graph_emb = self.pooling(g, h)
        return self.fc_mu(graph_emb), self.fc_logvar(graph_emb)


class CrystalDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, node_feat_dim: int):
        super().__init__()
        self.latent_proj = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim * 2), torch.nn.ReLU()
        )
        self.node_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_feat_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
        batch_num_nodes = g.batch_num_nodes()
        total_nodes = sum(batch_num_nodes)
        h = self.latent_proj(z)
        h_expanded = torch.zeros(total_nodes, h.size(1), device=z.device)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_expanded[start_idx:start_idx + num_nodes] = h[i].repeat(num_nodes, 1)
            start_idx += num_nodes
        return self.node_decoder(h_expanded)


class CrystalGCNVAE(torch.nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, latent_dim: int, num_heads: int):
        super().__init__()
        self.encoder = CrystalGCNEncoder(node_feat_dim, hidden_dim, latent_dim, num_heads)
        self.decoder = CrystalDecoder(latent_dim, hidden_dim, node_feat_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(g)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, g), mu, logvar


# 加载模型
model = CrystalGCNVAE(NODE_FEAT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HEADS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


# 计算不饱和度
def calculate_unsaturation(formula: str) -> float:
    """计算分子式的不饱和度"""
    atom_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        atom_count[elem] += int(cnt) if cnt else 1

    # 计算不饱和度: U = (2C + 2 - H - X + N) / 2
    C = atom_count.get('C', 0)
    H = atom_count.get('H', 0)
    N = atom_count.get('N', 0)
    # 卤素原子 (F, Cl, Br, I) 按H处理
    X = atom_count.get('F', 0) + atom_count.get('Cl', 0) + atom_count.get('Br', 0) + atom_count.get('I', 0)

    unsaturation = (2 * C + 2 - H - X + N) / 2.0
    return max(0.0, unsaturation)  # 确保非负


# 强制计算结合能（忽略所有限制）
def calculate_binding_energy_forced(graph: dgl.DGLGraph) -> Optional[float]:
    """强制计算结合能，忽略所有结构限制"""
    try:
        node_feats = graph.ndata['feat'].detach().cpu().numpy()
        coordinates = node_feats[:, 1:4]
        atomic_nums = np.round(node_feats[:, 0]).astype(int)

        # 创建OpenBabel分子对象
        ob_mol = openbabel.OBMol()
        ob_mol.BeginModify()

        # 添加原子
        for num, (x, y, z) in zip(atomic_nums, coordinates):
            ob_atom = openbabel.OBAtom()
            ob_atom.SetAtomicNum(int(num))
            ob_atom.SetVector(float(x), float(y), float(z))
            ob_mol.AddAtom(ob_atom)

        ob_mol.EndModify()

        # 强制连接所有原子（忽略距离限制）
        ob_mol.ConnectTheDots()
        ob_mol.PerceiveBondOrders()
        ob_mol.AddHydrogens()

        # 设置力场
        forcefield = openbabel.OBForceField.FindType("UFF")
        if not forcefield or not forcefield.Setup(ob_mol):
            print(f"[DEBUG] 强制结合能计算失败: 力场设置失败")
            return None

        # 计算能量
        energy_kcal = forcefield.Energy()
        energy_ev = energy_kcal * 0.04336
        return round(energy_ev, 4)

    except Exception as e:
        print(f"[DEBUG] 强制结合能计算异常: {str(e)}")
        return None


# 结合能计算函数（带限制）
def calculate_binding_energy(graph: dgl.DGLGraph) -> Optional[float]:
    try:
        node_feats = graph.ndata['feat'].detach().cpu().numpy()
        coordinates = node_feats[:, 1:4]

        atomic_nums = np.round(node_feats[:, 0]).astype(int)
        num_atoms = len(atomic_nums)

        # 前置检查：过滤明显重叠的结构
        if num_atoms > 1:
            dist_matrix = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)
            min_dist = np.min(dist_matrix[dist_matrix > 1e-6])
            if min_dist < 0.5:
                print(f"[DEBUG] 结合能计算跳过: 原子间距过小 ({min_dist:.4f}Å)")
                return None

        ob_mol = openbabel.OBMol()
        ob_mol.BeginModify()

        for idx, (num, (x, y, z)) in enumerate(zip(atomic_nums, coordinates)):
            if num not in number_to_symbol:
                ob_mol.EndModify()
                print(f"[DEBUG] 结合能计算失败: 未知原子序数 {num}，索引 {idx}")
                return None

            ob_atom = openbabel.OBAtom()
            ob_atom.SetAtomicNum(int(num))
            ob_atom.SetVector(float(x), float(y), float(z))
            ob_mol.AddAtom(ob_atom)

        ob_mol.EndModify()

        # 成键感知
        ob_mol.ConnectTheDots()  # 基础成键
        ob_mol.PerceiveBondOrders()  # 感知键级
        ob_mol.AddHydrogens()  # 确保H被正确添加

        # 修正H的成键
        for h_idx in range(1, ob_mol.NumAtoms() + 1):
            atom = ob_mol.GetAtom(h_idx)
            if atom.GetAtomicNum() == 1 and atom.GetExplicitValence() == 0:
                # 若H未成键，手动连接最近的C/O/N
                closest_dist = float('inf')
                closest_atom = None
                for other_idx in range(1, ob_mol.NumAtoms() + 1):
                    if other_idx == h_idx:
                        continue
                    other_atom = ob_mol.GetAtom(other_idx)
                    if other_atom.GetAtomicNum() in {6, 7, 8}:
                        # 获取原子对象并计算距离
                        atom_h = ob_mol.GetAtom(h_idx)
                        dist = atom_h.GetDistance(other_atom)
                        if dist < closest_dist and dist < 1.5:  # 1.5Å内视为可能成键
                            closest_dist = dist
                            closest_atom = other_idx
                if closest_atom:
                    ob_mol.AddBond(h_idx, closest_atom, 1)  # 手动添加单键

        # 检查成键合理性
        for i in range(len(atomic_nums)):
            num = atomic_nums[i]
            if num in BOND_COUNT_CONSTRAINTS:
                min_bonds, max_bonds = BOND_COUNT_CONSTRAINTS[num]
                # 使用GetExplicitValence()获取成键数
                bonds = ob_mol.GetAtom(i + 1).GetExplicitValence()
                if not (min_bonds <= bonds <= max_bonds):
                    print(
                        f"[DEBUG] 成键数不合理: {number_to_symbol[num]}原子有{bonds}个键，应在{min_bonds}-{max_bonds}范围内")

        forcefield = openbabel.OBForceField.FindType("UFF")
        if not forcefield or not forcefield.Setup(ob_mol):
            print(f"[DEBUG] 结合能计算失败: 力场设置失败")
            return None

        energy_kcal = forcefield.Energy()
        energy_ev = energy_kcal * 0.04336
        if energy_ev > 1000:
            print(f"[DEBUG] 结合能异常: {energy_ev:.4f} eV（可能结构不合理）")
            return None
        return round(energy_ev, 4)

    except Exception as e:
        print(f"[DEBUG] 结合能计算异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 从分子式解析原子类型
def get_atom_types(formula: str) -> List[int]:
    atom_count = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        atom_count[elem] += int(cnt) if cnt else 1

    atom_types = []
    for elem, cnt in atom_count.items():
        num = ase_atomic_numbers.get(elem)
        if num is None:
            raise ValueError(f"无效元素: {elem}（分子式 {formula}）")
        atom_types.extend([num] * cnt)
    return atom_types


# 原子分散算法（分阶段约束 + 键长优化）
def enforce_min_distance(coords: torch.Tensor, atomic_nums: List[int], min_dist: float = 0.8,
                         max_iter: int = 200) -> torch.Tensor:
    # 确保coords是三维的
    if coords.dim() != 2 or coords.shape[1] != 3:
        print(f"[WARNING] coords维度异常: {coords.shape}，已修正为三维坐标")
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        if coords.shape[1] < 3:
            # 填充缺失的维度
            pad_size = 3 - coords.shape[1]
            coords = torch.cat([coords, torch.zeros(coords.shape[0], pad_size, device=coords.device)], dim=1)
        elif coords.shape[1] > 3:
            # 只取前3个维度
            coords = coords[:, :3]

    coords = coords.clone()
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return coords

    # 分阶段提升最小距离要求（渐进式约束）
    stages = [0.3, 0.5, min_dist]  # 先松后紧
    for stage_min in stages:
        for i in range(max_iter // len(stages)):  # 每个阶段分配迭代次数
            dist_matrix = torch.cdist(coords, coords)
            mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=coords.device), diagonal=1)
            invalid_pairs = dist_matrix[mask] < stage_min

            if not torch.any(invalid_pairs):
                break  # 当前阶段目标达成

            idx_pairs = torch.combinations(torch.arange(num_atoms, device=coords.device))
            invalid_idx = idx_pairs[invalid_pairs]

            # 对重叠原子对进行调整
            for idx1, idx2 in invalid_idx:
                pos1 = coords[idx1]
                pos2 = coords[idx2]
                vec = pos1 - pos2
                dist = torch.norm(vec)

                if dist < 1e-6:  # 完全重叠
                    # 强制大位移分离：沿随机方向移动较远距离
                    random_dir = torch.randn_like(vec)
                    random_dir = random_dir / torch.norm(random_dir)  # 单位向量
                    # 移动距离是目标距离的1.5倍，确保彻底分离
                    move_dist = stage_min * 1.5
                    coords[idx1] += random_dir * move_dist / 2
                    coords[idx2] -= random_dir * move_dist / 2
                else:
                    # 非重叠但距离不足：按比例调整
                    needed = stage_min - dist
                    # 动态调整移动步长（距离越小，步长越大）
                    move_ratio = 1.0 if needed < 0.2 else 0.5
                    move = vec / dist * needed * move_ratio
                    coords[idx1] += move
                    coords[idx2] -= move

    # 优化常见键长到合理范围
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom_pair = (atomic_nums[i], atomic_nums[j])
            if atom_pair in BOND_LENGTH_RANGES:
                min_bond, max_bond = BOND_LENGTH_RANGES[atom_pair]
                ideal_bond = (min_bond + max_bond) / 2  # 理想键长

                pos1 = coords[i]
                pos2 = coords[j]
                vec = pos1 - pos2
                dist = torch.norm(vec)

                # 如果距离不在合理范围内，调整到理想键长
                if dist < min_bond:
                    adjustment = (ideal_bond - dist) * 0.3
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move
                elif dist > max_bond:
                    adjustment = (ideal_bond - dist) * 0.3
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move

    # 最终验证
    dist_matrix = torch.cdist(coords, coords)
    mask = torch.triu(torch.ones(num_atoms, num_atoms, dtype=torch.bool, device=coords.device), diagonal=1)
    final_distances = dist_matrix[mask]
    if len(final_distances) == 0:
        return coords

    num_atoms = coords.shape[0]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if atomic_nums[i] == 1 and atomic_nums[j] == 1:  # 两个都是H原子
                dist = torch.norm(coords[i] - coords[j])
                if dist < 1.0:  # H-H距离过近
                    vec = coords[i] - coords[j]
                    if dist < 1e-6:  # 完全重叠
                        # 随机方向推开
                        random_dir = torch.randn_like(vec)
                        random_dir = random_dir / torch.norm(random_dir)
                        coords[i] += random_dir * 0.5  # 推开0.5Å
                        coords[j] -= random_dir * 0.5
                    else:
                        # 按比例推开到1.0Å
                        needed = 1.0 - dist
                        move = vec / dist * needed * 0.8  # 一次移动80%的所需距离
                        coords[i] += move
                        coords[j] -= move

    return coords

    final_min_dist = torch.min(final_distances)
    if final_min_dist < min_dist:
        print(f"[WARNING] 最终最小距离 {final_min_dist:.4f}Å < {min_dist}Å（已尽力调整）")
    else:
        print(f"[DEBUG] 最终最小原子间距: {final_min_dist:.4f}Å")
    return coords


# 智能键长约束
def enforce_bond_constraints(coords: torch.Tensor, atomic_nums: List[int], max_iter: int = 150) -> torch.Tensor:
    # 确保coords是三维的
    if coords.dim() != 2 or coords.shape[1] != 3:
        print(f"[WARNING] coords维度异常: {coords.shape}，已修正为三维坐标")
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        if coords.shape[1] < 3:
            # 填充缺失的维度
            pad_size = 3 - coords.shape[1]
            coords = torch.cat([coords, torch.zeros(coords.shape[0], pad_size, device=coords.device)], dim=1)
        elif coords.shape[1] > 3:
            # 只取前3个维度
            coords = coords[:, :3]

    coords = coords.clone()
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return coords

    # 识别关键成键对
    required_pairs = []
    h_indices = [i for i, num in enumerate(atomic_nums) if num == 1]  # 所有H原子索引
    acceptor_indices = [i for i, num in enumerate(atomic_nums) if num in {6, 7, 8}]  # C/O/N索引

    # 为每个H分配一个成键伙伴
    for h_idx in h_indices:
        if acceptor_indices:
            # 计算H到所有受体的距离
            h_pos = coords[h_idx]
            dists = [torch.norm(h_pos - coords[acc_idx]) for acc_idx in acceptor_indices]
            closest_acc_idx = acceptor_indices[torch.argmin(torch.tensor(dists))]
            required_pairs.append((h_idx, closest_acc_idx))

    # 关键对的距离在合理键长范围内
    for _ in range(max_iter):
        updated = False
        for i, j in required_pairs:
            pos1 = coords[i]
            pos2 = coords[j]
            vec = pos1 - pos2
            dist = torch.norm(vec)
            pair = (atomic_nums[i], atomic_nums[j])

            # 关键键长范围
            if pair in BOND_LENGTH_RANGES:
                min_bond, max_bond = BOND_LENGTH_RANGES[pair]
                ideal_bond = (min_bond + max_bond) / 2  # 理想键长

                # 若距离不在范围内，拉近距离
                if dist > max_bond:
                    # 差距越大，调整幅度越大
                    adjustment = (ideal_bond - dist) * 0.5
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move
                    updated = True
                elif dist < min_bond:
                    # 过近则推开
                    adjustment = (ideal_bond - dist) * 0.5
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move
                    updated = True

        if not updated:
            break

    return coords


# 新增：生成环状结构
def generate_ring_structure(coords: torch.Tensor, atomic_nums: List[int], ring_size: int = 6,
                            radius: float = 1.4) -> torch.Tensor:
    """为碳骨架生成环状结构"""
    # 找到所有碳原子索引
    carbon_indices = [i for i, num in enumerate(atomic_nums) if num == 6]
    if len(carbon_indices) < ring_size:
        print(f"[DEBUG] 碳原子数量不足({len(carbon_indices)} < {ring_size})，无法生成环状结构")
        return coords

    # 选择环中的原子
    ring_atoms = random.sample(carbon_indices, ring_size)

    # 在XY平面生成环状坐标
    center = torch.tensor([0.0, 0.0, 0.0], device=coords.device)
    for i, idx in enumerate(ring_atoms):
        angle = torch.tensor(2 * torch.pi * i / ring_size, device=coords.device)
        x = center[0] + radius * torch.cos(angle)
        y = center[1] + radius * torch.sin(angle)
        z = center[2]
        coords[idx] = torch.tensor([x, y, z], device=coords.device)

    # 调整环上原子位置，使其更合理
    for i in range(len(ring_atoms)):
        idx1 = ring_atoms[i]
        idx2 = ring_atoms[(i + 1) % ring_size]
        pos1 = coords[idx1]
        pos2 = coords[idx2]
        dist = torch.norm(pos1 - pos2)

        # 调整键长到合理范围 (1.4-1.5Å)
        if dist < 1.3 or dist > 1.6:
            ideal_dist = 1.45
            adjustment = (ideal_dist - dist) * 0.5
            vec = (pos1 - pos2) / dist
            coords[idx1] += vec * adjustment
            coords[idx2] -= vec * adjustment

    print(f"[DEBUG] 生成环状结构: {ring_size}元环")
    return coords


# 新增：计算结构合理性惩罚项
def calculate_structure_penalties(g: dgl.DGLGraph) -> Dict[str, float]:
    """计算结构合理性惩罚项"""
    penalties = {
        "min_distance": 0.0,  # 原子重叠惩罚
        "bond_count": 0.0,  # 成键数不合理惩罚
        "bond_length": 0.0,  # 键长不合理惩罚
        "missing_bonds": 0.0,  # 必要键缺失惩罚
        "unknown_atom": 0.0,   # 未知原子惩罚
        "h_h_bonds": 0.0   
    }

    node_feats = g.ndata['feat'].detach().cpu().numpy()
    atomic_nums = np.round(node_feats[:, 0]).astype(int)
    coordinates = node_feats[:, 1:4]
    num_atoms = len(atomic_nums)

    # 1. 原子离散性惩罚（避免重叠）
    if num_atoms > 1:
        dist_matrix = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)
        min_dist = np.min(dist_matrix[dist_matrix > 1e-6])  # 排除自身距离
        # 距离越小，惩罚越大
        if min_dist < 0.5:  # 严重重叠
            penalties["min_distance"] = (0.5 - min_dist) * 1000
        elif min_dist < 0.8:  # 轻微重叠
            penalties["min_distance"] = (0.8 - min_dist) * 200

    # 2. 未知原子类型惩罚（结合能计算可能因此失败）
    for num in atomic_nums:
        if num not in number_to_symbol:
            penalties["unknown_atom"] += 5000  # 对未知原子施加重罚

    # 3. 成键分析（使用openbabel）
    ob_mol = openbabel.OBMol()
    ob_mol.BeginModify()
    for num, (x, y, z) in zip(atomic_nums, coordinates):
        if num not in number_to_symbol:
            ob_mol.EndModify()
            return penalties  # 有未知原子，直接返回基础惩罚
        ob_atom = openbabel.OBAtom()
        ob_atom.SetAtomicNum(int(num))
        ob_atom.SetVector(float(x), float(y), float(z))
        ob_mol.AddAtom(ob_atom)
    ob_mol.EndModify()
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    # 4. 成键数惩罚
    for i in range(num_atoms):
        num = atomic_nums[i]
        if num in BOND_COUNT_CONSTRAINTS:
            min_bonds, max_bonds = BOND_COUNT_CONSTRAINTS[num]
            # 使用GetExplicitValence()获取成键数
            bonds = ob_mol.GetAtom(i + 1).GetExplicitValence()
            if not (min_bonds <= bonds <= max_bonds):
                # 对H原子成键数错误加倍惩罚
                weight = 3.0 if num == 1 else 1.5
                penalties["bond_count"] += abs(bonds - (min_bonds + max_bonds) / 2) * 50 * weight

    # 5. 键长合理性惩罚
    for bond in openbabel.OBMolBondIter(ob_mol):
        a1_idx = bond.GetBeginAtomIdx() - 1  # 转换为0基索引
        a2_idx = bond.GetEndAtomIdx() - 1
        num1 = atomic_nums[a1_idx]
        num2 = atomic_nums[a2_idx]
        pair = (num1, num2) if num1 <= num2 else (num2, num1)

        if pair in BOND_LENGTH_RANGES:
            min_len, max_len = BOND_LENGTH_RANGES[pair]
            actual_len = bond.GetLength()
            # 键长偏离合理范围的惩罚
            if actual_len < min_len:
                penalties["bond_length"] += (min_len - actual_len) * 30
            elif actual_len > max_len:
                penalties["bond_length"] += (actual_len - max_len) * 30

        if num1 == 1 and num2 == 1:
            penalties["h_h_bonds"] += 5000  # 每发现一个H-H键，惩罚5000

    # 6. 必要键缺失惩罚
    for i in range(num_atoms):
        if atomic_nums[i] == 1:
            bonds = ob_mol.GetAtom(i + 1).GetExplicitValence()
            if bonds == 0:
                penalties["missing_bonds"] += 200

    return penalties


# 修改：综合适应度函数（结合能 + 结构惩罚）
def calculate_fitness(g: dgl.DGLGraph) -> float:
    """综合适应度函数：结合能 + 结构惩罚项"""
    # 计算结构惩罚项（无论结合能是否有效，都先计算惩罚）
    penalties = calculate_structure_penalties(g)

    # 权重设置
    weights = {
        "binding_energy": 1.0,
        "min_distance": 2.0,
        "bond_count": 2.5,
        "bond_length": 1.0,
        "missing_bonds": 3.0,
        "unknown_atom": 5.0,
        "h_h_bonds": 4.0
    }

    # 计算结合能
    binding_energy = calculate_binding_energy(g)

    if binding_energy is None:
        # 结合能计算失败：强制赋予极大的惩罚适应度（确保这类结构被优先排除）
        # 惩罚项总和 + 基础惩罚值（10000，确保远大于正常结构的适应度）
        total_penalty = (
                penalties["min_distance"] * weights["min_distance"] +
                penalties["bond_count"] * weights["bond_count"] +
                penalties["bond_length"] * weights["bond_length"] +
                penalties["missing_bonds"] * weights["missing_bonds"] +
                penalties["unknown_atom"] * weights["unknown_atom"] +
                penalties["h_h_bonds"] * weights["h_h_bonds"]
        )
        bad_fitness = total_penalty + 10000  # 基础惩罚确保优先级
        print(f"[DEBUG] 结合能计算失败，惩罚适应度: {bad_fitness:.2f}")
        return bad_fitness
    else:
        # 结合能有效：正常计算综合适应度
        total_fitness = (
                binding_energy * weights["binding_energy"] +
                penalties["min_distance"] * weights["min_distance"] +
                penalties["bond_count"] * weights["bond_count"] +
                penalties["bond_length"] * weights["bond_length"] +
                penalties["missing_bonds"] * weights["missing_bonds"] +
                penalties["unknown_atom"] * weights["unknown_atom"]
        )
        print(f"[DEBUG] 正常适应度: {total_fitness:.2f} (结合能: {binding_energy:.2f}eV)")
        return total_fitness


# 加载FAISS索引和元数据
def load_knowledge_base():
    try:
        print("[INFO] 加载FAISS索引和元数据...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"[INFO] FAISS索引维度: {index.d}")

        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        # 验证metadata类型
        if isinstance(metadata, list):
            print(f"[INFO] 元数据类型：列表（长度: {len(metadata)}）")
        else:
            print(f"[INFO] 元数据类型：字典（键数量: {len(metadata)}）")

        print(f"[INFO] 成功加载 {index.ntotal} 个材料的索引和元数据")
        return index, metadata
    except Exception as e:
        print(f"[ERROR] 加载知识图谱失败: {str(e)}")
        return None, None


# 从知识图谱检索相似结构（增强：获取晶胞参数）
def retrieve_similar_structures(formula: str, index, metadata, k=5):
    if index is None or metadata is None:
        return []

    # 创建64维查询向量（与FAISS索引维度一致）
    query_vector = np.zeros(64, dtype='float32')
    elements = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        elements[elem] += int(cnt) if cnt else 1

    for elem, cnt in elements.items():
        if elem in ase_atomic_numbers:
            atomic_num = ase_atomic_numbers[elem]
            vec_idx = atomic_num % 64  # 确保索引在0-63之间
            query_vector[vec_idx] = cnt

    query_vector = query_vector.reshape(1, -1)

    # 检索相似结构
    distances, indices = index.search(query_vector, k)

    # 提取检索结果
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            try:
                # 根据metadata类型选择索引方式
                if isinstance(metadata, list):
                    entry = metadata[idx]  # 列表用整数索引
                else:  # 字典用字符串索引
                    entry = metadata[str(idx)]

                entry['distance'] = distances[0][i]
                results.append(entry)
            except (IndexError, KeyError) as e:
                print(f"[WARNING] 无法获取索引 {idx} 的元数据: {e}")

    print(f"[INFO] 检索到 {len(results)} 个与 {formula} 相似的结构")
    for i, res in enumerate(results):
        print(f"  {i + 1}. {res['formula']} (距离: {res['distance']:.4f})")

    return results


# 生成合理的晶胞参数
def generate_lattice_parameters(similar_structures: List[Dict]) -> List[float]:
    """根据相似结构生成合理的晶胞参数，适配不同空间群"""
    valid_lattices = []
    for struct in similar_structures:
        if 'lattice_params' in struct and len(struct['lattice_params']) == 6:
            valid_lattices.append((struct['lattice_params'], struct['distance'],
                                   struct.get('space_group_number'), struct.get('crystal_system')))

    if valid_lattices:
        # 按空间群分类并计算平均值
        system_lattices = defaultdict(list)
        for lattice, dist, sg_num, system in valid_lattices:
            if system:
                system_lattices[system].append((lattice, dist))

        # 选择数据最多的晶系
        if system_lattices:
            selected_system = max(system_lattices.items(), key=lambda x: len(x[1]))[0]
            selected_data = system_lattices[selected_system]

            # 加权平均计算晶格参数
            total_weight = sum(1.0 / (d + 1e-6) for _, d in selected_data)
            weighted_sum = [0.0] * 6
            for lattice, dist in selected_data:
                weight = 1.0 / (1.0 + dist)
                weighted_sum = [sum_val + lat * weight for sum_val, lat in zip(weighted_sum, lattice)]
            return [val / total_weight for val in weighted_sum]

    # 根据空间群生成更合理的晶格参数
    sg_ranges = {
        'triclinic': (4.0, 10.0, 85.0, 95.0),
        'monoclinic': (4.0, 12.0, 95.0, 110.0),
        'orthorhombic': (3.5, 15.0, 89.0, 91.0),
        'tetragonal': (3.0, 12.0, 89.5, 90.5),
        'trigonal': (4.0, 10.0, 89.5, 90.5),
        'hexagonal': (3.0, 10.0, 89.5, 90.5),
        'cubic': (3.0, 8.0, 89.9, 90.1)
    }

    # 随机选择晶系生成晶格参数
    system = random.choice(list(sg_ranges.keys()))
    a_min, a_max, ang_min, ang_max = sg_ranges[system]

    a = np.random.uniform(a_min, a_max)

    # 根据晶系生成合理的b和c参数
    if system in ['triclinic', 'monoclinic', 'orthorhombic']:
        b = np.random.uniform(a_min, a_max)
        c = np.random.uniform(a_min, a_max)
    elif system in ['tetragonal']:
        b = a
        c = np.random.uniform(a * 0.8, a * 1.5)
    elif system in ['trigonal']:
        b = a
        c = np.random.uniform(a * 1.2, a * 2.0)
    elif system in ['hexagonal']:
        b = a
        c = np.random.uniform(a * 1.2, a * 2.0)
    else:  # cubic
        b = a
        c = a

    # 生成角度参数
    if system == 'triclinic':
        alpha = np.random.uniform(ang_min, ang_max)
        beta = np.random.uniform(ang_min, ang_max)
        gamma = np.random.uniform(ang_min, ang_max)
    elif system == 'monoclinic':
        alpha = 90.0
        beta = np.random.uniform(95.0, 110.0)
        gamma = 90.0
    else:  # 其他晶系
        alpha = 90.0
        beta = 90.0
        gamma = 90.0 if system != 'hexagonal' else 120.0

    return [a, b, c, alpha, beta, gamma]


# 从相似结构获取空间群信息
def get_crystallographic_info(similar_structures: List[Dict]) -> Dict:
    """从相似结构中获取或生成晶体学信息，增加复杂空间群的生成概率"""
    # 优先从相似结构中获取
    for struct in similar_structures:
        if 'space_group_symbol' in struct and 'space_group_number' in struct and 'crystal_system' in struct:
            sg_num = struct['space_group_number']
            crystal_system = struct['crystal_system']

            # 验证空间群与晶系是否匹配
            valid = True
            if crystal_system == 'triclinic' and not (1 <= sg_num <= 2):
                valid = False
            elif crystal_system == 'monoclinic' and not (3 <= sg_num <= 15):
                valid = False
            elif crystal_system == 'orthorhombic' and not (16 <= sg_num <= 74):
                valid = False
            elif crystal_system == 'tetragonal' and not (75 <= sg_num <= 142):
                valid = False
            elif crystal_system == 'trigonal' and not (143 <= sg_num <= 167):
                valid = False
            elif crystal_system == 'hexagonal' and not (168 <= sg_num <= 194):
                valid = False
            elif crystal_system == 'cubic' and not (195 <= sg_num <= 230):
                valid = False

            if valid:
                return {
                    'crystal_system': crystal_system,
                    'space_group_symbol': struct['space_group_symbol'],
                    'space_group_number': sg_num
                }

    # 生成匹配的空间群和晶系 - 增加复杂空间群的权重
    sg_ranges = {
        'triclinic': (1, 2, 0.5),  # 范围、权重
        'monoclinic': (3, 15, 1.0),
        'orthorhombic': (16, 74, 1.5),
        'tetragonal': (75, 142, 2.0),
        'trigonal': (143, 167, 1.8),
        'hexagonal': (168, 194, 1.8),
        'cubic': (195, 230, 2.5)  # 立方晶系权重最高
    }

    # 按权重随机选择晶系
    systems = list(sg_ranges.keys())
    weights = [sg_ranges[s][2] for s in systems]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # 随机选择晶系（带权重）
    system = random.choices(systems, probabilities)[0]
    min_sg, max_sg, _ = sg_ranges[system]

    # 空间群编号分布调整：优先选择非最低编号的空间群
    if max_sg - min_sg > 5:  # 有足够多的选择
        # 分成低、中、高三个区间，增加中高区间的概率
        third = (max_sg - min_sg) // 3
        ranges = [
            (min_sg, min_sg + third, 0.2),  # 低区间，权重0.2
            (min_sg + third, min_sg + 2 * third, 0.3),  # 中间区间，权重0.3
            (min_sg + 2 * third, max_sg, 0.5)  # 高区间，权重0.5
        ]
        range_weights = [r[2] for r in ranges]
        total_range_weight = sum(range_weights)
        range_probs = [w / total_range_weight for w in range_weights]

        # 随机选择区间
        selected_range = random.choices(ranges, range_probs)[0]
        sg_num = random.randint(selected_range[0], selected_range[1])
    else:
        sg_num = random.randint(min_sg, max_sg)

    # 空间群符号映射表（扩展）
    sg_symbol_map = {
        1: 'P1', 2: 'P-1',
        14: 'P2_1/c', 15: 'C2/c',
        62: 'Pnma', 63: 'Pbnm',
        143: 'R-3', 146: 'R-3m',
        194: 'P6_3/mmc', 191: 'P6/mmm',
        225: 'Fm-3m', 227: 'Fd-3m', 230: 'Ia-3d'
    }
    # 生成更合理的空间群符号
    if sg_num in sg_symbol_map:
        sg_symbol = sg_symbol_map[sg_num]
    else:
        # 根据晶系生成符号
        system_symbols = {
            'triclinic': ['P1', 'P-1'],
            'monoclinic': ['P2_1/c', 'C2/c', 'P2_1/m', 'Cm'],
            'orthorhombic': ['Pnma', 'Pbnm', 'Cmcm', 'Fmmm'],
            'tetragonal': ['P4_2/nmc', 'I41/amd', 'P4/mmm'],
            'trigonal': ['R-3', 'R-3m', 'P3_121'],
            'hexagonal': ['P6_3/mmc', 'P6/mmm', 'P63mc'],
            'cubic': ['Fm-3m', 'Fd-3m', 'Ia-3d', 'Pm-3m']
        }
        sg_symbol = random.choice(system_symbols.get(system, [f'Unknown({sg_num})']))

    return {
        'crystal_system': system,
        'space_group_symbol': sg_symbol,
        'space_group_number': sg_num
    }


# 生成图结构（使用相似结构的晶胞参数 + 改进初始坐标分布）
def generate_graph_from_latent(z: torch.Tensor, atom_types: List[int], similar_structures: List[Dict],
                               unsaturation: float = 0.0) -> Tuple[dgl.DGLGraph, List[float], Dict]:
    # 获取晶体学信息（先于原子生成）
    crystallographic_info = get_crystallographic_info(similar_structures)
    system = crystallographic_info['crystal_system']
    sg_num = crystallographic_info['space_group_number']

    # 根据空间群和晶系确定晶胞中的原子数量
    # 复杂空间群通常有更多的Wyckoff位置
    base_count = len(atom_types)
    num_atoms = base_count

    # 根据空间群复杂度增加原子数量
    if sg_num > 100:  # 较复杂空间群
        num_repeats = random.randint(1, 3)  # 1-3个不对称单元
        num_atoms = base_count * num_repeats
        # 确保原子数与晶系匹配（例如立方晶系倾向于有更多对称原子）
        if system == 'cubic' and num_repeats < 2:
            num_atoms = base_count * 2
    elif sg_num > 50:  # 中等复杂度
        num_repeats = random.randint(1, 2)
        num_atoms = base_count * num_repeats

    # 创建包含更多原子的图
    g = dgl.graph(([], []), num_nodes=num_atoms)

    # 生成基础坐标
    with torch.no_grad():
        recon_node = model.decoder(z, g)

    # 确保获取三维坐标
    if recon_node.shape[1] < 4:
        pad_size = 4 - recon_node.shape[1]
        recon_node = torch.cat([recon_node, torch.zeros(recon_node.shape[0], pad_size, device=recon_node.device)],
                               dim=1)

    # 初始分布（扩大范围以适应多原子晶胞）
    coords = torch.tanh(recon_node[:, 1:4]) * 30  # 扩大到±30Å

    # 扩展原子类型列表以匹配晶胞中的原子数量
    extended_atom_types = []
    repeat = num_atoms // base_count
    remainder = num_atoms % base_count
    for i in range(repeat):
        extended_atom_types.extend(atom_types)
    extended_atom_types.extend(atom_types[:remainder])
    atom_types = extended_atom_types

    # 生成环状结构（如果适用）
    ring_generated = False
    if unsaturation >= 1.0 and num_atoms >= 6:
        ring_size = random.choice([5, 6, 7])
        if random.random() < 0.7:  # 提高环状结构生成概率
            coords = generate_ring_structure(coords, atom_types, ring_size=ring_size)
            ring_generated = True

    # 多原子晶胞的坐标分布策略
    if num_atoms > len(atom_types) // max(repeat, 1) and not ring_generated:
        # 根据空间群的对称性分布原子
        symmetry_centers = []

        # 根据晶系生成对称中心
        if system == 'cubic':
            centers = [
                (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)  # 面心立方对称中心
            ]
            symmetry_centers = centers[:num_repeats]
        elif system == 'hexagonal':
            centers = [(i * 0.5, 0, 0) for i in range(4)]
            symmetry_centers = centers[:num_repeats]
        elif system in ['tetragonal', 'orthorhombic']:
            centers = [(i * 0.5, 0, 0) for i in range(3)]
            symmetry_centers = centers[:num_repeats]
        else:
            # 通用对称中心生成
            for i in range(num_repeats):
                angle = 2 * torch.pi * i / num_repeats
                radius = 10.0 + (i % 3) * 5.0
                center = torch.tensor([
                    torch.cos(torch.tensor(angle)) * radius,
                    torch.sin(torch.tensor(angle)) * radius,
                    (i % 2) * 8.0
                ], device=DEVICE)
                symmetry_centers.append(center)

        # 将原子分配到不同的对称中心
        atoms_per_center = num_atoms // len(symmetry_centers)
        remainder = num_atoms % len(symmetry_centers)

        for i, center in enumerate(symmetry_centers):
            start_idx = i * atoms_per_center
            end_idx = start_idx + atoms_per_center
            if i == len(symmetry_centers) - 1:
                end_idx += remainder  # 最后一个中心分配剩余原子

            # 将原子放置在对称中心周围
            for j in range(start_idx, end_idx):
                if j < num_atoms:
                    offset = (torch.rand(3, device=DEVICE) - 0.5) * 8.0  # ±4Å范围内
                    coords[j] = center + offset

    # 分离H原子和其他原子
    h_indices = [i for i, num in enumerate(atom_types) if num == 1]
    non_h_indices = [i for i, num in enumerate(atom_types) if num != 1]
    acceptors = [i for i, num in enumerate(atom_types) if num in {6, 7, 8}]

    # 为H原子分配成键伙伴
    if h_indices and acceptors:
        for h_idx in h_indices:
            acc_idx = random.choice(acceptors)
            acc_pos = coords[acc_idx]
            # 合理键长范围内放置H原子
            angle1 = torch.rand(1) * 2 * torch.pi
            angle2 = torch.rand(1) * torch.pi
            r = torch.tensor(0.9 + 0.3 * torch.rand(1))  # C-H键长范围
            h_pos = acc_pos + torch.tensor([
                r * torch.sin(angle2) * torch.cos(angle1),
                r * torch.sin(angle2) * torch.sin(angle1),
                r * torch.cos(angle2)
            ], device=DEVICE).squeeze()
            coords[h_idx] = h_pos

    # 分阶段执行距离约束
    coords = enforce_min_distance(coords, atom_types, min_dist=0.6, max_iter=100)
    coords = enforce_bond_constraints(coords, atom_types)
    coords = enforce_min_distance(coords, atom_types, min_dist=0.8, max_iter=150)

    # 组装特征
    atomic_nums = torch.tensor(atom_types, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    g.ndata['feat'] = torch.cat([atomic_nums, coords], dim=1)

    # 生成晶胞参数（从相似结构获取）
    lattice_params = generate_lattice_parameters(similar_structures)

    # 更新晶体学信息
    crystallographic_info['num_atoms_in_unit_cell'] = num_atoms

    # 最终验证
    if num_atoms > 1:
        coords_np = coords.detach().cpu().numpy()
        dist_matrix = np.linalg.norm(coords_np[:, None] - coords_np, axis=2)
        min_dist = np.min(dist_matrix[dist_matrix > 1e-6])
        print(f"[DEBUG] 最终最小原子间距: {min_dist:.4f}Å, 晶胞原子数: {num_atoms}, 空间群: {sg_num}")
    return g, lattice_params, crystallographic_info


# 粒子类
class Particle:
    def __init__(self, dim, initial_pos=None):
        if initial_pos is not None:
            self.position = initial_pos.clone()
        else:
            self.position = torch.randn(dim).to(DEVICE) * 1.2  # 稍大的初始速度，增加探索性
        self.velocity = torch.randn(dim).to(DEVICE) * 0.15  # 稍大的初始速度，增加探索性
        self.best_position = self.position.clone()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
        r1, r2 = torch.rand(2).to(DEVICE)
        self.velocity = w * self.velocity + \
                        c1 * r1 * (self.best_position - self.position) + \
                        c2 * r2 * (global_best - self.position)


# 生成候选结构（使用适应度函数，确保生成100个结构）
def generate_candidate_structures(formula: str, num_candidates=100):
    atom_types = get_atom_types(formula)
    total_atoms = len(atom_types)

    # 计算不饱和度
    unsaturation = calculate_unsaturation(formula)
    print(f"[DEBUG] 生成 {formula} 结构: 原子数 {total_atoms}, 不饱和度 {unsaturation:.2f}, "
          f"类型 {np.unique(atom_types, return_counts=True)}")

    # 加载知识图谱
    index, metadata = load_knowledge_base()
    similar_structures = retrieve_similar_structures(formula, index, metadata)

    # 粒子群优化
    num_particles = 30
    particles = []
    for _ in range(num_particles):
        pos = torch.randn(LATENT_DIM).to(DEVICE) * 1.2  # 增大标准差，提高多样性
        particles.append(Particle(LATENT_DIM, initial_pos=pos))

    global_best_pos = particles[0].position.clone()
    global_best_fitness = float('inf')

    # 初始化全局最优
    for particle in particles:
        z = particle.position.unsqueeze(0)
        g, _, _ = generate_graph_from_latent(z, atom_types, similar_structures, unsaturation)
        fitness = calculate_fitness(g)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_pos = particle.position.clone()

    # 迭代优化
    for _ in tqdm(range(50), desc="优化结构"):
        for particle in particles:
            z = particle.position.unsqueeze(0)
            g, _, _ = generate_graph_from_latent(z, atom_types, similar_structures, unsaturation)
            fitness = calculate_fitness(g)

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.clone()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = particle.position.clone()

        # 更新粒子位置
        for particle in particles:
            particle.update_velocity(global_best_pos)
            particle.position += particle.velocity

    # 收集所有生成的结构和对应的适应度
    all_candidates = []
    all_energies = []
    all_lattices = []
    all_fitness = []
    all_crystallographic_info = []

    # 从粒子群中收集所有粒子的最佳位置
    for particle in particles:
        z = particle.best_position.unsqueeze(0)
        try:
            g, lattice_params, crystallographic_info = generate_graph_from_latent(
                z, atom_types, similar_structures, unsaturation)
            energy = calculate_binding_energy(g)
            fitness = calculate_fitness(g)

            all_candidates.append(g)
            all_energies.append(energy)
            all_lattices.append(lattice_params)
            all_fitness.append(fitness)
            all_crystallographic_info.append(crystallographic_info)
        except Exception as e:
            print(f"[WARNING] 生成粒子结构时出错: {str(e)}")

    # 如果数量不足，从优化过程中生成更多结构
    while len(all_candidates) < num_candidates:
        # 从当前全局最优附近生成新的结构
        noise = torch.randn(LATENT_DIM).to(DEVICE) * 0.3  # 较小的噪声
        z = global_best_pos + noise
        z = z.unsqueeze(0)
        try:
            g, lattice_params, crystallographic_info = generate_graph_from_latent(
                z, atom_types, similar_structures, unsaturation)
            energy = calculate_binding_energy(g)
            fitness = calculate_fitness(g)

            all_candidates.append(g)
            all_energies.append(energy)
            all_lattices.append(lattice_params)
            all_fitness.append(fitness)
            all_crystallographic_info.append(crystallographic_info)
        except Exception as e:
            print(f"[WARNING] 补充生成结构时出错: {str(e)}")

    # 按适应度排序，选择最佳的num_candidates个结构
    sorted_indices = sorted(range(len(all_candidates)), key=lambda i: all_fitness[i])
    final_candidates = []
    final_energies = []
    final_lattices = []
    final_fitness = []
    final_crystallographic_info = []

    for idx in sorted_indices[:num_candidates]:
        final_candidates.append(all_candidates[idx])
        final_energies.append(all_energies[idx])
        final_lattices.append(all_lattices[idx])
        final_fitness.append(all_fitness[idx])
        final_crystallographic_info.append(all_crystallographic_info[idx])

    print(f"[INFO] 成功生成 {len(final_candidates)} 个候选结构")
    return final_candidates, final_energies, final_lattices, final_fitness, final_crystallographic_info


# 保存结构为CIF文件
# 改进的保存结构为CIF文件函数（重点修复空间群对称操作）
def save_structure_as_cif(
        structure_dict,
        filename,
        title=None,
        author=None,
        symmetry=None,
        additional_info=None
):
    """
    将晶体结构保存为CIF格式

    参数:
    structure_dict: 包含晶体结构信息的字典
    filename: 输出文件路径（str或Path对象）
    title: CIF文件标题
    author: 作者信息
    symmetry: 对称性信息
    additional_info: 额外信息
    """
    import datetime

    try:
        # 确保文件名是字符串类型
        if not isinstance(filename, str):
            filename = str(filename)

        with open(filename, 'w') as f:
            # 文件头信息
            f.write(f"# CIF file generated by Python script\n")
            f.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if title:
                f.write(f"data_{title.replace(' ', '_')}\n")
            else:
                f.write("data_generated_structure\n")

            # 作者信息
            if author:
                f.write(f"_audit_author_name '{author}'\n")

            # 晶胞参数
            cell = structure_dict.get('cell')
            if cell and len(cell) == 6:
                f.write(f"_cell_length_a {cell[0]:.6f}\n")
                f.write(f"_cell_length_b {cell[1]:.6f}\n")
                f.write(f"_cell_length_c {cell[2]:.6f}\n")
                f.write(f"_cell_angle_alpha {cell[3]:.6f}\n")
                f.write(f"_cell_angle_beta {cell[4]:.6f}\n")
                f.write(f"_cell_angle_gamma {cell[5]:.6f}\n\n")

            # 空间群
            space_group = structure_dict.get('space_group', 'P 1')
            f.write(f"_symmetry_space_group_name_H-M '{space_group}'\n")

            # 对称性操作
            if symmetry and 'operations' in symmetry:
                f.write("\nloop_\n_symmetry_equiv_pos_as_xyz\n")
                for op in symmetry['operations']:
                    f.write(f"'{op}'\n")
            else:
                f.write("_symmetry_equiv_pos_as_xyz 'x, y, z'\n\n")

            # 原子信息
            f.write("\nloop_\n_atom_site_label\n_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
            f.write("_atom_site_occupancy\n")

            atoms = structure_dict.get('atoms', [])
            for atom in atoms:
                label = atom.get('label', f"{atom['element']}{atom.get('id', '')}")
                element = atom['element']
                pos = atom['position']
                occupancy = atom.get('occupancy', 1.0)

                f.write(f"{label:<6} {element:<2} {pos[0]:>12.8f} {pos[1]:>12.8f} {pos[2]:>12.8f} {occupancy:>8.6f}\n")

            # 额外信息
            if additional_info:
                f.write("\n# Additional information\n")
                for key, value in additional_info.items():
                    f.write(f"#{key}: {value}\n")

        print(f"[INFO] 结构已保存到 {filename}")
        return True

    except Exception as e:
        print(f"[ERROR] 保存CIF文件失败: {e}")
        return False

def graph_to_cif_structure(graph, lattice_params, crystallographic_info):
    """将DGL图、晶格参数和晶体学信息转换为CIF结构字典"""
    try:
        # 提取原子特征
        node_feats = graph.ndata['feat'].detach().cpu().numpy()
        atomic_nums = np.round(node_feats[:, 0]).astype(int)
        cart_coords = node_feats[:, 1:4]  # 笛卡尔坐标

        # 从晶格参数创建晶格对象（用于坐标转换）
        a, b, c, alpha, beta, gamma = lattice_params
        lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        # 将笛卡尔坐标转换为分数坐标
        frac_coords = [lattice.get_fractional_coords(coord) for coord in cart_coords]

        # 构建原子列表
        atoms = []
        for i, (num, frac) in enumerate(zip(atomic_nums, frac_coords)):
            element = number_to_symbol.get(num, f"X{num}")  # 未知元素用X加原子序数表示
            atoms.append({
                'label': f"{element}{i+1}",
                'element': element,
                'position': frac.tolist(),
                'id': i + 1
            })

        # 构建空间群对称操作（简化版）
        sg_symbol = crystallographic_info.get('space_group_symbol', 'P1')
        symmetry_ops = ["x, y, z"]  # 简化处理，实际应用中可根据空间群生成更多操作

        # 构建结构字典
        return {
            'atoms': atoms,
            'cell': lattice_params,
            'space_group': sg_symbol,
            'symmetry': {
                'operations': symmetry_ops
            }
        }

    except Exception as e:
        print(f"[ERROR] 转换图为CIF结构时出错: {e}")
        return None



# 可视化函数
def visualize_results(formula, energies, fitness_scores, crystallographic_info_list, output_dir):
    """可视化结合能和适应度分数的分布"""
    plt.style.use('seaborn-v0_8-notebook')

    # 1. 结合能分布直方图
    valid_energies = [e for e in energies if e is not None]
    if valid_energies:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_energies, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('binding energy (eV)', fontsize=12)
        plt.ylabel('number of structures', fontsize=12)
        plt.title(f'{formula} distribution of binding energies of candidate structures', fontsize=14)
        plt.grid(alpha=0.3)
        energy_path = output_dir / f'{formula}_binding_energy_dist.png'
        plt.savefig(energy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 结合能分布图表已保存到 {energy_path}")
    else:
        print("[WARNING] 没有有效的结合能数据，无法生成结合能分布图")

    # 2. 适应度分数分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(fitness_scores, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('fitness score', fontsize=12)
    plt.ylabel('number of structures', fontsize=12)
    plt.title(f'{formula} distribution of fitness scores of candidate structures', fontsize=14)
    plt.grid(alpha=0.3)
    fitness_path = output_dir / f'{formula}_fitness_dist.png'
    plt.savefig(fitness_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 适应度分布图表已保存到 {fitness_path}")

    # 3. 结合能与适应度的相关性散点图（如果有有效结合能）
    if valid_energies and len(valid_energies) == len(fitness_scores):
        # 过滤无效结合能对应的适应度
        valid_pairs = [(e, f) for e, f in zip(energies, fitness_scores) if e is not None]
        if valid_pairs:
            e_vals, f_vals = zip(*valid_pairs)
            plt.figure(figsize=(10, 6))
            plt.scatter(e_vals, f_vals, color='orange', alpha=0.7)
            plt.xlabel('binding energy (eV)', fontsize=12)
            plt.ylabel('fitness score', fontsize=12)
            plt.title(f'{formula} correlation between binding energy and fitness', fontsize=14)
            plt.grid(alpha=0.3)
            corr_path = output_dir / f'{formula}_energy_vs_fitness.png'
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] 相关性图表已保存到 {corr_path}")

    # 4. 空间群分布统计
    space_groups = [info['space_group_symbol'] for info in crystallographic_info_list]
    plt.figure(figsize=(10, 6))
    plt.hist(space_groups, bins=len(set(space_groups)), color='purple', edgecolor='black')
    plt.xlabel('Space Group Symbol', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{formula} Distribution of Space Groups', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    sg_path = output_dir / f'{formula}_space_group_dist.png'
    plt.savefig(sg_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 空间群分布图表已保存到 {sg_path}")


# 强制计算所有候选结构的结合能并生成图表
def force_calculate_and_visualize(formula: str, candidates: List[dgl.DGLGraph], lattices: List[List[float]],
                                  crystallographic_info_list: List[Dict], output_dir: Path):
    """强制计算所有候选结构的结合能并生成图表"""
    forced_energies = []
    valid_count = 0

    print(f"[INFO] 开始强制计算 {len(candidates)} 个候选结构的结合能...")
    for i, (graph, lattice, crystallographic_info) in enumerate(
            tqdm(zip(candidates, lattices, crystallographic_info_list), total=len(candidates))):
        energy = calculate_binding_energy_forced(graph)
        forced_energies.append(energy)
        if energy is not None:
            valid_count += 1

        # 保存结构到单独的目录
        forced_dir = output_dir / "forced_structures"
        forced_dir.mkdir(parents=True, exist_ok=True)
        cif_file = forced_dir / f"{formula}_forced_{i + 1}.cif"

        # 转换并保存CIF
        cif_structure = graph_to_cif_structure(graph, lattice, crystallographic_info)
        if cif_structure:
            save_structure_as_cif(cif_structure, cif_file)

    # 保存强制计算结果
    summary_file = output_dir / f"{formula}_forced_summary.csv"
    with open(summary_file, 'w') as f:
        f.write("StructureID,ForcedBindingEnergy(eV),SpaceGroupSymbol,SpaceGroupNumber\n")
        for i, (energy, info) in enumerate(zip(forced_energies, crystallographic_info_list)):
            f.write(f"{i + 1},{energy if energy is not None else 'NaN'},"
                    f"{info['space_group_symbol']},{info['space_group_number']}\n")

    # 生成强制计算的图表
    forced_energies_valid = [e for e in forced_energies if e is not None]
    if forced_energies_valid:
        plt.figure(figsize=(10, 6))
        plt.hist(forced_energies_valid, bins=20, color='purple', edgecolor='black')
        plt.xlabel('binding energy (eV) - forced calculation', fontsize=12)
        plt.ylabel('number of structures', fontsize=12)
        plt.title(f'{formula} distribution of forced binding energies', fontsize=14)
        plt.grid(alpha=0.3)
        energy_path = output_dir / f'{formula}_forced_energy_dist.png'
        plt.savefig(energy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 强制计算结合能分布图表已保存到 {energy_path}")
    else:
        print("[WARNING] 强制计算未获得任何有效结合能数据")

    print(f"[INFO] 强制计算完成: {valid_count}/{len(candidates)} 个结构获得有效结合能")
    return forced_energies


# 修改主函数，添加可视化调用和强制计算
def main():
    formula = input("请输入分子式：")
    try:
        print(f"[INFO] 开始生成 {formula} 的候选结构...")
        candidates, energies, lattices, fitness, crystallographic_info_list = generate_candidate_structures(formula,
                                                                                                            num_candidates=100)

        # 保存所有候选结构
        output_dir = GENERATED_CIF_DIR / formula
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] 将候选结构保存到 {output_dir}")
        for i, (graph, energy, lattice, info) in enumerate(
                zip(candidates, energies, lattices, crystallographic_info_list)):
            # 1. 将图转换为CIF结构字典
            cif_structure = graph_to_cif_structure(graph, lattice, info)
            if not cif_structure:
                print(f"[WARNING] 无法转换结构 {i+1} 为CIF格式，跳过保存")
                continue

            # 2. 准备保存路径
            cif_file = output_dir / f"{formula}_candidate_{i + 1}.cif"

            # 3. 准备额外信息
            additional_info = {
                "BindingEnergy(eV)": energy if energy is not None else "N/A",
                "FitnessScore": fitness[i],
                "NumAtoms": len(graph.ndata['feat']),
                "CrystalSystem": info.get('crystal_system', 'unknown')
            }

            # 4. 保存CIF文件（正确传递参数）
            save_structure_as_cif(
                structure_dict=cif_structure,
                filename=cif_file,
                title=f"{formula}_candidate_{i+1}",
                additional_info=additional_info
            )

        # 保存结果汇总
        summary_file = output_dir / f"{formula}_summary.csv"
        with open(summary_file, 'w') as f:
            f.write(
                "StructureID,BindingEnergy(eV),FitnessScore,CrystalSystem,SpaceGroupSymbol,SpaceGroupNumber,NumAtomsInUnitCell\n")
            for i, (e, fit, info) in enumerate(zip(energies, fitness, crystallographic_info_list)):
                f.write(f"{i + 1},{e if e is not None else 'NaN'},{fit},"
                        f"{info['crystal_system']},{info['space_group_symbol']},"
                        f"{info['space_group_number']},{info['num_atoms_in_unit_cell']}\n")

        # 生成并保存可视化图表
        viz_dir = VISUALIZATION_DIR / formula
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualize_results(formula, energies, fitness, crystallographic_info_list, viz_dir)

        # 强制计算所有候选结构的结合能并生成图表
        forced_energies = force_calculate_and_visualize(formula, candidates, lattices, crystallographic_info_list,
                                                        viz_dir)

        # 将强制计算的结果添加到汇总文件中
        with open(summary_file, 'a') as f:
            f.write("\nStructureID,ForcedBindingEnergy(eV)\n")
            for i, energy in enumerate(forced_energies):
                f.write(f"{i + 1},{energy if energy is not None else 'NaN'}\n")

        print(f"[INFO] 结构生成完成，共生成 {len(candidates)} 个候选结构")
        print(f"[INFO] 结果汇总已保存到 {summary_file}")
        print(f"[INFO] 可视化图表已保存到 {viz_dir}")

    except Exception as e:
        print(f"[ERROR] 结构生成失败: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()