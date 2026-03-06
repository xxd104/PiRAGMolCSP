import os
import dgl
import torch
import numpy as np
import faiss
import json
import re
import random
import torch.nn as nn
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
# 添加缺少的导入
from dgl.nn import RelGraphConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling

# 全局配置
PROCESSED_GRAPH_DIR = "/home/nyx/RG-RAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/RG-RAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/RG-RAG/models"
VISUALIZATION_DIR = Path("/home/nyx/RG-RAG/generated_vis")
CIF_DIR = Path("/home/nyx/RG-RAG/raw_cifs")
MODEL_PATH = Path("/home/nyx/RG-RAG/models/best_model.pth")
FAISS_INDEX_PATH = Path("/home/nyx/RG-RAG/knowledage_base/material_index.faiss")
METADATA_PATH = Path("/home/nyx/RG-RAG/knowledage_base/material_metadata.json")
GENERATED_CIF_DIR = Path("/home/nyx/RG-RAG/generated_cifs")
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NODE_FEAT_DIM = 4
HIDDEN_DIM = 64
NUM_RELS = 4  # 关系类型数量：共价键、氢键、pi-pi堆积、范德华力
NUM_BASES = 4  # RGCN基函数数量
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

# 键类型映射
BOND_TYPES = {
    "covalent": 0,
    "hydrogen": 1,
    "pi_pi_stack": 2,
    "van_der_waals": 3
}

# 空间群与晶系的映射关系
SPACE_GROUP_RANGES = {
    'triclinic': (1, 2),
    'monoclinic': (3, 15),
    'orthorhombic': (16, 74),
    'tetragonal': (75, 142),
    'trigonal': (143, 167),
    'hexagonal': (168, 194),
    'cubic': (195, 230)
}

# 每个晶系的典型Wyckoff位置数范围（用于确定晶胞原子数）
WYCKOFF_COUNTS = {
    'triclinic': (1, 2),
    'monoclinic': (2, 4),
    'orthorhombic': (4, 8),
    'tetragonal': (4, 12),
    'trigonal': (3, 9),
    'hexagonal': (6, 12),
    'cubic': (8, 48)
}

# 空间群到典型晶格参数的映射
SG_LATTICE_TYPES = {
    'triclinic': 'aP',
    'monoclinic': 'mP, mC',
    'orthorhombic': 'oP, oC, oF, oI',
    'tetragonal': 'tP, tI',
    'trigonal': 'hP, hR',
    'hexagonal': 'hP',
    'cubic': 'cP, cF, cI'
}

sg_symbol_map = {
    'triclinic': ['P-1', 'P1'],  
    'monoclinic': ['C2/c', 'P21/c', 'P21/n', 'P2/c', 'C2'],
    'orthorhombic': ['Pnma', 'Cmcm', 'Pmmm', 'Cmc21', 'Pbca'],
    'tetragonal': ['P4/mmm', 'I41/amd', 'P42/nmc', 'P43212'],
    'trigonal': ['R-3m', 'R3c', 'P3121', 'P3221'],
    'hexagonal': ['P63/mmc', 'P6/mmm', 'P63mc'],
    'cubic': ['Fm-3m', 'Im-3m', 'Pm-3m', 'Fd-3m', 'Ia-3d']  # 增加高对称空间群
}

# 核心RGCN层（修正命名冲突）
class EdgeTypeRGCN(nn.Module):
    """带边类型的RGCN层，添加激活函数和归一化增强稳定性"""

    def __init__(self, in_feat: int, out_feat: int, num_rels: int, num_bases: int):
        super().__init__()
        self.num_rels = num_rels
        self.out_feat = out_feat

        self.rgcn = RelGraphConv(
            in_feat=in_feat,
            out_feat=out_feat,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=num_bases  # 减少基函数数量，降低复杂度
        )

        # 添加激活函数和归一化层，增强数值稳定性
        self.activation = nn.LeakyReLU(0.1)  # 比ELU更不易饱和
        self.norm = nn.LayerNorm(out_feat)  # 稳定输出范围

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        rgcn_out = self.rgcn(g, node_feats, edge_types)
        rgcn_out = self.activation(rgcn_out)
        rgcn_out = self.norm(rgcn_out)
        return rgcn_out


# 替换GCN编码器为RGCN编码器
class CrystalRGCNEncoder(nn.Module):
    def __init__(self,
                 node_feat_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 num_rels: int,
                 num_bases: int):
        super().__init__()

        # 减少层数和隐藏维度
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

        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_feats = g.ndata["feat"]  # (N, 4)

        h = self.rgcn1(g, node_feats, edge_types)
        h = self.rgcn2(g, h, edge_types)

        graph_emb = self.pooling(g, h)  # (B, hidden_dim*2)

        # 添加数值截断，防止极端值
        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        logvar = torch.clamp(logvar, min=-10, max=10)  # 限制logvar范围
        mu = torch.clamp(mu, min=-5, max=5)  # 限制mu范围

        return mu, logvar


# 解码器保持不变
class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, node_feat_dim: int = 4):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU()
        )

        # 确保输出维度为4
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)  # 强制输出4维
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
        # 获取批次中每个图的节点数
        batch_num_nodes = g.batch_num_nodes()  # 例如：[4, 92, 72, ...]
        B = len(batch_num_nodes)  # 批次大小
        total_nodes = sum(batch_num_nodes)  # 总节点数

        # 检查空图情况
        if total_nodes == 0:
            return torch.zeros(0, self.node_decoder[-1].out_features, device=z.device)

        # 潜在变量投影
        h = self.latent_proj(z)  # (B, hidden_dim*2)

        # 扩展潜在变量到节点级别
        h_expanded = torch.zeros(total_nodes, h.size(1), device=z.device)
        start_idx = 0
        for i in range(B):
            num_nodes = batch_num_nodes[i]
            end_idx = start_idx + num_nodes
            # 将第i个图的潜在变量扩展到num_nodes个节点
            h_expanded[start_idx:end_idx] = h[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        # 解码得到节点特征
        node_feats = self.node_decoder(h_expanded)  # (total_nodes, 4)

        # 严格校验维度和节点数
        assert node_feats.size(1) == 4, \
            f"解码器输出维度错误: {node_feats.size(1)}，预期: 4"
        assert node_feats.size(0) == total_nodes, \
            f"解码器节点数错误: {node_feats.size(0)}，预期: {total_nodes}"

        return node_feats


# VAE模型（适配RGCN）
class CrystalRGCNVAE(nn.Module):
    def __init__(self,
                 node_feat_dim: int = 4,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 num_rels: int = 4,
                 num_bases: int = 4):
        super().__init__()
        self.encoder = CrystalRGCNEncoder(node_feat_dim, hidden_dim, latent_dim, num_rels, num_bases)
        self.decoder = CrystalDecoder(latent_dim, hidden_dim, node_feat_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(g, edge_types)
        z = self.reparameterize(mu, logvar)
        recon_node = self.decoder(z, g)  # 输出节点数与g一致
        return recon_node, mu, logvar


def load_model(model_path, device, latent_dim=32, hidden_dim=64, num_rels=4, num_bases=4):
    """加载训练好的模型"""
    model = CrystalRGCNVAE(
        node_feat_dim=4,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_rels=num_rels,
        num_bases=num_bases
    ).to(device)

    # 尝试加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[INFO] 成功从 {model_path} 加载模型")
        return model
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {str(e)}")
        print("[INFO] 使用随机初始化的模型")
        return model


# 计算不饱和度（保持不变）
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

class Particle:
    def __init__(self, dim, initial_pos=None):
        if initial_pos is not None:
            self.position = initial_pos.clone()
        else:
            self.position = torch.randn(dim).to(DEVICE) * 1.2  # 初始位置
        self.velocity = torch.randn(dim).to(DEVICE) * 0.15  # 初始速度
        self.best_position = self.position.clone()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
        """更新粒子速度"""
        r1, r2 = torch.rand(2).to(DEVICE)
        self.velocity = w * self.velocity + \
                        c1 * r1 * (self.best_position - self.position) + \
                        c2 * r2 * (global_best - self.position)


def generate_candidate_structures(formula: str, model: CrystalRGCNVAE, num_candidates=100):
    """生成多个候选结构并通过PSO优化"""
    atom_types = get_atom_types(formula)
    total_atoms = len(atom_types)
    unsaturation = calculate_unsaturation(formula)
    print(f"[INFO] 生成 {formula} 候选结构: 原子数 {total_atoms}, 不饱和度 {unsaturation:.2f}")

    # 加载知识库并检索相似结构
    index, metadata = load_knowledge_base()
    similar_structures = retrieve_similar_structures(formula, index, metadata)

    # 初始化粒子群
    num_particles = 30
    particles = [Particle(LATENT_DIM) for _ in range(num_particles)]
    global_best_pos = particles[0].position.clone()
    global_best_fitness = float('inf')

    # 初始化全局最优
    for particle in particles:
        z = particle.position.unsqueeze(0)
        g, _, _ = generate_graph_from_latent(z, atom_types, similar_structures, unsaturation, model)
        fitness = calculate_fitness(g)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_pos = particle.position.clone()

    # PSO迭代优化
    for _ in tqdm(range(50), desc="PSO优化"):
        for particle in particles:
            # 生成当前粒子对应的结构
            z = particle.position.unsqueeze(0)
            g, _, _ = generate_graph_from_latent(z, atom_types, similar_structures, unsaturation, model)
            fitness = calculate_fitness(g)

            # 更新个体最优和全局最优
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.clone()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = particle.position.clone()

        # 更新粒子速度和位置
        for particle in particles:
            particle.update_velocity(global_best_pos)
            particle.position += particle.velocity

    # 收集候选结构
    all_candidates = []
    all_energies = []
    all_lattices = []
    all_fitness = []
    all_crystallographic_info = []

    # 从粒子群收集最优结构
    for particle in particles:
        z = particle.best_position.unsqueeze(0)
        try:
            g, lattice, info = generate_graph_from_latent(z, atom_types, similar_structures, unsaturation, model)
            energy = calculate_binding_energy(g)
            all_candidates.append(g)
            all_energies.append(energy)
            all_lattices.append(lattice)
            all_fitness.append(particle.best_fitness)
            all_crystallographic_info.append(info)
        except Exception as e:
            print(f"[WARNING] 生成粒子结构失败: {e}")

    # 补充生成足够数量的结构
    while len(all_candidates) < num_candidates:
        noise = torch.randn(LATENT_DIM).to(DEVICE) * 0.3
        z = global_best_pos + noise
        try:
            g, lattice, info = generate_graph_from_latent(z.unsqueeze(0), atom_types, similar_structures, unsaturation, model)
            energy = calculate_binding_energy(g)
            all_candidates.append(g)
            all_energies.append(energy)
            all_lattices.append(lattice)
            all_fitness.append(calculate_fitness(g))
            all_crystallographic_info.append(info)
        except Exception as e:
            print(f"[WARNING] 补充结构生成失败: {e}")

    # 按适应度排序
    sorted_indices = sorted(range(len(all_candidates)), key=lambda i: all_fitness[i])
    return (
        [all_candidates[i] for i in sorted_indices[:num_candidates]],
        [all_energies[i] for i in sorted_indices[:num_candidates]],
        [all_lattices[i] for i in sorted_indices[:num_candidates]],
        [all_fitness[i] for i in sorted_indices[:num_candidates]],
        [all_crystallographic_info[i] for i in sorted_indices[:num_candidates]]
    )


# 强制计算结合能（保持不变）
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


# 结合能计算函数（带限制）（保持不变）
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


# 从分子式解析原子类型（保持不变）
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
def enforce_min_distance(coords: torch.Tensor, atomic_nums: List[int], min_dist: float = 1.0,
                         max_iter: int = 300) -> torch.Tensor:
    num_atoms_coords = coords.shape[0]
    num_atoms_types = len(atomic_nums)
    if num_atoms_coords != num_atoms_types:
        raise ValueError(f"原子序数列表长度({num_atoms_types})与坐标数量({num_atoms_coords})不匹配！")
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
                    adjustment = (ideal_bond - dist) * 0.5
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move
                elif dist > max_bond:
                    adjustment = (ideal_bond - dist) * 0.5
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
                if dist < 1.2:  # 提高H-H最小距离到1.2Å
                    vec = coords[i] - coords[j]
                    if dist < 1e-6:  # 完全重叠
                        # 随机方向推开更远
                        random_dir = torch.randn_like(vec)
                        random_dir = random_dir / torch.norm(random_dir)
                        coords[i] += random_dir * 0.8  # 更大的推开距离
                        coords[j] -= random_dir * 0.8
                    else:
                        # 按比例推开到1.2Å
                        needed = 1.2 - dist
                        move = vec / dist * needed * 0.9  # 更大的调整比例
                        coords[i] += move
                        coords[j] -= move

    final_min_dist = torch.min(final_distances)
    if final_min_dist < min_dist:
        print(f"[WARNING] 最终最小距离 {final_min_dist:.4f}Å < {min_dist}Å（已尽力调整）")
    else:
        print(f"[DEBUG] 最终最小原子间距: {final_min_dist:.4f}Å")
    return coords


# 智能键长约束（保持不变）
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
                    adjustment = (ideal_bond - dist) * 0.8
                    move = vec / dist * adjustment
                    coords[i] += move
                    coords[j] -= move
                    updated = True
                elif dist < min_bond:
                    # 过近则推开
                    adjustment = (ideal_bond - dist) * 0.8
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
            adjustment = (ideal_dist - dist) * 0.8
            vec = (pos1 - pos2) / dist
            coords[idx1] += vec * adjustment
            coords[idx2] -= vec * adjustment

    print(f"[DEBUG] 生成环状结构: {ring_size}元环")
    return coords


# 计算结构合理性惩罚项（保持不变）
def calculate_structure_penalties(g: dgl.DGLGraph) -> Dict[str, float]:
    """计算结构合理性惩罚项"""
    penalties = {
        "min_distance": 0.0,  # 原子重叠惩罚
        "bond_count": 0.0,  # 成键数不合理惩罚
        "bond_length": 0.0,  # 键长不合理惩罚
        "missing_bonds": 0.0,  # 必要键缺失惩罚
        "unknown_atom": 0.0,  # 未知原子惩罚
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
        if min_dist < 0.8:  # 严重重叠
            penalties["min_distance"] = (0.8 - min_dist) * 2000
        elif min_dist < 1.0:  # 轻微重叠
            penalties["min_distance"] = (1.0 - min_dist) * 500

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


# 综合适应度函数（保持不变）
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

        # 加载FAISS索引
        if os.path.exists(FAISS_INDEX_PATH):
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            print(f"[INFO] 成功加载FAISS索引，维度: {index.d}, 向量数量: {index.ntotal}")
        else:
            print(f"[ERROR] FAISS索引文件不存在: {FAISS_INDEX_PATH}")
            return None, None

        # 加载元数据
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            print(f"[INFO] 成功加载元数据，条目数: {len(metadata)}")
        else:
            print(f"[ERROR] 元数据文件不存在: {METADATA_PATH}")
            return None, None

        return index, metadata

    except Exception as e:
        print(f"[ERROR] 加载知识库失败: {str(e)}")
        return None, None

def retrieve_similar_structures(formula: str, index, metadata, k=5) -> List[Dict]:
    """从FAISS索引检索相似结构"""
    if index is None or metadata is None:
        return []

    # 生成查询向量（适配32维索引）
    query_vector = np.zeros(32, dtype='float32')  # 修正为32维，匹配索引维度
    elements = defaultdict(int)
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, cnt in re.findall(pattern, formula):
        elements[elem] += int(cnt) if cnt else 1

    # 基于原子类型和数量生成权重向量（映射到32维）
    for elem, cnt in elements.items():
        if elem in ase_atomic_numbers:
            atomic_num = ase_atomic_numbers[elem]
            vec_idx = atomic_num % 32  # 映射到0-31维度
            query_vector[vec_idx] = cnt  # 数量作为权重

    # 检索相似结构
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            try:
                entry = metadata[idx] if isinstance(metadata, list) else metadata[str(idx)]
                entry['distance'] = distances[0][i]
                results.append(entry)
            except (IndexError, KeyError):
                print(f"[WARNING] 索引 {idx} 元数据缺失")
    print(f"[INFO] 检索到 {len(results)} 个相似结构")
    return results

import numpy as np
import random
from collections import defaultdict


def generate_lattice_parameters(similar_structures: List[Dict], crystal_system: str) -> List[float]:
    """生成与晶系匹配的晶格参数，确保非奇异矩阵"""
    valid_lattices = []
    for struct in similar_structures:
        if ('lattice_params' in struct and len(struct['lattice_params']) == 6 and
                struct.get('crystal_system') == crystal_system):
            valid_lattices.append((struct['lattice_params'], struct['distance']))

    # 尝试从相似结构生成
    if valid_lattices:
        total_weight = sum(1.0 / (d + 1e-6) for _, d in valid_lattices)
        weighted_sum = [0.0] * 6
        for lattice, dist in valid_lattices:
            weight = 1.0 / (1.0 + dist)
            weighted_sum = [sum_val + lat * weight for sum_val, lat in zip(weighted_sum, lattice)]
        lattice_params = [val / total_weight for val in weighted_sum]

        # 检查并修复潜在的奇异矩阵问题
        if is_singular_lattice(lattice_params):
            print("[WARNING] 从相似结构生成的晶格参数可能导致奇异矩阵，进行调整")
            return adjust_lattice_parameters(lattice_params, crystal_system)
        return lattice_params

    # 根据晶系生成合理的晶格参数
    lattice_type = SG_LATTICE_TYPES[crystal_system].split(',')[0].strip()

    # 不同晶系的典型晶格参数范围，确保参数不会导致奇异矩阵
    if crystal_system == 'triclinic':
        a = np.random.uniform(4.5, 10.0)
        b = np.random.uniform(a * 0.85, a * 1.15)
        c = np.random.uniform(a * 0.85, a * 1.15)
        alpha = np.random.uniform(82.0, 98.0)  # 避免90度，减少奇异可能性
        beta = np.random.uniform(82.0, 98.0)
        gamma = np.random.uniform(82.0, 98.0)

    elif crystal_system == 'monoclinic':
        a = np.random.uniform(4.5, 12.0)
        b = np.random.uniform(a * 0.85, a * 1.45)
        c = np.random.uniform(a * 0.85, a * 1.15)
        alpha = 90.0
        beta = np.random.uniform(93.0, 107.0)  # 远离90和180度
        gamma = 90.0

    elif crystal_system == 'orthorhombic':
        a = np.random.uniform(3.8, 15.0)
        b = np.random.uniform(a * 0.85, a * 1.45)
        c = np.random.uniform(b * 0.85, b * 1.45)  # 确保三个轴有明显差异
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    elif crystal_system == 'tetragonal':
        a = np.random.uniform(3.3, 12.0)
        b = a  # 四方晶系a=b
        c = np.random.uniform(a * 0.85, a * 1.8)  # c轴与a轴有明显差异
        alpha = beta = gamma = 90.0

    elif crystal_system == 'trigonal':
        a = np.random.uniform(4.3, 10.0)
        b = a
        c = np.random.uniform(a * 1.25, a * 1.95)
        alpha = beta = 90.0
        gamma = np.random.uniform(118.0, 122.0)  # 接近120但不完全等于

    elif crystal_system == 'hexagonal':
        a = np.random.uniform(4.3, 10.0)
        b = a
        c = np.random.uniform(a * 1.25, a * 1.95)
        alpha = beta = 90.0
        gamma = np.random.uniform(118.0, 122.0)  # 接近120但不完全等于

    elif crystal_system == 'cubic':
        a = np.random.uniform(3.3, 10.0)
        b = a  # 立方晶系a=b=c
        c = a
        alpha = beta = gamma = 90.0

    lattice_params = [a, b, c, alpha, beta, gamma]

    # 最终检查并调整
    if is_singular_lattice(lattice_params):
        print("[WARNING] 生成的晶格参数可能导致奇异矩阵，进行调整")
        return adjust_lattice_parameters(lattice_params, crystal_system)

    return lattice_params


def is_singular_lattice(lattice_params: List[float], eps: float = 1e-6) -> bool:
    """检查晶格参数是否可能导致奇异矩阵"""
    try:
        # 尝试创建晶格并计算体积
        lattice = Lattice.from_parameters(*lattice_params)
        volume = lattice.volume
        return abs(volume) < eps  # 体积接近零表示奇异
    except:
        return True


def adjust_lattice_parameters(lattice_params: List[float], crystal_system: str) -> List[float]:
    """调整晶格参数以避免奇异矩阵"""
    a, b, c, alpha, beta, gamma = lattice_params

    # 对不同晶系应用不同的调整策略
    if crystal_system == 'triclinic':
        # 轻微调整角度
        alpha += np.random.uniform(-2.0, 2.0)
        beta += np.random.uniform(-2.0, 2.0)
        gamma += np.random.uniform(-2.0, 2.0)
        # 确保角度在合理范围内
        alpha = np.clip(alpha, 75.0, 105.0)
        beta = np.clip(beta, 75.0, 105.0)
        gamma = np.clip(gamma, 75.0, 105.0)

    elif crystal_system == 'monoclinic':
        # 调整beta角
        beta += np.random.uniform(-3.0, 3.0)
        beta = np.clip(beta, 92.0, 108.0)

    elif crystal_system in ['orthorhombic', 'tetragonal', 'cubic']:
        # 调整轴长比例
        if crystal_system == 'orthorhombic':
            b = a * np.random.uniform(0.9, 1.4)
            c = b * np.random.uniform(0.9, 1.4)
        elif crystal_system == 'tetragonal':
            c = a * np.random.uniform(0.9, 1.7)
        # 立方晶系只需确保轴长不为零

    elif crystal_system in ['trigonal', 'hexagonal']:
        # 调整gamma角和c轴
        gamma += np.random.uniform(-1.5, 1.5)
        gamma = np.clip(gamma, 117.0, 123.0)
        c = a * np.random.uniform(1.2, 1.9)

    # 确保轴长为正
    a = max(3.5, a)
    b = max(3.5, b)
    c = max(3.5, c)

    return [a, b, c, alpha, beta, gamma]


def apply_symmetry_operations(coords: torch.Tensor, space_group_num: int, lattice: Lattice) -> torch.Tensor:
    """改进的对称操作：确保从单个分子正确扩展到晶胞"""
    try:
        coords_np = coords.cpu().numpy()
        num_molecule_atoms = len(coords_np)
        if num_molecule_atoms == 0:
            return coords

        # 创建分子结构作为基础
        dummy_species = ['H'] * num_molecule_atoms
        dummy_struct = Structure(lattice, dummy_species, coords_np)

        # 空间群分析
        sga = SpacegroupAnalyzer(dummy_struct, symprec=0.15, angle_tolerance=8.0)  # 放宽 tolerance 提高成功率
        sym_ops = sga.get_symmetry_operations()

        # 确保有足够的对称操作
        if len(sym_ops) < 2:
            from pymatgen.core.operations import SymmOp
            # 添加中心对称和镜像对称
            sym_ops.append(SymmOp.from_origin_rotation_reflection([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [0, 0, 0]))
            sym_ops.append(SymmOp.from_origin_rotation_reflection([[1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, 0]))

        # 应用对称操作生成晶胞原子
        unique_coords = set()
        for coord in coords_np:
            for op in sym_ops:
                # 应用对称操作
                sym_coord = op.operate(coord)
                # 转换为分数坐标并归一化到晶胞内
                frac_coord = lattice.get_fractional_coords(sym_coord)
                frac_coord = np.mod(frac_coord, 1.0)
                # 转换回笛卡尔坐标
                cart_coord = lattice.get_cartesian_coords(frac_coord)
                # 四舍五入去重
                rounded = tuple(np.round(cart_coord, 4))
                unique_coords.add(rounded)

        # 确保晶胞包含完整的分子对称拷贝
        num_sym_atoms = len(unique_coords)
        if num_sym_atoms < num_molecule_atoms * 2 and num_molecule_atoms < 50:
            # 强制添加对称拷贝（沿晶胞轴）
            axis_vecs = [
                lattice.get_cartesian_coords([1, 0, 0]),  # a轴
                lattice.get_cartesian_coords([0, 1, 0]),  # b轴
                lattice.get_cartesian_coords([0, 0, 1])  # c轴
            ]

            for coord in list(unique_coords):
                for vec in axis_vecs:
                    translated = tuple(np.round(np.array(coord) + vec, 4))
                    unique_coords.add(translated)

        # 转换回张量
        unit_cell_coords = torch.tensor(list(unique_coords), dtype=torch.float32, device=coords.device)
        print(f"[INFO] 对称操作完成: 从 {num_molecule_atoms} 个分子原子生成 {len(unit_cell_coords)} 个晶胞原子")
        return unit_cell_coords

    except Exception as e:
        print(f"[WARNING] 对称操作失败: {str(e)}，使用手动对称扩展")
        # 手动生成对称结构作为备选方案
        if len(coords) < 50:
            # 沿晶格轴生成对称拷贝
            a_axis = torch.tensor(lattice.get_cartesian_coords([1, 0, 0]), device=coords.device)
            b_axis = torch.tensor(lattice.get_cartesian_coords([0, 1, 0]), device=coords.device)

            mirrored1 = coords + a_axis
            mirrored2 = coords + b_axis
            mirrored3 = coords + a_axis + b_axis

            return torch.cat([coords, mirrored1, mirrored2, mirrored3], dim=0)
        return coords


def get_crystallographic_info(similar_structures: List[Dict]) -> Dict:
    """改进的晶体学信息生成，确保空间群多样性"""
    # 优先从相似结构中获取
    for struct in similar_structures:
        if ('space_group_symbol' in struct and 'space_group_number' in struct and
                'crystal_system' in struct):
            sg_num = struct['space_group_number']
            crystal_system = struct['crystal_system']
            if crystal_system in SPACE_GROUP_RANGES:
                min_sg, max_sg = SPACE_GROUP_RANGES[crystal_system]
                if min_sg <= sg_num <= max_sg:
                    return {
                        'crystal_system': crystal_system,
                        'space_group_symbol': struct['space_group_symbol'],
                        'space_group_number': sg_num,
                        'wyckoff_positions': struct.get('wyckoff_positions', {})
                    }

    # 调整晶系选择概率，降低三斜晶系概率
    systems = list(SPACE_GROUP_RANGES.keys())
    probabilities = [0.02, 0.1, 0.18, 0.2, 0.15, 0.15, 0.2]  # 三斜晶系概率降至2%
    crystal_system = random.choices(systems, probabilities)[0]

    min_sg, max_sg = SPACE_GROUP_RANGES[crystal_system]
    # 空间群编号偏向中间范围，避免极端简单空间群
    if max_sg - min_sg > 10:
        sg_num = random.randint(min_sg + 5, max_sg - 5)
    else:
        sg_num = random.randint(min_sg, max_sg)

    # 确保有可用的空间群符号
    sg_symbols = sg_symbol_map.get(crystal_system, ['P1'])
    sg_symbol = random.choice(sg_symbols)

    return {
        'crystal_system': crystal_system,
        'space_group_symbol': sg_symbol,
        'space_group_number': sg_num,
        'wyckoff_positions': {}
    }


def generate_graph_from_latent(z: torch.Tensor, atom_types: List[int], similar_structures: List[Dict],
                               unsaturation: float, model: CrystalRGCNVAE) -> Tuple[dgl.DGLGraph, List[float], Dict]:
    """改进的从潜在向量生成图结构的函数：先生成单个分子，再通过对称操作扩展到晶胞"""
    # 获取晶体学信息
    info = get_crystallographic_info(similar_structures)
    crystal_system = info['crystal_system']
    sg_num = info['space_group_number']

    # 1. 先生成单个分子结构（原子数量严格匹配分子式）
    num_molecule_atoms = len(atom_types)  # 单个分子的原子数
    print(f"[INFO] 生成单个{info['crystal_system']}分子结构，原子数: {num_molecule_atoms}")

    # 创建单个分子的图
    molecule_graph = dgl.graph(([], []), num_nodes=num_molecule_atoms)

    # 生成分子内边（共价键为主）
    src, dst, edge_types = [], [], []
    for i in range(num_molecule_atoms):
        for j in range(i + 1, num_molecule_atoms):
            # 分子内主要是共价键
            src.extend([i, j])
            dst.extend([j, i])
            edge_types.extend([BOND_TYPES["covalent"]] * 2)
    molecule_graph.add_edges(src, dst)
    molecule_graph.edata['etype'] = torch.tensor(edge_types, dtype=torch.long, device=DEVICE)

    # 生成分子节点特征
    with torch.no_grad():
        recon_node = model.decoder(z, molecule_graph)

    # 提取分子坐标并优化
    coords = recon_node[:, 1:4] * 5.0  # 分子尺度坐标范围
    coords = coords + torch.randn_like(coords) * 0.5  # 适度随机性

    # 2. 优化分子结构（关键步骤：确保分子结构合理）
    coords = enforce_min_distance(coords, atom_types, min_dist=0.9)  # 分子内原子间距
    coords = enforce_bond_constraints(coords, atom_types)  # 确保键长合理

    # 对不饱和分子生成环状结构（如苯环）
    if unsaturation >= 3 and 6 in atom_types:  # 含碳且不饱和度高
        carbon_count = atom_types.count(6)
        ring_size = min(6, max(3, carbon_count // 2))  # 合理的环大小
        coords = generate_ring_structure(coords, atom_types, ring_size=ring_size)

    # 3. 生成晶格参数（适配分子大小）
    # 根据分子大小调整晶格参数
    molecule_size = torch.max(coords, dim=0)[0] - torch.min(coords, dim=0)[0]
    lattice_scaling = max(1.5, torch.mean(molecule_size).item() * 0.8)  # 确保分子间有合理间距

    lattice_params = generate_lattice_parameters(similar_structures, crystal_system)
    # 基于分子大小调整晶格尺寸
    lattice_params = [param * lattice_scaling for param in lattice_params]
    lattice = Lattice.from_parameters(*lattice_params)

    # 4. 应用对称操作生成晶胞（从单个分子扩展到晶胞）
    print(f"[INFO] 应用空间群{sg_num}对称操作生成晶胞...")
    unit_cell_coords = apply_symmetry_operations(coords, sg_num, lattice)
    num_unit_cell_atoms = len(unit_cell_coords)  # 晶胞总原子数

    # 5. 构建完整晶胞图
    g = dgl.graph(([], []), num_nodes=num_unit_cell_atoms)

    # 扩展原子类型列表（对称操作后保持原子类型一致）
    extended_atom_types = []
    base_len = len(atom_types)
    for i in range(num_unit_cell_atoms):
        extended_atom_types.append(atom_types[i % base_len])  # 循环使用分子的原子类型

    # 添加晶胞内边（区分分子内和分子间相互作用）
    src, dst, edge_types = [], [], []
    for i in range(num_unit_cell_atoms):
        for j in range(i + 1, num_unit_cell_atoms):
            # 计算原子间距
            dist = torch.norm(unit_cell_coords[i] - unit_cell_coords[j])

            # 判断是分子内还是分子间相互作用
            if dist < CUTOFF_DISTANCE:
                # 同一分子内的原子（通过索引判断）
                if (i % base_len < num_molecule_atoms and
                        j % base_len < num_molecule_atoms and
                        abs(i - j) < num_molecule_atoms):
                    # 分子内共价键
                    bond_type = BOND_TYPES["covalent"]
                else:
                    # 分子间相互作用（范德华力为主）
                    bond_type = BOND_TYPES["van_der_waals"]

                src.extend([i, j])
                dst.extend([j, i])
                edge_types.extend([bond_type] * 2)

    g.add_edges(src, dst)
    g.edata['etype'] = torch.tensor(edge_types, dtype=torch.long, device=DEVICE)

    # 6. 组装节点特征
    atomic_nums = torch.tensor(extended_atom_types, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    g.ndata['feat'] = torch.cat([atomic_nums, unit_cell_coords], dim=1)

    # 更新晶体学信息
    info['num_atoms_in_unit_cell'] = num_unit_cell_atoms
    info['num_molecules_in_unit_cell'] = num_unit_cell_atoms // num_molecule_atoms
    info['lattice_type'] = SG_LATTICE_TYPES[crystal_system].split(',')[0]

    print(f"[INFO] 晶胞生成完成: 原子数 {num_unit_cell_atoms}, 包含 {info['num_molecules_in_unit_cell']} 个分子")
    return g, lattice_params, info


# 生成新的分子结构
def generate_structure(model: CrystalRGCNVAE, formula: str, num_samples: int = 10,
                       max_attempts: int = 100, device: torch.device = DEVICE) -> Optional[Dict]:
    """基于VAE模型生成符合分子式的分子结构"""
    print(f"[INFO] 开始生成分子式为 {formula} 的分子结构...")

    # 获取原子类型列表
    try:
        atom_types = get_atom_types(formula)
    except ValueError as e:
        print(f"[ERROR] 分子式解析错误: {str(e)}")
        return None

    num_atoms = len(atom_types)
    if num_atoms > MAX_ATOMS:
        print(f"[ERROR] 原子数量过多 ({num_atoms} > {MAX_ATOMS})，生成可能不稳定")
        return None

    # 计算不饱和度，用于指导结构生成
    unsaturation = calculate_unsaturation(formula)
    print(f"[INFO] 分子式 {formula} 的不饱和度: {unsaturation:.2f}")

    # 记录最佳结构及其适应度
    best_structure = None
    best_fitness = float('inf')
    valid_structures = []

    # 生成多个样本，选择最佳结构
    for sample_idx in range(num_samples):
        print(f"[INFO] 正在生成样本 {sample_idx + 1}/{num_samples}...")

        # 创建一个简单的图结构作为输入
        g = dgl.graph(([], []))
        g.add_nodes(num_atoms)

        # 原子类型特征
        atom_type_tensor = torch.tensor(atom_types, dtype=torch.float32, device=device).view(-1, 1)

        # 初始坐标（随机分布）
        coords = torch.randn(num_atoms, 3, device=device) * 2.0  # 初始分布在较大空间

        # 构建节点特征 [原子类型, x, y, z]
        node_feats = torch.cat([atom_type_tensor, coords], dim=1)
        g.ndata['feat'] = node_feats

        # 为所有节点对创建边（全连接图）
        src = []
        dst = []
        edge_types = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                # 根据原子类型和距离确定边类型
                # 简化处理：所有边默认为共价键
                src.append(i)
                dst.append(j)
                edge_types.append(BOND_TYPES["covalent"])

                # 反向边
                src.append(j)
                dst.append(i)
                edge_types.append(BOND_TYPES["covalent"])

        g.add_edges(src, dst)
        g.edata['etype'] = torch.tensor(edge_types, dtype=torch.long, device=device)

        # 尝试多次优化
        for attempt in range(max_attempts):
            # 模型推理
            model.eval()
            with torch.no_grad():
                recon_node, mu, logvar = model(g, g.edata['etype'])

            # 关键修改：只更新坐标，保留原始原子类型
            recon_coords = recon_node[:, 1:4]  # 只取坐标部分
            # 保留原始原子类型
            updated_feats = torch.cat([atom_type_tensor, recon_coords], dim=1)
            g.ndata['feat'] = updated_feats

            # 获取坐标并优化（使用更新后的特征）
            new_coords = g.ndata['feat'][:, 1:4]

            # 应用原子间距约束
            new_coords = enforce_min_distance(new_coords, atom_types)

            # 应用键长约束
            new_coords = enforce_bond_constraints(new_coords, atom_types)

            # 对于不饱和结构，尝试生成环状结构
            if unsaturation >= 1.0 and random.random() < 0.3 and len(atom_types) >= 3:
                # 只对含碳结构生成环
                if 6 in atom_types:
                    ring_size = min(6, len([i for i, num in enumerate(atom_types) if num == 6]))
                    if ring_size >= 3:
                        new_coords = generate_ring_structure(new_coords, atom_types, ring_size=ring_size)

            # 再次保留原子类型，只更新优化后的坐标
            final_feats = torch.cat([atom_type_tensor, new_coords], dim=1)
            g.ndata['feat'] = final_feats

            # 计算适应度
            fitness = calculate_fitness(g)

            # 记录最佳结构
            if fitness < best_fitness:
                best_fitness = fitness
                best_structure = g.clone()
                print(f"[INFO] 尝试 {attempt + 1}/{max_attempts}: 适应度改善至 {fitness:.4f}")

            # 打印进度
            if (attempt + 1) % 10 == 0:
                print(
                    f"[INFO] 尝试 {attempt + 1}/{max_attempts}: 当前适应度 {fitness:.4f}, 最佳适应度 {best_fitness:.4f}")

        # 保存此样本的最佳结构
        if best_structure is not None:
            valid_structures.append((best_structure, best_fitness))

        # 重置最佳结构，为下一个样本做准备
        best_structure = None
        best_fitness = float('inf')

    # 如果没有找到有效结构
    if not valid_structures:
        print(f"[ERROR] 未能生成有效的 {formula} 结构")
        return None

    # 按适应度排序，选择最佳结构
    valid_structures.sort(key=lambda x: x[1])
    best_g, best_fitness = valid_structures[0]

    # 提取最终结构信息
    final_node_feats = best_g.ndata['feat'].detach().cpu().numpy()
    final_atomic_nums = np.round(final_node_feats[:, 0]).astype(int)
    final_coordinates = final_node_feats[:, 1:4]

    # 计算最终结合能
    final_energy = calculate_binding_energy(best_g)

    # 构建结果字典
    result = {
        'formula': formula,
        'atomic_numbers': final_atomic_nums.tolist(),
        'coordinates': final_coordinates.tolist(),
        'binding_energy': final_energy,
        'fitness': best_fitness,
        'unsaturation': unsaturation
    }

    print(f"[INFO] 成功生成分子式为 {formula} 的结构")
    print(f"[INFO] 结合能: {final_energy:.4f} eV, 适应度: {best_fitness:.4f}")

    return result


def graph_to_cif_structure(graph: dgl.DGLGraph, lattice_params: List[float], info: Dict) -> Dict:
    """将图转换为CIF结构字典，添加奇异矩阵处理"""
    try:
        # 检查并调整晶格参数
        if is_singular_lattice(lattice_params):
            print("[WARNING] 检测到奇异晶格，尝试调整参数")
            lattice_params = adjust_lattice_parameters(lattice_params, info['crystal_system'])

            # 如果仍然奇异，使用备用晶格
            if is_singular_lattice(lattice_params):
                print("[WARNING] 调整后仍为奇异晶格，使用备用晶格参数")
                lattice_params = generate_lattice_parameters([], info['crystal_system'])

        node_feats = graph.ndata['feat'].detach().cpu().numpy()
        atomic_nums = np.round(node_feats[:, 0]).astype(int)
        cart_coords = node_feats[:, 1:4]

        # 创建晶格并转换为分数坐标
        lattice = Lattice.from_parameters(*lattice_params)
        frac_coords = [lattice.get_fractional_coords(coord) for coord in cart_coords]

        # 构建原子列表
        atoms = []
        for i, (num, frac) in enumerate(zip(atomic_nums, frac_coords)):
            elem = number_to_symbol.get(num, f'X{num}')
            atoms.append({
                'label': f'{elem}{i + 1}',
                'element': elem,
                'position': frac.tolist(),
                'occupancy': 1.0
            })

        return {
            'atoms': atoms,
            'cell': lattice_params,
            'space_group': info['space_group_symbol'],
            'lattice_volume': lattice.volume
        }

    except Exception as e:
        print(f"[ERROR] 转换为CIF结构失败: {str(e)}")
        # 生成一个简单的备用结构
        lattice = Lattice.from_parameters(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)  # 简单立方晶格
        return {
            'atoms': [{'label': f'X{i}', 'element': 'X', 'position': [0.0, 0.0, 0.0], 'occupancy': 1.0}
                      for i in range(len(atomic_nums))],
            'cell': [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
            'space_group': 'P1',
            'lattice_volume': lattice.volume
        }

# 保存生成的结构为CIF文件
def save_structure_as_cif(structure_dict: Dict, filename: str) -> bool:
    """保存CIF文件（适配graph_to_cif_structure生成的结构字典）"""
    try:
        # 提取结构数据
        atoms = structure_dict['atoms']
        cell_params = structure_dict['cell']
        space_group = structure_dict['space_group']

        # 创建pymatgen Structure对象
        lattice = Lattice.from_parameters(*cell_params)

        # 提取元素和分数坐标
        species = [atom['element'] for atom in atoms]
        frac_coords = [atom['position'] for atom in atoms]

        # 创建Structure对象
        structure = Structure(lattice, species, frac_coords)

        # 保存为CIF文件
        cif_writer = CifWriter(structure)
        cif_writer.write_file(filename)

        print(f"[INFO] 结构已保存为CIF文件: {filename}")
        return True

    except Exception as e:
        print(f"[ERROR] 保存CIF文件失败: {str(e)}")
        return False

def visualize_results(formula: str, energies: List[float], fitness: List[float],
                      crystallographic_info: List[Dict], output_dir: Path):
    """生成结构分析图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 结合能分布
    valid_energies = [e for e in energies if e is not None]
    if valid_energies:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_energies, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Binding Energy (eV)')
        plt.ylabel('Number of Structures')
        plt.title(f'{formula} Binding Energy Distribution')
        plt.savefig(output_dir / f'{formula}_energy.png', dpi=300)
        plt.close()

    # 2. 适应度分布
    plt.figure(figsize=(10, 6))
    plt.hist(fitness, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Fitness Score')
    plt.ylabel('Number of Structures')
    plt.title(f'{formula} Fitness Distribution')
    plt.savefig(output_dir / f'{formula}_fitness.png', dpi=300)
    plt.close()

    # 3. 空间群分布
    sg_symbols = [info['space_group_symbol'] for info in crystallographic_info]
    plt.figure(figsize=(10, 6))
    plt.hist(sg_symbols, bins=len(set(sg_symbols)), color='purple', edgecolor='black')
    plt.xlabel('Space Group Symbol')
    plt.ylabel('Count')
    plt.title(f'{formula} Space Group Distribution')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / f'{formula}_spacegroups.png', dpi=300)
    plt.close()


def force_calculate_and_visualize(formula: str, candidates: List[dgl.DGLGraph], lattices: List[List[float]],
                                  info_list: List[Dict], energies: List[float], output_dir: Path):
    """强制计算所有结构的结合能并可视化"""
    forced_energies = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (g, lattice, info) in enumerate(zip(candidates, lattices, info_list)):
        energy = calculate_binding_energy_forced(g)
        forced_energies.append(energy)
        # 保存强制计算的结构
        cif_struct = graph_to_cif_structure(g, lattice, info)
        save_structure_as_cif(cif_struct, output_dir / f'{formula}_forced_{i}.cif')

    # 生成强制计算能量分布
    valid_forced = [e for e in forced_energies if e is not None]
    if valid_forced:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_forced, bins=20, color='red', edgecolor='black')
        plt.xlabel('Binding Energy (eV)')
        plt.ylabel('Number of Structures')
        plt.title(f'{formula} Forced Calculation Binding Energy Distribution')
        plt.savefig(output_dir / f'{formula}_forced_energy.png', dpi=300)
        plt.close()

    # 保存汇总表
    with open(output_dir / f'{formula}_summary.csv', 'w') as f:
        f.write('ID,Binding Energy (eV),Forced Binding Energy (eV),Space Group\n')
        for i, (e, fe, info) in enumerate(zip(energies, forced_energies, info_list)):
            f.write(f'{i},{e},{fe},{info["space_group_symbol"]}\n')

def main():
    formula = input("请输入分子式: ")
    model = load_model(MODEL_PATH, DEVICE)
    if model is None:
        return

    # 生成候选结构
    candidates, energies, lattices, fitness, info_list = generate_candidate_structures(formula, model)

    # 保存结果
    output_dir = GENERATED_CIF_DIR / formula
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (g, lattice, info) in enumerate(zip(candidates, lattices, info_list)):
        cif_struct = graph_to_cif_structure(g, lattice, info)
        save_structure_as_cif(cif_struct, output_dir / f'{formula}_candidate_{i}.cif')

    # 可视化（英文标签）
    viz_dir = VISUALIZATION_DIR / formula
    visualize_results(formula, energies, fitness, info_list, viz_dir)
    # 修复force_calculate_and_visualize调用，传入energies参数
    force_calculate_and_visualize(formula, candidates, lattices, info_list, energies, viz_dir)

    print(f"[INFO] 所有结果保存至 {output_dir} 和 {viz_dir}")


if __name__ == "__main__":
    main()