import os
import re
import dgl
import numpy as np
import torch
from ase.io import read
from ase.geometry import get_distances
from typing import List, Dict, Tuple, Union, Optional, Set
import logging
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count, Manager
from multiprocessing.dummy import Pool as ThreadPool
import psutil
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.linalg import norm
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from tqdm import tqdm
import networkx as nx
import time

# ==================== 警告过滤 ====================
warnings.filterwarnings(
    "ignore",
    message=r"crystal system '(orthorhombic|triclinic|monoclinic|tetragonal|hexagonal|cubic|trigonal)' is not interpreted for space group",
    module="ase.io.cif"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="ase.spacegroup.spacegroup",
    message="scaled_positions \d+ and \d+ are equivalent"
)

# ==================== 全局配置 ====================
RAW_CIF_DIR = "/home/nyx/GRAG/raw_cifs"
PROCESSED_GRAPH_DIR = "/home/nyx/RG-RAG/dgl_graphs"
FEATURE_VIS_DIR = "/home/nyx/RG-RAG/feature_vis"
GRAPH_VIS_DIR = "/home/nyx/RG-RAG/graph_vis"
MAX_ATOM_LIMIT = 1000
MAX_BOND_PER_ATOM = {  # 每种原子的最大成键数限制
    "C": 4, "N": 3, "O": 2, "H": 1, "S": 6,
    "F": 1, "Cl": 1, "Br": 1, "I": 1
}

BOND_TYPES = {
    "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4
}

COVALENT_BONDS = {
    (("C", "C"), BOND_TYPES["SINGLE"]): (1.48, 1.56),
    (("C", "C"), BOND_TYPES["DOUBLE"]): (1.31, 1.35),
    (("C", "C"), BOND_TYPES["TRIPLE"]): (1.18, 1.22),
    (("C", "C"), BOND_TYPES["AROMATIC"]): (1.38, 1.42),
    (("C", "N"), BOND_TYPES["SINGLE"]): (1.38, 1.46),
    (("C", "N"), BOND_TYPES["DOUBLE"]): (1.25, 1.30),
    (("C", "N"), BOND_TYPES["AROMATIC"]): (1.35, 1.40),
    (("C", "O"), BOND_TYPES["SINGLE"]): (1.35, 1.43),
    (("C", "O"), BOND_TYPES["DOUBLE"]): (1.18, 1.23),
    (("C", "H"), BOND_TYPES["SINGLE"]): (1.05, 1.12),
    (("N", "H"), BOND_TYPES["SINGLE"]): (0.98, 1.05),
    (("O", "H"), BOND_TYPES["SINGLE"]): (0.94, 1.00),
    (("N", "N"), BOND_TYPES["SINGLE"]): (1.38, 1.46),
    (("N", "N"), BOND_TYPES["DOUBLE"]): (1.22, 1.26),
    (("N", "N"), BOND_TYPES["TRIPLE"]): (1.09, 1.13),
    (("O", "O"), BOND_TYPES["SINGLE"]): (1.40, 1.48),
    (("C", "S"), BOND_TYPES["SINGLE"]): (1.75, 1.85),
    (("C", "S"), BOND_TYPES["DOUBLE"]): (1.55, 1.60)
}

COVALENT_BOND_MAP = defaultdict(list)
for (elements, bond_type), (min_len, max_len) in COVALENT_BONDS.items():
    elem_pair = frozenset(elements)
    COVALENT_BOND_MAP[elem_pair].append((bond_type, min_len, max_len))

INTERMOLECULAR_PARAMS = {
    "hydrogen_bond": {
        "donor_groups": {"O-H", "N-H", "S-H"},
        "acceptors": {"O", "N", "S"},
        "distance_range": (1.5, 3.5),
        "angle_range": (120, 180)
    },
    "pi_pi_stack": {
        "ring_size": 5,
        "distance_range": (3.0, 4.5),
        "angle_range": (0, 30)
    },
    "van_der_waals": {
        "vdw_radius": {
            "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
            "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98
        },
        "distance_factor": 1.2,
        "max_bonds": 3  # 每个原子的最大范德华键数
    }
}

# 元素颜色映射（基于CPK颜色方案）
ELEMENT_COLORS = {
    "H": "white",
    "C": "#232323",  # 黑色/深灰色
    "N": "#3050F8",  # 蓝色
    "O": "#FF0D0D",  # 红色
    "S": "#FFFF30",  # 黄色
    "F": "#90E050",  # 浅绿色
    "Cl": "#1FF01F",  # 绿色
    "Br": "#A62929",  # 棕色
    "I": "#940094",  # 紫色
    "default": "#808080"  # 默认灰色
}

ATOMIC_NUMBERS = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16,
    "F": 9, "Cl": 17, "Br": 35, "I": 53
}
VALID_ELEMENTS = set(ATOMIC_NUMBERS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cif_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GLOBAL_STATS = {
    "min_z": 1, "max_z": 16, "coord_min": -10.0, "coord_max": 10.0
}


# ==================== 工具函数 ====================
def remove_duplicate_atoms(ase_atoms):
    """去除坐标重复的原子（保留6位小数精度）"""
    unique_positions = set()
    filtered_atoms = []
    for atom in ase_atoms:
        pos_key = tuple(np.round(atom.position, decimals=6))
        if pos_key not in unique_positions:
            unique_positions.add(pos_key)
            filtered_atoms.append(atom)
    return type(ase_atoms)(
        symbols=[atom.symbol for atom in filtered_atoms],
        positions=[atom.position for atom in filtered_atoms],
        cell=ase_atoms.cell,
        pbc=ase_atoms.pbc
    )


def validate_cif(cif_path: str) -> Tuple[bool, str]:
    """验证CIF文件有效性"""
    try:
        atoms = read(cif_path, format="cif")
        if len(atoms) > MAX_ATOM_LIMIT:
            return False, f"原子数量{len(atoms)}超过最大限制{MAX_ATOM_LIMIT}"
        invalid_elements = set(atoms.symbols) - VALID_ELEMENTS
        if invalid_elements:
            return False, f"包含无效元素: {invalid_elements}"
        if any(re.search(r"\?|#", sym) for sym in atoms.symbols):
            return False, "存在无序原子（符号包含?或#）"
        return True, "CIF文件验证通过"
    except FileNotFoundError:
        return False, "文件未找到"
    except Exception as e:
        return False, f"CIF读取失败: {str(e)}"


# ==================== 分子结构分析 ====================
def get_molecular_groups(atoms: List[Dict], cell: np.ndarray) -> List[List[int]]:
    """通过并查集划分分子基团"""
    coords = np.array([atom["position"] for atom in atoms])
    n_atoms = len(atoms)
    max_bond_length = max(v[1] for v in COVALENT_BONDS.values())
    adj_list = [[] for _ in range(n_atoms)]

    # 记录每个原子的成键数
    bond_count = defaultdict(int)

    for i in range(n_atoms):
        sym_i = atoms[i]["symbol"]
        max_bonds_i = MAX_BOND_PER_ATOM.get(sym_i, 0)

        for j in range(i + 1, n_atoms):
            # 检查j原子是否已经达到最大成键数
            sym_j = atoms[j]["symbol"]
            max_bonds_j = MAX_BOND_PER_ATOM.get(sym_j, 0)
            if bond_count[j] >= max_bonds_j:
                continue

            distance = get_distances(coords[i].reshape(1, 3), coords[j].reshape(1, 3), cell=cell, pbc=True)[1][0][0]
            elem_pair = frozenset({sym_i, sym_j})
            if elem_pair not in COVALENT_BOND_MAP:
                continue

            # 检查是否有匹配的键长范围
            matched = False
            for bond_type, min_len, max_len in COVALENT_BOND_MAP[elem_pair]:
                if min_len <= distance <= max_len:
                    matched = True
                    break

            if matched:
                # 检查i原子是否可以继续成键
                if bond_count[i] < max_bonds_i:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
                    bond_count[i] += 1
                    bond_count[j] += 1

    # 并查集操作
    parent = list(range(n_atoms))
    rank = [1] * n_atoms

    def find(u: int) -> int:
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u: int, v: int):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                if rank[root_u] == rank[root_v]:
                    rank[root_u] += 1

    for i in range(n_atoms):
        for j in adj_list[i]:
            if j > i:
                union(i, j)

    groups = defaultdict(list)
    for idx in range(n_atoms):
        groups[find(idx)].append(idx)

    return list(groups.values())


def find_covalent_bonds(atoms: List[Dict], cell: np.ndarray) -> List[Dict]:
    """识别共价键，添加键长限制和成键数量限制"""
    bonds = []
    coords = np.array([atom["position"] for atom in atoms])
    n_atoms = len(atoms)

    # 记录每个原子的成键数
    bond_count = defaultdict(int)

    for i in range(n_atoms):
        sym_i = atoms[i]["symbol"]
        max_bonds_i = MAX_BOND_PER_ATOM.get(sym_i, 0)

        # 跳过已经达到最大成键数的原子
        if bond_count[i] >= max_bonds_i:
            continue

        for j in range(i + 1, n_atoms):
            # 跳过已经达到最大成键数的原子
            sym_j = atoms[j]["symbol"]
            max_bonds_j = MAX_BOND_PER_ATOM.get(sym_j, 0)
            if bond_count[j] >= max_bonds_j:
                continue

            distance = get_distances(coords[i].reshape(1, 3), coords[j].reshape(1, 3), cell=cell, pbc=True)[1][0][0]
            elem_pair = frozenset({sym_i, sym_j})
            if elem_pair not in COVALENT_BOND_MAP:
                continue

            # 查找匹配的键长范围
            matched_bonds = []
            for bond_type, min_len, max_len in COVALENT_BOND_MAP[elem_pair]:
                if min_len <= distance <= max_len:
                    matched_bonds.append((bond_type, min_len, max_len))

            if matched_bonds:
                # 选择最匹配的键长类型
                best_bond = min(matched_bonds, key=lambda x: abs(distance - ((x[1] + x[2]) / 2)))
                bond_type_name = {
                    BOND_TYPES["SINGLE"]: "single",
                    BOND_TYPES["DOUBLE"]: "double",
                    BOND_TYPES["TRIPLE"]: "triple",
                    BOND_TYPES["AROMATIC"]: "aromatic"
                }[best_bond[0]]

                # 添加键并更新成键数
                bonds.append({
                    "u": i, "v": j,
                    "type": "covalent",
                    "subtype": bond_type_name,
                    "distance": distance,
                    "symbols": f"{sym_i}-{sym_j}"
                })
                bond_count[i] += 1
                bond_count[j] += 1

    return bonds


# ==================== 芳香环识别 ====================
def ase_atoms_to_rdkit_mol(atoms, covalent_bonds):
    """将ASE原子和共价键转换为带键信息的RDKit分子（关键修正）"""
    mol = Chem.RWMol()
    atom_idx_map = {}  # 映射ASE原子索引到RDKit原子索引

    # 添加原子
    for idx, atom in enumerate(atoms):
        rd_atom = Chem.Atom(atom["symbol"])
        mol_idx = mol.AddAtom(rd_atom)
        atom_idx_map[idx] = mol_idx

    # 添加共价键（关键：RDKit需要键信息才能判断芳香性）
    for bond in covalent_bonds:
        u = atom_idx_map[bond["u"]]
        v = atom_idx_map[bond["v"]]
        bond_order = {
            "single": Chem.BondType.SINGLE,
            "double": Chem.BondType.DOUBLE,
            "triple": Chem.BondType.TRIPLE,
            "aromatic": Chem.BondType.AROMATIC
        }[bond["subtype"]]
        mol.AddBond(u, v, bond_order)

    # 添加坐标
    conf = Chem.Conformer(mol.GetNumAtoms())
    for ase_idx, rd_idx in atom_idx_map.items():
        x, y, z = atoms[ase_idx]["position"]
        conf.SetAtomPosition(rd_idx, (x, y, z))
    mol.AddConformer(conf)

    # 计算芳香性（关键步骤）
    Chem.SanitizeMol(mol)
    rdmolops.AssignStereochemistryFrom3D(mol)
    return mol


def calculate_bond_length_uniformity(ring_atoms, bonds, atoms):
    """计算环的键长均匀性"""
    ring_bonds = [bond["distance"] for bond in bonds if {bond["u"], bond["v"]}.issubset(set(ring_atoms))]
    if not ring_bonds:
        return 1.0
    return np.std(ring_bonds) / np.mean(ring_bonds)


def calculate_ring_planarity(ring_atoms, atoms):
    """计算环的平面性"""
    coords = np.array([atoms[i]["position"] for i in ring_atoms])
    centroid = np.mean(coords, axis=0)
    cov = np.cov(coords - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0] if eigenvalues[0] < eigenvalues[1] else eigenvectors[:, 1]
    return np.sqrt(np.mean(np.square(np.dot(coords - centroid, normal))))


def find_aromatic_rings(atoms: List[Dict], covalent_bonds: List[Dict]) -> List[List[int]]:
    """识别芳香环（增强版：RDKit为主，几何特征为辅）"""
    try:
        # 1. 尝试RDKit识别（带键信息的分子）
        rdkit_mol = ase_atoms_to_rdkit_mol(atoms, covalent_bonds)

        # 宽容模式处理分子（避免因轻微不规范导致失败）
        try:
            Chem.SanitizeMol(rdkit_mol,
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            # 允许不严格的kekulize
            Chem.SanitizeMol(rdkit_mol,
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        ring_info = rdkit_mol.GetRingInfo()
        all_rings = ring_info.AtomRings()
        if not all_rings:
            return []

        # 映射RDKit索引到ASE索引
        rd_to_ase = {i: idx for idx, i in enumerate(range(rdkit_mol.GetNumAtoms()))}
        aromatic_rings = []

        for ring in all_rings:
            if len(ring) < 5:  # 过滤小环
                continue

            # 检查环原子是否均为芳香性且无明显非环原子
            is_aromatic = True
            for atom_idx in ring:
                atom = rdkit_mol.GetAtomWithIdx(atom_idx)
                if not atom.GetIsAromatic():
                    is_aromatic = False
                    break
                # 排除非环原子被错误标记的情况
                if not ring_info.IsAtomInRingOfSize(atom_idx, len(ring)):
                    is_aromatic = False
                    break

            if is_aromatic:
                ase_ring = [rd_to_ase[atom_idx] for atom_idx in ring]
                aromatic_rings.append(ase_ring)

        # 去重
        unique_rings = []
        seen = set()
        for ring in aromatic_rings:
            sorted_ring = tuple(sorted(ring))
            if sorted_ring not in seen:
                seen.add(sorted_ring)
                unique_rings.append(list(sorted_ring))

        if unique_rings:
            return unique_rings

    except Exception as e:
        logger.warning(f"RDKit芳香环识别失败，使用备选方案: {str(e)}")

    # 2. 备选方案：基于几何特征识别芳香环（键长均匀性+平面性）
    # 先通过共价键构建环（简单DFS）
    adj = defaultdict(list)
    for bond in covalent_bonds:
        adj[bond["u"]].append(bond["v"])
        adj[bond["v"]].append(bond["u"])

    visited = set()
    all_rings = []

    # DFS找环
    def dfs(start, current, path, depth):
        if depth > 2 and current == start:
            ring = sorted(path[:-1])
            if len(ring) >= 5 and ring not in all_rings:
                all_rings.append(ring)
            return
        if current in path[:-1]:  # 避免重复
            return
        if depth > 10:  # 限制环大小
            return
        for neighbor in adj[current]:
            dfs(start, neighbor, path + [neighbor], depth + 1)

    for i in range(len(atoms)):
        if i not in visited:
            dfs(i, i, [i], 0)

    # 过滤非芳香环（基于键长均匀性和平面性）
    aromatic_candidates = []
    for ring in all_rings:
        # 计算键长均匀性（芳香环键长差异小）
        bond_lengths = []
        valid_ring = True
        for j in range(len(ring)):
            u = ring[j]
            v = ring[(j + 1) % len(ring)]
            bond = next((b for b in covalent_bonds if (b["u"] == u and b["v"] == v) or (b["u"] == v and b["v"] == u)),
                        None)
            if not bond:
                valid_ring = False
                break
            bond_lengths.append(bond["distance"])
        if not valid_ring:
            continue

        # 键长标准差/均值 < 0.05（均匀性高）
        if np.std(bond_lengths) / np.mean(bond_lengths) > 0.05:
            continue

        # 平面性（芳香环接近平面）
        planarity = calculate_ring_planarity(ring, atoms)
        if planarity > 0.15:  # 平面性差（单位：Å）
            continue

        aromatic_candidates.append(ring)

    return aromatic_candidates


# ==================== 分子间相互作用识别 ====================
def find_hydrogen_bonds(atoms: List[Dict], cell: np.ndarray, mol_groups: List[List[int]], covalent_bonds: List[Dict]) -> \
        List[Dict]:
    """识别氢键（修正供体/受体识别和角度计算）"""
    h_bonds = []
    params = INTERMOLECULAR_PARAMS["hydrogen_bond"]
    coords = np.array([atom["position"] for atom in atoms])

    # 1. 识别所有氢键供体（O-H, N-H, S-H）
    donors = []  # (供体原子索引, 氢原子索引, 供体类型)
    for bond in covalent_bonds:
        if bond["subtype"] != "single":
            continue  # 仅单键可作为氢键供体
        sym_pair = bond["symbols"]
        u, v = bond["u"], bond["v"]
        sym_u, sym_v = atoms[u]["symbol"], atoms[v]["symbol"]

        # 判断供体-氢对
        if (sym_u == "H" and sym_v in {"O", "N", "S"}):
            donors.append((v, u, f"{sym_v}-H"))  # (供体原子, 氢原子, 类型)
        elif (sym_v == "H" and sym_u in {"O", "N", "S"}):
            donors.append((u, v, f"{sym_u}-H"))

    # 2. 识别所有氢键受体（O, N, S，需有孤对电子）
    acceptors = [i for i, atom in enumerate(atoms) if atom["symbol"] in params["acceptors"]]

    # 3. 筛选跨分子的供体-受体对
    for donor_idx, h_idx, donor_type in donors:
        # 供体所在分子组
        donor_group = next(g for g in mol_groups if donor_idx in g)

        # 遍历所有可能的受体
        for acceptor_idx in acceptors:
            # 跳过同一分子内的相互作用
            if acceptor_idx in donor_group:
                continue
            acceptor_sym = atoms[acceptor_idx]["symbol"]
            if acceptor_sym not in params["acceptors"]:
                continue

            # 4. 计算H-受体距离（考虑周期性）
            h_pos = coords[h_idx]
            acceptor_pos = coords[acceptor_idx]
            dist_h_acceptor = get_distances(
                h_pos.reshape(1, 3),
                acceptor_pos.reshape(1, 3),
                cell=cell,
                pbc=True
            )[1][0][0]

            # 距离过滤
            if not (params["distance_range"][0] <= dist_h_acceptor <= params["distance_range"][1]):
                continue

            # 5. 计算氢键角度（供体-H-受体）
            donor_pos = coords[donor_idx]
            # 向量：供体->H，H->受体
            vec_donor_h = h_pos - donor_pos
            vec_h_acceptor = acceptor_pos - h_pos
            # 单位向量
            vec_dh_norm = vec_donor_h / (np.linalg.norm(vec_donor_h) + 1e-8)
            vec_ha_norm = vec_h_acceptor / (np.linalg.norm(vec_h_acceptor) + 1e-8)
            # 角度计算（180度为理想氢键）
            angle = np.degrees(np.arccos(np.clip(np.dot(vec_dh_norm, vec_ha_norm), -1.0, 1.0)))

            # 角度过滤
            if not (params["angle_range"][0] <= angle <= params["angle_range"][1]):
                continue

            # 添加氢键
            h_bonds.append({
                "u": donor_idx,
                "v": acceptor_idx,
                "type": "hydrogen",
                "subtype": f"{donor_type}...{acceptor_sym}",
                "distance": dist_h_acceptor,
                "angle": angle,
                "h_idx": h_idx
            })

    # 限制每个原子的氢键数量（避免冗余）
    atom_hbond_count = defaultdict(int)
    filtered_hbonds = []
    for hb in h_bonds:
        if atom_hbond_count[hb["u"]] < 3 and atom_hbond_count[hb["v"]] < 3:
            filtered_hbonds.append(hb)
            atom_hbond_count[hb["u"]] += 1
            atom_hbond_count[hb["v"]] += 1

    return filtered_hbonds


def find_pi_pi_stacks(atoms: List[Dict], cell: np.ndarray, mol_groups: List[List[int]],
                      aromatic_rings: List[List[int]]) -> List[Dict]:
    """识别π-π堆积（修正环间距离/角度计算和分子过滤）"""
    stacks = []
    if len(aromatic_rings) < 2:
        return stacks  # 至少需要2个环
    params = INTERMOLECULAR_PARAMS["pi_pi_stack"]

    # 1. 预处理每个芳香环：计算中心、法向量、所属分子
    ring_info = []
    for ring in aromatic_rings:
        # 环原子坐标
        ring_coords = np.array([atoms[i]["position"] for i in ring])
        # 环中心
        center = np.mean(ring_coords, axis=0)
        # 环平面法向量（通过主成分分析）
        cov = np.cov(ring_coords - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # 最小特征值对应的 eigenvector 为法向量
        normal /= np.linalg.norm(normal) + 1e-8
        # 环所属分子组
        ring_mol_group = next(g for g in mol_groups if any(idx in g for idx in ring))
        ring_info.append({
            "indices": ring,
            "center": center,
            "normal": normal,
            "mol_group": ring_mol_group
        })

    # 2. 计算环对之间的相互作用
    for i in range(len(ring_info)):
        for j in range(i + 1, len(ring_info)):
            r1, r2 = ring_info[i], ring_info[j]

            # 过滤同一分子内的环
            if r1["mol_group"] == r2["mol_group"]:
                continue

            # 环间垂直距离（沿法向量方向）
            delta = r2["center"] - r1["center"]
            distance = np.abs(np.dot(delta, r1["normal"]))  # 垂直距离

            # 距离过滤
            if not (params["distance_range"][0] <= distance <= params["distance_range"][1]):
                continue

            # 环平面夹角（法向量夹角）
            angle = np.degrees(np.arccos(np.clip(np.dot(r1["normal"], r2["normal"]), -1.0, 1.0)))
            # 允许平行（0-30度）或垂直（T型，80-100度）
            if not ((params["angle_range"][0] <= angle <= params["angle_range"][1]) or
                    (80 <= angle <= 100)):
                continue

            # 添加π-π堆积作用
            stacks.append({
                "u": r1["indices"][0],  # 用环中第一个原子代表环
                "v": r2["indices"][0],
                "type": "pi_pi_stack",
                "distance": distance,
                "angle": angle,
                "ring1": r1["indices"],
                "ring2": r2["indices"],
                "stack_type": "parallel" if angle <= 30 else "T-shaped"
            })

    return stacks


def find_van_der_waals(atoms: List[Dict], cell: np.ndarray, mol_groups: List[List[int]]) -> List[Dict]:
    """识别范德华力，添加成键数量限制"""
    vdw_bonds = []
    params = INTERMOLECULAR_PARAMS["van_der_waals"]
    coords = np.array([atom["position"] for atom in atoms])
    n_atoms = len(atoms)

    # 记录每个原子的范德华键数
    vdw_bond_count = defaultdict(int)

    for i in range(n_atoms):
        # 跳过已经达到最大范德华键数的原子
        if vdw_bond_count[i] >= params["max_bonds"]:
            continue

        sym_i = atoms[i]["symbol"]
        if sym_i not in params["vdw_radius"]:
            continue

        for j in range(i + 1, n_atoms):
            # 跳过已经达到最大范德华键数的原子
            if vdw_bond_count[j] >= params["max_bonds"]:
                continue

            sym_j = atoms[j]["symbol"]
            if sym_j not in params["vdw_radius"]:
                continue

            # 检查是否在同一分子内
            if any(i in g and j in g for g in mol_groups):
                continue

            distance = get_distances(coords[i].reshape(1, 3), coords[j].reshape(1, 3), cell=cell, pbc=True)[1][0][0]
            vdw_sum = params["vdw_radius"][sym_i] + params["vdw_radius"][sym_j]

            if 1.0 < distance < params["distance_factor"] * vdw_sum:
                vdw_bonds.append({
                    "u": i, "v": j,
                    "type": "van_der_waals",
                    "distance": distance
                })
                vdw_bond_count[i] += 1
                vdw_bond_count[j] += 1

    return vdw_bonds


# ==================== 特征可视化 ====================
def visualize_atom_features(graph, cif_name):
    """可视化原子特征分布"""
    os.makedirs(FEATURE_VIS_DIR, exist_ok=True)
    node_feats = graph.ndata["feat"].numpy()

    # 创建画布
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 可视化原子序数特征
    axes[0].hist(node_feats[:, 0], bins=20, color='skyblue')
    axes[0].set_title('Atomic Number Feature')
    axes[0].set_xlabel('Normalized Value')
    axes[0].set_ylabel('Frequency')

    # 可视化坐标特征
    for i in range(3):
        axes[i + 1].hist(node_feats[:, i + 1], bins=20, color=f'C{i + 1}')
        axes[i + 1].set_title(f'Coordinate Feature {i + 1}')
        axes[i + 1].set_xlabel('Normalized Value')
        axes[i + 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_VIS_DIR, f"{cif_name}_atom_features.png"), dpi=300)
    plt.close()


def visualize_bond_features(graph, cif_name):
    """可视化键特征分布"""
    os.makedirs(FEATURE_VIS_DIR, exist_ok=True)

    if 'feat' not in graph.edata:
        logger.warning(f"{cif_name}: 图中没有边特征，无法生成键特征可视化")
        return

    edge_feats = graph.edata["feat"].numpy()

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 可视化距离特征
    axes[0].hist(edge_feats[:, 0], bins=20, color='orange')
    axes[0].set_title('Distance Feature')
    axes[0].set_xlabel('Normalized Value')
    axes[0].set_ylabel('Frequency')

    # 可视化角度特征
    axes[1].hist(edge_feats[:, 1], bins=20, color='green')
    axes[1].set_title('Angle Feature')
    axes[1].set_xlabel('Normalized Value')
    axes[1].set_ylabel('Frequency')

    # 可视化键类型特征
    axes[2].hist(edge_feats[:, 2], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], align='mid', rwidth=0.8)
    axes[2].set_title('Bond Type Feature')
    axes[2].set_xlabel('Bond Type')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xticks([0, 1, 2, 3])
    axes[2].set_xticklabels(['Covalent', 'Hydrogen', 'Pi-Pi', 'Van der Waals'])

    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_VIS_DIR, f"{cif_name}_bond_features.png"), dpi=300)
    plt.close()


# ==================== 图构建与可视化 ====================
def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """特征归一化"""
    range_val = max_val - min_val
    return (value - min_val) / range_val if range_val > 1e-6 else 0.0


def build_dgl_graph(atoms: List[Dict], interactions: List[Dict], cell: np.ndarray, cif_name: str) -> dgl.DGLGraph:
    """构建DGL图"""
    n_atoms = len(atoms)
    node_feats = []
    for atom in atoms:
        atomic_num = ATOMIC_NUMBERS.get(atom["symbol"], 0)
        feat_z = normalize_feature(atomic_num, GLOBAL_STATS["min_z"], GLOBAL_STATS["max_z"])
        feat_pos = [normalize_feature(coord, GLOBAL_STATS["coord_min"], GLOBAL_STATS["coord_max"]) for coord in
                    atom["position"]]
        node_feats.append([feat_z] + feat_pos)

    edge_types = {"covalent": 0, "hydrogen": 1, "pi_pi_stack": 2, "van_der_waals": 3}
    src, dst, edge_feats = [], [], []
    bond_stats = defaultdict(lambda: {"dists": [], "angles": []})
    for inter in interactions:
        bond_stats[inter["type"]]["dists"].append(inter["distance"])
        if "angle" in inter:
            bond_stats[inter["type"]]["angles"].append(inter["angle"])

    for inter in interactions:
        u, v = inter["u"], inter["v"]
        edge_type = edge_types[inter["type"]]
        dists = bond_stats[inter["type"]]["dists"]
        min_d, max_d = min(dists) if dists else 0.8, max(dists) if dists else 5.0
        feat_dist = normalize_feature(inter["distance"], min_d, max_d)
        feat_angle = normalize_feature(inter.get("angle", 0),
                                       min(bond_stats[inter["type"]]["angles"]) if bond_stats[inter["type"]][
                                           "angles"] else 0,
                                       max(bond_stats[inter["type"]]["angles"]) if bond_stats[inter["type"]][
                                           "angles"] else 180) if "angle" in inter else 0.0
        edge_feats.append([feat_dist, feat_angle, edge_type])
        src.append(u)
        dst.append(v)
        src.append(v)
        dst.append(u)
        edge_feats.append([feat_dist, feat_angle, edge_type])

    graph = dgl.graph((src, dst), num_nodes=n_atoms)
    graph.ndata["feat"] = torch.tensor(node_feats, dtype=torch.float32).to(DEVICE)

    if edge_feats:
        graph.edata["feat"] = torch.tensor(edge_feats, dtype=torch.float32).to(DEVICE)

    graph.graphdata = {
        'feat': torch.tensor(cell, dtype=torch.float32).to(DEVICE)
    }
    return graph


def visualize_graph(graph: dgl.DGLGraph, cif_name: str):
    """可视化单个图结构，根据元素类型使用不同颜色"""
    os.makedirs(GRAPH_VIS_DIR, exist_ok=True)

    # 将DGL图转换为networkx图
    nx_graph = graph.to_networkx()
    pos = nx.spring_layout(nx_graph)  # 使用networkx的spring_layout
    fig, ax = plt.subplots(figsize=(10, 8))

    # 根据边类型设置不同颜色
    edge_colors = []
    for u, v, data in nx_graph.edges(data=True):
        if 'feat' in data and len(data['feat']) > 2:
            edge_type = int(data['feat'][2])
            if edge_type == 0:  # 共价键
                edge_colors.append('black')
            elif edge_type == 1:  # 氢键
                edge_colors.append('blue')
            elif edge_type == 2:  # π-π堆积
                edge_colors.append('green')
            elif edge_type == 3:  # 范德华力
                edge_colors.append('red')
        else:
            edge_colors.append('gray')

    # 根据原子类型设置不同颜色
    node_colors = []
    # 假设节点特征的第一个值是归一化的原子序数
    node_feats = graph.ndata["feat"].numpy()
    for i in range(graph.num_nodes()):
        # 尝试从特征中获取原子序数并映射到元素符号
        atomic_num = int(
            round(node_feats[i, 0] * (GLOBAL_STATS["max_z"] - GLOBAL_STATS["min_z"]) + GLOBAL_STATS["min_z"]))
        element = next((k for k, v in ATOMIC_NUMBERS.items() if v == atomic_num), "default")
        color = ELEMENT_COLORS.get(element, ELEMENT_COLORS["default"])
        node_colors.append(color)

    # 绘制节点和边
    nx.draw(nx_graph, pos=pos, ax=ax, node_size=100,
            node_color=node_colors, edge_color=edge_colors,
            with_labels=False, alpha=0.8, width=0.8)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='k', label=element)
                       for element, color in ELEMENT_COLORS.items() if element != "default"]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title(f"Crystal Structure: {cif_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_VIS_DIR, f"{cif_name}_graph.png"), dpi=300)
    plt.close()


# ==================== 并行处理流程 ====================
from multiprocessing import JoinableQueue


def add_task(task_queue, cif_path):
    """添加任务到队列"""
    task_queue.put(cif_path)


def producer(task_queue: JoinableQueue, cif_files: List[str]):
    """生产者进程:放入待处理CIF文件,使用多线程"""
    pool = ThreadPool(psutil.cpu_count(logical=False))
    pool.starmap(add_task, [(task_queue, cif_path) for cif_path in cif_files])
    pool.close()
    pool.join()
    n_consumers = psutil.cpu_count(logical=False)
    for _ in range(n_consumers):
        task_queue.put(None)  # 结束标志


def consumer(task_queue: JoinableQueue, result_queue: Queue, progress_dict: Dict):
    """消费者进程:处理单个CIF文件"""
    while True:
        cif_path = task_queue.get()
        if cif_path is None:
            task_queue.task_done()
            break

        cif_name = os.path.basename(cif_path).split('.')[0]
        try:
            valid, msg = validate_cif(cif_path)
            if not valid:
                result_queue.put((cif_name, False, msg, None))
                task_queue.task_done()
                progress_dict[cif_name] = (False, msg)
                logger.error(f"{cif_name} 处理失败: {msg}")
                continue

            # 读取并处理CIF文件
            ase_atoms = read(cif_path, format="cif")
            logger.info(f"{cif_name}: 成功读取CIF文件")
            ase_atoms = remove_duplicate_atoms(ase_atoms)
            cell = ase_atoms.cell.array
            atoms = [{"symbol": sym, "position": pos.tolist()} for sym, pos in
                     zip(ase_atoms.symbols, ase_atoms.positions)]

            # 分析分子结构和相互作用
            mol_groups = get_molecular_groups(atoms, cell)
            logger.info(f"{cif_name}: 分子基团数量: {len(mol_groups)}")
            covalent_bonds = find_covalent_bonds(atoms, cell)
            logger.info(f"{cif_name}: 共价键数量: {len(covalent_bonds)}")
            hydrogen_bonds = find_hydrogen_bonds(atoms, cell, mol_groups, covalent_bonds)
            logger.info(f"{cif_name}: 氢键数量: {len(hydrogen_bonds)}")
            aromatic_rings = find_aromatic_rings(atoms, covalent_bonds)
            logger.info(f"{cif_name}: 芳香环数量: {len(aromatic_rings)}")
            pi_pi_stacks = find_pi_pi_stacks(atoms, cell, mol_groups, aromatic_rings)
            logger.info(f"{cif_name}: π - π 堆积数量: {len(pi_pi_stacks)}")
            van_der_waals = find_van_der_waals(atoms, cell, mol_groups)
            logger.info(f"{cif_name}: 范德华力数量: {len(van_der_waals)}")

            # 合并所有相互作用
            interactions = covalent_bonds + hydrogen_bonds + pi_pi_stacks + van_der_waals
            if not interactions:
                result_queue.put((cif_name, False, "无有效相互作用", None))
                task_queue.task_done()
                progress_dict[cif_name] = (False, "无有效相互作用")
                logger.error(f"{cif_name} 处理失败: 无有效相互作用")
                continue

            # 构建图并保存
            graph = build_dgl_graph(atoms, interactions, cell, cif_name)
            logger.info(f"{cif_name}: 成功构建图")

            # 创建保存目录
            os.makedirs(PROCESSED_GRAPH_DIR, exist_ok=True)
            graph_path = os.path.join(PROCESSED_GRAPH_DIR, f"{cif_name}.bin")
            dgl.save_graphs(graph_path, [graph])
            logger.info(f"{cif_name}: 成功保存图文件")

            # 可视化特征和图结构
            visualize_atom_features(graph, cif_name)
            visualize_bond_features(graph, cif_name)
            visualize_graph(graph, cif_name)

            result_queue.put((cif_name, True, "处理成功", graph))
            task_queue.task_done()
            progress_dict[cif_name] = (True, "处理成功")
            logger.info(f"{cif_name} 处理成功")

        except Exception as e:
            result_queue.put((cif_name, False, str(e), None))
            task_queue.task_done()
            progress_dict[cif_name] = (False, str(e))
            logger.error(f"{cif_name} 处理失败: {str(e)}")


def main():
    """主函数"""
    # 确保目录存在
    os.makedirs(RAW_CIF_DIR, exist_ok=True)
    os.makedirs(PROCESSED_GRAPH_DIR, exist_ok=True)
    os.makedirs(FEATURE_VIS_DIR, exist_ok=True)
    os.makedirs(GRAPH_VIS_DIR, exist_ok=True)

    # 获取所有CIF文件
    cif_files = [os.path.join(RAW_CIF_DIR, f) for f in os.listdir(RAW_CIF_DIR)
                 if f.lower().endswith('.cif')]

    if not cif_files:
        logger.warning("未找到CIF文件")
        return

    logger.info(f"找到 {len(cif_files)} 个CIF文件")

    # 创建进程间通信队列
    task_queue = JoinableQueue()
    result_queue = Queue()

    # 用于跟踪进度的共享字典
    with Manager() as manager:
        progress_dict = manager.dict()

        # 启动生产者进程
        producer_process = Process(target=producer, args=(task_queue, cif_files))
        producer_process.start()

        # 启动消费者进程
        n_consumers = psutil.cpu_count(logical=False)
        consumer_processes = []
        for _ in range(n_consumers):
            p = Process(target=consumer, args=(task_queue, result_queue, progress_dict))
            p.start()
            consumer_processes.append(p)

        # 等待生产者完成
        producer_process.join()

        # 等待所有任务完成（添加进度条）
        total = len(cif_files)
        with tqdm(total=total, desc="处理进度") as pbar:
            processed = 0
            while processed < total:
                # 检查已处理的数量
                current_processed = len(progress_dict)
                if current_processed > processed:
                    pbar.update(current_processed - processed)
                    processed = current_processed
                time.sleep(0.5)  # 避免频繁检查

        # 终止消费者进程
        for p in consumer_processes:
            p.terminate()
            p.join()

        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # 统计结果
        success_count = sum(1 for _, success, _, _ in results if success)
        failed_count = len(results) - success_count

        logger.info(f"处理完成: 成功 {success_count}, 失败 {failed_count}")

        # 输出失败的文件
        if failed_count > 0:
            logger.info("失败的文件:")
            for name, success, msg, _ in results:
                if not success:
                    logger.info(f"  {name}: {msg}")


if __name__ == "__main__":
    main()