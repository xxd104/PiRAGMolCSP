import os
import re
import dgl
import numpy as np
import torch
import warnings
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, Counter
from multiprocessing import Pool, Manager, cpu_count, set_start_method
from scipy.spatial import KDTree
from scipy.linalg import norm
from ase.io import read
from ase.geometry import get_distances
from ase import Atoms

# ==================== 全局配置 ====================
# 目录配置
RAW_CIF_DIR = "/home/nyx/N-RGAG/raw_cifs"
DGL_GRAPH_DIR = "/home/nyx/N-RGAG/dgl_graphs"
GRAPH_VIS_DIR = "/home/nyx/N-RGAG/graph_vis"
GRAPH_DIS_DIR = "/home/nyx/N-RGAG/graph_dis"
# 化学参数（通用型，基于IUPAC数据）
# 共价半径 (Å) - 通用单键共价半径
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
    'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
    'Si': 1.11, 'B': 0.84, 'Li': 1.28, 'Na': 1.66, 'K': 2.03
}

# 范德华半径 (Å)
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'P': 1.80,
    'Si': 2.10, 'B': 1.92, 'Li': 1.82, 'Na': 2.27, 'K': 2.75
}

# 键类型映射（用于编码）
BOND_TYPE_MAP = {
    'single': 0, 'double': 1, 'triple': 2, 'aromatic': 3, 'ring': 4,
    'hydrogen': 5, 'pi_pi_stack': 6, 'van_der_waals': 7
}

# 键长比例阈值（相对于单键共价半径和）- 通用判断标准
BOND_LENGTH_THRESHOLDS = {
    'triple': (0.70, 0.85),  # 三键：0.70-0.85倍单键半径和
    'double': (0.85, 1.00),  # 双键：0.85-1.00倍单键半径和
    'aromatic': (0.95, 1.05),  # 芳香键：0.95-1.05倍单键半径和
    'single': (1.00, 1.20),  # 单键：1.00-1.20倍单键半径和
    'ring': (1.00, 1.20)  # 环键：同单键范围，额外通过环结构判断
}

# 分子间作用力参数
INTERMOLECULAR_PARAMS = {
    'hydrogen_bond': {
        'donor_elements': {'N', 'O', 'S'},
        'acceptor_elements': {'N', 'O', 'S'},
        'h_distance': (1.5, 3.5),  # H到受体距离
        'angle': (120, 180)  # 供体-H-受体角度
    },
    'pi_pi_stack': {
        'distance': (3.0, 4.5),  # 环平面间距
        'angle': (0, 30)  # 平面夹角
    },
    'van_der_waals': {
        'factor': 1.2  # 范德华半径和的1.2倍为上限
    }
}

# 元素颜色映射（CPK配色）
ELEMENT_COLORS = {
    'H': 'white', 'C': '#333333', 'N': '#3050F8', 'O': '#FF0D0D',
    'S': '#FFFF30', 'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929',
    'I': '#940094', 'P': '#FF8000', 'Si': '#FAC000', 'B': '#909090',
    'default': '#808080'
}

# 原子序数映射
ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'F': 9, 'Cl': 17,
    'Br': 35, 'I': 53, 'P': 15, 'Si': 14, 'B': 5, 'Li': 3, 'Na': 11, 'K': 19
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("cif2dgl.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 过滤无关警告
warnings.filterwarnings("ignore")


# ==================== 工具函数 ====================
def parse_cif_file(cif_path):
    """
    解析CIF文件，提取所有需要的信息
    返回：字典包含晶胞参数、原子信息、受力、应力张量、总能量
    """
    parsed_data = {
        'cell_params': {},  # 晶胞参数：a,b,c,alpha,beta,gamma
        'atoms': [],  # 原子信息：symbol, label, position, occupancy
        'forces': [],  # 原子受力：按原子顺序的(x,y,z)
        'stress_tensor': [],  # 应力张量：3x3矩阵
        'total_energy': {}  # 总能量：hartree, eV
    }

    # 先用ASE读取基础结构
    try:
        ase_atoms = read(cif_path, format='cif')
        parsed_data['cell_params'] = {
            'a': ase_atoms.cell[0][0],
            'b': ase_atoms.cell[1][1],
            'c': ase_atoms.cell[2][2],
            'alpha': np.degrees(np.arccos(np.clip(
                np.dot(ase_atoms.cell[1], ase_atoms.cell[2]) /
                (norm(ase_atoms.cell[1]) * norm(ase_atoms.cell[2])), -1, 1))),
            'beta': np.degrees(np.arccos(np.clip(
                np.dot(ase_atoms.cell[0], ase_atoms.cell[2]) /
                (norm(ase_atoms.cell[0]) * norm(ase_atoms.cell[2])), -1, 1))),
            'gamma': np.degrees(np.arccos(np.clip(
                np.dot(ase_atoms.cell[0], ase_atoms.cell[1]) /
                (norm(ase_atoms.cell[0]) * norm(ase_atoms.cell[1])), -1, 1)))
        }

        # 提取原子基础信息
        for atom in ase_atoms:
            parsed_data['atoms'].append({
                'symbol': atom.symbol,
                'position': atom.position.tolist(),
                'atomic_number': ATOMIC_NUMBERS.get(atom.symbol, 0)
            })
    except Exception as e:
        logger.error(f"读取CIF结构失败 {cif_path}: {e}")
        return None

    # 手动解析DFTB+计算结果（受力、应力、能量）
    with open(cif_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    force_pattern = re.compile(r'Force_(\d+):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)')
    stress_pattern = re.compile(r'Stress_Row_(\d+):\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)')
    energy_pattern_hartree = re.compile(r'Total Energy \(Hartree\):\s*([-\d.]+)')
    energy_pattern_ev = re.compile(r'Total Energy \(eV\):\s*([-\d.]+)')

    forces = {}
    stress_rows = {}

    for line in lines:
        # 提取受力
        force_match = force_pattern.search(line)
        if force_match:
            idx = int(force_match.group(1)) - 1  # 转换为0索引
            fx = float(force_match.group(2))
            fy = float(force_match.group(3))
            fz = float(force_match.group(4))
            forces[idx] = [fx, fy, fz]

        # 提取应力张量
        stress_match = stress_pattern.search(line)
        if stress_match:
            row = int(stress_match.group(1)) - 1
            s1 = float(stress_match.group(2))
            s2 = float(stress_match.group(3))
            s3 = float(stress_match.group(4))
            stress_rows[row] = [s1, s2, s3]

        # 提取总能量（Hartree）
        energy_h_match = energy_pattern_hartree.search(line)
        if energy_h_match:
            parsed_data['total_energy']['hartree'] = float(energy_h_match.group(1))

        # 提取总能量（eV）
        energy_ev_match = energy_pattern_ev.search(line)
        if energy_ev_match:
            parsed_data['total_energy']['eV'] = float(energy_ev_match.group(1))

    # 按原子顺序整理受力（确保顺序一致）
    parsed_data['forces'] = [forces.get(i, [0.0, 0.0, 0.0]) for i in range(len(parsed_data['atoms']))]

    # 整理应力张量（3x3）
    parsed_data['stress_tensor'] = [stress_rows.get(i, [0.0, 0.0, 0.0]) for i in range(3)]

    return parsed_data


def find_covalent_bonds(atoms, cell):
    """
    通用方法识别共价键（基于共价半径，非硬编码键长）
    参数：
        atoms: 原子列表，包含symbol和position
        cell: 晶胞矩阵（aseAtoms.cell.array）
    返回：
        bonds: 共价键列表，包含u/v索引、类型、距离等
    """
    bonds = []
    n_atoms = len(atoms)
    coords = np.array([atom['position'] for atom in atoms])

    # 构建KDTree加速距离计算
    kdtree = KDTree(coords)

    for i in range(n_atoms):
        sym_i = atoms[i]['symbol']
        if sym_i not in COVALENT_RADII:
            continue

        # 获取i原子的共价半径
        r_i = COVALENT_RADII[sym_i]

        # 搜索可能成键的原子（1.2倍最大共价半径和范围内）
        max_search_radius = 1.2 * (r_i + max(COVALENT_RADII.values()))
        neighbors = kdtree.query_ball_point(coords[i], max_search_radius)

        for j in neighbors:
            if j <= i:  # 避免重复
                continue

            sym_j = atoms[j]['symbol']
            if sym_j not in COVALENT_RADII:
                continue

            # 计算实际距离（考虑周期性）
            distance = get_distances(
                coords[i].reshape(1, 3),
                coords[j].reshape(1, 3),
                cell=cell,
                pbc=True
            )[1][0][0]

            # 计算单键共价半径和
            r_j = COVALENT_RADII[sym_j]
            r_sum = r_i + r_j

            # 判断键类型
            bond_type = None
            distance_ratio = distance / r_sum

            # 按键型阈值判断
            for btype, (min_ratio, max_ratio) in BOND_LENGTH_THRESHOLDS.items():
                if min_ratio <= distance_ratio <= max_ratio:
                    bond_type = btype
                    break

            if bond_type:
                bonds.append({
                    'u': i,
                    'v': j,
                    'type': bond_type,
                    'distance': distance,
                    'distance_ratio': distance_ratio,
                    'symbols': f"{sym_i}-{sym_j}"
                })

    # 识别环键（在共价键基础上，通过闭环结构判断）
    ring_bonds = identify_ring_bonds(bonds, n_atoms)
    for bond in bonds:
        if any(rb['u'] == bond['u'] and rb['v'] == bond['v'] for rb in ring_bonds):
            bond['type'] = 'ring'

    return bonds


def identify_ring_bonds(covalent_bonds, n_atoms):
    """
    识别环键（基于共价键的闭环结构）
    """
    # 构建邻接表
    adj = defaultdict(list)
    for bond in covalent_bonds:
        adj[bond['u']].append(bond['v'])
        adj[bond['v']].append(bond['u'])

    ring_bonds = []
    visited = set()

    # DFS查找环
    def dfs(start, current, path):
        if current == start and len(path) >= 4:  # 至少4个原子形成环
            # 提取环中的键
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if u < v:  # 避免重复
                    ring_bonds.append({'u': u, 'v': v})
            return

        if len(path) > 10:  # 限制环大小
            return

        for neighbor in adj[current]:
            if neighbor not in path or (neighbor == start and len(path) >= 4):
                dfs(start, neighbor, path + [neighbor])

    for i in range(n_atoms):
        if i not in visited:
            dfs(i, i, [i])
            visited.add(i)

    return ring_bonds


def find_hydrogen_bonds(atoms, cell, covalent_bonds):
    """
    识别氢键（通用方法）
    """
    h_bonds = []
    coords = np.array([atom['position'] for atom in atoms])
    params = INTERMOLECULAR_PARAMS['hydrogen_bond']

    # 1. 识别氢键供体（N-H/O-H/S-H）
    donors = []  # (供体原子索引, H原子索引)
    for bond in covalent_bonds:
        if bond['type'] != 'single':
            continue
        u, v = bond['u'], bond['v']
        sym_u, sym_v = atoms[u]['symbol'], atoms[v]['symbol']
        if sym_u == 'H' and sym_v in params['donor_elements']:
            donors.append((v, u))
        elif sym_v == 'H' and sym_u in params['donor_elements']:
            donors.append((u, v))

    # 2. 识别氢键受体（N/O/S）
    acceptors = [i for i, atom in enumerate(atoms) if atom['symbol'] in params['acceptor_elements']]

    # 3. 筛选有效氢键
    for donor_idx, h_idx in donors:
        if h_idx not in range(len(atoms)) or donor_idx not in range(len(atoms)):
            continue
        # 供体和H的坐标
        donor_pos = np.array(atoms[donor_idx]['position'])
        h_pos = np.array(atoms[h_idx]['position'])

        for acceptor_idx in acceptors:
            if acceptor_idx == donor_idx or acceptor_idx == h_idx:
                continue
            acceptor_pos = np.array(atoms[acceptor_idx]['position'])

            # 计算H到受体的距离（考虑周期性）
            h_acceptor_dist = get_distances(
                h_pos.reshape(1, 3),
                acceptor_pos.reshape(1, 3),
                cell=cell,
                pbc=True
            )[1][0][0]

            # 距离过滤
            if not (params['h_distance'][0] <= h_acceptor_dist <= params['h_distance'][1]):
                continue

            # 计算供体-H-受体角度
            vec_donor_h = h_pos - donor_pos
            vec_h_acceptor = acceptor_pos - h_pos

            # 单位向量
            vec_dh_norm = vec_donor_h / (norm(vec_donor_h) + 1e-8)
            vec_ha_norm = vec_h_acceptor / (norm(vec_h_acceptor) + 1e-8)

            # 计算角度（度）
            angle = np.degrees(np.arccos(np.clip(np.dot(vec_dh_norm, vec_ha_norm), -1, 1)))

            # 角度过滤
            if params['angle'][0] <= angle <= params['angle'][1]:
                h_bonds.append({
                    'u': donor_idx,
                    'v': acceptor_idx,
                    'type': 'hydrogen',
                    'distance': h_acceptor_dist,
                    'angle': angle,
                    'h_idx': h_idx
                })

    return h_bonds


def find_pi_pi_stacks(atoms, cell, covalent_bonds):
    """
    识别π-π堆积作用
    """
    stacks = []
    params = INTERMOLECULAR_PARAMS['pi_pi_stack']

    # 1. 先识别芳香环
    aromatic_rings = find_aromatic_rings(atoms, covalent_bonds)
    if len(aromatic_rings) < 2:
        return stacks

    # 2. 计算每个环的中心和法向量
    ring_info = []
    for ring in aromatic_rings:
        ring_coords = np.array([atoms[i]['position'] for i in ring])
        center = np.mean(ring_coords, axis=0)

        # 计算环平面法向量（PCA）
        cov = np.cov(ring_coords - center, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, np.argmin(eig_vals)]
        normal /= norm(normal) + 1e-8

        ring_info.append({
            'indices': ring,
            'center': center,
            'normal': normal
        })

    # 3. 计算环间相互作用
    for i in range(len(ring_info)):
        for j in range(i + 1, len(ring_info)):
            r1, r2 = ring_info[i], ring_info[j]

            # 计算环中心距离（考虑周期性）
            dist = get_distances(
                r1['center'].reshape(1, 3),
                r2['center'].reshape(1, 3),
                cell=cell,
                pbc=True
            )[1][0][0]

            # 计算环平面垂直距离
            plane_dist = np.abs(np.dot(r2['center'] - r1['center'], r1['normal']))

            # 计算平面夹角
            angle = np.degrees(np.arccos(np.clip(np.dot(r1['normal'], r2['normal']), -1, 1)))

            # 过滤条件
            if (params['distance'][0] <= plane_dist <= params['distance'][1] and
                    params['angle'][0] <= angle <= params['angle'][1]):
                stacks.append({
                    'u': r1['indices'][0],
                    'v': r2['indices'][0],
                    'type': 'pi_pi_stack',
                    'distance': plane_dist,
                    'angle': angle,
                    'ring1': r1['indices'],
                    'ring2': r2['indices']
                })

    return stacks


def find_aromatic_rings(atoms, covalent_bonds):
    """
    识别芳香环（基于几何特征）
    """
    # 先获取所有环结构
    adj = defaultdict(list)
    for bond in covalent_bonds:
        adj[bond['u']].append(bond['v'])
        adj[bond['v']].append(bond['u'])

    all_rings = []
    visited = set()

    # DFS找环
    def dfs(start, current, path):
        if current == start and len(path) >= 5:  # 芳香环至少5元环
            ring = sorted(path[:-1])
            if ring not in all_rings:
                all_rings.append(ring)
            return
        if len(path) > 10:
            return
        for neighbor in adj[current]:
            if neighbor not in path or (neighbor == start and len(path) >= 5):
                dfs(start, neighbor, path + [neighbor])

    for i in range(len(atoms)):
        if i not in visited:
            dfs(i, i, [i])
            visited.add(i)

    # 筛选芳香环（键长均匀+平面性好）
    aromatic_rings = []
    for ring in all_rings:
        # 计算环内键长
        ring_bond_lengths = []
        for j in range(len(ring)):
            u = ring[j]
            v = ring[(j + 1) % len(ring)]
            # 查找u-v之间的键
            bond = next((b for b in covalent_bonds if
                         (b['u'] == u and b['v'] == v) or (b['u'] == v and b['v'] == u)), None)
            if bond:
                ring_bond_lengths.append(bond['distance'])

        if not ring_bond_lengths:
            continue

        # 键长均匀性（标准差/均值 < 0.05）
        length_std = np.std(ring_bond_lengths) / np.mean(ring_bond_lengths)
        if length_std > 0.05:
            continue

        # 平面性（所有原子到环平面的平均距离 < 0.15 Å）
        ring_coords = np.array([atoms[i]['position'] for i in ring])
        centroid = np.mean(ring_coords, axis=0)
        cov = np.cov(ring_coords - centroid, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, np.argmin(eig_vals)]
        plane_distances = np.abs(np.dot(ring_coords - centroid, normal))
        if np.mean(plane_distances) < 0.15:
            aromatic_rings.append(ring)

    return aromatic_rings


def find_van_der_waals(atoms, cell, covalent_bonds):
    """
    识别范德华力（通用方法）
    """
    vdw_bonds = []
    coords = np.array([atom['position'] for atom in atoms])
    params = INTERMOLECULAR_PARAMS['van_der_waals']

    # 先构建共价键邻接表（排除共价键连接的原子）
    covalent_adj = defaultdict(set)
    for bond in covalent_bonds:
        covalent_adj[bond['u']].add(bond['v'])
        covalent_adj[bond['v']].add(bond['u'])

    # KDTree加速搜索
    kdtree = KDTree(coords)

    for i in range(len(atoms)):
        sym_i = atoms[i]['symbol']
        if sym_i not in VDW_RADII:
            continue

        r_i = VDW_RADII[sym_i]
        max_search_radius = params['factor'] * (r_i + max(VDW_RADII.values()))
        neighbors = kdtree.query_ball_point(coords[i], max_search_radius)

        for j in neighbors:
            if j <= i or j in covalent_adj[i]:
                continue

            sym_j = atoms[j]['symbol']
            if sym_j not in VDW_RADII:
                continue

            r_j = VDW_RADII[sym_j]
            vdw_sum = r_i + r_j

            # 计算实际距离（考虑周期性）
            distance = get_distances(
                coords[i].reshape(1, 3),
                coords[j].reshape(1, 3),
                cell=cell,
                pbc=True
            )[1][0][0]

            if distance < params['factor'] * vdw_sum:
                vdw_bonds.append({
                    'u': i,
                    'v': j,
                    'type': 'van_der_waals',
                    'distance': distance
                })

    return vdw_bonds


def build_dgl_graph(parsed_data, cell, all_interactions):
    """
    构建DGL图
    """
    atoms = parsed_data['atoms']
    n_atoms = len(atoms)

    DEVICE = torch.device("cpu")  # 指定设备

    # 1. 构建节点特征：原子序数 + 受力（归一化）
    node_features = []
    forces = parsed_data['forces']

    # 归一化受力（全局最大最小值）
    all_forces = np.array(forces).flatten()
    force_min, force_max = all_forces.min(), all_forces.max()
    # 添加epsilon避免除以零
    force_range = force_max - force_min if force_max != force_min else 1e-8

    for i in range(n_atoms):
        # 原子序数（归一化到0-1）
        atomic_num = ATOMIC_NUMBERS.get(atoms[i]['symbol'], 0)
        norm_atomic_num = atomic_num / max(ATOMIC_NUMBERS.values())

        # 受力（归一化到0-1）
        fx, fy, fz = forces[i]
        norm_fx = (fx - force_min) / force_range
        norm_fy = (fy - force_min) / force_range
        norm_fz = (fz - force_min) / force_range

        node_features.append([norm_atomic_num, norm_fx, norm_fy, norm_fz])

    # 2. 构建边特征
    src_nodes = []
    dst_nodes = []
    edge_features = []

    # 防御性检查：空交互列表
    if not all_interactions:
        logger.warning("无相互作用，返回空图")
        dgl_graph = dgl.graph(([], []), num_nodes=n_atoms)
        dgl_graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
    else:
        # 归一化距离（基于所有相互作用的距离）
        all_distances = [inter['distance'] for inter in all_interactions]
        dist_min, dist_max = min(all_distances), max(all_distances)
        # 添加epsilon避免除以零
        dist_range = dist_max - dist_min if dist_max != dist_min else 1e-8

        for inter in all_interactions:
            u, v = inter['u'], inter['v']
            bond_type = BOND_TYPE_MAP[inter['type']]

            # 归一化距离
            norm_dist = (inter['distance'] - dist_min) / dist_range

            # 角度（无则为0）
            angle = inter.get('angle', 0.0)
            norm_angle = angle / 180.0  # 归一化到0-1

            # 边特征：归一化距离 + 归一化角度 + 键类型
            edge_feat = [norm_dist, norm_angle, bond_type]

            # 添加边（无向图，双向）
            src_nodes.extend([u, v])
            dst_nodes.extend([v, u])
            edge_features.extend([edge_feat, edge_feat])

        # 3. 创建DGL图
        dgl_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=n_atoms)
        dgl_graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
        dgl_graph.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32).to(DEVICE)

    # ========== 将图级属性存储到ndata中 ==========
    # 晶胞参数：a,b,c,alpha,beta,gamma
    cell_params = [
        parsed_data['cell_params']['a'],
        parsed_data['cell_params']['b'],
        parsed_data['cell_params']['c'],
        parsed_data['cell_params']['alpha'],
        parsed_data['cell_params']['beta'],
        parsed_data['cell_params']['gamma']
    ]

    # 应力张量（展平为9维）
    stress_flat = [val for row in parsed_data['stress_tensor'] for val in row]

    # 总能量（hartree和eV）
    energy = [
        parsed_data['total_energy'].get('hartree', 0.0),
        parsed_data['total_energy'].get('eV', 0.0)
    ]

    # 组合图特征
    graph_feat = cell_params + stress_flat + energy

    # 所有节点共享相同的图级属性（形状：[n_atoms, len(graph_feat)]）
    graph_attr_tensor = torch.tensor(graph_feat, dtype=torch.float32).to(DEVICE)
    dgl_graph.ndata['graph_attr'] = graph_attr_tensor.unsqueeze(0).repeat(n_atoms, 1)

    # 保存原始属性结构（方便访问）
    dgl_graph.ndata['cell_params'] = torch.tensor(cell_params, dtype=torch.float32).to(DEVICE).unsqueeze(0).repeat(
        n_atoms, 1)
    dgl_graph.ndata['stress_tensor_flat'] = torch.tensor(stress_flat, dtype=torch.float32).to(DEVICE).unsqueeze(
        0).repeat(n_atoms, 1)
    dgl_graph.ndata['total_energy'] = torch.tensor(energy, dtype=torch.float32).to(DEVICE).unsqueeze(0).repeat(n_atoms,
                                                                                                               1)

    return dgl_graph


# ========== 处理batch/unbatch后的图级属性 ==========
def extract_graph_attrs_from_batch(batched_graph):
    """
    从批量图中提取每个子图的图级属性
    参数：
        batched_graph: dgl.batch后的批量图
    返回：
        graph_attrs_list: 每个子图的图级属性列表（与unbatch后的子图一一对应）
    """
    # 获取每个子图的节点数
    batch_num_nodes = batched_graph.batch_num_nodes()
    graph_attrs_list = []

    # 按子图分割节点属性
    start_idx = 0
    for n_nodes in batch_num_nodes:
        end_idx = start_idx + n_nodes
        # 取第一个节点的图级属性即可（所有节点共享）
        graph_attr = batched_graph.ndata['graph_attr'][start_idx]
        cell_params = batched_graph.ndata['cell_params'][start_idx]
        stress_flat = batched_graph.ndata['stress_tensor_flat'][start_idx]
        total_energy = batched_graph.ndata['total_energy'][start_idx]

        # 恢复应力张量为3x3
        stress_tensor = stress_flat.reshape(3, 3)

        graph_attrs_list.append({
            'cell_params': cell_params,
            'stress_tensor': stress_tensor,
            'total_energy': total_energy,
            'all_feat': graph_attr
        })
        start_idx = end_idx

    return graph_attrs_list


def restore_graph_attrs_to_subgraphs(subgraphs, graph_attrs_list):
    """
    给unbatch后的子图恢复便捷的属性访问方式
    参数：
        subgraphs: dgl.unbatch后的子图列表
        graph_attrs_list: extract_graph_attrs_from_batch返回的属性列表
    """
    for subgraph, attrs in zip(subgraphs, graph_attrs_list):
        subgraph.cell_params = attrs['cell_params']
        subgraph.stress_tensor = attrs['stress_tensor']
        subgraph.total_energy = attrs['total_energy']
        subgraph.graph_feat = attrs['all_feat']
    return subgraphs


def visualize_dgl_graph(dgl_graph, atoms, cif_name):
    """
    可视化DGL图并保存为PNG
    """
    os.makedirs(GRAPH_VIS_DIR, exist_ok=True)

    nx_graph = dgl_graph.to_networkx(edge_attrs=['feat'])

    # 生成布局
    pos = nx.spring_layout(nx_graph, seed=42)

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))

    # 节点颜色（按元素）
    node_colors = []
    for i in range(len(atoms)):
        sym = atoms[i]['symbol']
        node_colors.append(ELEMENT_COLORS.get(sym, ELEMENT_COLORS['default']))

    # 边颜色（按类型）
    edge_colors = []
    edge_types = {
        0: 'black',  # single
        1: 'red',  # double
        2: 'blue',  # triple
        3: 'green',  # aromatic
        4: 'orange',  # ring
        5: 'cyan',  # hydrogen
        6: 'magenta',  # pi_pi_stack
        7: 'gray'  # van_der_waals
    }

    for u, v, data in nx_graph.edges(data=True):
        bond_type = int(data['feat'][2]) if len(data['feat']) >= 3 else 0
        edge_colors.append(edge_types.get(bond_type, 'gray'))

    # 绘制图
    nx.draw(
        nx_graph, pos, ax=ax,
        node_size=200,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=False,
        alpha=0.8,
        width=1.0
    )

    # 添加图例
    legend_elements = [
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                          for label, color in ELEMENT_COLORS.items() if label != 'default'
                      ] + [
                          plt.Line2D([0], [0], color=color, lw=2, label=label)
                          for label, color in {
            'Single Bond': 'black', 'Double Bond': 'red', 'Triple Bond': 'blue',
            'Aromatic Bond': 'green', 'Ring Bond': 'orange', 'Hydrogen Bond': 'cyan',
            'π-π Stack': 'magenta', 'Van der Waals': 'gray'
        }.items()
                      ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title(f'Molecular Graph: {cif_name}', fontsize=14)

    # 保存图片
    plt.tight_layout()
    save_path = os.path.join(GRAPH_VIS_DIR, f'{cif_name}_graph.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_single_cif(cif_path, stats_dict):
    """
    处理单个CIF文件（供多进程调用）
    修改点：适配新的图级属性存储方式，简化保存逻辑
    """
    cif_name = os.path.basename(cif_path).split('.')[0]

    try:
        # 1. 解析CIF文件
        parsed_data = parse_cif_file(cif_path)
        if not parsed_data:
            logger.error(f"解析失败: {cif_name}")
            return False

        # 2. 读取ASE原子对象
        ase_atoms = read(cif_path, format='cif')
        cell = ase_atoms.cell.array

        # 3. 识别所有相互作用
        # 共价键
        covalent_bonds = find_covalent_bonds(parsed_data['atoms'], cell)
        # 氢键
        hydrogen_bonds = find_hydrogen_bonds(parsed_data['atoms'], cell, covalent_bonds)
        # π-π堆积
        pi_pi_stacks = find_pi_pi_stacks(parsed_data['atoms'], cell, covalent_bonds)
        # 范德华力
        van_der_waals = find_van_der_waals(parsed_data['atoms'], cell, covalent_bonds)

        # 合并所有相互作用
        all_interactions = covalent_bonds + hydrogen_bonds + pi_pi_stacks + van_der_waals

        # 4. 构建DGL图
        dgl_graph = build_dgl_graph(parsed_data, cell, all_interactions)

        # 5. 保存DGL图
        os.makedirs(DGL_GRAPH_DIR, exist_ok=True)
        graph_save_path = os.path.join(DGL_GRAPH_DIR, f'{cif_name}.bin')

        # 修改点：无需单独保存graph_attrs，直接保存图即可（属性在ndata中）
        torch.save(dgl_graph, graph_save_path)

        # 6. 可视化图
        if all_interactions:
            visualize_dgl_graph(dgl_graph, parsed_data['atoms'], cif_name)

        # 7. 更新统计信息
        # 元素分布
        element_counter = stats_dict['element_dist']
        elements = [atom['symbol'] for atom in parsed_data['atoms']]
        for elem, count in Counter(elements).items():
            element_counter[elem] = element_counter.get(elem, 0) + count

        # 键类型分布
        bond_counter = stats_dict['bond_dist']
        for inter in all_interactions:
            bond_counter[inter['type']] = bond_counter.get(inter['type'], 0) + 1

        # 能量分布
        energy_list = stats_dict['energy_dist']
        energy_ev = parsed_data['total_energy'].get('eV', 0.0)
        energy_list.append(energy_ev)

        # 分子间作用力分布
        intermolecular_counter = stats_dict['intermolecular_dist']
        for inter in hydrogen_bonds + pi_pi_stacks + van_der_waals:
            intermolecular_counter[inter['type']] = intermolecular_counter.get(inter['type'], 0) + 1

        logger.info(f"处理成功: {cif_name}")
        return True

    except Exception as e:
        logger.error(f"处理失败 {cif_name}: {str(e)}")
        return False


def generate_distribution_plots(stats_dict):
    """
    生成各类分布图
    """
    os.makedirs(GRAPH_DIS_DIR, exist_ok=True)

    # 1. 元素分布图（饼图）
    element_counter = dict(stats_dict['element_dist'])
    if element_counter:
        fig, ax = plt.subplots(figsize=(10, 8))
        elements = list(element_counter.keys())
        counts = list(element_counter.values())
        colors = [ELEMENT_COLORS.get(elem, ELEMENT_COLORS['default']) for elem in elements]

        ax.pie(counts, labels=elements, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Element Distribution', fontsize=14)
        plt.savefig(os.path.join(GRAPH_DIS_DIR, 'element_distribution.png'), dpi=300)
        plt.close()

    # 2. 键类型分布图（柱状图）
    bond_counter = dict(stats_dict['bond_dist'])
    if bond_counter:
        fig, ax = plt.subplots(figsize=(12, 6))
        bond_types = list(bond_counter.keys())
        counts = list(bond_counter.values())

        ax.bar(bond_types, counts, color=['black', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta', 'gray'])
        ax.set_title('Bond Type Distribution', fontsize=14)
        ax.set_xlabel('Bond Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIS_DIR, 'bond_type_distribution.png'), dpi=300)
        plt.close()

    # 3. 能量分布图（直方图）
    energy_list = list(stats_dict['energy_dist'])
    if energy_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(energy_list, bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Total Energy Distribution (eV)', fontsize=14)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIS_DIR, 'energy_distribution.png'), dpi=300)
        plt.close()

    # 4. 分子间作用力分布图（柱状图）
    intermolecular_counter = dict(stats_dict['intermolecular_dist'])
    if intermolecular_counter:
        fig, ax = plt.subplots(figsize=(10, 6))
        force_types = list(intermolecular_counter.keys())
        counts = list(intermolecular_counter.values())
        colors = ['cyan', 'magenta', 'gray']

        ax.bar(force_types, counts, color=colors[:len(force_types)])
        ax.set_title('Intermolecular Force Distribution', fontsize=14)
        ax.set_xlabel('Force Type')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIS_DIR, 'intermolecular_force_distribution.png'), dpi=300)
        plt.close()


def main():
    """
    主函数：批量处理CIF文件
    """
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # 1. 准备工作
    # 检查输入目录
    if not os.path.exists(RAW_CIF_DIR):
        logger.error(f"输入目录不存在: {RAW_CIF_DIR}")
        return

    # 获取所有CIF文件
    cif_files = [os.path.join(RAW_CIF_DIR, f) for f in os.listdir(RAW_CIF_DIR)
                 if f.lower().endswith('.cif')]

    if not cif_files:
        logger.warning("未找到CIF文件")
        return

    logger.info(f"找到 {len(cif_files)} 个CIF文件")

    # 2. 初始化共享统计字典
    manager = Manager()
    stats_dict = manager.dict()
    stats_dict['element_dist'] = manager.dict()
    stats_dict['bond_dist'] = manager.dict()
    stats_dict['energy_dist'] = manager.list()
    stats_dict['intermolecular_dist'] = manager.dict()

    # 3. 多进程处理
    n_processes = min(cpu_count(), len(cif_files))
    logger.info(f"启动 {n_processes} 个进程处理")

    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.starmap(process_single_cif, [(f, stats_dict) for f in cif_files]),
            total=len(cif_files),
            desc="处理进度"
        ))

    # 4. 统计处理结果
    success_count = sum(results)
    fail_count = len(results) - success_count
    logger.info(f"处理完成 - 成功: {success_count}, 失败: {fail_count}")

    # 5. 生成分布图
    logger.info("生成分布图...")
    generate_distribution_plots(stats_dict)

    logger.info("所有任务完成！")


# 加载DGL图的辅助函数
def load_dgl_graph_with_attrs(graph_path):
    """
    加载包含图级属性的DGL图
    修改点：无需单独恢复graph_attrs，直接从ndata读取
    """
    DEVICE = torch.device("cpu")  # 根据需要改为cuda
    dgl_graph = torch.load(graph_path, map_location=DEVICE)

    # 便捷访问：提取第一个节点的图级属性（所有节点共享）
    dgl_graph.cell_params = dgl_graph.ndata['cell_params'][0]
    dgl_graph.stress_tensor = dgl_graph.ndata['stress_tensor_flat'][0].reshape(3, 3)
    dgl_graph.total_energy = dgl_graph.ndata['total_energy'][0]
    dgl_graph.graph_feat = dgl_graph.ndata['graph_attr'][0]

    return dgl_graph


# ========== 使用batch/unbatch并保留图级属性 ==========
def example_batch_unbatch_usage(graph_paths):
    """
    批量加载图、batch、提取属性、unbatch、恢复属性
    """
    # 1. 加载多个图
    graphs = [load_dgl_graph_with_attrs(path) for path in graph_paths]

    # 2. batch图
    batched_graph = dgl.batch(graphs)

    # 3. 提取每个子图的图级属性
    graph_attrs_list = extract_graph_attrs_from_batch(batched_graph)

    # 4. unbatch图
    subgraphs = dgl.unbatch(batched_graph)

    # 5. 恢复子图的便捷属性访问
    subgraphs = restore_graph_attrs_to_subgraphs(subgraphs, graph_attrs_list)

    # 6. 访问属性示例
    for i, subgraph in enumerate(subgraphs):
        print(f"子图{i} - 晶胞参数: {subgraph.cell_params}")
        print(f"子图{i} - 总能量(eV): {subgraph.total_energy[1]}")


if __name__ == "__main__":
    main()