import os
import dgl
import faiss
import torch
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.data import atomic_numbers
from collections import defaultdict
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch.nn as nn
import dgl.nn as dglnn
from dgl.nn import GlobalAttentionPooling, RelGraphConv
import traceback
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from openbabel import openbabel
from sklearn.decomposition import PCA, IncrementalPCA


# ==================== 核心RGCN层 ====================
class EdgeTypeRGCN(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_rels: int, num_bases: int):
        super().__init__()
        self.rgcn = RelGraphConv(
            in_feat=in_feat,
            out_feat=out_feat,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=num_bases
        )
        self.activation = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        out = self.rgcn(g, node_feat, edge_types)
        out = self.activation(out)
        return self.norm(out)


# ==================== 编码器 ====================
class CrystalRGCNEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, latent_dim: int, num_rels: int, num_bases: int):
        super().__init__()
        self.rgcn1 = EdgeTypeRGCN(node_feat_dim, hidden_dim, num_rels, num_bases)
        self.rgcn2 = EdgeTypeRGCN(hidden_dim, hidden_dim * 2, num_rels, num_bases)
        self.rgcn3 = EdgeTypeRGCN(hidden_dim * 2, hidden_dim * 2, num_rels, num_bases)

        final_dim = hidden_dim * 2
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        ))
        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor):
        h = self.rgcn1(g, g.ndata['feat'], edge_types)
        h = self.rgcn2(g, h, edge_types)
        h = self.rgcn3(g, h, edge_types)
        graph_emb = self.pooling(g, h)
        return self.fc_mu(graph_emb), self.fc_logvar(graph_emb)


# ==================== 解码器 ====================
class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, node_feat_dim: int):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2), nn.ReLU()
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
        batch_num_nodes = g.batch_num_nodes()
        total_nodes = sum(batch_num_nodes)
        h_expanded = torch.zeros(total_nodes, z.size(1) * 2, device=z.device)
        start_idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            h_expanded[start_idx:start_idx + num_nodes] = z[i].repeat(num_nodes, 1)
            start_idx += num_nodes
        return self.node_decoder(h_expanded)


# ==================== 完整VAE模型 ====================
class CrystalRGCNVAE(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, latent_dim: int, num_rels: int, num_bases: int):
        super().__init__()
        self.encoder = CrystalRGCNEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_rels=num_rels,
            num_bases=num_bases
        )
        self.decoder = CrystalDecoder(latent_dim, hidden_dim, node_feat_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(g, edge_types)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, g), mu, logvar


# ==================== 配置参数类 ====================
class KnowledgeBaseConfig:
    CIF_DIR = Path("/home/nyx/GRAG/raw_cifs")
    MODEL_PATH = Path("/home/nyx/RG-RAG/models/best_model.pth")
    FAISS_INDEX_PATH = Path("/home/nyx/RG-RAG/knowledage_base/material_index.faiss")
    METADATA_PATH = Path("/home/nyx/RG-RAG/knowledage_base/material_metadata.json")
    LATENT_DIM = 32  # 与训练模型匹配
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NODE_FEAT_DIM = 4
    HIDDEN_DIM = 64  # 与训练模型匹配
    NUM_RELS = 4  # 关系类型数量
    NUM_BASES = 4  # 基函数数量
    CUTOFF_DISTANCE = 5.0
    PROCESS_NUM = 44  # 进程数（针对48核CPU优化，保留4个核心给系统）
    MAX_ATOMS = 1000
    VISUALIZATION_DIR = Path("/home/nyx/RG-RAG/know_vis")
    MAX_T_SNE_SAMPLES = 2000
    # 移除超时配置（多进程环境下改用wait+timeout）


# ==================== CIF转DGL图（增强容错性） ====================
def cif_to_dgl_graph(cif_path: Path) -> Optional[dgl.DGLGraph]:
    try:
        # 忽略PyMatGen的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(cif_path)
            structures = parser.get_structures()

        if not structures:
            print(f"[警告] {cif_path.name} 无有效结构，跳过")
            return None

        structure = structures[0]
        if len(structure.sites) > KnowledgeBaseConfig.MAX_ATOMS:
            print(f"[警告] {cif_path.name} 原子数过多（{len(structure.sites)}），跳过")
            return None

        node_feats = []
        atomic_nums = []
        symbols = []
        number_to_symbol = {v: k for k, v in atomic_numbers.items()}  # 原子序数到符号的映射

        # 从structure.species获取原子信息（与sites一一对应）
        for i, site in enumerate(structure.sites):
            try:
                # 关键修复：使用structure.species[i]替代site.specie
                specie = structure.species[i]

                # 获取原子序数
                if hasattr(specie, 'Z'):
                    atomic_num = specie.Z
                elif hasattr(specie, 'atomic_number'):
                    atomic_num = specie.atomic_number
                else:
                    # 从元素符号反查原子序数
                    elem_symbol = str(specie).strip()
                    atomic_num = atomic_numbers.get(elem_symbol, 0)

                if atomic_num <= 0 or atomic_num > 118:
                    print(f"[警告] {cif_path.name} 原子序数无效（{atomic_num}），跳过该位点")
                    continue  # 跳过无效原子

                # 生成节点特征
                norm_atomic_num = atomic_num / 118.0
                node_feats.append([norm_atomic_num, site.x, site.y, site.z])
                atomic_nums.append(atomic_num)

                # 生成元素符号（基于原子序数，确保与node_feats同步）
                symbol = number_to_symbol.get(atomic_num, "Unknown")
                symbols.append(symbol)

            except Exception as e:
                print(f"[位点警告] {cif_path.name} 位点{i}处理警告: {str(e)}")
                continue  # 仅跳过当前异常位点

        # 若所有位点都被跳过，返回None
        if not node_feats:
            print(f"[警告] {cif_path.name} 所有位点均无效，跳过")
            return None

        # 构建边和边类型
        edges = []
        edge_types = []
        num_atoms = len(node_feats)  # 确保与节点特征数量一致
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                try:
                    dist = structure.get_distance(i, j)
                except:
                    continue  # 忽略距离计算失败的边
                if dist <= KnowledgeBaseConfig.CUTOFF_DISTANCE:
                    # 简化键类型分类
                    if dist < 1.5:
                        bond_type = 0  # 共价键
                    elif 1.5 <= dist < 2.5:
                        bond_type = 1  # 氢键
                    elif 2.5 <= dist < 4.0:
                        bond_type = 2  # pi-pi堆积
                    else:
                        bond_type = 3  # 范德华力
                    bond_type = max(0, min(bond_type, KnowledgeBaseConfig.NUM_RELS - 1))

                    edges.append((i, j))
                    edges.append((j, i))
                    edge_types.append(bond_type)
                    edge_types.append(bond_type)

        # 创建图
        u, v = zip(*edges) if edges else ([], [])
        graph = dgl.graph((u, v), num_nodes=num_atoms)
        graph.ndata['feat'] = torch.tensor(node_feats, dtype=torch.float32)
        graph.edata['edge_type'] = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.tensor([],
                                                                                                              dtype=torch.long)

        # 存储元素符号（使用整数编码，修复torch.string错误）
        symbol_set = list(set(symbols))
        symbol_to_idx = {s: i for i, s in enumerate(symbol_set)}
        graph.ndata['symbol'] = torch.tensor(
            [symbol_to_idx.get(sym, 0) for sym in symbols],
            dtype=torch.long
        )

        return graph

    except Exception as e:
        print(f"[CIF处理错误] {cif_path.name}: {str(e)}")
        traceback.print_exc()
        return None


# ==================== 结合能计算函数 ====================
def calculate_binding_energy(graph: dgl.DGLGraph) -> Optional[float]:
    try:
        node_feats = graph.ndata['feat'].cpu().numpy()
        coordinates = node_feats[:, 1:4]  # x, y, z坐标
        atomic_nums = (node_feats[:, 0] * 118).astype(int)

        if not np.isfinite(coordinates).all():
            return None

        number_to_symbol = {v: k for k, v in atomic_numbers.items()}

        ob_mol = openbabel.OBMol()
        ob_mol.BeginModify()

        for num, (x, y, z) in zip(atomic_nums, coordinates):
            py_num = int(num)
            if py_num < 1 or py_num > 118:
                ob_mol.EndModify()
                return None

            ob_atom = openbabel.OBAtom()
            ob_atom.SetAtomicNum(py_num)
            ob_atom.SetVector(float(x), float(y), float(z))
            ob_mol.AddAtom(ob_atom)

        ob_mol.EndModify()

        forcefield = openbabel.OBForceField.FindType("UFF")
        if not forcefield:
            return None

        if not forcefield.Setup(ob_mol):
            return None

        energy_kcal = forcefield.Energy()
        return round(energy_kcal * 0.04336, 4)  # 转换为eV

    except Exception as e:
        print(f"结合能计算失败: {str(e)}")
        return None


# ==================== 辅助函数 ====================
def get_molecular_formula(graph: dgl.DGLGraph) -> str:
    try:
        symbols = graph.ndata['symbol']
        if symbols.dtype == torch.long:
            # 从节点特征反推元素符号
            node_feats = graph.ndata['feat'].cpu().numpy()
            atomic_nums = (node_feats[:, 0] * 118).astype(int).tolist()
            number_to_symbol = {v: k for k, v in atomic_numbers.items()}
            symbols = [number_to_symbol.get(num, "X") for num in atomic_nums]
        else:
            symbols = [s.decode() if isinstance(s, bytes) else str(s) for s in symbols.cpu().numpy()]

        element_counts = defaultdict(int)
        for sym in symbols:
            element_counts[sym] += 1

        sorted_elements = sorted(element_counts.items(), key=lambda x: x[0])
        return ''.join([f"{elem}{cnt}" if cnt > 1 else elem for elem, cnt in sorted_elements])
    except:
        return "Unknown"


def extract_elements_from_formula(formula: str) -> List[str]:
    if formula == "Unknown":
        return []
    try:
        elements = re.findall(r'[A-Z][a-z]*', formula)
        return list(set(elements))
    except:
        return []


# ==================== 模型加载函数 ====================
def load_vae_encoder(config: KnowledgeBaseConfig) -> nn.Module:
    try:
        model = CrystalRGCNVAE(
            node_feat_dim=config.NODE_FEAT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_rels=config.NUM_RELS,
            num_bases=config.NUM_BASES
        ).to(config.DEVICE)

        # 安全加载模型
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)

        # 处理参数不匹配
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}

        missing_keys = [k for k in model_dict if k not in pretrained_dict]
        if missing_keys:
            print(f"警告: 预训练模型缺少键: {missing_keys[:5]}（最多显示5个）")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        return model.encoder
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def extract_graph_embedding(graph: dgl.DGLGraph, encoder: nn.Module, config: KnowledgeBaseConfig) -> np.ndarray:
    try:
        with torch.no_grad():
            edge_types = graph.edata['edge_type'].to(config.DEVICE) if 'edge_type' in graph.edata else torch.tensor([],
                                                                                                                    dtype=torch.long,
                                                                                                                    device=config.DEVICE)
            mu, _ = encoder(graph.to(config.DEVICE), edge_types)
            embedding = mu.cpu().numpy().reshape(1, -1).astype(np.float32)
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                print(f"[警告] 生成NaN/Inf嵌入，跳过")
                return np.array([])
            return embedding
    except Exception as e:
        print(f"嵌入提取错误: {str(e)}")
        return np.array([])


# ==================== 处理单个CIF文件（移除线程超时装饰器） ====================
def process_single_cif(cif_path: Path, encoder: nn.Module, config: KnowledgeBaseConfig) -> Optional[Dict]:
    try:
        print(f"[处理] 开始解析 {cif_path.name}（大小: {os.path.getsize(cif_path) / 1024:.1f}KB）")
        graph = cif_to_dgl_graph(cif_path)
        if not graph:
            print(f"[结果] {cif_path.name} → 生成图失败")
            return None

        # 提取基本信息
        formula = get_molecular_formula(graph)
        num_atoms = graph.num_nodes()

        # 计算嵌入（核心步骤）
        embedding = extract_graph_embedding(graph, encoder, config)
        if embedding.size == 0:
            print(f"[警告] {cif_path.name} 嵌入提取失败，跳过")
            return None

        # 尝试提取晶体学信息
        crystal_system = None
        space_group_symbol = None
        space_group_number = None
        lattice_params = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0]
        try:
            # 从原始CIF重新解析结构
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = CifParser(cif_path)
                structures = parser.get_structures()

            if structures:
                structure = structures[0]
                # 晶格参数
                lattice = structure.lattice
                lattice_params = [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma]
                # 晶系和空间群
                sga = SpacegroupAnalyzer(structure)
                crystal_system = sga.get_crystal_system()
                space_group_symbol = sga.get_space_group_symbol()
                space_group_number = sga.get_space_group_number()
        except:
            pass  # 允许提取失败

        # 计算结合能
        binding_energy = calculate_binding_energy(graph)

        print(f"[结果] {cif_path.name} → 处理成功（原子数: {num_atoms}, 公式: {formula}）")
        return {
            "material_id": cif_path.stem,
            "formula": formula,
            "binding_energy": binding_energy,
            "lattice_params": lattice_params,
            "crystal_system": crystal_system,
            "space_group_symbol": space_group_symbol,
            "space_group_number": space_group_number,
            "num_atoms_in_unit_cell": num_atoms,
            "embedding": embedding
        }

    except Exception as e:
        print(f"[错误] {cif_path.name} → {str(e)}")
        traceback.print_exc()
        return None


# ==================== 多进程适配的包装函数 ====================
def process_single_cif_wrapper(cif_path: Path, config: KnowledgeBaseConfig) -> Optional[Dict]:
    """多进程环境下的包装函数，每个进程独立加载模型"""
    try:
        # 每个进程单独加载模型（避免多进程间模型共享冲突）
        encoder = load_vae_encoder(config)
        return process_single_cif(cif_path, encoder, config)
    except Exception as e:
        print(f"[进程初始化错误] {cif_path.name}: {str(e)}")
        return None


# ==================== 可视化函数 ====================
def visualize_embeddings(embeddings, config):
    if embeddings.size == 0 or len(embeddings) < 2:
        print("跳过嵌入可视化：样本不足")
        return

    try:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # PCA可视化
        if len(embeddings) < 5000:
            pca = PCA(n_components=2)
        else:
            pca = IncrementalPCA(n_components=2, batch_size=min(1000, len(embeddings)))
        pca_result = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.6)

        # t-SNE可视化
        if 10 < len(embeddings) < config.MAX_T_SNE_SAMPLES:
            sample_indices = np.random.choice(
                len(embeddings),
                size=min(config.MAX_T_SNE_SAMPLES, len(embeddings)),
                replace=False
            )
            sample_embeddings = embeddings[sample_indices]

            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=min(30, len(sample_embeddings) - 1),
                        n_iter=1000)
            tsne_result = tsne.fit_transform(sample_embeddings)

            plt.subplot(1, 2, 2)
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5)
            plt.title('t-SNE Visualization (Sampled)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            plt.subplot(1, 2, 2)
            if len(embeddings) <= 10:
                msg = '样本不足(≤10)\n跳过t-SNE'
            else:
                msg = f'样本过大(≥{config.MAX_T_SNE_SAMPLES})\n使用PCA替代'
            plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(config.VISUALIZATION_DIR / 'embedding_visualization.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"嵌入可视化失败: {str(e)}")


def visualize_metadata(metadata, config):
    plt.figure(figsize=(20, 6))

    # 1. 结合能分布
    binding_energies = [data["binding_energy"] for data in metadata if data["binding_energy"] is not None]
    plt.subplot(1, 4, 1)
    if binding_energies:
        sns.histplot(binding_energies, bins=min(100, len(binding_energies) // 10), kde=True, color='skyblue')
        plt.axvline(np.mean(binding_energies), color='red', linestyle='dashed', linewidth=1)
        plt.title('Binding Energy Distribution')
        plt.xlabel('Binding Energy (eV)')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No binding energy data', ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 2. 元素分布
    plt.subplot(1, 4, 2)
    element_counter = defaultdict(int)
    for data in metadata:
        if data["formula"] and data["formula"] != "Unknown":
            elements = extract_elements_from_formula(data["formula"])
            for element in elements:
                element_counter[element] += 1
    if element_counter:
        sorted_elements = sorted(element_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        elements, counts = zip(*sorted_elements)
        plt.bar(elements, counts, color='lightgreen')
        plt.title('Top 20 Elements')
        plt.xlabel('Element')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No element data', ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 3. 晶系分布
    plt.subplot(1, 4, 3)
    crystal_system_counter = defaultdict(int)
    for data in metadata:
        cs = data.get("crystal_system")
        if cs:
            crystal_system_counter[cs] += 1
    if crystal_system_counter:
        sorted_cs = sorted(crystal_system_counter.items(), key=lambda x: x[1], reverse=True)
        cs_names, cs_counts = zip(*sorted_cs)
        plt.bar(cs_names, cs_counts, color=plt.cm.Set3(np.linspace(0, 1, len(sorted_cs))))
        plt.title('Crystal System Distribution')
        plt.xlabel('Crystal System')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No crystal system data', ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 4. 晶胞原子数量分布
    plt.subplot(1, 4, 4)
    num_atoms_list = [data["num_atoms_in_unit_cell"] for data in metadata if "num_atoms_in_unit_cell" in data]
    if num_atoms_list:
        filtered_num_atoms = [n for n in num_atoms_list if n <= KnowledgeBaseConfig.MAX_ATOMS]
        sns.histplot(filtered_num_atoms, bins=min(50, len(filtered_num_atoms) // 5), kde=False, color='salmon')
        plt.axvline(np.mean(filtered_num_atoms), color='purple', linestyle='dashed', linewidth=1)
        plt.title('Unit Cell Atom Count')
        plt.xlabel('Number of Atoms')
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No atom count data', ha='center', va='center', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(config.VISUALIZATION_DIR / 'metadata_visualization.png', dpi=150)
    plt.close()


# ==================== 主构建流程 ====================
def build_knowledge_base(config: Optional[KnowledgeBaseConfig] = None):
    config = config or KnowledgeBaseConfig()
    if not config.CIF_DIR.exists():
        raise FileNotFoundError(f"CIF目录不存在: {config.CIF_DIR}")

    cif_files = list(config.CIF_DIR.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"目录{config.CIF_DIR}中无有效.cif文件")

    print(f"开始处理 {len(cif_files)} 个CIF文件（进程数: {config.PROCESS_NUM}）...")

    metadata = []
    embeddings_list = []
    processed_count = 0
    skipped_count = 0

    # 多进程处理CIF文件
    with ProcessPoolExecutor(max_workers=config.PROCESS_NUM) as executor:
        # 为每个任务准备参数（避免进程间传递大对象）
        tasks = [(cif_path, config) for cif_path in cif_files]

        # 修复：正确映射任务和cif_path
        futures = {executor.submit(process_single_cif_wrapper, *task): task[0] for task in tasks}

        progress_bar = tqdm(total=len(cif_files), desc="处理CIF文件", unit="file")

        for future in as_completed(futures):
            cif_path = futures[future]
            try:
                result = future.result()
                if result:
                    metadata.append({
                        "material_id": result["material_id"],
                        "formula": result["formula"],
                        "binding_energy": result["binding_energy"],
                        "lattice_params": result["lattice_params"],
                        "crystal_system": result["crystal_system"],
                        "space_group_symbol": result["space_group_symbol"],
                        "space_group_number": result["space_group_number"],
                        "num_atoms_in_unit_cell": result["num_atoms_in_unit_cell"]
                    })
                    embeddings_list.append(result["embedding"].flatten())
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"[进程错误] {cif_path.name}: {str(e)}")
                skipped_count += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                "已处理": processed_count,
                "已跳过": skipped_count
            })
        progress_bar.close()

    # 检查有效数据
    if not embeddings_list:
        print("错误: 无有效嵌入数据，可能所有CIF文件处理失败")
        return

    # 构建FAISS索引
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    print(f"成功处理 {len(metadata)} 个材料，跳过 {skipped_count} 个文件")

    # 保存成功处理的材料ID列表
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.VISUALIZATION_DIR / 'processed_materials.txt', 'w') as f:
        for item in metadata:
            f.write(f"{item['material_id']}\t{item['formula']}\n")

    try:
        index = faiss.IndexFlatL2(config.LATENT_DIM)
        index.add(embeddings)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))
        print(f"FAISS索引已保存: {config.FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"FAISS索引创建失败: {str(e)}")
        return

    # 保存元数据
    try:
        with open(config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"元数据已保存: {config.METADATA_PATH}")
    except Exception as e:
        print(f"元数据保存失败: {str(e)}")

    # 生成可视化
    print("生成可视化图表...")
    visualize_embeddings(embeddings, config)
    visualize_metadata(metadata, config)
    print(f"可视化图表已保存至: {config.VISUALIZATION_DIR}")

    # 最终报告
    print("\n" + "=" * 50)
    print("知识库构建完成!")
    print(f"总文件数: {len(cif_files)}")
    print(f"成功处理: {len(metadata)}")
    print(f"跳过文件: {skipped_count}")
    print(f"嵌入维度: {config.LATENT_DIM}")
    print(f"FAISS索引: {config.FAISS_INDEX_PATH}")
    print(f"元数据: {config.METADATA_PATH}")
    print(f"成功材料列表: {config.VISUALIZATION_DIR / 'processed_materials.txt'}")
    print("=" * 50)


if __name__ == "__main__":
    # 确保在主进程中执行
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('spawn', force=True)

    try:
        build_knowledge_base()
    except Exception as e:
        print(f"\n知识库构建失败: {str(e)}")
        traceback.print_exc()
        exit(1)