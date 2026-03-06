import os
import dgl
import faiss
import torch
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.data import atomic_numbers
from collections import defaultdict
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch.nn as nn
import dgl.nn as dglnn
from dgl.nn import GlobalAttentionPooling
from openbabel import openbabel
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings


# ==================== 核心GAT层 ====================
class EdgeTypeGAT(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat
        self.gat = dglnn.GATConv(
            in_feats=in_feat,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True
        )

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        return self.gat(g, node_feat).flatten(1)


# ==================== 编码器 ====================
class CrystalGCNEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, latent_dim: int, num_heads: int):
        super().__init__()
        self.gat1 = EdgeTypeGAT(node_feat_dim, hidden_dim // 2, num_heads)
        self.gat2 = EdgeTypeGAT((hidden_dim // 2) * num_heads, hidden_dim, num_heads)
        self.gat3 = EdgeTypeGAT(hidden_dim * num_heads, hidden_dim * 2, num_heads)
        final_dim = num_heads * hidden_dim * 2
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        ))
        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

    def forward(self, g: dgl.DGLGraph):
        h = nn.ELU()(self.gat1(g, g.ndata['feat']))
        h = nn.ELU()(self.gat2(g, h))
        h = nn.ELU()(self.gat3(g, h))
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
class CrystalGCNVAE(nn.Module):
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


# ==================== 配置参数类 ====================
class KnowledgeBaseConfig:
    CIF_DIR = Path("/home/nyx/GRAG/raw_cifs")
    MODEL_PATH = Path("/home/nyx/GRAG/models/best_model.pth")
    FAISS_INDEX_PATH = Path("/home/nyx/GRAG/knowledge_base/material_index.faiss")
    METADATA_PATH = Path("/home/nyx/GRAG/knowledge_base/material_metadata.json")
    LATENT_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NODE_FEAT_DIM = 4
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    CUTOFF_DISTANCE = 5.0
    NUM_WORKERS = 4
    MAX_ATOMS = 1000
    VISUALIZATION_DIR = Path("/home/nyx/GRAG/know_vis")
    MAX_T_SNE_SAMPLES = 2000  # t-SNE处理的最大样本数


# ==================== 增强的CIF转DGL图函数 ====================
def cif_to_dgl_graph(cif_path: Path) -> Optional[dgl.DGLGraph]:
    try:
        # 忽略PyMatGen的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(cif_path)
            structures = parser.get_structures()

        if not structures:
            print(f"[警告] CIF文件 {cif_path.name} 无有效结构，跳过")
            return None

        # 使用第一个有效结构
        structure = structures[0]

        # 检查原子数量
        if len(structure.sites) > KnowledgeBaseConfig.MAX_ATOMS:
            print(f"[警告] CIF文件 {cif_path.name} 原子数 {len(structure.sites)} 超过限制，跳过")
            return None

        node_feats = []
        symbols = []
        valid_structure = True

        # 处理每个原子
        for idx, site in enumerate(structure.sites):
            # 检查坐标是否有效
            if not np.isfinite(site.coords).all():
                print(f"[警告] CIF文件 {cif_path.name} 原子 {idx} 坐标无效，跳过")
                valid_structure = False
                break

            atomic_num = site.specie.Z
            # 检查原子序数是否在有效范围
            if atomic_num <= 0 or atomic_num > 118:
                print(f"[警告] CIF文件 {cif_path.name} 原子 {idx} 序数 {atomic_num} 无效，跳过")
                valid_structure = False
                break

            norm_atomic_num = atomic_num / 118.0
            coords = site.coords
            node_feats.append([norm_atomic_num, coords[0], coords[1], coords[2]])
            symbols.append(str(site.specie.symbol))

        if not valid_structure:
            return None

        # 构建图结构
        edges = []
        num_atoms = len(structure.sites)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                try:
                    dist = structure.get_distance(i, j)
                except Exception as e:
                    continue  # 静默跳过距离计算错误
                if dist <= KnowledgeBaseConfig.CUTOFF_DISTANCE:
                    edges.append((i, j))
                    edges.append((j, i))

        u, v = zip(*edges) if edges else ([], [])
        graph = dgl.graph((u, v), num_nodes=num_atoms)
        graph.ndata['feat'] = torch.tensor(node_feats, dtype=torch.float32)

        # 符号存储兼容处理
        try:
            graph.ndata['symbol'] = torch.tensor(symbols, dtype=torch.string)
        except:
            symbol_to_idx = {sym: i for i, sym in enumerate(set(symbols))}
            graph.ndata['symbol'] = torch.tensor([symbol_to_idx[sym] for sym in symbols], dtype=torch.long)

        return graph

    except Exception as e:
        print(f"[CIF处理错误] 文件 {cif_path.name}: {str(e)}")
        return None


# ==================== 结合能计算函数 ====================
def calculate_binding_energy(graph: dgl.DGLGraph) -> Optional[float]:
    try:
        # 提取原子信息
        node_feats = graph.ndata['feat'].cpu().numpy()
        coordinates = node_feats[:, 1:4]
        atomic_nums = (node_feats[:, 0] * 118).astype(int)

        # 检查坐标是否有效
        if not np.isfinite(coordinates).all():
            return None

        # 原子序数到元素符号的映射
        number_to_symbol = {v: k for k, v in atomic_numbers.items()}

        # 创建Open Babel分子
        ob_mol = openbabel.OBMol()
        ob_mol.BeginModify()

        for num, (x, y, z) in zip(atomic_nums, coordinates):
            py_num = int(num)
            # 检查原子序数是否在有效范围
            if py_num < 1 or py_num > 118:
                ob_mol.EndModify()
                return None

            # 创建原子
            ob_atom = openbabel.OBAtom()
            ob_atom.SetAtomicNum(py_num)
            ob_atom.SetVector(float(x), float(y), float(z))
            ob_mol.AddAtom(ob_atom)

        ob_mol.EndModify()

        # 初始化UFF力场
        forcefield = openbabel.OBForceField.FindType("UFF")
        if not forcefield:
            return None

        # 设置力场并计算能量
        if not forcefield.Setup(ob_mol):
            return None

        energy_kcal = forcefield.Energy()
        return round(energy_kcal * 0.04336, 4)  # kcal/mol -> eV

    except Exception as e:
        return None


# ==================== 辅助函数 ====================
def get_molecular_formula(graph: dgl.DGLGraph) -> str:
    try:
        symbols = graph.ndata['symbol']
        if symbols.dtype == torch.long:
            node_feats = graph.ndata['feat'].cpu().numpy()
            atomic_nums = (node_feats[:, 0] * 118).astype(int).tolist()
            number_to_symbol = {v: k for k, v in atomic_numbers.items()}
            symbols = [number_to_symbol.get(num, "X") for num in atomic_nums]
        else:
            symbols = [s.decode() if isinstance(s, bytes) else str(s) for s in symbols.cpu().numpy()]

        element_counts = defaultdict(int)
        for sym in symbols:
            element_counts[sym] += 1

        # 按元素符号排序
        sorted_elements = sorted(element_counts.items(), key=lambda x: x[0])
        return ''.join([f"{elem}{cnt}" if cnt > 1 else elem for elem, cnt in sorted_elements])
    except:
        return "Unknown"


def extract_elements_from_formula(formula: str) -> List[str]:
    """从分子式中提取唯一元素列表"""
    if formula == "Unknown":
        return []

    try:
        elements = re.findall(r'[A-Z][a-z]*', formula)
        return list(set(elements))
    except:
        return []


def load_vae_encoder(config: KnowledgeBaseConfig) -> nn.Module:
    try:
        model = CrystalGCNVAE(
            node_feat_dim=config.NODE_FEAT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_heads=config.NUM_HEADS
        ).to(config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model.eval()
        return model.encoder
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def extract_graph_embedding(graph: dgl.DGLGraph, encoder: nn.Module, config: KnowledgeBaseConfig) -> np.ndarray:
    try:
        with torch.no_grad():
            mu, _ = encoder(graph.to(config.DEVICE))
            embedding = mu.cpu().numpy().reshape(1, -1).astype(np.float32)
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return np.array([])
            return embedding
    except:
        return np.array([])


def process_single_cif(cif_path: Path, encoder: nn.Module, config: KnowledgeBaseConfig) -> Optional[Dict]:
    try:
        graph = cif_to_dgl_graph(cif_path)
        if not graph:
            return None

        # 重新解析结构以提取完整信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(cif_path)
            structures = parser.get_structures()
        if not structures:
            return None
        structure = structures[0]

        # 提取晶胞中的原子数量（核心新增）
        num_atoms_in_unit_cell = len(structure.sites)

        formula = get_molecular_formula(graph)
        binding_energy = calculate_binding_energy(graph)
        embedding = extract_graph_embedding(graph, encoder, config)
        if embedding.size == 0:
            return None

        # 提取晶系信息
        crystal_system = None
        try:
            crystal_system = structure.lattice.crystal_system
        except Exception as e:
            print(f"[警告] {cif_path.name} 无法提取晶系: {str(e)}")

        # 提取对称性信息
        space_group_symbol = None
        space_group_number = None
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group_symbol = sga.get_space_group_symbol()
            space_group_number = sga.get_space_group_number()
        except Exception as e:
            print(f"[警告] {cif_path.name} 无法提取空间群: {str(e)}")

        # 提取晶格参数
        lattice_params = [0, 0, 0, 0, 0, 0]
        try:
            lattice = structure.lattice
            lattice_params = [
                lattice.a, lattice.b, lattice.c,
                lattice.alpha, lattice.beta, lattice.gamma
            ]
        except:
            try:
                coords = graph.ndata['feat'][:, 1:].numpy()
                if len(coords) > 0:
                    min_coords = np.min(coords, axis=0)
                    max_coords = np.max(coords, axis=0)
                    lattice_params = [
                        max_coords[0] - min_coords[0],
                        max_coords[1] - min_coords[1],
                        max_coords[2] - min_coords[2],
                        90.0, 90.0, 90.0
                    ]
            except:
                pass

        return {
            "material_id": cif_path.stem,
            "cif_path": str(cif_path),
            "formula": formula,
            "binding_energy": binding_energy,
            "embedding": embedding,
            "lattice_params": lattice_params,
            "crystal_system": crystal_system,
            "space_group_symbol": space_group_symbol,
            "space_group_number": space_group_number,
            "num_atoms_in_unit_cell": num_atoms_in_unit_cell  # 新增：晶胞中的原子数量
        }

    except Exception as e:
        print(f"[处理错误] {cif_path.name}: {str(e)}")
        return None


# ==================== 可视化函数 ====================
def visualize_embeddings(embeddings, config):
    """嵌入向量可视化"""
    if embeddings.size == 0 or len(embeddings) < 2:
        print("跳过嵌入可视化：样本不足")
        return

    try:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

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
    """元数据可视化（新增晶胞原子数量分布）"""
    plt.figure(figsize=(20, 6))  # 加宽画布以容纳4个子图

    # 1. 结合能分布
    binding_energies = [data["binding_energy"] for data in metadata if data["binding_energy"] is not None]
    plt.subplot(1, 4, 1)
    if binding_energies:
        min_val = min(binding_energies)
        max_val = max(binding_energies)
        mean_val = np.mean(binding_energies)
        num_bins = min(100, len(binding_energies) // 10)
        if num_bins < 10:
            num_bins = 10

        sns.histplot(binding_energies, bins=num_bins, kde=True, color='skyblue')
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        plt.text(mean_val * 1.05, plt.ylim()[1] * 0.9, f'Mean: {mean_val:.2f} eV', color='red')
        plt.title('Binding Energy Distribution')
        plt.xlabel('Binding Energy (eV)')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No binding energy data',
                 ha='center', va='center', fontsize=12)
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
        plt.text(0.5, 0.5, 'No element data',
                 ha='center', va='center', fontsize=12)
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
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_cs)))
        plt.bar(cs_names, cs_counts, color=colors)
        plt.title('Crystal System Distribution')
        plt.xlabel('Crystal System')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No crystal system data',
                 ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 4. 新增：晶胞原子数量分布
    plt.subplot(1, 4, 4)
    num_atoms_list = [data["num_atoms_in_unit_cell"] for data in metadata if "num_atoms_in_unit_cell" in data]
    if num_atoms_list:
        # 过滤异常值（超过MAX_ATOMS的可能是错误数据）
        filtered_num_atoms = [n for n in num_atoms_list if n <= KnowledgeBaseConfig.MAX_ATOMS]
        if not filtered_num_atoms:
            filtered_num_atoms = num_atoms_list  # 若全是异常值则保留原始数据

        min_n = min(filtered_num_atoms)
        max_n = max(filtered_num_atoms)
        mean_n = np.mean(filtered_num_atoms)
        num_bins = min(50, len(filtered_num_atoms) // 5)  # 动态分箱
        if num_bins < 5:
            num_bins = 5

        sns.histplot(filtered_num_atoms, bins=num_bins, kde=False, color='salmon')
        plt.axvline(mean_n, color='purple', linestyle='dashed', linewidth=1)
        plt.text(mean_n * 1.05, plt.ylim()[1] * 0.9, f'Mean: {mean_n:.1f}', color='purple')
        plt.title('Unit Cell Atom Count')
        plt.xlabel('Number of Atoms')
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No atom count data',
                 ha='center', va='center', fontsize=12)
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

    print(f"开始处理 {len(cif_files)} 个CIF文件...")

    try:
        encoder = load_vae_encoder(config)
        print("VAE编码器加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    metadata = []
    embeddings_list = []
    processed_count = 0
    skipped_count = 0

    progress_bar = tqdm(total=len(cif_files), desc="处理CIF文件", unit="file")

    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = {}
        for cif_path in cif_files:
            future = executor.submit(process_single_cif, cif_path, encoder, config)
            futures[future] = cif_path

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
                        "num_atoms_in_unit_cell": result["num_atoms_in_unit_cell"]  # 保存原子数量
                    })
                    embeddings_list.append(result["embedding"].flatten())
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"处理文件 {cif_path.name} 时出错: {str(e)}")
                skipped_count += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                "已处理": processed_count,
                "已跳过": skipped_count,
                "进度": f"{progress_bar.n}/{progress_bar.total}"
            })

    progress_bar.close()

    if not embeddings_list:
        print("错误: 无有效嵌入数据，终止构建")
        return

    embeddings = np.vstack(embeddings_list).astype(np.float32)
    print(f"成功处理 {len(metadata)} 个材料，跳过 {skipped_count} 个文件")

    # 创建FAISS索引
    try:
        index = faiss.IndexFlatL2(config.LATENT_DIM)
        index.add(embeddings)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))
        print(f"FAISS索引已保存: {config.FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"FAISS索引创建失败: {str(e)}")
        return

    # 保存元数据（包含原子数量）
    try:
        with open(config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"元数据已保存: {config.METADATA_PATH}（包含晶胞原子数量）")
    except Exception as e:
        print(f"元数据保存失败: {str(e)}")

    # 创建可视化目录
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # 执行可视化
    print("生成可视化图表...")
    visualize_embeddings(embeddings, config)
    visualize_metadata(metadata, config)  # 包含原子数量分布
    print(f"可视化图表已保存至: {config.VISUALIZATION_DIR}")

    # 最终报告
    print("\n" + "=" * 50)
    print("知识库构建完成!")
    print(f"总文件数: {len(cif_files)}")
    print(f"成功处理: {len(metadata)}")
    print(f"跳过文件: {skipped_count}")
    print(f"嵌入维度: {config.LATENT_DIM}")
    print(f"元数据包含: 分子式、结合能、晶格参数、晶系、空间群、晶胞原子数量")
    print(f"FAISS索引: {config.FAISS_INDEX_PATH}")
    print(f"元数据: {config.METADATA_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        build_knowledge_base()
    except Exception as e:
        print(f"\n知识库构建失败: {str(e)}")
        traceback.print_exc()
        exit(1)