import os
import dgl
import torch
import numpy as np
import faiss
import logging
import argparse
import json
from torch import nn
from dgl.nn import GATConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
from typing import List, Tuple, Dict, Optional
import warnings
from pathlib import Path
import re

# ========== RAG知识库配置 ==========
# 向量库存储路径
VECTOR_DB_DIR = "/home/nyx/N-RGAG/know_base"
# 知识库索引名称
VECTOR_INDEX_NAME = "crystal_latent_index.faiss"
# 元数据存储路径
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "crystal_metadata.json")
# 向量维度
VECTOR_DIM = 64
# 批量处理大小
BATCH_SIZE_RAG = 32

# 新增：cif文件根目录
CIF_ROOT_DIR = "/home/nyx/N-RGAG/raw_cifs"

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
# 路径配置
PROCESSED_GRAPH_DIR = "/home/nyx/N-RGAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/N-RGAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/N-RGAG/models"

# 特征维度配置
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
GRAPH_ATTR_DIM = 17  # 6晶胞+9应力+2能量
ENERGY_DIM = 2  # hartree + eV（索引0: hartree，索引1: eV）
STRESS_DIM = 9  # 应力张量展平
CELL_PARAMS_DIM = 6  # 晶胞参数

# 数值稳定配置
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [RAG构建] %(message)s",
    handlers=[
        logging.FileHandler("rag_know_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== IF文件解析函数====================
def parse_cif_metadata(cif_path: str) -> Dict[str, any]:
    """
    解析cif文件中的原始元数据：Total Energy (eV) 和 Stress Tensor (3x3)
    :param cif_path: cif文件路径
    :return: 包含原始能量、应力的字典
    """
    metadata = {
        "total_energy_ev": None,
        "total_energy_ev_text": "Unknown",
        "stress_tensor_3x3": None,
        "stress_tensor_flat": None,
        "stress_tensor_text": "Unknown"
    }

    if not os.path.exists(cif_path):
        logger.warning(f"CIF文件不存在: {cif_path}")
        return metadata

    try:
        with open(cif_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        # 1. 解析Total Energy (eV)
        energy_pattern = re.compile(r"# Total Energy \(eV\):\s*(-?\d+\.\d+)")
        for line in lines:
            match = energy_pattern.search(line)
            if match:
                energy_val = float(match.group(1))  # 修复：强制转为Python float
                metadata["total_energy_ev"] = energy_val
                metadata["total_energy_ev_text"] = line  # 保留原始文本
                break

        # 2. 解析Stress Tensor (3x3) [atomic units]
        stress_lines = []
        stress_start = False
        stress_row_count = 0
        stress_pattern = re.compile(r"# Stress_Row_\d+:\s*([\d\-\.e\s]+)")

        for line in lines:
            if "# Stress Tensor (3x3) [atomic units]" in line:
                stress_start = True
                metadata["stress_tensor_text"] = line  # 保留标题行
                continue
            if stress_start and stress_row_count < 3:
                match = stress_pattern.search(line)
                if match:
                    stress_vals = [float(x) for x in match.group(1).split()]  # 修复：强制转为Python float
                    if len(stress_vals) == 3:
                        stress_lines.append(stress_vals)
                        metadata["stress_tensor_text"] += "\n" + line  # 拼接3行应力
                        stress_row_count += 1

        # 转换为3x3数组和扁平9维数组
        if len(stress_lines) == 3:
            stress_3x3 = np.array(stress_lines, dtype=np.float64)
            metadata["stress_tensor_3x3"] = stress_3x3.tolist()  # 修复：直接转为list
            metadata["stress_tensor_flat"] = stress_3x3.flatten().tolist()  # 修复：直接转为list

    except Exception as e:
        logger.error(f"解析CIF文件{cif_path}失败: {str(e)}")

    return metadata


# ==================== CrystalDataScaler类定义 ====================
class CrystalDataScaler:
    """晶体数据归一化器"""

    def __init__(self):
        # 能量归一化参数（energy: hartree + eV）
        self.energy_mean = np.zeros(ENERGY_DIM)
        self.energy_std = np.ones(ENERGY_DIM)
        # 应力归一化参数（stress tensor）
        self.stress_mean = np.zeros(STRESS_DIM)
        self.stress_std = np.ones(STRESS_DIM)
        # 节点特征归一化参数
        self.node_feat_mean = np.zeros(NODE_FEAT_DIM)
        self.node_feat_std = np.ones(NODE_FEAT_DIM)
        # 边特征归一化参数
        self.edge_feat_mean = np.zeros(EDGE_FEAT_DIM)
        self.edge_feat_std = np.ones(EDGE_FEAT_DIM)

    def fit_energy(self, energy_data: np.ndarray):
        """拟合能量数据的归一化参数"""
        self.energy_mean = np.mean(energy_data, axis=0)
        self.energy_std = np.std(energy_data, axis=0)
        # 防止标准差为0
        self.energy_std[self.energy_std < 1e-8] = 1.0

    def fit_stress(self, stress_data: np.ndarray):
        """拟合应力数据的归一化参数"""
        self.stress_mean = np.mean(stress_data, axis=0)
        self.stress_std = np.std(stress_data, axis=0)
        self.stress_std[self.stress_std < 1e-8] = 1.0

    def fit_node_feats(self, node_feat_data: np.ndarray):
        """拟合节点特征的归一化参数"""
        self.node_feat_mean = np.mean(node_feat_data, axis=0)
        self.node_feat_std = np.std(node_feat_data, axis=0)
        self.node_feat_std[self.node_feat_std < 1e-8] = 1.0

    def fit_edge_feats(self, edge_feat_data: np.ndarray):
        """拟合边特征的归一化参数"""
        self.edge_feat_mean = np.mean(edge_feat_data, axis=0)
        self.edge_feat_std = np.std(edge_feat_data, axis=0)
        self.edge_feat_std[self.edge_feat_std < 1e-8] = 1.0

    def transform_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """能量归一化（训练时用）"""
        mean = torch.tensor(self.energy_mean, dtype=energy.dtype, device=energy.device)
        std = torch.tensor(self.energy_std, dtype=energy.dtype, device=energy.device)
        return (energy - mean) / std

    def inverse_transform_energy(self, energy_norm: torch.Tensor) -> torch.Tensor:
        """能量反归一化（推理/构建RAG时用）"""
        mean = torch.tensor(self.energy_mean, dtype=energy_norm.dtype, device=energy_norm.device)
        std = torch.tensor(self.energy_std, dtype=energy_norm.dtype, device=energy_norm.device)
        return energy_norm * std + mean

    def transform_stress(self, stress: torch.Tensor) -> torch.Tensor:
        """应力归一化（训练时用）"""
        mean = torch.tensor(self.stress_mean, dtype=stress.dtype, device=stress.device)
        std = torch.tensor(self.stress_std, dtype=stress.dtype, device=stress.device)
        return (stress - mean) / std

    def inverse_transform_stress(self, stress_norm: torch.Tensor) -> torch.Tensor:
        """应力反归一化（推理/构建RAG时用）"""
        mean = torch.tensor(self.stress_mean, dtype=stress_norm.dtype, device=stress_norm.device)
        std = torch.tensor(self.stress_std, dtype=stress_norm.dtype, device=stress_norm.device)
        return stress_norm * std + mean

    def transform_node_feats(self, node_feats: torch.Tensor) -> torch.Tensor:
        """节点特征归一化"""
        mean = torch.tensor(self.node_feat_mean, dtype=node_feats.dtype, device=node_feats.device)
        std = torch.tensor(self.node_feat_std, dtype=node_feats.dtype, device=node_feats.device)
        return (node_feats - mean) / std

    def transform_edge_feats(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """边特征归一化"""
        mean = torch.tensor(self.edge_feat_mean, dtype=edge_feats.dtype, device=edge_feats.device)
        std = torch.tensor(self.edge_feat_std, dtype=edge_feats.dtype, device=edge_feats.device)
        return (edge_feats - mean) / std


# ==================== 模型定义 ====================
class EdgeAttentionGAT(nn.Module):
    """带边特征注意力的GAT层"""

    def __init__(self, in_feat: int, out_feat: int, num_heads: int, edge_feat_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat

        self.gat = GATConv(
            in_feats=in_feat + edge_feat_dim,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            feat_drop=0.1,
            attn_drop=0.1
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_feat_dim * 2, edge_feat_dim)
        )

        self.node_proj = nn.Linear(in_feat, in_feat)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        edge_feats = g.edata["feat"]
        edge_feats = self.edge_mlp(edge_feats)

        src, dst = g.edges()
        node_feats_proj = self.node_proj(node_feats)

        src_feats = node_feats_proj[src]
        dst_feats = node_feats_proj[dst]

        src_input = torch.cat([src_feats, edge_feats], dim=1)
        dst_input = torch.cat([dst_feats, edge_feats], dim=1)

        temp_feats = torch.zeros((g.num_nodes(), src_input.shape[1]), device=node_feats.device)
        temp_feats = temp_feats.scatter_add(0, src.unsqueeze(1).repeat(1, src_input.shape[1]), src_input)
        temp_feats = temp_feats.scatter_add(0, dst.unsqueeze(1).repeat(1, dst_input.shape[1]), dst_input)

        if g.num_edges() == 0:
            temp_feats = torch.cat(
                [node_feats_proj, torch.zeros((g.num_nodes(), EDGE_FEAT_DIM), device=node_feats.device)], dim=1)

        gat_out = self.gat(g, temp_feats)
        return gat_out.flatten(1)


class CrystalGATEncoder(nn.Module):
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
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_gat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(final_gat_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_gat_dim, latent_dim)

        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

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

        graph_emb = self.pooling(g, h)
        node_emb = h

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


class CrystalDecoder(nn.Module):
    """解码器"""

    def __init__(self,
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        node_emb_input_dim = num_heads * (hidden_dim * 2)
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

        self.energy_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, energy_dim)
        )

        self.stress_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stress_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, node_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = len(g.batch_num_nodes())
        total_nodes = g.num_nodes()

        z_proj = self.latent_proj(z)

        z_expanded = torch.zeros(total_nodes, z_proj.size(1), device=z.device)
        start_idx = 0
        for i in range(batch_size):
            num_nodes = g.batch_num_nodes()[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        node_emb_proj = self.node_emb_proj(node_emb)
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)

        src, dst = g.edges()
        z_src = z_expanded[src]
        z_dst = z_expanded[dst]
        edge_input = torch.cat([z_src, z_dst], dim=1)
        recon_edge_feats = self.edge_decoder(edge_input)

        pred_energy = self.energy_predictor(z)
        pred_stress = self.stress_predictor(z)

        return {
            'recon_node': recon_node_feats,
            'recon_edge': recon_edge_feats,
            'pred_energy': pred_energy,
            'pred_stress': pred_stress
        }


class CrystalGATVAE(nn.Module):
    """完整GAT-VAE模型"""

    def __init__(self, args):
        super().__init__()
        self.encoder = CrystalGATEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_heads=args.num_heads
        )

        self.decoder = CrystalDecoder(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM
        )

        self.latent_dim = args.latent_dim

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
            'z': z
        }
        output.update(decode_out)

        return output


# ==================== RAG知识库核心功能====================
class CrystalRAGBuilder:
    """晶体RAG外部知识库构建器"""

    def __init__(self, vector_dim: int, device: torch.device):
        self.vector_dim = vector_dim
        self.device = device
        # 创建向量库目录（确保目录存在）
        Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)

        # 初始化FAISS索引（适配晶体向量的高效检索）
        self.index = faiss.IndexFlatL2(vector_dim)
        # 存储元数据（晶体ID、能量、应力等）
        self.metadata = []

    def extract_latent_features(self, model: nn.Module, data_loader: GraphDataLoader,
                                cif_metadata_dict: Dict[str, Dict], scaler=None):
        """
        从模型中提取潜在特征，同时结合cif原始元数据
        :param model: GAT-VAE模型
        :param data_loader: 图数据加载器
        :param cif_metadata_dict: {晶体ID: cif元数据}
        :param scaler: 归一化器
        """
        model.eval()
        all_latent = []
        all_meta = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                output = model(batch)

                # 提取潜在向量（用mu，更稳定）
                latent = output['mu'].cpu().numpy()
                all_latent.append(latent)

                # 提取元数据（结合cif原始数据）
                batch_num_nodes = batch.batch_num_nodes()
                start_idx = 0
                for idx, num_nodes in enumerate(batch_num_nodes):
                    # 基础元数据
                    crystal_id = f"crystal_{batch_idx}_{idx}"
                    # 从cif元数据中读取原始值（优先）
                    cif_meta = cif_metadata_dict.get(crystal_id, cif_metadata_dict.get(f"{batch_idx}_{idx}", {}))

                    # 能量：优先用cif原始值，无则用反归一化值
                    if cif_meta.get("total_energy_ev") is not None:
                        energy_ev = float(cif_meta["total_energy_ev"])  # 修复：强制转为Python float
                        energy_ev_text = cif_meta["total_energy_ev_text"]
                    else:
                        # 备用：从图数据反归一化
                        energy_norm = batch.ndata["total_energy"][start_idx][1].cpu().item()  # 确保是Python float
                        if scaler:
                            energy_tensor = scaler.inverse_transform_energy(torch.tensor([[0, energy_norm]]))
                            energy_ev = float(energy_tensor[0][1].item())  # 修复：强制转为Python float
                            energy_ev_text = f"# Total Energy (eV): {energy_ev:.4f}"
                        else:
                            energy_ev = float(energy_norm)  # 修复：强制转为Python float
                            energy_ev_text = f"# Total Energy (eV): {energy_ev:.4f}"

                    # 应力：优先用cif原始值，无则用反归一化值
                    if cif_meta.get("stress_tensor_3x3") is not None:
                        stress_3x3 = cif_meta["stress_tensor_3x3"]
                        stress_flat = cif_meta["stress_tensor_flat"]
                        stress_text = cif_meta["stress_tensor_text"]
                    else:
                        # 备用：从图数据反归一化
                        stress_norm = batch.ndata["stress_tensor_flat"][start_idx].cpu().numpy()
                        if scaler:
                            stress_tensor = scaler.inverse_transform_stress(torch.tensor(stress_norm))
                            stress_flat = stress_tensor.cpu().numpy().tolist()  # 修复：转为list
                            stress_3x3 = np.array(stress_flat).reshape(3, 3).tolist()  # 修复：转为list
                        else:
                            stress_flat = stress_norm.tolist()  # 修复：转为list
                            stress_3x3 = np.array(stress_flat).reshape(3, 3).tolist()  # 修复：转为list
                        # 生成应力文本
                        stress_text = "# Stress Tensor (3x3) [atomic units]\n"
                        for i in range(3):
                            row = stress_3x3[i]
                            stress_text += f"# Stress_Row_{i + 1}: {float(row[0]):.12f} {float(row[1]):.12f} {float(row[2]):.12f}\n"  # 修复：强制float

                    all_meta.append({
                        "crystal_id": crystal_id,
                        "num_atoms": int(num_nodes),  # 修复：强制转为Python int
                        "Total Energy (eV)": energy_ev_text,  # 原始文本
                        "Stress Tensor (3x3) [atomic units]": stress_text,  # 原始文本
                        "energy_eV": float(energy_ev),  # 修复：强制转为Python float
                        "stress_tensor_3x3": stress_3x3,  # 已确保是list
                        "stress_tensor_flat": stress_flat  # 已确保是list
                    })
                    start_idx += num_nodes

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"已处理{batch_idx + 1}批次，提取{(batch_idx + 1) * BATCH_SIZE_RAG}个晶体向量")

        # 拼接所有向量和元数据
        self.latent_features = np.concatenate(all_latent, axis=0).astype(np.float32)
        self.metadata = all_meta
        logger.info(f"特征提取完成，共{len(self.latent_features)}个晶体向量，维度{self.latent_features.shape}")

    def build_vector_db(self):
        """构建FAISS向量库（RAG外部知识库核心）- 修改为保存JSON元数据"""
        # 添加向量到索引
        self.index.add(self.latent_features)
        # 保存索引
        faiss.write_index(self.index, os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME))

        # 保存元数据为JSON文件（核心修改）
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"✅ RAG向量库构建完成！")
        logger.info(f"   - 向量索引路径: {os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME)}")
        logger.info(f"   - 元数据路径: {METADATA_PATH}")
        logger.info(f"   - 向量总数: {self.index.ntotal}")
        logger.info(f"   - 向量维度: {self.vector_dim}")

    def load_vector_db(self):
        """加载已构建的向量库（用于后续RAG检索）"""
        index_path = os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME)
        if os.path.exists(index_path) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(index_path)
            # 加载JSON格式的元数据
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ 加载已有的RAG向量库，共{self.index.ntotal}个向量")
        else:
            raise FileNotFoundError(
                f"RAG向量库文件缺失！请检查：\n1. 索引文件: {index_path}\n2. 元数据文件: {METADATA_PATH}")

    def retrieve_similar(self, query_vector: np.ndarray, top_k: int = 5):
        """RAG核心检索功能：根据查询向量找最相似的晶体"""
        # 归一化查询向量
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        # 检索top-k相似向量
        distances, indices = self.index.search(query_vector, top_k)

        # 整理检索结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "rank": int(i + 1),
                    "distance": float(distances[0][i]),  # 转换为JSON可序列化的float
                    "metadata": self.metadata[idx]
                })

        return results


# ==================== 数据集加载 ====================
class CrystalDataset(dgl.data.DGLDataset):
    """晶体数据集加载器（关联cif文件解析原始元数据）"""

    def __init__(self, split_type: str):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        # 存储cif元数据：{晶体ID: 解析后的元数据}
        self.cif_metadata = {}
        super().__init__(name=f'crystal_{split_type}')
        self.load()

    def process(self):
        self.graphs = []

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"划分文件不存在: {self.split_path}")

        with open(self.split_path, "r") as f:
            graph_files = [line.strip() for line in f.readlines() if line.strip()]

        for fname in graph_files:
            # 1. 加载图文件
            graph_path = os.path.join(PROCESSED_GRAPH_DIR, fname)
            if not os.path.exists(graph_path):
                logger.warning(f"图文件不存在: {graph_path}")
                continue

            # 2. 找到对应的cif文件（确保文件名匹配）
            cif_name = os.path.splitext(fname)[0] + ".cif"
            cif_path = os.path.join(CIF_ROOT_DIR, cif_name)

            # 3. 解析cif元数据
            crystal_id = os.path.splitext(fname)[0]  # 用文件名作为晶体ID
            self.cif_metadata[crystal_id] = parse_cif_metadata(cif_path)

            # 4. 加载图数据
            try:
                dgl_graph = torch.load(graph_path)
                assert dgl_graph.ndata["feat"].size(1) == NODE_FEAT_DIM, "节点特征维度不匹配"
                if dgl_graph.num_edges() > 0:
                    assert dgl_graph.edata["feat"].size(1) == EDGE_FEAT_DIM, "边特征维度不匹配"
                self.graphs.append(dgl_graph)
            except Exception as e:
                logger.error(f"加载图{graph_path}失败: {str(e)}")

        if len(self.graphs) == 0:
            raise RuntimeError(f"{self.split_type}数据集为空")
        logger.info(f"{self.split_type}数据集加载完成，共{len(self.graphs)}个图，解析{len(self.cif_metadata)}个cif文件")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_cif_metadata(self):
        """返回所有晶体的cif元数据"""
        return self.cif_metadata


# ==================== 参数解析（适配RAG场景） ====================
def parse_args():
    parser = argparse.ArgumentParser(description='晶体RAG外部知识库构建（对齐GAT-VAE模型）')
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'], help='计算设备')
    # 模型参数
    parser.add_argument('--latent_dim', type=int, default=64, help='潜在变量维度（必须匹配训练值）')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度（必须匹配训练值）')
    parser.add_argument('--num_heads', type=int, default=4, help='GAT注意力头数（必须匹配训练值）')
    # 路径参数
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_SAVE_DIR, "best_gat_vae.pth"),
                        help='预训练模型路径')
    parser.add_argument('--split_type', type=str, default='train', choices=['train', 'val', 'test'],
                        help='用于构建RAG的数据集类型')
    # RAG参数
    parser.add_argument('--top_k', type=int, default=5, help='RAG检索返回的相似晶体数')
    # 新增：cif路径参数
    parser.add_argument('--cif_dir', type=str, default=CIF_ROOT_DIR, help='cif文件存储目录')
    return parser.parse_args()


# ==================== 主函数（RAG知识库构建核心） ====================
def main():
    args = parse_args()
    # 更新cif根目录
    global CIF_ROOT_DIR
    CIF_ROOT_DIR = args.cif_dir

    # 1. 设备配置
    device = torch.device(args.device)
    logger.info(f"使用设备：{device}")

    # 2. 加载数据集（包含cif元数据解析）
    logger.info(f"加载{args.split_type}数据集，用于构建RAG知识库...")
    dataset = CrystalDataset(args.split_type)
    data_loader = GraphDataLoader(dataset, batch_size=BATCH_SIZE_RAG, shuffle=False)
    # 获取cif元数据
    cif_metadata = dataset.get_cif_metadata()

    # 3. 初始化模型
    logger.info("加载预训练GAT-VAE模型（参数1:1对齐）...")
    model = CrystalGATVAE(args).to(device)

    # 4. 加载模型权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        logger.info("✅ Checkpoint加载成功，开始处理模型权重")
    except Exception as e:
        raise RuntimeError(f"加载Checkpoint失败: {str(e)}")

    # 处理多GPU训练的module.前缀
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[key] = v

    # 严格加载
    model.load_state_dict(new_state_dict, strict=True)
    logger.info("✅ 模型权重加载成功！参数名100%匹配")

    # 5. 加载归一化器（用于元数据反归一化）
    scaler = None
    try:
        scaler = checkpoint.get('scaler', None)
        if scaler:
            logger.info("✅ 归一化器加载成功")
        else:
            logger.warning("⚠️ Checkpoint中未找到scaler，元数据将使用归一化后的值")
    except Exception as e:
        logger.warning(f"⚠️ 加载scaler失败: {str(e)}，元数据将使用归一化后的值")

    # 6. 构建RAG外部知识库
    logger.info("开始构建晶体RAG外部知识库...")
    rag_builder = CrystalRAGBuilder(vector_dim=args.latent_dim, device=device)

    # 6.1 提取潜在特征（结合cif原始元数据）
    rag_builder.extract_latent_features(model, data_loader, cif_metadata, scaler)

    # 6.2 构建FAISS向量库
    rag_builder.build_vector_db()

    # 7. 测试RAG检索（验证知识库可用性）
    logger.info("测试RAG检索功能...")
    # 取第一个晶体的向量作为查询
    test_query = rag_builder.latent_features[0]
    results = rag_builder.retrieve_similar(test_query, top_k=args.top_k)

    logger.info("RAG检索测试结果（Top-5相似晶体）:")
    for res in results:
        logger.info(f"\n=== 排名{res['rank']} | 距离{res['distance']:.4f} ===")
        logger.info(f"晶体ID: {res['metadata']['crystal_id']}")
        logger.info(f"原子数: {res['metadata']['num_atoms']}")
        logger.info(f"总能量: {res['metadata']['Total Energy (eV)']}")
        logger.info(f"应力张量: {res['metadata']['Stress Tensor (3x3) [atomic units]']}")

    logger.info("🎉 晶体RAG外部知识库构建完成！")


if __name__ == "__main__":
    main()