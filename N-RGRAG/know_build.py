import os
import dgl
import torch
import numpy as np
import faiss
import logging
import argparse
import json
from torch import nn
from dgl.nn import GlobalAttentionPooling, RelGraphConv
from dgl.dataloading import GraphDataLoader
from typing import List, Tuple, Dict, Optional
import warnings
from pathlib import Path
import re

# ========== RAG知识库专属配置 ==========
VECTOR_DB_DIR = "/home/nyx/N-RGRAG/know_base"
VECTOR_INDEX_NAME = "crystal_latent_index.faiss"
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "crystal_metadata.json")
VECTOR_DIM = 32 
BATCH_SIZE_RAG = 32

# 新增：cif文件根目录
CIF_ROOT_DIR = "/home/nyx/N-RGRAG/raw_cifs"

# ==================== 全局配置 ====================
PROCESSED_GRAPH_DIR = "/home/nyx/N-RGRAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/N-RGRAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/N-RGRAG/models"

# 特征维度配置
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
GRAPH_ATTR_DIM = 17  # 6晶胞+9应力+2能量
ENERGY_DIM = 2  # hartree + eV
STRESS_DIM = 9  # 应力张量展平
CELL_PARAMS_DIM = 6  # 晶胞参数
ENERGY_EV_DIM = 1  # 固定eV维度的索引

# 数值稳定配置
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
KL_LOSS_CLAMP = 100.0

# RGCN专属配置
NUM_RELATIONS = 4
NUM_BASES = 4
NUM_BLOCKS = None

warnings.filterwarnings("ignore")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [RGCN-RAG构建] %(message)s",
    handlers=[
        logging.FileHandler("rgcn_rag_know_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== CIF文件解析函数 ====================
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
                energy_val = float(match.group(1))
                metadata["total_energy_ev"] = energy_val
                metadata["total_energy_ev_text"] = line
                break

        # 2. 解析Stress Tensor (3x3) [atomic units]
        stress_lines = []
        stress_start = False
        stress_row_count = 0
        stress_pattern = re.compile(r"# Stress_Row_\d+:\s*([\d\-\.e\s]+)")

        for line in lines:
            if "# Stress Tensor (3x3) [atomic units]" in line:
                stress_start = True
                metadata["stress_tensor_text"] = line
                continue
            if stress_start and stress_row_count < 3:
                match = stress_pattern.search(line)
                if match:
                    stress_vals = [float(x) for x in match.group(1).split()]
                    if len(stress_vals) == 3:
                        stress_lines.append(stress_vals)
                        metadata["stress_tensor_text"] += "\n" + line
                        stress_row_count += 1

        # 转换为3x3数组和扁平9维数组
        if len(stress_lines) == 3:
            stress_3x3 = np.array(stress_lines, dtype=np.float64)
            metadata["stress_tensor_3x3"] = stress_3x3.tolist()
            metadata["stress_tensor_flat"] = stress_3x3.flatten().tolist()

    except Exception as e:
        logger.error(f"解析CIF文件{cif_path}失败: {str(e)}")

    return metadata


# ==================== CrystalDataScaler类 ====================
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
        """基于训练集计算归一化统计量，NaN过滤"""
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

        # 计算均值和标准差（添加小epsilon避免除0）
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
        # 节点特征归一化
        g.ndata["feat"] = (g.ndata["feat"] - self.node_feat_mean.to(g.device)) / self.node_feat_std.to(g.device)

        # 边特征归一化与边类型提取
        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean.to(g.device)) / self.edge_feat_std.to(g.device)
            # 提取边类型并限制范围
            g.edata["edge_type"] = g.edata["feat"][:, 2].round().long()
            g.edata["edge_type"] = torch.clamp(g.edata["edge_type"], 0, NUM_RELATIONS - 1)

        # 能量与应力归一化
        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean.to(g.device)) / self.energy_std.to(
            g.device)
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean.to(
            g.device)) / self.stress_std.to(g.device)

        return g

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
        """能量反归一化"""
        # 校验dim的有效性
        if self.energy_std is None or self.energy_mean is None:
            logger.warning("归一化器的energy_mean/energy_std未初始化，直接返回原始能量")
            return energy

        if dim >= len(self.energy_std) or dim < 0:
            logger.warning(f"无效的维度dim={dim}，自动切换为dim=0")
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
            logger.warning(f"不支持的能量张量维度: {energy.dim()}，返回原始张量")
            return energy

    def inverse_transform_stress(self, stress: torch.Tensor) -> torch.Tensor:
        """应力反归一化"""
        if self.stress_std is None or self.stress_mean is None:
            logger.warning("归一化器的stress_mean/stress_std未初始化，直接返回原始应力")
            return stress
        return stress * self.stress_std.to(stress.device) + self.stress_mean.to(stress.device)


# ==================== RGCN核心层 ====================
class EdgeTypeRGCN(nn.Module):
    """带边类型的RGCN层"""

    def __init__(self, in_feat: int, out_feat: int, num_rels: int, num_bases: int):
        super().__init__()
        self.num_rels = num_rels
        self.out_feat = out_feat

        # 核心：使用RelGraphConv
        self.rgcn = RelGraphConv(
            in_feat=in_feat,
            out_feat=out_feat,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=num_bases
        )

        # 激活函数和归一化，增强数值稳定性
        self.activation = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        """前向传播：RGCN编码 + 激活 + 归一化"""
        rgcn_out = self.rgcn(g, node_feats, edge_types)
        rgcn_out = self.activation(rgcn_out)
        rgcn_out = self.norm(rgcn_out)
        return rgcn_out


# ==================== RGCN编码器====================
class CrystalRGCNEncoder(nn.Module):
    """简化版RGCN编码器"""

    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 num_rels: int = NUM_RELATIONS,
                 num_bases: int = NUM_BASES):
        super().__init__()

        # 简化层数
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

        # 图池化
        final_dim = hidden_dim * 2
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        # 潜在变量分布（融入数值截断）
        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

        # 初始化
        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播：返回mu、logvar、节点嵌入"""
        node_feats = g.ndata["feat"]  # (N, 4)

        # 简化前向
        h = self.rgcn1(g, node_feats, edge_types)
        h = self.rgcn2(g, h, edge_types)

        # 图池化得到全局嵌入
        graph_emb = self.pooling(g, h)  # (B, hidden_dim*2)
        node_emb = h  # 保留节点嵌入，用于解码

        # 数值截断，防止极端值
        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        mu = torch.clamp(mu, min=-5, max=5)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


# ==================== 解码器====================
class CrystalDecoder(nn.Module):
    """解码器：能量/应力预测，融入节点数匹配和维度校验"""

    def __init__(self,
                 latent_dim: int = 32,
                 hidden_dim: int = 64,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()

        # 潜在变量投影
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU()
        )

        # 节点嵌入投影
        node_emb_input_dim = hidden_dim * 2
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 节点特征解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # 能量和应力预测头
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
        """前向传播：保留重构和预测功能，增加节点数校验"""
        batch_num_nodes = g.batch_num_nodes()
        B = len(batch_num_nodes)
        total_nodes = g.num_nodes()

        # 空图处理，避免报错
        if total_nodes == 0:
            logger.warning("批次中总节点数为0")
            return {
                'recon_node': torch.zeros(0, NODE_FEAT_DIM, device=z.device),
                'pred_energy': self.energy_predictor(z),
                'pred_stress': self.stress_predictor(z)
            }

        # 潜在变量投影与扩展
        z_proj = self.latent_proj(z)  # (B, hidden_dim*2)
        z_expanded = torch.zeros(total_nodes, z_proj.size(1), device=z.device)
        start_idx = 0
        for i in range(B):
            num_nodes = batch_num_nodes[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        # 节点嵌入投影与重构
        node_emb_proj = self.node_emb_proj(node_emb)  # (N, hidden_dim*2)
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)  # (total_nodes, 4)

        # 严格校验维度和节点数
        assert recon_node_feats.size(1) == NODE_FEAT_DIM, \
            f"解码器输出维度错误: {recon_node_feats.size(1)}，预期: {NODE_FEAT_DIM}"
        assert recon_node_feats.size(0) == total_nodes, \
            f"解码器节点数错误: {recon_node_feats.size(0)}，预期: {total_nodes}"

        # 能量和应力预测
        pred_energy = self.energy_predictor(z)
        pred_stress = self.stress_predictor(z)

        return {
            'recon_node': recon_node_feats,
            'pred_energy': pred_energy,
            'pred_stress': pred_stress
        }


# ==================== 完整RGCN-VAE模型 ====================
class CrystalRGCNVAE(nn.Module):
    """RGCN-VAE（能量/应力预测）"""

    def __init__(self, args):
        super().__init__()
        self.encoder = CrystalRGCNEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_rels=args.num_rels,
            num_bases=args.num_bases
        )

        self.decoder = CrystalDecoder(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            node_feat_dim=NODE_FEAT_DIM
        )

        self.latent_dim = args.latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧（融入std裁剪）"""
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 1e6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播：对齐RGCN输入要求（需要边类型）"""
        mu, logvar, node_emb = self.encoder(g, edge_types)
        z = self.reparameterize(mu, logvar)
        decode_out = self.decoder(z, g, node_emb)

        # 整合输出
        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
        output.update(decode_out)

        return output


# ==================== RAG知识库核心功能 ====================
class CrystalRAGBuilder:
    """晶体RAG外部知识库构建器"""

    def __init__(self, vector_dim: int, device: torch.device):
        self.vector_dim = vector_dim
        self.device = device
        # 创建向量库目录
        Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)

        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(vector_dim)
        # 存储元数据
        self.metadata = []

    def extract_latent_features(self, model: nn.Module, data_loader: GraphDataLoader,
                                cif_metadata_dict: Dict[str, Dict], scaler=None):
        """
        从RGCN-VAE模型中提取潜在特征，结合cif原始元数据（修复ID匹配，优化反归一化）
        :param model: RGCN-VAE模型
        :param data_loader: 带边类型和真实ID的图数据加载器
        :param cif_metadata_dict: {晶体ID: cif元数据}
        :param scaler: 归一化器
        """
        model.eval()
        all_latent = []
        all_meta = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 接收批量真实晶体ID
                batch_graph, batch_edge_types, batch_crystal_ids = batch
                batch_graph = batch_graph.to(self.device)
                batch_edge_types = batch_edge_types.to(self.device).long()

                # RGCN-VAE前向传播（需要传入边类型）
                output = model(batch_graph, batch_edge_types)

                # 提取潜在向量
                latent = output['mu'].cpu().numpy()
                all_latent.append(latent)

                # 提取元数据（结合cif原始数据）
                batch_num_nodes = batch_graph.batch_num_nodes()
                start_idx = 0
                # 遍历批量真实ID，不再拼接虚假ID
                for idx, (crystal_id, num_nodes) in enumerate(zip(batch_crystal_ids, batch_num_nodes)):
                    # 用真实晶体ID查询CIF元数据，优先路径生效
                    cif_meta = cif_metadata_dict.get(crystal_id, {})

                    # 能量：优先用cif原始值，无则用反归一化值
                    if cif_meta.get("total_energy_ev") is not None:
                        energy_ev = float(cif_meta["total_energy_ev"])
                        energy_ev_text = cif_meta["total_energy_ev_text"]
                        logger.debug(f"样本{crystal_id}：从CIF获取能量 {energy_ev:.4f} eV")
                    else:
                        # 从图数据反归一化（优化逻辑，增加详细日志）
                        energy_ev = 0.0
                        energy_ev_text = "# Total Energy (eV): 0.0000"
                        try:
                            # 步骤1：校验total_energy是否存在且有效
                            if not hasattr(batch_graph.ndata, "total_energy"):
                                logger.warning(f"样本{crystal_id} 无total_energy图数据，使用默认能量值")
                                continue
                            total_energy_tensor = batch_graph.ndata["total_energy"]
                            if total_energy_tensor.numel() == 0:
                                logger.warning(f"样本{crystal_id} total_energy张量为空，使用默认能量值")
                                continue

                            # 步骤2：正确获取图级能量（批量图中，取每个图的第一个节点能量作为图级能量）
                            # 批量图节点索引：start_idx 到 start_idx + num_nodes - 1
                            graph_energy_norm = total_energy_tensor[start_idx]  # 取第一个节点作为图级能量
                            if graph_energy_norm.dim() > 1:
                                graph_energy_norm = graph_energy_norm.squeeze()  # 压缩多余维度

                            # 步骤3：安全反归一化
                            if scaler:
                                energy_input = torch.tensor([graph_energy_norm], dtype=torch.float32)
                                energy_tensor = scaler.inverse_transform_energy(energy_input, dim=ENERGY_EV_DIM)
                                if energy_tensor.numel() > 0:
                                    energy_ev = float(energy_tensor.item())
                                    energy_ev = max(min(energy_ev, 1e6), -1e6)  # 限制极端值
                                    energy_ev_text = f"# Total Energy (eV): {energy_ev:.4f}"
                                    logger.debug(f"样本{crystal_id}：从图数据反归一化获取能量 {energy_ev:.4f} eV")
                                else:
                                    logger.warning(f"样本{crystal_id} 反归一化后能量张量为空，使用默认能量值")
                            else:
                                logger.warning(f"样本{crystal_id} 无归一化器scaler，无法反归一化，使用默认能量值")

                        except Exception as e:
                            logger.warning(f"样本{crystal_id} 反归一化获取能量失败: {str(e)}，使用默认能量值")

                    # 应力：优先用cif原始值，无则用反归一化值（保持逻辑，优化ID匹配）
                    if cif_meta.get("stress_tensor_3x3") is not None:
                        stress_3x3 = cif_meta["stress_tensor_3x3"]
                        stress_flat = cif_meta["stress_tensor_flat"]
                        stress_text = cif_meta["stress_tensor_text"]
                    else:
                        # 备用：从图数据反归一化
                        stress_3x3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        stress_flat = [0.0] * 9
                        try:
                            stress_norm = batch_graph.ndata["stress_tensor_flat"][start_idx].cpu().numpy()
                            if scaler:
                                stress_tensor = scaler.inverse_transform_stress(torch.tensor(stress_norm))
                                stress_flat = stress_tensor.cpu().numpy().tolist()
                                stress_3x3 = np.array(stress_flat).reshape(3, 3).tolist()
                            else:
                                stress_flat = stress_norm.tolist()
                                stress_3x3 = np.array(stress_flat).reshape(3, 3).tolist()
                        except Exception as e:
                            logger.warning(f"样本{crystal_id} 反归一化获取应力失败: {str(e)}，使用默认应力值")
                        # 生成应力文本
                        stress_text = "# Stress Tensor (3x3) [atomic units]\n"
                        for i in range(3):
                            row = stress_3x3[i]
                            stress_text += f"# Stress_Row_{i + 1}: {float(row[0]):.12f} {float(row[1]):.12f} {float(row[2]):.12f}\n"

                    # 整理元数据（使用真实晶体ID）
                    all_meta.append({
                        "crystal_id": crystal_id,
                        "num_atoms": int(num_nodes),
                        "Total Energy (eV)": energy_ev_text,
                        "Stress Tensor (3x3) [atomic units]": stress_text,
                        "energy_eV": float(energy_ev),
                        "stress_tensor_3x3": stress_3x3,
                        "stress_tensor_flat": stress_flat
                    })
                    start_idx += num_nodes

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"已处理{batch_idx + 1}批次，提取{(batch_idx + 1) * BATCH_SIZE_RAG}个晶体向量")

        # 拼接所有向量和元数据（增加非空校验）
        if all_latent:
            self.latent_features = np.concatenate(all_latent, axis=0).astype(np.float32)
        else:
            self.latent_features = np.array([], dtype=np.float32).reshape(0, self.vector_dim)
            logger.warning("未提取到任何潜在特征，向量库为空")
        self.metadata = all_meta
        logger.info(f"特征提取完成，共{len(self.latent_features)}个晶体向量，维度{self.latent_features.shape}")

        # 拼接所有向量和元数据（增加非空校验）
        if all_latent:
            self.latent_features = np.concatenate(all_latent, axis=0).astype(np.float32)
        else:
            self.latent_features = np.array([], dtype=np.float32).reshape(0, self.vector_dim)
            logger.warning("未提取到任何潜在特征，向量库为空")
        self.metadata = all_meta
        logger.info(f"特征提取完成，共{len(self.latent_features)}个晶体向量，维度{self.latent_features.shape}")

    def build_vector_db(self):
        """构建FAISS向量库，保存JSON元数据"""
        # 添加向量到索引（非空校验）
        if self.latent_features is not None and len(self.latent_features) > 0:
            self.index.add(self.latent_features)
        else:
            logger.warning("无有效潜在特征，无法构建向量索引")

        # 保存索引
        faiss.write_index(self.index, os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME))

        # 保存元数据为JSON文件
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"✅ RGCN-RAG向量库构建完成！")
        logger.info(f"   - 向量索引路径: {os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME)}")
        logger.info(f"   - 元数据路径: {METADATA_PATH}")
        logger.info(f"   - 向量总数: {self.index.ntotal}")
        logger.info(f"   - 向量维度: {self.vector_dim}")

    def load_vector_db(self):
        """加载已构建的RGCN-RAG向量库"""
        index_path = os.path.join(VECTOR_DB_DIR, VECTOR_INDEX_NAME)
        if os.path.exists(index_path) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(index_path)
            # 加载JSON格式的元数据
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ 加载已有的RGCN-RAG向量库，共{self.index.ntotal}个向量")
        else:
            raise FileNotFoundError(
                f"RGCN-RAG向量库文件缺失！请检查：\n1. 索引文件: {index_path}\n2. 元数据文件: {METADATA_PATH}")

    def retrieve_similar(self, query_vector: np.ndarray, top_k: int = 5):
        """RAG核心检索功能：根据查询向量找最相似的晶体"""
        # 归一化查询向量
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        # 检索top-k相似向量（增加索引非空校验）
        if self.index.ntotal == 0:
            logger.warning("向量库为空，无法进行检索")
            return []

        distances, indices = self.index.search(query_vector, top_k)

        # 整理检索结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "rank": int(i + 1),
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results


# ==================== 数据集加载 ====================
class CrystalDataset(dgl.data.DGLDataset):
    """晶体数据集加载器（关联cif文件，返回图和边类型，优化ID存储）"""
    def __init__(self, split_type: str):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        # 存储cif元数据：{晶体ID: 解析后的元数据}
        self.cif_metadata = {}
        # 存储边类型列表（RGCN必需）
        self.edge_types_list = []
        # 新增：存储图对应的原始ID列表，便于后续匹配
        self.graph_id_list = []
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

            # 2. 找到对应的cif文件
            cif_name = os.path.splitext(fname)[0] + ".cif"
            cif_path = os.path.join(CIF_ROOT_DIR, cif_name)

            # 3. 解析cif元数据（优化晶体ID存储，保留原始文件名ID）
            crystal_id = os.path.splitext(fname)[0]
            self.graph_id_list.append(crystal_id)
            self.cif_metadata[crystal_id] = parse_cif_metadata(cif_path)

            # 4. 加载图数据，处理边类型（RGCN必需）
            try:
                dgl_graph = torch.load(graph_path)
                assert dgl_graph.ndata["feat"].size(1) == NODE_FEAT_DIM, "节点特征维度不匹配"

                # 添加自环，避免节点入度为0
                dgl_graph = dgl.add_self_loop(dgl_graph)

                # 提取并处理边类型
                edge_types = torch.tensor([], dtype=torch.long, device=dgl_graph.device)
                if dgl_graph.num_edges() > 0 and 'feat' in dgl_graph.edata:
                    if torch.isnan(dgl_graph.edata["feat"]).any():
                        logger.warning(f"图{graph_path}边特征含NaN/Inf，跳过")
                        continue
                    # 提取边类型并限制范围
                    edge_types = dgl_graph.edata["feat"][:, 2].round().long()
                    edge_types = torch.clamp(edge_types, 0, NUM_RELATIONS - 1)

                self.graphs.append(dgl_graph)
                self.edge_types_list.append(edge_types)

            except Exception as e:
                logger.error(f"加载图{graph_path}失败: {str(e)}")

        if len(self.graphs) == 0:
            raise RuntimeError(f"{self.split_type}数据集为空")
        logger.info(f"{self.split_type}数据集加载完成，共{len(self.graphs)}个图，解析{len(self.cif_metadata)}个cif文件")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """返回图、对应的边类型、真实晶体ID"""
        return self.graphs[idx], self.edge_types_list[idx], self.graph_id_list[idx]

    def get_cif_metadata(self):
        """返回所有晶体的cif元数据"""
        return self.cif_metadata

    def get_graph_ids(self):
        """返回所有图的原始ID列表"""
        return self.graph_id_list


# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='晶体RGCN-RAG外部知识库构建（对齐RGCN-VAE模型）')
    #默认值改为cpu，避免DGL CUDA报错
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='计算设备（cuda需要安装对应版本的DGL-CUDA）')
    # RGCN模型参数
    parser.add_argument('--latent_dim', type=int, default=32, help='潜在变量维度（匹配RGCN-VAE训练值）')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度（匹配RGCN-VAE训练值）')
    parser.add_argument('--num_rels', type=int, default=4, help='RGCN关系类型数量（匹配训练值）')
    parser.add_argument('--num_bases', type=int, default=4, help='RGCN基函数数量（匹配训练值）')
    # 路径参数
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_SAVE_DIR, "best_rgcn_vae_stable.pth"),
                        help='预训练RGCN-VAE模型路径')
    parser.add_argument('--split_type', type=str, default='train', choices=['train', 'val', 'test'],
                        help='用于构建RAG的数据集类型')
    # RAG参数
    parser.add_argument('--top_k', type=int, default=5, help='RAG检索返回的相似晶体数')
    parser.add_argument('--cif_dir', type=str, default=CIF_ROOT_DIR, help='cif文件存储目录')
    return parser.parse_args()


# ==================== 主函数 ====================
def main():
    args = parse_args()
    # 更新cif根目录
    global CIF_ROOT_DIR
    CIF_ROOT_DIR = args.cif_dir

    # 1.  强制使用cpu设备，屏蔽cuda相关操作
    device = torch.device(args.device)
    logger.info(f"使用设备：{device}（若需cuda加速，请安装对应版本的DGL-CUDA）")

    # 2. 加载数据集（包含cif元数据解析和RGCN边类型）
    logger.info(f"加载{args.split_type}数据集，用于构建RGCN-RAG知识库...")
    dataset = CrystalDataset(args.split_type)

    # 自定义collate_fn（批量处理图、边类型、真实晶体ID 【关键修改】）
    def collate_fn(batch):
        graphs, edge_types, crystal_ids = zip(*batch)
        batched_graph = dgl.batch(graphs)
        batched_edge_types = torch.cat(edge_types, dim=0) if edge_types[0].numel() > 0 else torch.tensor([],
                                                                                                         dtype=torch.long)
        return batched_graph, batched_edge_types, crystal_ids  # 返回批量真实ID

    data_loader = GraphDataLoader(dataset, batch_size=BATCH_SIZE_RAG, shuffle=False, collate_fn=collate_fn)
    # 获取cif元数据
    cif_metadata = dataset.get_cif_metadata()

    # 3. 初始化RGCN-VAE模型： 模型加载到cpu设备
    logger.info("加载预训练RGCN-VAE模型（参数1:1对齐）...")
    model = CrystalRGCNVAE(args).to(device)

    # 4. 加载模型权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    # 加载checkpoint，指定map_location为cpu，避免CUDA设备不匹配
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
    logger.info("✅ RGCN-VAE模型权重加载成功！参数名100%匹配")

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

    # 6. 构建RGCN-RAG外部知识库
    logger.info("开始构建晶体RGCN-RAG外部知识库...")
    rag_builder = CrystalRAGBuilder(vector_dim=args.latent_dim, device=device)

    # 6.1 提取潜在特征（结合cif原始元数据）
    rag_builder.extract_latent_features(model, data_loader, cif_metadata, scaler)

    # 6.2 构建FAISS向量库
    rag_builder.build_vector_db()

    # 7. 测试RGCN-RAG检索（验证知识库可用性，增加非空校验）
    logger.info("测试RGCN-RAG检索功能...")
    # 取第一个晶体的向量作为查询（增加非空校验）
    if len(rag_builder.latent_features) > 0:
        test_query = rag_builder.latent_features[0]
        results = rag_builder.retrieve_similar(test_query, top_k=args.top_k)

        logger.info("RGCN-RAG检索测试结果（Top-5相似晶体）:")
        for res in results:
            logger.info(f"\n=== 排名{res['rank']} | 距离{res['distance']:.4f} ===")
            logger.info(f"晶体ID: {res['metadata']['crystal_id']}")
            logger.info(f"原子数: {res['metadata']['num_atoms']}")
            logger.info(f"总能量: {res['metadata']['Total Energy (eV)']}")
            logger.info(f"应力张量: {res['metadata']['Stress Tensor (3x3) [atomic units]'][:100]}...")
    else:
        logger.warning("无有效潜在特征，跳过检索测试")

    logger.info("🎉 晶体RGCN-RAG外部知识库构建完成！")


if __name__ == "__main__":
    main()