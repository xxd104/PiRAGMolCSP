import os
import dgl
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from torch import nn
from dgl.nn import GlobalAttentionPooling, RelGraphConv
from dgl.dataloading import GraphDataLoader
from typing import List, Tuple, Dict, Optional
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
# 路径配置
PROCESSED_GRAPH_DIR = "/home/nyx/N-RGRAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/N-RGRAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/N-RGRAG/models"
VISUALIZATION_DIR = "/home/nyx/N-RGRAG/models_vis"

# 特征维度配置（完全保留）
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
GRAPH_ATTR_DIM = 17  # 6晶胞+9应力+2能量
ENERGY_DIM = 2  # hartree + eV
STRESS_DIM = 9  # 应力张量展平
CELL_PARAMS_DIM = 6  # 晶胞参数
ENERGY_EV_DIM = 1  # 固定eV维度的索引

# 训练配置（完全保留）
RANDOM_SEED = 24
MAX_FORCE_NORM = 1.0
MAX_STRESS_NORM = 1.0
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
KL_LOSS_CLAMP = 100.0
GRAD_CLIP_MAX_NORM = 0.1
KL_MAX_WEIGHT = 0.01

# RGCN专属配置（完全保留）
NUM_RELATIONS = 4
NUM_BASES = 4
NUM_BLOCKS = None

# 可视化配置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rgcn_vae_optimized_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 数据归一化工具 ====================
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
        """基于训练集计算归一化统计量，增加NaN过滤"""
        node_feats = []
        edge_feats = []
        energy_list = []
        stress_list = []

        for g in graphs:
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

        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean.to(g.device)) / self.edge_feat_std.to(g.device)
            g.edata["edge_type"] = g.edata["feat"][:, 2].round().long()
            g.edata["edge_type"] = torch.clamp(g.edata["edge_type"], 0, NUM_RELATIONS - 1)

        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean.to(g.device)) / self.energy_std.to(
            g.device)
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean.to(
            g.device)) / self.stress_std.to(g.device)

        return g

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
        """能量反归一化（适配所有维度输入）"""
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
        """应力反归一化"""
        return stress * self.stress_std.to(stress.device) + self.stress_mean.to(stress.device)


# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='优化版RGCN-VAE模型训练')
    # 设备与基础参数（完全保留）
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')

    # 模型参数（完全保留）
    parser.add_argument('--latent_dim', type=int, default=32, help='潜在变量维度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='兼容冗余参数')
    parser.add_argument('--kl_warmup', type=int, default=50, help='KL退火轮数')

    # RGCN参数（完全保留）
    parser.add_argument('--num_rels', type=int, default=4, help='关系类型数量')
    parser.add_argument('--num_bases', type=int, default=4, help='RGCN基函数数量')

    # 保存与可视化参数（完全保留）
    parser.add_argument('--model_save_interval', type=int, default=5, help='模型保存间隔')
    parser.add_argument('--vis_interval', type=int, default=10, help='可视化保存间隔')

    # 损失权重（完全保留）
    parser.add_argument('--gen_loss_weight', type=float, default=1.0, help='生成损失权重')
    parser.add_argument('--energy_loss_weight', type=float, default=0.5, help='能量损失权重')
    parser.add_argument('--force_loss_weight', type=float, default=0.3, help='力场损失权重')

    return parser.parse_args()


# ==================== 核心RGCN层（优化版：残差连接+GELU） ====================
class EdgeTypeRGCN(nn.Module):
    """带边类型、残差连接和GELU的RGCN层"""

    def __init__(self, in_feat: int, out_feat: int, num_rels: int, num_bases: int):
        super().__init__()
        self.num_rels = num_rels
        self.out_feat = out_feat

        self.rgcn = RelGraphConv(
            in_feat=in_feat,
            out_feat=out_feat,
            num_rels=num_rels,
            regularizer='basis',
            num_bases=num_bases
        )

        self.activation = nn.GELU()  # 替换为GELU，提升梯度流动性
        self.norm = nn.LayerNorm(out_feat)

        # 残差连接：维度对齐
        self.residual = nn.Linear(in_feat, out_feat) if in_feat != out_feat else nn.Identity()

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        residual = self.residual(node_feats)
        rgcn_out = self.rgcn(g, node_feats, edge_types)
        rgcn_out = rgcn_out + residual  # 残差连接
        rgcn_out = self.activation(rgcn_out)
        rgcn_out = self.norm(rgcn_out)
        return rgcn_out


# ==================== VAE编码器（优化版：更好的初始化） ====================
class CrystalRGCNEncoder(nn.Module):
    """RGCN编码器（带残差和Kaiming初始化）"""

    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 num_rels: int = 4,
                 num_bases: int = 4):
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
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Kaiming初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # 特殊初始化logvar，确保初始稳定
        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feats = g.ndata["feat"]

        h = self.rgcn1(g, node_feats, edge_types)
        h = self.rgcn2(g, h, edge_types)

        graph_emb = self.pooling(g, h)
        node_emb = h

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        mu = torch.clamp(mu, min=-5, max=5)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


# ==================== VAE解码器（优化版：LayerNorm+增强边解码） ====================
class CrystalDecoder(nn.Module):
    """解码器（带LayerNorm和增强边特征重构）"""

    def __init__(self,
                 latent_dim: int = 32,
                 hidden_dim: int = 64,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2)
        )

        node_emb_input_dim = hidden_dim * 2
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1)
        )

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # 增强边解码器：增加隐藏层
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

        self.energy_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, energy_dim)
        )

        self.stress_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stress_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, node_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_num_nodes = g.batch_num_nodes()
        B = len(batch_num_nodes)
        total_nodes = g.num_nodes()

        if total_nodes == 0:
            logger.warning("批次中总节点数为0")
            return {
                'recon_node': torch.zeros(0, NODE_FEAT_DIM, device=z.device),
                'recon_edge': torch.zeros(0, EDGE_FEAT_DIM, device=z.device),
                'pred_energy': self.energy_predictor(z),
                'pred_stress': self.stress_predictor(z)
            }

        z_proj = self.latent_proj(z)
        z_expanded = torch.zeros(total_nodes, z_proj.size(1), device=z.device)
        start_idx = 0
        for i in range(B):
            num_nodes = batch_num_nodes[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        node_emb_proj = self.node_emb_proj(node_emb)
        node_input = z_expanded + node_emb_proj
        recon_node_feats = self.node_decoder(node_input)

        recon_edge_feats = torch.tensor([], device=z.device)
        if g.num_edges() > 0:
            src, dst = g.edges()
            z_src = z_expanded[src]
            z_dst = z_expanded[dst]
            edge_input = torch.cat([z_src, z_dst], dim=1)
            recon_edge_feats = self.edge_decoder(edge_input)

        assert recon_node_feats.size(1) == NODE_FEAT_DIM
        assert recon_node_feats.size(0) == total_nodes
        if g.num_edges() > 0:
            assert recon_edge_feats.size(1) == EDGE_FEAT_DIM
            assert recon_edge_feats.size(0) == g.num_edges()

        pred_energy = self.energy_predictor(z)
        pred_stress = self.stress_predictor(z)

        return {
            'recon_node': recon_node_feats,
            'recon_edge': recon_edge_feats,
            'pred_energy': pred_energy,
            'pred_stress': pred_stress
        }


# ==================== 完整RGCN-VAE模型 ====================
class CrystalRGCNVAE(nn.Module):
    """RGCN-VAE（保持原有架构）"""

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
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM
        )

        self.latent_dim = args.latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 1e6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph, edge_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, node_emb = self.encoder(g, edge_types)
        z = self.reparameterize(mu, logvar)
        decode_out = self.decoder(z, g, node_emb)

        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
        output.update(decode_out)

        return output


# ==================== 损失函数（优化版：Huber Loss+平滑KL退火） ====================
class CrystalVAELoss(nn.Module):
    """复合损失函数：Huber Loss提高鲁棒性"""

    def __init__(self, args):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)  # 对异常值更鲁棒
        self.kl_warmup = args.kl_warmup
        self.current_kl_weight = 0.0

        self.gen_weight = args.gen_loss_weight
        self.energy_weight = args.energy_loss_weight
        self.force_weight = args.force_loss_weight

    def set_kl_weight(self, epoch: int):
        """更平滑的KL退火"""
        if epoch <= self.kl_warmup:
            self.current_kl_weight = KL_MAX_WEIGHT * (epoch / self.kl_warmup)
        else:
            self.current_kl_weight = KL_MAX_WEIGHT

    def compute_force_loss(self, pred_stress: torch.Tensor, true_stress: torch.Tensor,
                           pred_forces: torch.Tensor, true_forces: torch.Tensor) -> torch.Tensor:
        pred_stress = torch.clamp(pred_stress, -1e3, 1e3)
        pred_forces = torch.clamp(pred_forces, -1e3, 1e3)

        stress_loss = self.huber(pred_stress, true_stress)
        force_loss = self.huber(pred_forces, true_forces)

        return (stress_loss + force_loss) / 2

    def forward(self, model_output: Dict[str, torch.Tensor],
                g: dgl.DGLGraph, edge_types: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.set_kl_weight(epoch)

        true_node = g.ndata["feat"]
        true_edge = g.edata["feat"] if g.num_edges() > 0 else torch.tensor([], device=true_node.device,
                                                                           dtype=true_node.dtype)

        if torch.isnan(true_node).any():
            logger.warning("真实节点特征包含NaN")
        if g.num_edges() > 0 and torch.isnan(true_edge).any():
            logger.warning("真实边特征包含NaN")

        batch_num_nodes = g.batch_num_nodes()
        true_energy_list = []
        true_stress_list = []
        true_forces_list = []

        start_idx = 0
        for num_nodes in batch_num_nodes:
            end_idx = start_idx + num_nodes
            true_energy_list.append(g.ndata["total_energy"][start_idx])
            true_stress_list.append(g.ndata["stress_tensor_flat"][start_idx])
            true_forces_list.append(true_node[start_idx:end_idx, 1:4])
            start_idx = end_idx

        true_energy = torch.stack(true_energy_list)
        true_stress = torch.stack(true_stress_list)
        true_forces = torch.cat(true_forces_list)

        # 节点重构损失（Huber）
        recon_node_pred = torch.clamp(model_output['recon_node'], -1e3, 1e3)
        if recon_node_pred.size(0) != true_node.size(0):
            min_nodes = min(recon_node_pred.size(0), true_node.size(0))
            recon_node_pred = recon_node_pred[:min_nodes]
            true_node = true_node[:min_nodes]
            logger.warning(f"重构节点数不匹配，截断至{min_nodes}")
        recon_node_loss = self.huber(recon_node_pred, true_node)

        # 边重构损失（Huber）
        recon_edge_loss = torch.tensor(0.0, device=true_node.device)
        if g.num_edges() > 0 and model_output['recon_edge'].numel() > 0:
            recon_edge_pred = torch.clamp(model_output['recon_edge'], -1e3, 1e3)
            if recon_edge_pred.size(0) != true_edge.size(0):
                min_edges = min(recon_edge_pred.size(0), true_edge.size(0))
                recon_edge_pred = recon_edge_pred[:min_edges]
                true_edge = true_edge[:min_edges]
                logger.warning(f"重构边数不匹配，截断至{min_edges}")
            recon_edge_loss = self.huber(recon_edge_pred, true_edge)

        # KL损失
        mu = model_output['mu']
        logvar = model_output['logvar']
        kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss_per_sample = torch.clamp(kl_loss_per_sample, 0, KL_LOSS_CLAMP)
        kl_loss = kl_loss_per_sample.mean()

        gen_loss = (recon_node_loss + recon_edge_loss) + self.current_kl_weight * kl_loss

        # 能量和力场损失
        pred_energy = torch.clamp(model_output['pred_energy'], -1e3, 1e3)
        energy_loss = self.huber(pred_energy, true_energy)

        pred_forces = model_output['recon_node'][:, 1:4]
        force_loss = self.compute_force_loss(model_output['pred_stress'], true_stress, pred_forces, true_forces)

        # NaN保护
        if torch.isnan(energy_loss) or torch.isinf(energy_loss):
            energy_loss = torch.tensor(0.0, device=true_node.device)
            logger.warning("能量损失为NaN/Inf，重置为0")
        if torch.isnan(force_loss) or torch.isinf(force_loss):
            force_loss = torch.tensor(0.0, device=true_node.device)
            logger.warning("力场损失为NaN/Inf，重置为0")
        if torch.isnan(recon_edge_loss) or torch.isinf(recon_edge_loss):
            recon_edge_loss = torch.tensor(0.0, device=true_node.device)
            logger.warning("边重构损失为NaN/Inf，重置为0")

        total_loss = (
                self.gen_weight * gen_loss +
                self.energy_weight * energy_loss +
                self.force_weight * force_loss
        )

        loss_dict = {
            'total_loss': total_loss.item() if torch.isfinite(total_loss) else 1e3,
            'gen_loss': gen_loss.item() if torch.isfinite(gen_loss) else 0.0,
            'recon_node_loss': recon_node_loss.item() if torch.isfinite(recon_node_loss) else 0.0,
            'recon_edge_loss': recon_edge_loss.item() if torch.isfinite(recon_edge_loss) else 0.0,
            'kl_loss': kl_loss.item() if torch.isfinite(kl_loss) else 0.0,
            'energy_loss': energy_loss.item() if torch.isfinite(energy_loss) else 0.0,
            'force_loss': force_loss.item() if torch.isfinite(force_loss) else 0.0
        }

        return total_loss, loss_dict


# ==================== 数据集加载（优化版：数据增强） ====================
class CrystalDataset(dgl.data.DGLDataset):
    """增强数据验证和数据增强的晶体数据集"""

    def __init__(self, split_type: str, args):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        self.args = args
        super().__init__(name=f'crystal_{split_type}')
        self.load()

    def process(self):
        self.graphs = []
        self.edge_types_list = []

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"划分文件不存在: {self.split_path}")

        with open(self.split_path, "r") as f:
            graph_files = [line.strip() for line in f.readlines() if line.strip()]

        for fname in graph_files:
            path = os.path.join(PROCESSED_GRAPH_DIR, fname)
            if not os.path.exists(path):
                logger.warning(f"图文件不存在: {path}")
                continue

            try:
                dgl_graph = torch.load(path)

                if 'feat' not in dgl_graph.ndata:
                    logger.warning(f"图{path}无节点特征，跳过")
                    continue
                if dgl_graph.ndata["feat"].size(1) != NODE_FEAT_DIM:
                    logger.warning(f"图{path}节点维度错误，跳过")
                    continue
                if torch.isnan(dgl_graph.ndata["feat"]).any() or torch.isinf(dgl_graph.ndata["feat"]).any():
                    logger.warning(f"图{path}节点特征含NaN/Inf，跳过")
                    continue

                dgl_graph = dgl.add_self_loop(dgl_graph)

                edge_types = torch.tensor([], dtype=torch.long, device=dgl_graph.device)
                if dgl_graph.num_edges() > 0 and 'feat' in dgl_graph.edata:
                    if torch.isnan(dgl_graph.edata["feat"]).any():
                        logger.warning(f"图{path}边特征含NaN/Inf，跳过")
                        continue
                    edge_types = dgl_graph.edata["feat"][:, 2].round().long()
                    edge_types = torch.clamp(edge_types, 0, self.args.num_rels - 1)

                self.graphs.append(dgl_graph)
                self.edge_types_list.append(edge_types)

            except Exception as e:
                logger.error(f"加载图{path}失败: {str(e)}")

        if len(self.graphs) == 0:
            raise RuntimeError(f"{self.split_type}数据集为空")
        logger.info(f"{self.split_type}数据集加载完成，共{len(self.graphs)}个图")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """返回图和边类型（训练时加节点噪声增强）"""
        graph = self.graphs[idx]
        edge_types = self.edge_types_list[idx]

        # 数据增强：仅训练集加小高斯噪声
        if self.split_type == 'train':
            node_feat = graph.ndata['feat']
            noise = torch.randn_like(node_feat) * 0.01  # 不破坏数据分布的小噪声
            graph.ndata['feat'] = node_feat + noise

        return graph, edge_types


# ==================== 可视化函数（保持不变） ====================
def visualize_loss_curves(train_losses: Dict[str, List[float]],
                          val_losses: Dict[str, List[float]],
                          save_path: str):
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses['total_loss'], label='Train', color='blue')
    plt.plot(val_losses['total_loss'], label='Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(train_losses['gen_loss'], label='Train', color='green')
    plt.plot(val_losses['gen_loss'], label='Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Generation Loss')
    plt.title('Generation Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(train_losses['recon_node_loss'], label='Train', color='purple')
    plt.plot(val_losses['recon_node_loss'], label='Validation', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Node Reconstruction Loss')
    plt.title('Node Recon Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(train_losses['recon_edge_loss'], label='Train', color='cyan')
    plt.plot(val_losses['recon_edge_loss'], label='Validation', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Edge Reconstruction Loss')
    plt.title('Edge Recon Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(train_losses['energy_loss'], label='Train', color='darkgreen')
    plt.plot(val_losses['energy_loss'], label='Validation', color='darkred')
    plt.xlabel('Epochs')
    plt.ylabel('Energy Loss')
    plt.title('Energy Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.plot(train_losses['force_loss'], label='Train', color='darkblue')
    plt.plot(val_losses['force_loss'], label='Validation', color='darkorange')
    plt.xlabel('Epochs')
    plt.ylabel('Force Loss')
    plt.title('Force Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"损失曲线已保存至: {save_path}")


def visualize_tsne_latent_space(model: nn.Module, val_loader: GraphDataLoader, scaler: CrystalDataScaler,
                                device: torch.device, save_path: str):
    model.eval()
    mu_list = []
    energy_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch_graph, batch_edge_types = batch
            batch_graph = batch_graph.to(device)
            batch_edge_types = batch_edge_types.to(device).long()

            output = model(batch_graph, batch_edge_types)
            mu_list.append(output['mu'].cpu())

            batch_num_nodes = batch_graph.batch_num_nodes()
            start_idx = 0
            for num_nodes in batch_num_nodes:
                energy_ev_norm = batch_graph.ndata["total_energy"][start_idx][ENERGY_EV_DIM]
                energy_ev = scaler.inverse_transform_energy(energy_ev_norm).item()
                energy_list.append(energy_ev)
                start_idx += num_nodes

    if not mu_list:
        logger.warning("无潜在变量数据，跳过t-SNE可视化")
        return
    mu = torch.cat(mu_list, dim=0).numpy()
    energy = np.array(energy_list)

    if mu.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, mu.shape[0] - 1))
        mu_tsne = tsne.fit_transform(mu)
    else:
        mu_tsne = mu

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], c=energy, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Total Energy (eV)')
    plt.title('t-SNE Visualization of Latent Space (Colored by Energy)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"t-SNE潜在空间可视化已保存至: {save_path}")


def visualize_force_heatmap(model: nn.Module, val_loader: GraphDataLoader, scaler: CrystalDataScaler,
                            device: torch.device, save_path: str):
    model.eval()
    true_forces_all = []
    pred_forces_all = []

    with torch.no_grad():
        for batch in val_loader:
            batch_graph, batch_edge_types = batch
            batch_graph = batch_graph.to(device)
            batch_edge_types = batch_edge_types.to(device).long()

            output = model(batch_graph, batch_edge_types)

            true_node_norm = batch_graph.ndata["feat"].cpu()
            true_node = true_node_norm * scaler.node_feat_std + scaler.node_feat_mean
            true_forces = true_node[:, 1:4].numpy()

            pred_node_norm = output['recon_node'][:, :4].cpu()
            pred_node = pred_node_norm * scaler.node_feat_std + scaler.node_feat_mean
            pred_forces = pred_node[:, 1:4].numpy()

            n_nodes = min(100, true_forces.shape[0])
            true_forces_all = true_forces[:n_nodes]
            pred_forces_all = pred_forces[:n_nodes]
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    im1 = ax1.imshow(true_forces_all.T, cmap='RdBu_r', aspect='auto')
    ax1.set_title('True Atomic Forces (X/Y/Z)')
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Force Dimension')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['X', 'Y', 'Z'])
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(pred_forces_all.T, cmap='RdBu_r', aspect='auto')
    ax2.set_title('Predicted Atomic Forces (X/Y/Z)')
    ax2.set_xlabel('Atom Index')
    ax2.set_ylabel('Force Dimension')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['X', 'Y', 'Z'])
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"受力热力图已保存至: {save_path}")


def visualize_energy_prediction(model: nn.Module, val_loader: GraphDataLoader, scaler: CrystalDataScaler,
                                device: torch.device, save_path: str):
    model.eval()
    true_energy = []
    pred_energy = []

    with torch.no_grad():
        for batch in val_loader:
            batch_graph, batch_edge_types = batch
            batch_graph = batch_graph.to(device)
            batch_edge_types = batch_edge_types.to(device).long()

            output = model(batch_graph, batch_edge_types)

            batch_num_nodes = batch_graph.batch_num_nodes()
            start_idx = 0
            for num_nodes in batch_num_nodes:
                energy_ev_norm = batch_graph.ndata["total_energy"][start_idx][ENERGY_EV_DIM]
                true_e = scaler.inverse_transform_energy(energy_ev_norm).item()
                true_energy.append(true_e)
                start_idx += num_nodes

            pred_e_ev_norm = output['pred_energy'][:, ENERGY_EV_DIM].cpu()
            pred_e = scaler.inverse_transform_energy(pred_e_ev_norm).numpy()
            pred_energy.extend(pred_e)

    true_energy = np.array(true_energy)
    pred_energy = np.array(pred_energy)
    r2 = r2_score(true_energy, pred_energy)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_energy, pred_energy, alpha=0.7, s=20)
    min_e = min(true_energy.min(), pred_energy.min())
    max_e = max(true_energy.max(), pred_energy.max())
    plt.plot([min_e, max_e], [min_e, max_e], 'r--', label=f'R² = {r2:.4f}')
    plt.xlabel('True Energy (eV)')
    plt.ylabel('Predicted Energy (eV)')
    plt.title('Energy Prediction Comparison (eV)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"能量预测对比图已保存至: {save_path}")


def visualize_stress_heatmap(model: nn.Module, val_loader: GraphDataLoader, scaler: CrystalDataScaler,
                             device: torch.device, save_path: str):
    model.eval()
    true_stress = []
    pred_stress = []

    with torch.no_grad():
        count = 0
        for batch in val_loader:
            batch_graph, batch_edge_types = batch
            batch_graph = batch_graph.to(device)
            batch_edge_types = batch_edge_types.to(device).long()

            output = model(batch_graph, batch_edge_types)

            batch_num_nodes = batch_graph.batch_num_nodes()
            start_idx = 0
            for num_nodes in batch_num_nodes:
                if count >= 10:
                    break
                stress_norm = batch_graph.ndata["stress_tensor_flat"][start_idx].cpu()
                true_s = scaler.inverse_transform_stress(stress_norm).numpy().reshape(3, 3)
                true_stress.append(true_s)
                count += 1
                start_idx += num_nodes

            pred_s_norm = output['pred_stress'][:10 - count].cpu()
            pred_s = scaler.inverse_transform_stress(pred_s_norm).numpy().reshape(-1, 3, 3)
            pred_stress.extend(pred_s)

            if count >= 10:
                break

    n_plots = min(5, len(true_stress))
    fig, axes = plt.subplots(2, n_plots, figsize=(15, 8))

    for i in range(n_plots):
        im1 = axes[0, i].imshow(true_stress[i], cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, i].set_title(f'True Stress - Graph {i + 1}')
        axes[0, i].set_xticks([0, 1, 2])
        axes[0, i].set_yticks([0, 1, 2])

        im2 = axes[1, i].imshow(pred_stress[i], cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Pred Stress - Graph {i + 1}')
        axes[1, i].set_xticks([0, 1, 2])
        axes[1, i].set_yticks([0, 1, 2])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"应力张量热力图已保存至: {save_path}")


# ==================== 训练主函数（优化版：Warmup+梯度累积+早停） ====================
def train_model(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dgl.random.seed(RANDOM_SEED)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"使用设备: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    logger.info("加载数据集...")
    train_dataset = CrystalDataset('train', args)
    val_dataset = CrystalDataset('val', args)
    test_dataset = CrystalDataset('test', args)

    logger.info("计算数据归一化统计量...")
    scaler = CrystalDataScaler()
    scaler.fit(train_dataset.graphs)

    logger.info("应用数据归一化...")
    train_dataset.graphs = [scaler.transform(g) for g in train_dataset.graphs]
    val_dataset.graphs = [scaler.transform(g) for g in val_dataset.graphs]
    test_dataset.graphs = [scaler.transform(g) for g in test_dataset.graphs]

    def collate_fn(batch):
        graphs, edge_types = zip(*batch)
        batched_graph = dgl.batch(graphs)
        batched_edge_types = torch.cat(edge_types, dim=0) if edge_types[0].numel() > 0 else torch.tensor([],
                                                                                                         dtype=torch.long)
        return batched_graph, batched_edge_types

    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CrystalRGCNVAE(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 改用AdamW
    criterion = CrystalVAELoss(args).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # 学习率Warmup
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 梯度累积
    accumulation_steps = 2
    logger.info(f"梯度累积步数: {accumulation_steps}")

    # 早停
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_val_loss = float('inf')

    train_losses = {
        'total_loss': [], 'gen_loss': [], 'recon_node_loss': [],
        'recon_edge_loss': [], 'kl_loss': [], 'energy_loss': [], 'force_loss': []
    }
    val_losses = {k: [] for k in train_losses.keys()}

    if len(train_dataset) > 0:
        test_batch = next(iter(train_loader))
        batch_graph, batch_edge_types = test_batch
        batch_graph = batch_graph.to(device)
        batch_edge_types = batch_edge_types.to(device).long()
        model.eval()
        with torch.no_grad():
            output = model(batch_graph, batch_edge_types)
            assert not torch.isnan(output['recon_node']).any(), "重构输出有NaN"
            assert not torch.isnan(output['mu']).any(), "mu有NaN"
        logger.info("单批次测试通过，无NaN")

    logger.info(f"开始训练，共{args.epochs}轮，KL退火轮数: {args.kl_warmup}，Warmup轮数: {warmup_epochs}")
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_rgcn_vae_optimized.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_dict = {k: 0.0 for k in train_losses.keys()}
        train_batch_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_graph, batch_edge_types = batch
            batch_graph = batch_graph.to(device)
            batch_edge_types = batch_edge_types.to(device).long()

            # 边Dropout：训练时随机保留90%边
            if batch_graph.num_edges() > 0:
                edge_mask = torch.rand(batch_graph.num_edges(), device=device) > 0.1
                batch_graph = dgl.edge_subgraph(batch_graph, edge_mask, relabel_nodes=False)
                batch_edge_types = batch_edge_types[edge_mask]

            output = model(batch_graph, batch_edge_types)
            loss, loss_details = criterion(output, batch_graph, batch_edge_types, epoch)

            if torch.isnan(loss):
                logger.warning(f"批次{train_batch_count}损失为NaN，跳过更新")
                train_batch_count += 1
                continue

            loss = loss / accumulation_steps
            loss.backward()

            for k in train_loss_dict.keys():
                train_loss_dict[k] += loss_details[k]
            train_batch_count += 1

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()

        if train_batch_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()

        for k in train_loss_dict.keys():
            train_loss_dict[k] /= train_batch_count if train_batch_count > 0 else 1.0
            train_losses[k].append(train_loss_dict[k])

        if epoch <= warmup_epochs:
            warmup_scheduler.step()

        model.eval()
        val_loss_dict = {k: 0.0 for k in val_losses.keys()}
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_graph, batch_edge_types = batch
                batch_graph = batch_graph.to(device)
                batch_edge_types = batch_edge_types.to(device).long()
                output = model(batch_graph, batch_edge_types)
                loss, loss_details = criterion(output, batch_graph, batch_edge_types, epoch)

                for k in val_loss_dict.keys():
                    val_loss_dict[k] += loss_details[k]
                val_batch_count += 1

        for k in val_loss_dict.keys():
            val_loss_dict[k] /= val_batch_count if val_batch_count > 0 else 1.0
            val_losses[k].append(val_loss_dict[k])

        scheduler.step(val_loss_dict['total_loss'])

        if val_loss_dict['total_loss'] < best_val_loss:
            best_val_loss = val_loss_dict['total_loss']
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'val_loss': best_val_loss
            }, best_model_path)
            logger.info(f"✅ 最佳模型已更新（Epoch {epoch}），验证损失: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"验证损失{early_stopping_patience}轮未下降，触发早停！")
                break

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss_dict['total_loss']:.4f} | "
            f"Val Loss: {val_loss_dict['total_loss']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if epoch % args.vis_interval == 0 or epoch == args.epochs or early_stopping_counter >= early_stopping_patience:
            vis_epoch_dir = os.path.join(VISUALIZATION_DIR, f"epoch_{epoch}")
            os.makedirs(vis_epoch_dir, exist_ok=True)

            loss_vis_path = os.path.join(vis_epoch_dir, "loss_curves.png")
            visualize_loss_curves(train_losses, val_losses, loss_vis_path)

            tsne_vis_path = os.path.join(vis_epoch_dir, "latent_tsne.png")
            visualize_tsne_latent_space(model, val_loader, scaler, device, tsne_vis_path)

            force_vis_path = os.path.join(vis_epoch_dir, "force_heatmap.png")
            visualize_force_heatmap(model, val_loader, scaler, device, force_vis_path)

            energy_vis_path = os.path.join(vis_epoch_dir, "energy_prediction.png")
            visualize_energy_prediction(model, val_loader, scaler, device, energy_vis_path)

            stress_vis_path = os.path.join(vis_epoch_dir, "stress_heatmap.png")
            visualize_stress_heatmap(model, val_loader, scaler, device, stress_vis_path)

    logger.info("开始测试最佳模型...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        logger.info(f"加载最佳模型（Epoch {checkpoint['epoch']}），最佳验证损失: {checkpoint['val_loss']:.4f}")

        model.eval()
        test_loss_dict = {k: 0.0 for k in train_losses.keys()}
        test_batch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                batch_graph, batch_edge_types = batch
                batch_graph = batch_graph.to(device)
                batch_edge_types = batch_edge_types.to(device).long()
                output = model(batch_graph, batch_edge_types)
                loss, loss_details = criterion(output, batch_graph, batch_edge_types, args.epochs)

                for k in test_loss_dict.keys():
                    test_loss_dict[k] += loss_details[k]
                test_batch_count += 1

        for k in test_loss_dict.keys():
            test_loss_dict[k] /= test_batch_count if test_batch_count > 0 else 1.0

        logger.info("=" * 50)
        logger.info(f"测试结果（最佳模型 Epoch {checkpoint['epoch']}）:")
        logger.info(f"总损失: {test_loss_dict['total_loss']:.4f}")
        logger.info(f"生成损失: {test_loss_dict['gen_loss']:.4f}")
        logger.info(f"能量损失: {test_loss_dict['energy_loss']:.4f}")
        logger.info(f"力场损失: {test_loss_dict['force_loss']:.4f}")
        logger.info("=" * 50)

        test_vis_dir = os.path.join(VISUALIZATION_DIR, "test_results")
        os.makedirs(test_vis_dir, exist_ok=True)

        test_energy_path = os.path.join(test_vis_dir, "test_energy_prediction.png")
        visualize_energy_prediction(model, test_loader, scaler, device, test_energy_path)

        test_tsne_path = os.path.join(test_vis_dir, "test_latent_tsne.png")
        visualize_tsne_latent_space(model, test_loader, scaler, device, test_tsne_path)

    logger.info("训练完成！")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)