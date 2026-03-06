import os
import dgl
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from torch import nn
from dgl.nn import GATConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
from typing import List, Tuple, Dict, Optional
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
# 路径配置
PROCESSED_GRAPH_DIR = "/home/nyx/N-RGAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/N-RGAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/N-RGAG/models"
VISUALIZATION_DIR = "/home/nyx/N-RGAG/models_vis"

# 特征维度配置（匹配DGL构建代码）
NODE_FEAT_DIM = 4  # 原子序数+受力x/y/z
EDGE_FEAT_DIM = 3  # 距离+角度+键类型
GRAPH_ATTR_DIM = 17  # 6晶胞+9应力+2能量
ENERGY_DIM = 2  # hartree + eV
STRESS_DIM = 9  # 应力张量展平
CELL_PARAMS_DIM = 6  # 晶胞参数

# 训练配置
RANDOM_SEED = 24
MAX_FORCE_NORM = 1.0  # 受力归一化上限
MAX_STRESS_NORM = 1.0  # 应力归一化上限

# 数值稳定配置
LOGVAR_CLAMP_MIN = -10  # logvar最小值，防止exp(logvar)过小
LOGVAR_CLAMP_MAX = 10  # logvar最大值，防止exp(logvar)溢出
KL_LOSS_CLAMP = 100.0  # KL损失裁剪上限
GRAD_CLIP_MAX_NORM = 5.0  # 梯度裁剪阈值（增大以适配晶体数据）

# 可视化配置
ENERGY_EV_DIM = 1  # 固定eV维度的索引
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gat_vae_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 数据归一化工具 ====================
class CrystalDataScaler:
    """晶体数据归一化器（适配DGL图数据）"""

    def __init__(self):
        # 统计量初始化（均值/标准差）
        self.node_feat_mean = None
        self.node_feat_std = None
        self.edge_feat_mean = None
        self.edge_feat_std = None
        self.energy_mean = None
        self.energy_std = None
        self.stress_mean = None
        self.stress_std = None

    def fit(self, graphs: List[dgl.DGLGraph]):
        """基于训练集计算归一化统计量"""
        # 收集所有特征
        node_feats = []
        edge_feats = []
        energy_list = []
        stress_list = []

        for g in graphs:
            # 节点特征
            node_feats.append(g.ndata["feat"])
            # 边特征（跳过无边图）
            if g.num_edges() > 0:
                edge_feats.append(g.edata["feat"])
            # 能量（取第一个节点的能量）
            energy_list.append(g.ndata["total_energy"][0])
            # 应力（取第一个节点的应力）
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
        """归一化单张图数据"""
        # 节点特征归一化
        g.ndata["feat"] = (g.ndata["feat"] - self.node_feat_mean.to(g.device)) / self.node_feat_std.to(g.device)
        # 边特征归一化
        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean.to(g.device)) / self.edge_feat_std.to(g.device)
        # 能量归一化
        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean.to(g.device)) / self.energy_std.to(
            g.device)
        # 应力归一化
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean.to(
            g.device)) / self.stress_std.to(g.device)
        return g

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
        """
        能量反归一化（完美适配所有维度输入）
        Args:
            energy: 归一化后的能量张量（标量/1维/2维）
            dim: 要反归一化的维度（0=hartree, 1=eV）
        Returns:
            反归一化后的能量值（保持原维度）
        """
        # 获取对应维度的统计量
        std = self.energy_std[dim]
        mean = self.energy_mean[dim]

        # 按维度分支处理（核心修复）
        if energy.dim() == 0:
            # 0维（标量）：直接计算
            return energy * std + mean
        elif energy.dim() == 1:
            # 1维（批量eV值）：直接逐元素计算，无需二维索引
            return energy * std + mean
        elif energy.dim() == 2:
            # 2维（hartree+eV）：取指定维度列
            return energy[:, dim] * std + mean
        else:
            raise ValueError(f"不支持的能量张量维度: {energy.dim()}")

    def inverse_transform_stress(self, stress: torch.Tensor) -> torch.Tensor:
        """应力反归一化"""
        return stress * self.stress_std.to(stress.device) + self.stress_mean.to(stress.device)


# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='GAT-VAE模型训练（带边注意力+能量/力场损失）')
    # 设备与基础参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（降低至1e-4）')

    # 模型参数
    parser.add_argument('--latent_dim', type=int, default=64, help='潜在变量维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='GAT注意力头数')
    parser.add_argument('--kl_warmup', type=int, default=50, help='KL退火轮数')

    # 保存与可视化参数
    parser.add_argument('--model_save_interval', type=int, default=5, help='模型保存间隔（轮数）')
    parser.add_argument('--vis_interval', type=int, default=10, help='可视化保存间隔（轮数）')

    # 损失权重
    parser.add_argument('--gen_loss_weight', type=float, default=1.0, help='生成损失权重')
    parser.add_argument('--energy_loss_weight', type=float, default=0.5, help='能量损失权重')
    parser.add_argument('--force_loss_weight', type=float, default=0.3, help='力场损失权重')

    return parser.parse_args()


# ==================== 带边注意力的GAT层 ====================
class EdgeAttentionGAT(nn.Module):
    """带边特征注意力的GAT层（不依赖edge_attr参数）"""

    def __init__(self, in_feat: int, out_feat: int, num_heads: int, edge_feat_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat

        # 基础GAT层
        self.gat = GATConv(
            in_feats=in_feat + edge_feat_dim,  # 节点特征+边特征拼接
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            feat_drop=0.1,
            attn_drop=0.1
        )

        # 边特征投影
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_feat_dim * 2, edge_feat_dim)
        )

        # 节点特征投影（匹配维度）
        self.node_proj = nn.Linear(in_feat, in_feat)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将边特征整合到节点特征中
        """
        # 1. 提取并处理边特征
        edge_feats = g.edata["feat"]  # (E, 3)
        edge_feats = self.edge_mlp(edge_feats)  # (E, 3)

        # 2. 复制边特征到源节点和目标节点
        src, dst = g.edges()
        node_feats_proj = self.node_proj(node_feats)  # (N, in_feat)

        # 3. 为源节点和目标节点添加边特征
        src_feats = node_feats_proj[src]  # (E, in_feat)
        dst_feats = node_feats_proj[dst]  # (E, in_feat)

        # 4. 拼接节点特征和边特征
        src_input = torch.cat([src_feats, edge_feats], dim=1)  # (E, in_feat + edge_feat_dim)
        dst_input = torch.cat([dst_feats, edge_feats], dim=1)  # (E, in_feat + edge_feat_dim)

        # 5. 创建临时特征矩阵
        temp_feats = torch.zeros((g.num_nodes(), src_input.shape[1]), device=node_feats.device)

        # 6. 聚合边特征到节点（求和）
        temp_feats = temp_feats.scatter_add(0, src.unsqueeze(1).repeat(1, src_input.shape[1]), src_input)
        temp_feats = temp_feats.scatter_add(0, dst.unsqueeze(1).repeat(1, dst_input.shape[1]), dst_input)

        # 7. 如果没有边，直接使用节点特征
        if g.num_edges() == 0:
            temp_feats = torch.cat(
                [node_feats_proj, torch.zeros((g.num_nodes(), EDGE_FEAT_DIM), device=node_feats.device)], dim=1)

        # 8. GAT前向传播
        gat_out = self.gat(g, temp_feats)  # (N, num_heads, out_feat)
        return gat_out.flatten(1)  # (N, num_heads*out_feat)


# ==================== VAE编码器 ====================
class CrystalGATEncoder(nn.Module):
    """GAT编码器（处理节点+边特征，输出潜在变量分布）"""

    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 hidden_dim: int = 128,
                 latent_dim: int = 64,
                 num_heads: int = 4):
        super().__init__()

        # GAT层
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

        # 图池化（得到图级嵌入）
        final_gat_dim = num_heads * (hidden_dim * 2)
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_gat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        # 潜在变量分布（mu和logvar）- 增加初始化以保证数值稳定
        self.fc_mu = nn.Linear(final_gat_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_gat_dim, latent_dim)

        # 初始化logvar为较小值，防止初始exp(logvar)过大
        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

        # 激活函数
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 提取节点特征
        node_feats = g.ndata["feat"]  # (N, 4)

        # GAT层前向
        h = self.gat1(g, node_feats)  # (N, num_heads*(hidden_dim//2))
        h = self.elu(h)
        h = self.dropout(h)

        h = self.gat2(g, h)  # (N, num_heads*hidden_dim)
        h = self.elu(h)
        h = self.dropout(h)

        h = self.gat3(g, h)  # (N, num_heads*(hidden_dim*2))
        h = self.elu(h)

        # 图池化得到全局嵌入
        graph_emb = self.pooling(g, h)  # (B, final_gat_dim)

        # 节点级嵌入保留（用于解码）
        node_emb = h

        # 潜在变量 - 增加数值裁剪
        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)

        # 裁剪logvar防止溢出
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


# ==================== VAE解码器 ====================
class CrystalDecoder(nn.Module):
    """解码器：重构节点/边特征 + 预测能量/应力"""

    def __init__(self,
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,  # 新增：接收注意力头数参数
                 node_feat_dim: int = NODE_FEAT_DIM,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM,
                 stress_dim: int = STRESS_DIM):
        super().__init__()

        # 潜在变量投影
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 新增：节点嵌入投影层（将encoder输出的node_emb维度投影到hidden_dim*2）
        node_emb_input_dim = num_heads * (hidden_dim * 2)  # encoder输出的node_emb维度
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 节点特征解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # 边特征解码器（修复：输入维度改为hidden_dim*4，匹配z_expanded拼接后的维度）
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 修复：latent_dim*2 → hidden_dim*4
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

        # 能量预测头
        self.energy_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, energy_dim)
        )

        # 应力张量预测头
        self.stress_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stress_dim)
        )

    def forward(self, z: torch.Tensor, g: dgl.DGLGraph, node_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        返回：包含重构特征和预测值的字典
        """
        batch_size = len(g.batch_num_nodes())
        total_nodes = g.num_nodes()

        # 1. 投影潜在变量
        z_proj = self.latent_proj(z)  # (B, hidden_dim*2)

        # 2. 扩展到节点级别
        z_expanded = torch.zeros(total_nodes, z_proj.size(1), device=z.device)
        start_idx = 0
        for i in range(batch_size):
            num_nodes = g.batch_num_nodes()[i]
            end_idx = start_idx + num_nodes
            z_expanded[start_idx:end_idx] = z_proj[i].unsqueeze(0).repeat(num_nodes, 1)
            start_idx = end_idx

        # 3. 投影节点嵌入（核心修复：匹配z_expanded维度）
        node_emb_proj = self.node_emb_proj(node_emb)  # (N, hidden_dim*2)

        # 4. 重构节点特征（使用投影后的node_emb）
        recon_node_feats = self.node_decoder(z_expanded + node_emb_proj)  # 维度匹配后相加

        # 5. 重构边特征
        src, dst = g.edges()
        z_src = z_expanded[src]
        z_dst = z_expanded[dst]
        edge_input = torch.cat([z_src, z_dst], dim=1)  # (E, hidden_dim*4)
        recon_edge_feats = self.edge_decoder(edge_input)

        # 6. 预测能量和应力
        pred_energy = self.energy_predictor(z)  # (B, 2)
        pred_stress = self.stress_predictor(z)  # (B, 9)

        return {
            'recon_node': recon_node_feats,
            'recon_edge': recon_edge_feats,
            'pred_energy': pred_energy,
            'pred_stress': pred_stress
        }


# ==================== 完整GAT-VAE模型 ====================
class CrystalGATVAE(nn.Module):
    """带能量/力场预测的GAT-VAE模型"""

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
            num_heads=args.num_heads,  # 新增：传递num_heads参数
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM
        )

        self.latent_dim = args.latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧 - 增加数值稳定"""
        std = torch.exp(0.5 * logvar)
        # 裁剪std防止过大
        std = torch.clamp(std, 1e-6, 1e6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        # 编码得到潜在分布
        mu, logvar, node_emb = self.encoder(g)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码得到重构和预测结果
        decode_out = self.decoder(z, g, node_emb)

        # 整合输出
        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
        output.update(decode_out)

        return output


# ==================== 损失函数（三部分） ====================
class CrystalVAELoss(nn.Module):
    """包含生成损失、能量损失、力场损失的复合损失函数 - 增强数值稳定"""

    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl_warmup = args.kl_warmup
        self.current_kl_weight = 0.0

        # 损失权重
        self.gen_weight = args.gen_loss_weight
        self.energy_weight = args.energy_loss_weight
        self.force_weight = args.force_loss_weight

    def set_kl_weight(self, epoch: int):
        """KL退火"""
        if epoch <= self.kl_warmup:
            self.current_kl_weight = min(1.0, epoch / self.kl_warmup)
        else:
            self.current_kl_weight = 1.0

    def compute_force_loss(self, pred_stress: torch.Tensor, true_stress: torch.Tensor,
                           pred_forces: torch.Tensor, true_forces: torch.Tensor) -> torch.Tensor:
        """计算力场损失：应力张量 + 原子受力 - 增加数值裁剪"""
        # 裁剪预测值防止溢出
        pred_stress = torch.clamp(pred_stress, -1e3, 1e3)
        pred_forces = torch.clamp(pred_forces, -1e3, 1e3)

        # 应力张量损失
        stress_loss = self.mse(pred_stress, true_stress)

        # 原子受力损失
        force_loss = self.mse(pred_forces, true_forces)

        return (stress_loss + force_loss) / 2

    def forward(self, model_output: Dict[str, torch.Tensor],
                g: dgl.DGLGraph, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失 - 增强数值稳定
        返回：总损失，各部分损失字典
        """
        self.set_kl_weight(epoch)

        # 1. 提取真实特征
        true_node = g.ndata["feat"]  # (N, 4)
        # 修复：无边图时保证tensor在正确设备上
        true_edge = g.edata["feat"] if g.num_edges() > 0 else torch.tensor([], device=true_node.device,
                                                                           dtype=true_node.dtype)

        # 提取图级属性（每个图取第一个节点的属性）
        batch_num_nodes = g.batch_num_nodes()
        true_energy_list = []
        true_stress_list = []
        true_forces_list = []

        start_idx = 0
        for num_nodes in batch_num_nodes:
            end_idx = start_idx + num_nodes
            # 真实能量 (hartree, eV)
            true_energy = g.ndata["total_energy"][start_idx]
            true_energy_list.append(true_energy)

            # 真实应力张量
            true_stress = g.ndata["stress_tensor_flat"][start_idx]
            true_stress_list.append(true_stress)

            # 真实受力（节点特征的第2-4维）
            true_forces = true_node[start_idx:end_idx, 1:4]
            true_forces_list.append(true_forces)

            start_idx = end_idx

        true_energy = torch.stack(true_energy_list)  # (B, 2)
        true_stress = torch.stack(true_stress_list)  # (B, 9)
        true_forces = torch.cat(true_forces_list)  # (N, 3)

        # 2. 生成损失（重构损失 + KL损失）
        # 节点重构损失 - 裁剪预测值
        recon_node_pred = torch.clamp(model_output['recon_node'], -1e3, 1e3)
        recon_node_loss = self.mse(recon_node_pred, true_node)

        # 边重构损失（处理无边情况）
        if g.num_edges() > 0:
            recon_edge_pred = torch.clamp(model_output['recon_edge'], -1e3, 1e3)
            recon_edge_loss = self.mse(recon_edge_pred, true_edge)
        else:
            recon_edge_loss = torch.tensor(0.0, device=true_node.device)

        # KL损失 - 增强数值稳定
        mu = model_output['mu']
        logvar = model_output['logvar']

        # 裁剪mu和logvar防止溢出
        mu = torch.clamp(mu, -1e3, 1e3)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        # 计算KL损失（逐样本计算后平均，避免批量求和溢出）
        kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # 裁剪KL损失值
        kl_loss_per_sample = torch.clamp(kl_loss_per_sample, 0, KL_LOSS_CLAMP)
        kl_loss = kl_loss_per_sample.mean()

        gen_loss = (recon_node_loss + recon_edge_loss) + self.current_kl_weight * kl_loss

        # 3. 能量损失 - 裁剪预测值
        pred_energy = torch.clamp(model_output['pred_energy'], -1e3, 1e3)
        energy_loss = self.mse(pred_energy, true_energy)

        # 4. 力场损失
        # 预测受力（节点重构特征的第2-4维）
        pred_forces = model_output['recon_node'][:, 1:4]
        force_loss = self.compute_force_loss(
            model_output['pred_stress'], true_stress,
            pred_forces, true_forces
        )

        # 5. 总损失 - 检查NaN并替换
        total_loss = (
                self.gen_weight * gen_loss +
                self.energy_weight * energy_loss +
                self.force_weight * force_loss
        )

        # 关键修复：检测并替换NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1e3, device=total_loss.device)  # 替换为大的有限值
            logger.warning(f"检测到NaN/Inf损失，已替换为1e3")

        # 整理损失字典（确保所有值都是有限的）
        loss_dict = {
            'total_loss': total_loss.item() if torch.isfinite(total_loss) else 1e3,
            'gen_loss': gen_loss.item() if torch.isfinite(gen_loss) else 0.0,
            'recon_node_loss': recon_node_loss.item() if torch.isfinite(recon_node_loss) else 0.0,
            'recon_edge_loss': recon_edge_loss.item() if (
                        g.num_edges() > 0 and torch.isfinite(recon_edge_loss)) else 0.0,
            'kl_loss': kl_loss.item() if torch.isfinite(kl_loss) else 0.0,
            'energy_loss': energy_loss.item() if torch.isfinite(energy_loss) else 0.0,
            'force_loss': force_loss.item() if torch.isfinite(force_loss) else 0.0
        }

        return total_loss, loss_dict


# ==================== 数据集加载 ====================
class CrystalDataset(dgl.data.DGLDataset):
    """晶体DGL图数据集加载器"""

    def __init__(self, split_type: str):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        super().__init__(name=f'crystal_{split_type}')
        self.load()

    def process(self):
        self.graphs = []

        # 检查划分文件
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"划分文件不存在: {self.split_path}")

        # 读取文件列表
        with open(self.split_path, "r") as f:
            graph_files = [line.strip() for line in f.readlines() if line.strip()]

        # 加载图文件
        for fname in graph_files:
            path = os.path.join(PROCESSED_GRAPH_DIR, fname)
            if not os.path.exists(path):
                logger.warning(f"图文件不存在: {path}")
                continue

            try:
                # 加载DGL图（匹配构建代码的保存方式）
                dgl_graph = torch.load(path)
                # 确保特征维度正确
                assert dgl_graph.ndata["feat"].size(
                    1) == NODE_FEAT_DIM, f"节点特征维度错误: {dgl_graph.ndata['feat'].size(1)}"
                if dgl_graph.num_edges() > 0:
                    assert dgl_graph.edata["feat"].size(
                        1) == EDGE_FEAT_DIM, f"边特征维度错误: {dgl_graph.edata['feat'].size(1)}"

                # 检查数据中是否有NaN/Inf
                if torch.isnan(dgl_graph.ndata["feat"]).any() or torch.isinf(dgl_graph.ndata["feat"]).any():
                    logger.warning(f"图{path}包含NaN/Inf节点特征，已过滤")
                    continue
                if dgl_graph.num_edges() > 0 and (
                        torch.isnan(dgl_graph.edata["feat"]).any() or torch.isinf(dgl_graph.edata["feat"]).any()):
                    logger.warning(f"图{path}包含NaN/Inf边特征，已过滤")
                    continue

                self.graphs.append(dgl_graph)
            except Exception as e:
                logger.error(f"加载图{path}失败: {str(e)}")

        if len(self.graphs) == 0:
            raise RuntimeError(f"{self.split_type}数据集为空")
        logger.info(f"{self.split_type}数据集加载完成，共{len(self.graphs)}个图")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ==================== 可视化函数 ====================
def visualize_loss_curves(train_losses: Dict[str, List[float]],
                          val_losses: Dict[str, List[float]],
                          save_path: str):
    """可视化训练/验证损失曲线（包含各部分损失）"""
    plt.figure(figsize=(15, 10))

    # 子图1：总损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses['total_loss'], label='Train', color='blue')
    plt.plot(val_losses['total_loss'], label='Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：生成损失
    plt.subplot(2, 2, 2)
    plt.plot(train_losses['gen_loss'], label='Train', color='green')
    plt.plot(val_losses['gen_loss'], label='Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Generation Loss')
    plt.title('Generation Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图3：能量损失
    plt.subplot(2, 2, 3)
    plt.plot(train_losses['energy_loss'], label='Train', color='purple')
    plt.plot(val_losses['energy_loss'], label='Validation', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Energy Loss')
    plt.title('Energy Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图4：力场损失
    plt.subplot(2, 2, 4)
    plt.plot(train_losses['force_loss'], label='Train', color='cyan')
    plt.plot(val_losses['force_loss'], label='Validation', color='magenta')
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
    """t-SNE可视化潜在空间（仅使用eV维度）"""
    model.eval()
    mu_list = []
    energy_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            mu_list.append(output['mu'].cpu())

            # 提取能量值（仅eV维度）作为颜色标签
            batch_num_nodes = batch.batch_num_nodes()
            start_idx = 0
            for num_nodes in batch_num_nodes:
                # 仅提取eV维度的标量值（索引1）
                energy_ev_norm = batch.ndata["total_energy"][start_idx][ENERGY_EV_DIM]  # 标量
                # 反归一化eV维度
                energy_ev = scaler.inverse_transform_energy(energy_ev_norm).item()
                energy_list.append(energy_ev)
                start_idx += num_nodes

    # 拼接所有潜在变量
    mu = torch.cat(mu_list, dim=0).numpy()
    energy = np.array(energy_list)

    # t-SNE降维
    if mu.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, mu.shape[0] - 1))
        mu_tsne = tsne.fit_transform(mu)
    else:
        mu_tsne = mu

    # 绘制散点图
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
    """原子受力热力图（真实vs预测）"""
    model.eval()
    true_forces_all = []
    pred_forces_all = []

    with torch.no_grad():
        # 取第一个batch的数据
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)

            # 提取真实受力（反归一化）
            true_node_norm = batch.ndata["feat"].cpu()
            true_node = true_node_norm * scaler.node_feat_std + scaler.node_feat_mean
            true_forces = true_node[:, 1:4].numpy()

            # 提取预测受力（反归一化）
            pred_node_norm = output['recon_node'][:, :4].cpu()
            pred_node = pred_node_norm * scaler.node_feat_std + scaler.node_feat_mean
            pred_forces = pred_node[:, 1:4].numpy()

            # 只取前100个节点（便于可视化）
            n_nodes = min(100, true_forces.shape[0])
            true_forces_all = true_forces[:n_nodes]
            pred_forces_all = pred_forces[:n_nodes]
            break

    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 真实受力
    im1 = ax1.imshow(true_forces_all.T, cmap='RdBu_r', aspect='auto')
    ax1.set_title('True Atomic Forces (X/Y/Z)')
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Force Dimension')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['X', 'Y', 'Z'])
    plt.colorbar(im1, ax=ax1)

    # 预测受力
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
    """能量预测对比散点图（仅eV维度）"""
    model.eval()
    true_energy = []
    pred_energy = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)

            # 提取真实能量（仅eV维度，反归一化）
            batch_num_nodes = batch.batch_num_nodes()
            start_idx = 0
            for num_nodes in batch_num_nodes:
                # 仅提取eV维度的标量
                energy_ev_norm = batch.ndata["total_energy"][start_idx][ENERGY_EV_DIM]
                true_e = scaler.inverse_transform_energy(energy_ev_norm).item()
                true_energy.append(true_e)
                start_idx += num_nodes

            # 提取预测能量（仅eV维度，反归一化）
            pred_e_ev_norm = output['pred_energy'][:, ENERGY_EV_DIM].cpu()
            pred_e = scaler.inverse_transform_energy(pred_e_ev_norm).numpy()
            pred_energy.extend(pred_e)

    # 转换为数组
    true_energy = np.array(true_energy)
    pred_energy = np.array(pred_energy)

    # 计算R2分数
    r2 = r2_score(true_energy, pred_energy)

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(true_energy, pred_energy, alpha=0.7, s=20)
    # 绘制对角线
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


# ==================== 训练主函数 ====================
def train_model(args):
    # 1. 初始化种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dgl.random.seed(RANDOM_SEED)

    # 2. 设备配置
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 3. 创建保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 4. 加载数据集
    logger.info("加载数据集...")
    train_dataset = CrystalDataset('train')
    val_dataset = CrystalDataset('val')
    test_dataset = CrystalDataset('test')

    # 数据归一化
    logger.info("计算数据归一化统计量...")
    scaler = CrystalDataScaler()
    scaler.fit(train_dataset.graphs)

    # 应用归一化
    logger.info("应用数据归一化...")
    train_dataset.graphs = [scaler.transform(g) for g in train_dataset.graphs]
    val_dataset.graphs = [scaler.transform(g) for g in val_dataset.graphs]
    test_dataset.graphs = [scaler.transform(g) for g in test_dataset.graphs]

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. 初始化模型、优化器、损失函数
    model = CrystalGATVAE(args).to(device)
    # 使用AdamW优化器（更稳定），并降低学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = CrystalVAELoss(args).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    # 6. 损失记录
    train_losses = {
        'total_loss': [], 'gen_loss': [], 'recon_node_loss': [],
        'recon_edge_loss': [], 'kl_loss': [], 'energy_loss': [], 'force_loss': []
    }
    val_losses = train_losses.copy()

    # 7. 最佳模型跟踪
    best_val_loss = float('inf')
    best_epoch = 0

    # 8. 训练循环
    logger.info(f"开始训练，共{args.epochs}轮")
    for epoch in range(1, args.epochs + 1):
        # ========== 训练阶段 ==========
        model.train()
        train_loss_dict = {k: 0.0 for k in train_losses.keys()}
        train_batch_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 前向传播
            output = model(batch)

            # 计算损失
            loss, loss_details = criterion(output, batch, epoch)

            # 反向传播 - 检查梯度是否为NaN
            loss.backward()

            # 梯度裁剪（增大阈值）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            # 检查梯度
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            if epoch == 1 and train_batch_count == 0:
                logger.info(f"初始梯度范数: {grad_norm:.4f}")

            # 仅当损失为有限值时更新参数
            if torch.isfinite(loss):
                optimizer.step()
            else:
                logger.warning(f"批次{train_batch_count}损失为NaN/Inf，跳过参数更新")

            # 累加损失
            for k in train_loss_dict.keys():
                train_loss_dict[k] += loss_details[k]
            train_batch_count += 1

        # 计算平均训练损失
        for k in train_loss_dict.keys():
            train_loss_dict[k] /= train_batch_count if train_batch_count > 0 else 1.0
            train_losses[k].append(train_loss_dict[k])

        # ========== 验证阶段 ==========
        model.eval()
        val_loss_dict = {k: 0.0 for k in val_losses.keys()}
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss, loss_details = criterion(output, batch, epoch)

                for k in val_loss_dict.keys():
                    val_loss_dict[k] += loss_details[k]
                val_batch_count += 1

        # 计算平均验证损失
        for k in val_loss_dict.keys():
            val_loss_dict[k] /= val_batch_count if val_batch_count > 0 else 1.0
            val_losses[k].append(val_loss_dict[k])

        # 更新学习率
        scheduler.step(val_loss_dict['total_loss'])

        # ========== 日志记录 ==========
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss_dict['total_loss']:.4f} | "
            f"Val Loss: {val_loss_dict['total_loss']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ========== 每轮检查并更新最佳模型 ==========
        if val_loss_dict['total_loss'] < best_val_loss:
            best_val_loss = val_loss_dict['total_loss']
            best_epoch = epoch
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_gat_vae.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,  # 保存归一化器
                'val_loss': best_val_loss
            }, best_model_path)
            logger.info(f"✅ 最佳模型已更新（Epoch {epoch}），验证损失: {best_val_loss:.4f}，保存至: {best_model_path}")

        # ========== 每5轮保存当前epoch的模型（用于存档） ==========
        if epoch % args.model_save_interval == 0:
            model_path = os.path.join(MODEL_SAVE_DIR, f"gat_vae_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,  # 保存归一化器
                'train_loss': train_loss_dict['total_loss'],
                'val_loss': val_loss_dict['total_loss']
            }, model_path)
            logger.info(f"📁 第{epoch}轮模型已保存至: {model_path}")

        # ========== 可视化 ==========
        if epoch % args.vis_interval == 0 or epoch == args.epochs:
            vis_epoch_dir = os.path.join(VISUALIZATION_DIR, f"epoch_{epoch}")
            os.makedirs(vis_epoch_dir, exist_ok=True)

            # 1. 损失曲线
            loss_vis_path = os.path.join(vis_epoch_dir, "loss_curves.png")
            visualize_loss_curves(train_losses, val_losses, loss_vis_path)

            # 2. t-SNE潜在空间
            tsne_vis_path = os.path.join(vis_epoch_dir, "latent_tsne.png")
            visualize_tsne_latent_space(model, val_loader, scaler, device, tsne_vis_path)

            # 3. 受力热力图
            force_vis_path = os.path.join(vis_epoch_dir, "force_heatmap.png")
            visualize_force_heatmap(model, val_loader, scaler, device, force_vis_path)

            # 4. 能量预测对比
            energy_vis_path = os.path.join(vis_epoch_dir, "energy_prediction.png")
            visualize_energy_prediction(model, val_loader, scaler, device, energy_vis_path)


    # ========== 测试阶段 ==========
    logger.info("开始测试最佳模型...")
    # 加载最佳模型
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_gat_vae.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']  # 加载归一化器
        logger.info(f"加载最佳模型（Epoch {checkpoint['epoch']}），最佳验证损失: {checkpoint['val_loss']:.4f}")

        # 测试
        model.eval()
        test_loss_dict = {k: 0.0 for k in train_losses.keys()}
        test_batch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                loss, loss_details = criterion(output, batch, args.epochs)

                for k in test_loss_dict.keys():
                    test_loss_dict[k] += loss_details[k]
                test_batch_count += 1

        # 计算平均测试损失
        for k in test_loss_dict.keys():
            test_loss_dict[k] /= test_batch_count if test_batch_count > 0 else 1.0

        # 记录测试结果
        logger.info("=" * 50)
        logger.info(f"测试结果（最佳模型 Epoch {best_epoch}）:")
        logger.info(f"总损失: {test_loss_dict['total_loss']:.4f}")
        logger.info(f"生成损失: {test_loss_dict['gen_loss']:.4f}")
        logger.info(f"能量损失: {test_loss_dict['energy_loss']:.4f}")
        logger.info(f"力场损失: {test_loss_dict['force_loss']:.4f}")
        logger.info("=" * 50)

        # 保存测试可视化
        test_vis_dir = os.path.join(VISUALIZATION_DIR, "test_results")
        os.makedirs(test_vis_dir, exist_ok=True)

        # 能量预测对比
        test_energy_path = os.path.join(test_vis_dir, "test_energy_prediction.png")
        visualize_energy_prediction(model, test_loader, scaler, device, test_energy_path)

        # t-SNE潜在空间
        test_tsne_path = os.path.join(test_vis_dir, "test_latent_tsne.png")
        visualize_tsne_latent_space(model, test_loader, scaler, device, test_tsne_path)

    logger.info("训练完成！")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)