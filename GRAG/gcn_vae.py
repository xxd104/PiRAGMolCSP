import os
import dgl
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import argparse
from dgl.nn import GATConv, GlobalAttentionPooling, RelGraphConv  # 假设这里引入了 RelGraphConv
from dgl.dataloading import GraphDataLoader
from typing import List, Tuple, Dict
import logging
from sklearn.manifold import TSNE  # 导入t-SNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局配置 ====================
PROCESSED_GRAPH_DIR = "/home/nyx/GRAG/dgl_graphs"  # 处理后图文件目录
SPLIT_BASE_DIR = "/home/nyx/GRAG/dgl_xxx"  # 数据集划分目录
MODEL_SAVE_DIR = "/home/nyx/GRAG/models"  # 模型保存目录
VISUALIZATION_DIR = "/home/nyx/GRAG/visualizations"  # 可视化结果保存目录
RANDOM_SEED = 24  # 随机种子

# 节点特征维度（固定为4）
NODE_FEAT_DIM = 4
# 边特征最大维度
EDGE_FEAT_DIM = 3

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='GCN-VAE模型训练（含可视化）')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--latent_dim', type=int, default=64, help='潜在变量维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_heads', type=int, default=4, help='GAT注意力头数')
    parser.add_argument('--kl_warmup', type=int, default=10, help='KL退火轮数')
    parser.add_argument('--vis_interval', type=int, default=5, help='可视化间隔轮数')
    return parser.parse_args()


# ==================== 核心GAT层 ====================
class EdgeTypeGAT(nn.Module):
    """带边类型注意力的GAT层"""

    def __init__(self, in_feat: int, out_feat: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat

        self.gat = GATConv(
            in_feats=in_feat,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True
        )

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        gat_out = self.gat(g, node_feats)  # (N, num_heads, out_feat)
        return gat_out.flatten(1)  # 展平为(N, num_heads*out_feat)




# ==================== 编码器 ====================
class CrystalGCNEncoder(nn.Module):
    """GCN编码器（匹配4维节点特征）"""

    def __init__(self,
                 node_feat_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 num_heads: int):
        super().__init__()

        self.gat1 = EdgeTypeGAT(
            in_feat=node_feat_dim,
            out_feat=hidden_dim // 2,
            num_heads=num_heads
        )  # 输出维度: num_heads*(hidden_dim//2)

        self.gat2 = EdgeTypeGAT(
            in_feat=(hidden_dim // 2) * num_heads,
            out_feat=hidden_dim,
            num_heads=num_heads
        )  # 输出维度: num_heads*hidden_dim

        self.gat3 = EdgeTypeGAT(
            in_feat=hidden_dim * num_heads,
            out_feat=hidden_dim * 2,
            num_heads=num_heads
        )  # 输出维度: num_heads*(hidden_dim*2)

        final_dim = num_heads * (hidden_dim * 2)
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(final_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_dim, latent_dim)

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        node_feats = g.ndata["feat"]  # (N, 4)

        h = self.gat1(g, node_feats)  # (N, num_heads*(hidden_dim//2))
        h = nn.ELU()(h)

        h = self.gat2(g, h)  # (N, num_heads*hidden_dim)
        h = nn.ELU()(h)

        h = self.gat3(g, h)  # (N, num_heads*(hidden_dim*2))
        h = nn.ELU()(h)

        graph_emb = self.pooling(g, h)  # (B, num_heads*(hidden_dim*2))

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        return mu, logvar


# ==================== 解码器（修复动态节点匹配问题） ====================
class CrystalDecoder(nn.Module):
    """潜在变量解码器（确保节点数和维度匹配）"""

    def __init__(self, latent_dim: int, hidden_dim: int, node_feat_dim: int = NODE_FEAT_DIM):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU()
        )

        # 确保输出维度为NODE_FEAT_DIM（4）
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

        # 潜在变量投影
        h = self.latent_proj(z)  # (B, hidden_dim*2)

        # 扩展潜在变量到节点级别（核心修复）
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
        assert node_feats.size(1) == NODE_FEAT_DIM, \
            f"解码器输出维度错误: {node_feats.size(1)}，预期: {NODE_FEAT_DIM}"
        assert node_feats.size(0) == total_nodes, \
            f"解码器节点数错误: {node_feats.size(0)}，预期: {total_nodes}"

        return node_feats


# ==================== 完整VAE模型 ====================
class CrystalGCNVAE(nn.Module):
    def __init__(self,
                 node_feat_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = 128,
                 latent_dim: int = 64,
                 num_heads: int = 2):
        super().__init__()
        self.encoder = CrystalGCNEncoder(node_feat_dim, hidden_dim, latent_dim, num_heads)
        self.decoder = CrystalDecoder(latent_dim, hidden_dim, node_feat_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(g)
        z = self.reparameterize(mu, logvar)
        recon_node = self.decoder(z, g)  # 输出节点数与g一致
        return recon_node, mu, logvar


# ==================== 损失函数（含KL退火） ====================
class VAELoss(nn.Module):
    def __init__(self, kl_warmup: int = 10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl_warmup = kl_warmup
        self.current_kl_weight = 0.0

    def set_kl_weight(self, epoch: int):
        if epoch <= self.kl_warmup:
            self.current_kl_weight = min(0.1, epoch / self.kl_warmup)
        else:
            self.current_kl_weight = 0.1

    def forward(self, recon_node: torch.Tensor, real_node: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor, epoch: int) -> torch.Tensor:
        self.set_kl_weight(epoch)

        # 确保重构节点数与真实节点数一致（双重保险）
        if recon_node.size(0) != real_node.size(0):
            min_nodes = min(recon_node.size(0), real_node.size(0))
            recon_node = recon_node[:min_nodes]
            real_node = real_node[:min_nodes]
            logger.warning(f"损失计算时节点数不匹配，截断至{min_nodes}")

        recon_loss = self.mse(recon_node, real_node)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / real_node.size(0)  # 归一化KL损失

        total_loss = recon_loss + self.current_kl_weight * kl_loss
        return total_loss


# ==================== 数据集加载 ====================
class CrystalDataset(dgl.data.DGLDataset):
    def __init__(self, split_type: str):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        super().__init__(name='crystal_dataset')
        self.load()

    def process(self):
        self.graphs = []
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"划分文件不存在: {self.split_path}")

        with open(self.split_path, "r") as f:
            graph_files = [line.strip() for line in f.readlines() if line.strip()]

        for fname in graph_files:
            path = os.path.join(PROCESSED_GRAPH_DIR, f"{fname}")
            if not os.path.exists(path):
                logger.warning(f"图文件不存在: {path}")
                continue
            try:
                graphs, _ = dgl.load_graphs(path)
                self.graphs.extend(graphs)
            except Exception as e:
                logger.error(f"加载图{path}失败: {str(e)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ==================== 可视化功能（全英文批注） ====================
def visualize_training_loss(train_losses: List[float], val_losses: List[float], save_path: str):
    """可视化训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to {save_path}")


def visualize_latent_space_tSNE(mu: torch.Tensor, labels: List[str], save_path: str, title: str):
    """使用t-SNE可视化潜在空间（簇类分析）"""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 检查NaN值
    if torch.isnan(mu).any():
        logger.warning("Latent space contains NaN, skipping t-SNE visualization")
        return

    # t-SNE降维（簇类分析核心）
    mu_np = mu.cpu().numpy()
    if mu_np.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, mu_np.shape[0] - 1))
        mu_tsne = tsne.fit_transform(mu_np)
    else:
        mu_tsne = mu_np

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], alpha=0.7, s=30, cmap='viridis')

    # 为少量点添加标签
    if len(labels) < 50 and mu.size(0) < 100:
        for i, label in enumerate(labels):
            plt.annotate(label, (mu_tsne[i, 0], mu_tsne[i, 1]), fontsize=8)

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"t-SNE latent space visualization saved to {save_path}")


def visualize_reconstruction(real_feats: torch.Tensor, recon_feats: torch.Tensor, save_path: str):
    """可视化原始与重构特征差异（英文批注）"""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 维度强制校验与修正
    if real_feats.size(1) != NODE_FEAT_DIM:
        raise ValueError(f"Real feature dim error: {real_feats.size(1)}, expected {NODE_FEAT_DIM}")
    if recon_feats.size(1) != NODE_FEAT_DIM:
        logger.error(f"Reconstructed feature dim error: {recon_feats.size(1)}, expected {NODE_FEAT_DIM}")
        # 强制截断到4维
        recon_feats = recon_feats[:, :NODE_FEAT_DIM]
        logger.warning(f"Forced truncated to {NODE_FEAT_DIM} dims")

    # 节点数强制匹配
    if real_feats.size(0) != recon_feats.size(0):
        min_nodes = min(real_feats.size(0), recon_feats.size(0))
        real_feats = real_feats[:min_nodes]
        recon_feats = recon_feats[:min_nodes]
        logger.warning(f"Node count mismatch, truncated to {min_nodes} nodes")

    # 计算每个特征维度的平均差异
    diff = torch.abs(real_feats - recon_feats).mean(dim=0).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.bar(range(NODE_FEAT_DIM), diff)
    plt.title('Mean Absolute Difference: Original vs Reconstructed Features')
    plt.xlabel('Feature Dimensions')
    plt.ylabel('Mean Absolute Difference')
    plt.xticks(range(NODE_FEAT_DIM), ['Atomic Number', 'X Coordinate', 'Y Coordinate', 'Z Coordinate'])
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Reconstruction difference visualization saved to {save_path}")


# ==================== 训练主函数 ====================
def train_model(args):
    # 初始化种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dgl.random.seed(RANDOM_SEED)

    # 设备配置
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"使用设备: {device}, 图数据目录: {PROCESSED_GRAPH_DIR}")

    # 确保目录存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 加载数据集
    train_dataset = CrystalDataset('train')
    val_dataset = CrystalDataset('val')
    test_dataset = CrystalDataset('test')

    # 验证数据集有效性
    for dataset in [train_dataset, val_dataset, test_dataset]:
        if len(dataset) == 0:
            raise RuntimeError(f"{dataset.split_type}数据集为空")
        logger.info(f"{dataset.split_type}数据集大小: {len(dataset)}")

    # 数据加载器（动态节点兼容）
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型初始化
    model = CrystalGCNVAE(
        node_feat_dim=NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads
    ).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = VAELoss(kl_warmup=args.kl_warmup).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # 记录损失
    train_losses = []
    val_losses = []

    # 验证集可视化样本（限制节点数）
    val_graphs = []
    val_graphs = val_dataset.graphs
    max_nodes_per_graph = 10000
    val_graphs = [g for g in val_graphs if g.num_nodes() <= max_nodes_per_graph]
    val_graphs = [g.to(device) for g in val_graphs]
    logger.info(f"验证集可视化样本数: {len(val_graphs)}")

    # 训练循环
    logger.info(f"开始训练，共{args.epochs}轮，KL退火轮数: {args.kl_warmup}")
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            real_node = batch.ndata["feat"]  # (总节点数, 4)

            # 输入维度校验
            if real_node.size(1) != NODE_FEAT_DIM:
                raise ValueError(f"输入特征维度错误: {real_node.size(1)}，预期: {NODE_FEAT_DIM}")

            optimizer.zero_grad()
            recon_node, mu, logvar = model(batch)  # 重构节点

            # 计算损失
            loss = criterion(recon_node, real_node, mu, logvar, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 严格梯度裁剪
            optimizer.step()

            train_loss += loss.item() * batch.batch_size

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                real_node = batch.ndata["feat"]
                recon_node, mu, logvar = model(batch)
                loss = criterion(recon_node, real_node, mu, logvar, epoch)
                val_loss += loss.item() * batch.batch_size

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # 保存最佳模型
        best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Epoch {epoch}: 验证损失下降至{val_loss:.4f}，保存最佳模型")

        logger.info(
            f"Epoch {epoch}/{args.epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 可视化（间隔轮次）
        if epoch % args.vis_interval == 0 or epoch == args.epochs:
            # 1. 损失曲线
            loss_vis_path = os.path.join(VISUALIZATION_DIR, f"loss_curve_epoch{epoch}.png")
            visualize_training_loss(train_losses, val_losses, loss_vis_path)

            # 2. t-SNE潜在空间簇类分析
            with torch.no_grad():
                val_mus = []
                val_labels = [f"Graph_{i}" for i in range(len(val_graphs))]  # 简化标签
                for g in val_graphs:
                    mu, _ = model.encoder(g)
                    val_mus.append(mu)
                val_mu = torch.cat(val_mus, dim=0)  # (样本数, latent_dim)

                tsne_vis_path = os.path.join(VISUALIZATION_DIR, f"latent_tsne_epoch{epoch}.png")
                visualize_latent_space_tSNE(val_mu, val_labels, tsne_vis_path, f"t-SNE Latent Space (Epoch {epoch})")

            # 3. 重构特征差异
            with torch.no_grad():
                recon_feats = []
                real_feats = []
                for g in val_graphs:
                    real_feats.append(g.ndata["feat"])
                    recon_node, _, _ = model(g)
                    recon_feats.append(recon_node)

                real_feats = torch.cat(real_feats, dim=0)
                recon_feats = torch.cat(recon_feats, dim=0)

                recon_vis_path = os.path.join(VISUALIZATION_DIR, f"reconstruction_epoch{epoch}.png")
                visualize_reconstruction(real_feats, recon_feats, recon_vis_path)

    # 测试最佳模型
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                real_node = batch.ndata["feat"]
                recon_node, mu, logvar = model(batch)
                loss = criterion(recon_node, real_node, mu, logvar, args.epochs)
                test_loss += loss.item() * batch.batch_size

        test_loss /= len(test_dataset)
        logger.info(f"\n测试集最终损失: {test_loss:.4f}")
        logger.info(f"最佳模型路径: {best_model_path}")

        # 最终可视化
        final_tsne_path = os.path.join(VISUALIZATION_DIR, "final_latent_tsne.png")
        with torch.no_grad():
            test_mus = []
            for batch in test_loader:
                batch = batch.to(device)
                mu, _ = model.encoder(batch)
                test_mus.append(mu)
            test_mu = torch.cat(test_mus, dim=0)
            visualize_latent_space_tSNE(test_mu, [f"Test_{i}" for i in range(len(test_mu))],
                                        final_tsne_path, "Final t-SNE Latent Space (Test Set)")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)