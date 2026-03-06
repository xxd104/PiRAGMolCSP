import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from dgl.nn import GATConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
import warnings

warnings.filterwarnings("ignore")

# ==================== 全局配置 (完全保持原样) ====================
PROCESSED_GRAPH_DIR = "/home/nyx/N-RGAG/dgl_graphs"
SPLIT_BASE_DIR = "/home/nyx/N-RGAG/dgl_xxx"
MODEL_SAVE_DIR = "/home/nyx/N-RGAG/models"
VISUALIZATION_DIR = "/home/nyx/N-RGAG/models_vis"

NODE_FEAT_DIM = 4
EDGE_FEAT_DIM = 3
GRAPH_ATTR_DIM = 17
ENERGY_DIM = 2
STRESS_DIM = 9
CELL_PARAMS_DIM = 6

RANDOM_SEED = 24
MAX_FORCE_NORM = 1.0
MAX_STRESS_NORM = 1.0

LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
KL_LOSS_CLAMP = 100.0
GRAD_CLIP_MAX_NORM = 5.0

ENERGY_EV_DIM = 1
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gat_vae_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== Lookahead 优化器 (新增) ====================
class LookaheadOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self._step = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['slow'] = p.data.clone()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(), 'lookahead': self.state, 'step': self._step}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.state = state_dict['lookahead']
        self._step = state_dict['step']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    slow = self.state[p]['slow']
                    slow.data += self.alpha * (p.data - slow.data)
                    p.data.copy_(slow.data)
        return loss


# ==================== 数据归一化工具 (保持原样，增加鲁棒性) ====================
class CrystalDataScaler:
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
        node_feats = []
        edge_feats = []
        energy_list = []
        stress_list = []

        for g in graphs:
            node_feats.append(g.ndata["feat"])
            if g.num_edges() > 0:
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
        g.ndata["feat"] = (g.ndata["feat"] - self.node_feat_mean.to(g.device)) / self.node_feat_std.to(g.device)
        if g.num_edges() > 0:
            g.edata["feat"] = (g.edata["feat"] - self.edge_feat_mean.to(g.device)) / self.edge_feat_std.to(g.device)
        g.ndata["total_energy"] = (g.ndata["total_energy"] - self.energy_mean.to(g.device)) / self.energy_std.to(
            g.device)
        g.ndata["stress_tensor_flat"] = (g.ndata["stress_tensor_flat"] - self.stress_mean.to(
            g.device)) / self.stress_std.to(g.device)
        return g

    def inverse_transform_energy(self, energy: torch.Tensor, dim: int = ENERGY_EV_DIM) -> torch.Tensor:
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
        return stress * self.stress_std.to(stress.device) + self.stress_mean.to(stress.device)


# ==================== 参数解析 (保持原样) ====================
def parse_args():
    parser = argparse.ArgumentParser(description='GAT-VAE模型训练 (优化版)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--kl_warmup', type=int, default=50)
    parser.add_argument('--model_save_interval', type=int, default=5)
    parser.add_argument('--vis_interval', type=int, default=10)
    parser.add_argument('--gen_loss_weight', type=float, default=1.0)
    parser.add_argument('--energy_loss_weight', type=float, default=0.5)
    parser.add_argument('--force_loss_weight', type=float, default=0.3)
    return parser.parse_args()


# ==================== 优化后的带边注意力GAT层 (新增残差+LayerNorm+SiLU) ====================
class EdgeAttentionGAT(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_heads: int, edge_feat_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat

        # 边特征处理 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_feat_dim * 2, edge_feat_dim)
        )

        # 节点特征投影
        self.node_proj = nn.Linear(in_feat, in_feat)

        # 核心 GAT 层
        self.gat = GATConv(
            in_feats=in_feat + edge_feat_dim,
            out_feats=out_feat,
            num_heads=num_heads,
            allow_zero_in_degree=True,
            feat_drop=0.1,
            attn_drop=0.1
        )

        # 残差连接投影
        self.residual_proj = nn.Linear(in_feat,
                                       out_feat * num_heads) if in_feat != out_feat * num_heads else nn.Identity()

        # Layer Norm
        self.layer_norm = nn.LayerNorm(out_feat * num_heads)
        self.silu = nn.SiLU()

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(node_feats)
        node_feats_proj = self.node_proj(node_feats)

        if g.num_edges() > 0:
            edge_feats = g.edata["feat"]
            edge_feats = self.edge_mlp(edge_feats)
            src, dst = g.edges()

            src_feats = node_feats_proj[src]
            dst_feats = node_feats_proj[dst]

            src_input = torch.cat([src_feats, edge_feats], dim=1)
            dst_input = torch.cat([dst_feats, edge_feats], dim=1)

            temp_feats = torch.zeros((g.num_nodes(), src_input.shape[1]), device=node_feats.device)
            temp_feats = temp_feats.scatter_add(0, src.unsqueeze(1).repeat(1, src_input.shape[1]), src_input)
            temp_feats = temp_feats.scatter_add(0, dst.unsqueeze(1).repeat(1, dst_input.shape[1]), dst_input)
        else:
            temp_feats = torch.cat(
                [node_feats_proj, torch.zeros((g.num_nodes(), EDGE_FEAT_DIM), device=node_feats.device)], dim=1)

        gat_out = self.gat(g, temp_feats).flatten(1)

        # 残差 + 归一化 + 激活
        out = gat_out + residual
        out = self.layer_norm(out)
        out = self.silu(out)
        return out


# ==================== 优化后的VAE编码器 ====================
class CrystalGATEncoder(nn.Module):
    def __init__(self, node_feat_dim: int = NODE_FEAT_DIM, edge_feat_dim: int = EDGE_FEAT_DIM,
                 hidden_dim: int = 128, latent_dim: int = 64, num_heads: int = 4):
        super().__init__()

        self.gat1 = EdgeAttentionGAT(node_feat_dim, hidden_dim // 2, num_heads, edge_feat_dim)
        self.gat2 = EdgeAttentionGAT((hidden_dim // 2) * num_heads, hidden_dim, num_heads, edge_feat_dim)
        self.gat3 = EdgeAttentionGAT(hidden_dim * num_heads, hidden_dim * 2, num_heads, edge_feat_dim)

        final_gat_dim = num_heads * (hidden_dim * 2)
        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(final_gat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ))

        self.fc_mu = nn.Linear(final_gat_dim, latent_dim)
        self.fc_logvar = nn.Linear(final_gat_dim, latent_dim)

        # 更安全的初始化
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_logvar.weight, 0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feats = g.ndata["feat"]

        h = self.gat1(g, node_feats)
        h = self.gat2(g, h)
        h = self.gat3(g, h)

        graph_emb = self.pooling(g, h)
        node_emb = h

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        logvar = torch.clamp(logvar, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        return mu, logvar, node_emb


# ==================== 优化后的VAE解码器 ====================
class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, num_heads: int = 4,
                 node_feat_dim: int = NODE_FEAT_DIM, edge_feat_dim: int = EDGE_FEAT_DIM,
                 energy_dim: int = ENERGY_DIM, stress_dim: int = STRESS_DIM):
        super().__init__()

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        node_emb_input_dim = num_heads * (hidden_dim * 2)
        self.node_emb_proj = nn.Sequential(
            nn.Linear(node_emb_input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        # 节点解码器 (带残差思想)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # 边解码器
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_feat_dim)
        )

        self.energy_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, energy_dim)
        )

        self.stress_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
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


# ==================== 完整GAT-VAE模型 ====================
class CrystalGATVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = CrystalGATEncoder(
            node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
            hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, num_heads=args.num_heads
        )
        self.decoder = CrystalDecoder(
            latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM
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
        output = {'mu': mu, 'logvar': logvar, 'z': z}
        output.update(decode_out)
        return output


# ==================== 优化后的损失函数 (Huber Loss + 循环KL退火) ====================
class CrystalVAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)  # 对异常值更鲁棒
        self.kl_warmup = args.kl_warmup
        self.current_kl_weight = 0.0
        self.gen_weight = args.gen_loss_weight
        self.energy_weight = args.energy_loss_weight
        self.force_weight = args.force_loss_weight

    def set_kl_weight(self, epoch: int):
        """循环 KL 退火 (Cyclical Annealing)"""
        cycle_len = self.kl_warmup * 2
        e_in_cycle = epoch % cycle_len
        self.current_kl_weight = min(1.0, e_in_cycle / max(1, self.kl_warmup))

    def compute_force_loss(self, pred_stress, true_stress, pred_forces, true_forces):
        pred_stress = torch.clamp(pred_stress, -1e3, 1e3)
        pred_forces = torch.clamp(pred_forces, -1e3, 1e3)
        stress_loss = self.huber(pred_stress, true_stress)
        force_loss = self.huber(pred_forces, true_forces)
        return (stress_loss + force_loss) / 2

    def forward(self, model_output, g, epoch):
        self.set_kl_weight(epoch)

        true_node = g.ndata["feat"]
        true_edge = g.edata["feat"] if g.num_edges() > 0 else torch.tensor([], device=true_node.device)

        batch_num_nodes = g.batch_num_nodes()
        true_energy_list, true_stress_list, true_forces_list = [], [], []
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

        # 1. 生成损失
        recon_node_pred = torch.clamp(model_output['recon_node'], -1e3, 1e3)
        recon_node_loss = self.huber(recon_node_pred, true_node)

        if g.num_edges() > 0:
            recon_edge_pred = torch.clamp(model_output['recon_edge'], -1e3, 1e3)
            recon_edge_loss = self.huber(recon_edge_pred, true_edge)
        else:
            recon_edge_loss = torch.tensor(0.0, device=true_node.device)

        mu = torch.clamp(model_output['mu'], -1e3, 1e3)
        logvar = torch.clamp(model_output['logvar'], LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)
        kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss_per_sample = torch.clamp(kl_loss_per_sample, 0, KL_LOSS_CLAMP)
        kl_loss = kl_loss_per_sample.mean()

        gen_loss = (recon_node_loss + recon_edge_loss) + self.current_kl_weight * kl_loss

        # 2. 能量损失
        pred_energy = torch.clamp(model_output['pred_energy'], -1e3, 1e3)
        energy_loss = self.huber(pred_energy, true_energy)

        # 3. 力场损失
        pred_forces = model_output['recon_node'][:, 1:4]
        force_loss = self.compute_force_loss(
            model_output['pred_stress'], true_stress, pred_forces, true_forces
        )

        total_loss = (
                self.gen_weight * gen_loss +
                self.energy_weight * energy_loss +
                self.force_weight * force_loss
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1e3, device=total_loss.device)
            logger.warning(f"检测到NaN/Inf损失，已替换")

        safe_item = lambda x: x.item() if torch.isfinite(x) else 0.0
        loss_dict = {
            'total_loss': safe_item(total_loss),
            'gen_loss': safe_item(gen_loss),
            'recon_node_loss': safe_item(recon_node_loss),
            'recon_edge_loss': safe_item(recon_edge_loss) if g.num_edges() > 0 else 0.0,
            'kl_loss': safe_item(kl_loss),
            'energy_loss': safe_item(energy_loss),
            'force_loss': safe_item(force_loss)
        }

        return total_loss, loss_dict


# ==================== 数据集加载 (保持原样) ====================
class CrystalDataset(dgl.data.DGLDataset):
    def __init__(self, split_type: str):
        self.split_type = split_type
        self.split_path = os.path.join(SPLIT_BASE_DIR, f"{split_type}_list.txt")
        super().__init__(name=f'crystal_{split_type}')
        self.load()

    def process(self):
        self.graphs = []
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
                assert dgl_graph.ndata["feat"].size(1) == NODE_FEAT_DIM
                if dgl_graph.num_edges() > 0:
                    assert dgl_graph.edata["feat"].size(1) == EDGE_FEAT_DIM

                if torch.isnan(dgl_graph.ndata["feat"]).any() or torch.isinf(dgl_graph.ndata["feat"]).any():
                    continue
                if dgl_graph.num_edges() > 0 and (
                        torch.isnan(dgl_graph.edata["feat"]).any() or torch.isinf(dgl_graph.edata["feat"]).any()):
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


# ==================== 可视化函数 (保持原样) ====================
def visualize_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(15, 10))
    keys = ['total_loss', 'gen_loss', 'energy_loss', 'force_loss']
    titles = ['Total Loss', 'Generation Loss', 'Energy Loss', 'Force Loss']
    colors = [['blue', 'red'], ['green', 'orange'], ['purple', 'brown'], ['cyan', 'magenta']]

    for i, (k, t, c) in enumerate(zip(keys, titles, colors)):
        plt.subplot(2, 2, i + 1)
        plt.plot(train_losses[k], label='Train', color=c[0])
        plt.plot(val_losses[k], label='Val', color=c[1])
        plt.title(t), plt.legend(), plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_tsne_latent_space(model, val_loader, scaler, device, save_path):
    model.eval()
    mu_list, energy_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            mu_list.append(output['mu'].cpu())

            bn_nodes = batch.batch_num_nodes()
            start_idx = 0
            for nn in bn_nodes:
                e_norm = batch.ndata["total_energy"][start_idx][ENERGY_EV_DIM]
                e = scaler.inverse_transform_energy(e_norm).item()
                energy_list.append(e)
                start_idx += nn

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
    plt.title('t-SNE Visualization of Latent Space')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_energy_prediction(model, val_loader, scaler, device, save_path):
    model.eval()
    true_e, pred_e = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)

            bn_nodes = batch.batch_num_nodes()
            start_idx = 0
            for nn in bn_nodes:
                e_norm = batch.ndata["total_energy"][start_idx][ENERGY_EV_DIM]
                true_e.append(scaler.inverse_transform_energy(e_norm).item())
                start_idx += nn

            p_e_norm = output['pred_energy'][:, ENERGY_EV_DIM].cpu()
            pred_e.extend(scaler.inverse_transform_energy(p_e_norm).numpy())

    true_e = np.array(true_e)
    pred_e = np.array(pred_e)
    r2 = r2_score(true_e, pred_e)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_e, pred_e, alpha=0.7, s=20)
    min_val = min(true_e.min(), pred_e.min())
    max_val = max(true_e.max(), pred_e.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label=f'R² = {r2:.4f}')
    plt.xlabel('True Energy (eV)'), plt.ylabel('Predicted Energy (eV)')
    plt.legend(), plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 权重初始化函数 (新增) ====================
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='silu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ==================== 训练主函数 (优化版) ====================
def train_model(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dgl.random.seed(RANDOM_SEED)

    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    logger.info("加载数据集...")
    train_dataset = CrystalDataset('train')
    val_dataset = CrystalDataset('val')
    test_dataset = CrystalDataset('test')

    logger.info("计算数据归一化统计量...")
    scaler = CrystalDataScaler()
    scaler.fit(train_dataset.graphs)

    logger.info("应用数据归一化...")
    train_dataset.graphs = [scaler.transform(g) for g in train_dataset.graphs]
    val_dataset.graphs = [scaler.transform(g) for g in val_dataset.graphs]
    test_dataset.graphs = [scaler.transform(g) for g in test_dataset.graphs]

    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CrystalGATVAE(args).to(device)
    model.apply(weights_init)  # 应用自定义初始化

    # 优化器配置：AdamW + Lookahead
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer = LookaheadOptimizer(base_optimizer, k=5, alpha=0.5)

    criterion = CrystalVAELoss(args).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    train_losses = {k: [] for k in
                    ['total_loss', 'gen_loss', 'recon_node_loss', 'recon_edge_loss', 'kl_loss', 'energy_loss',
                     'force_loss']}
    val_losses = {k: [] for k in train_losses.keys()}

    best_val_loss = float('inf')
    warmup_epochs = 10  # 新增 Warmup
    base_lr = args.lr

    logger.info(f"开始训练，共{args.epochs}轮")
    for epoch in range(1, args.epochs + 1):
        # 1. 学习率 Warmup 调整
        if epoch <= warmup_epochs:
            lr = base_lr * (epoch / warmup_epochs)
            for param_group in base_optimizer.param_groups:
                param_group['lr'] = lr

        # ========== 训练阶段 ==========
        model.train()
        train_loss_dict = {k: 0.0 for k in train_losses.keys()}
        train_batch_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss, loss_details = criterion(output, batch, epoch)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

            for k in train_loss_dict.keys():
                train_loss_dict[k] += loss_details[k]
            train_batch_count += 1

        for k in train_loss_dict.keys():
            train_loss_dict[k] /= max(train_batch_count, 1)
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

        for k in val_loss_dict.keys():
            val_loss_dict[k] /= max(val_batch_count, 1)
            val_losses[k].append(val_loss_dict[k])

        # Warmup 结束后才调整 ReduceLROnPlateau
        if epoch > warmup_epochs:
            scheduler.step(val_loss_dict['total_loss'])

        current_lr = base_optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss_dict['total_loss']:.4f} | "
            f"Val Loss: {val_loss_dict['total_loss']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最佳模型
        if val_loss_dict['total_loss'] < best_val_loss:
            best_val_loss = val_loss_dict['total_loss']
            best_epoch = epoch
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_gat_vae.pth")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': base_optimizer.state_dict(),
                'scaler': scaler, 'val_loss': best_val_loss
            }, best_model_path)
            logger.info(f"✅ 最佳模型已更新 (Epoch {epoch})")

        # 定期保存
        if epoch % args.model_save_interval == 0:
            model_path = os.path.join(MODEL_SAVE_DIR, f"gat_vae_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'scaler': scaler
            }, model_path)

        # 可视化
        if epoch % args.vis_interval == 0 or epoch == args.epochs:
            vis_epoch_dir = os.path.join(VISUALIZATION_DIR, f"epoch_{epoch}")
            os.makedirs(vis_epoch_dir, exist_ok=True)
            visualize_loss_curves(train_losses, val_losses, os.path.join(vis_epoch_dir, "loss_curves.png"))
            visualize_tsne_latent_space(model, val_loader, scaler, device,
                                        os.path.join(vis_epoch_dir, "latent_tsne.png"))
            visualize_energy_prediction(model, val_loader, scaler, device,
                                        os.path.join(vis_epoch_dir, "energy_prediction.png"))

    # ========== 测试 ==========
    logger.info("开始测试最佳模型...")
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_gat_vae.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']

        model.eval()
        test_vis_dir = os.path.join(VISUALIZATION_DIR, "test_results")
        os.makedirs(test_vis_dir, exist_ok=True)
        visualize_energy_prediction(model, test_loader, scaler, device,
                                    os.path.join(test_vis_dir, "test_energy_prediction.png"))

    logger.info("训练完成！")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)