"""主模型的可控变体与对照基线。

- `src/models/ablations.py` : 4 个消融变体（仅切换 `PT_GNO` 的一个数学项）
- `src/models/baselines.py` : 缝合基线与代表性 SOTA（同参数量级）

所有基线都遵守同一接口：
    forward(batch) -> {"topo_prob","level_logits","cont","kl"(可选),"phase_gate"(可选)}
其中 `batch` 由 `src/engine/data.py` 构造，与 PT-GNO 完全一致，保证公平对比。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pt_gno import (NUM_CONT, NUM_DISASTERS, NUM_LEVELS, GNOConfig, PT_GNO)


def _mlp(i, h, o, drop=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


def _scatter_msg(H, edge_index, edge_w, aggregate: str = "mean"):
    """标准图消息聚合：把邻居 (src) 的信息聚到中心节点 (dst)。

    `H` 形状 (B, N, d)；`edge_index[0]=src(邻居)`、`edge_index[1]=dst(中心)`。
    """
    src, dst = edge_index[0], edge_index[1]
    B, N, D = H.shape
    w = edge_w.view(1, -1).clamp_min(0.0).expand(B, -1)
    deg = w.new_zeros((B, N)).scatter_add_(1, dst.view(1, -1).expand(B, -1), w).clamp_min(1e-6)
    msg = w.unsqueeze(-1) * H[:, src]
    out = H.new_zeros((B, N, D)).scatter_add_(
        1, dst.view(1, -1, 1).expand(B, -1, D), msg)
    return out / deg.unsqueeze(-1) if aggregate == "mean" else out


# ==========================================================================
# 1) Stitched-GNN-LLM：GCN + 文本 MLP 末端拼接（论文要批判的对象）
# ==========================================================================
class StitchedGNNLLM(nn.Module):
    """审稿人对照：`torch.cat([h_gnn, h_llm])` 式末端融合 + 无物理内置结构。

    与 PT-GNO 参数量对齐（±10%），差异仅在"拼接而非本征参数注入"。
    """

    def __init__(self, n_features=15, text_dim=256, d_model=128, hidden=128, dropout=0.1):
        super().__init__()
        self.enc = _mlp(n_features, hidden, d_model, dropout)
        self.gcn = nn.Linear(d_model, d_model)
        self.text = _mlp(text_dim * 2, hidden, d_model, dropout)
        self.norm = nn.LayerNorm(d_model * 2)
        self.topo = _mlp(d_model * 2, hidden, NUM_DISASTERS, dropout)
        self.level = _mlp(d_model * 2, hidden, NUM_LEVELS, dropout)
        self.cont = _mlp(d_model * 2, hidden, NUM_CONT, dropout)

    def forward(self, batch):
        x = batch["x"]
        mask = batch.get("mask")
        if mask is not None:
            x = x * mask
        h = self.enc(x)                                  # (B,T,N,d)
        T_ = h.shape[1]
        H = torch.stack([self.gcn(h[:, t]) for t in range(T_)], dim=1)
        e = batch["edge_index"]
        w = batch["edge_w"]
        H = torch.stack([_scatter_msg(H[:, t], e, w) + H[:, t]
                         for t in range(T_)], dim=1)
        # GNN 分支给出**全局**图状态（节点维被均值池化掉），
        # LLM 分支给出**节点**文本，二者在末端拼接 —— 正是缝合基线的缺陷所在。
        g_state = H.mean(dim=2)[:, -1]                                  # (B,d)
        tvec = self.text(torch.cat([batch["h_cur"][:, -1],
                                    batch.get("h_rag", batch["h_cur"])[:, -1]], -1))  # (B,N,d)
        fuse = self.norm(torch.cat([g_state.unsqueeze(1).expand(-1, tvec.shape[1], -1),
                                    tvec], dim=-1))                     # (B,N,2d)
        fz = fuse                                            # (B,N,2d)
        return {
            "topo_prob": torch.sigmoid(self.topo(fz)),
            "level_logits": self.level(fz),
            "cont": F.relu(self.cont(fz)),
            "phase_gate": torch.ones(x.shape[0], x.shape[2], device=x.device),
        }


# ==========================================================================
# 2) GraphWaveNet 风格：自适应邻接 + 膨胀因果卷积
# ==========================================================================
class GraphWaveNetLite(nn.Module):
    def __init__(self, n_features=15, text_dim=256, d_model=128, hidden=128,
                 dropout=0.1, n_nodes=400, dilation=(1, 2, 4)):
        super().__init__()
        self.inp = nn.Conv1d(n_features, d_model, 1)
        self.tcn = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 2, dilation=d, padding=d) for d in dilation])
        self.skip = nn.Conv1d(d_model, d_model, 1)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, 10) * 0.1)
        self.adj_proj = nn.Linear(10, d_model)
        self.adj_proj2 = nn.Linear(10, d_model)
        self.text = _mlp(text_dim * 2, hidden, d_model, dropout)
        self.head = _mlp(d_model, hidden, NUM_DISASTERS + NUM_LEVELS + NUM_CONT, dropout)

    def forward(self, batch):
        x = batch["x"]                                    # (B,T,N,F)
        if batch.get("mask") is not None:
            x = x * batch["mask"]
        B, T, N, _ = x.shape
        h = self.inp(x.permute(0, 2, 3, 1).reshape(B * N, -1, T))  # (B*N,d,T)
        for c in self.tcn:
            h = h + F.gelu(c(h)[..., :T])
        h = h[..., -1].reshape(B, N, -1)
        emb = self.node_emb[:N]
        a = F.softmax(F.relu(self.adj_proj(emb) @ self.adj_proj2(emb).T), dim=-1)
        h = h + torch.einsum("nm,bmd->bnd", a, h)
        hc = batch["h_cur"][:, -1]                      # (B,N,D)
        hr = batch.get("h_rag", batch["h_cur"])[:, -1]
        tvec = self.text(torch.cat([hc, hr], -1))       # (B,N,d)
        h = h + tvec
        o = self.head(h)
        return {"topo_prob": torch.sigmoid(o[..., :NUM_DISASTERS]),
                "level_logits": o[..., NUM_DISASTERS:NUM_DISASTERS + NUM_LEVELS],
                "cont": F.relu(o[..., NUM_DISASTERS + NUM_LEVELS:]),
                "phase_gate": torch.ones(B, N, device=x.device)}


# ==========================================================================
# 3) GRU-GCN：时空循环聚合
# ==========================================================================
class GRUGCN(nn.Module):
    def __init__(self, n_features=15, text_dim=256, d_model=128, hidden=128, dropout=0.1):
        super().__init__()
        self.enc = _mlp(n_features, hidden, d_model, dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.gcn = nn.Linear(d_model, d_model)
        self.text = _mlp(text_dim * 2, hidden, d_model, dropout)
        self.head = _mlp(d_model, hidden, NUM_DISASTERS + NUM_LEVELS + NUM_CONT, dropout)

    def forward(self, batch):
        x = batch["x"]
        if batch.get("mask") is not None:
            x = x * batch["mask"]
        B, T, N, _ = x.shape
        h = self.enc(x)                                   # (B,T,N,d)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, hN = self.gru(h)                  # out:(B*N,T,d), hN:(1,B*N,d)
        hT = hN[0].reshape(B, N, -1)           # (B,N,d)
        hT = hT + _scatter_msg(self.gcn(hT), batch["edge_index"], batch["edge_w"])
        hc = batch["h_cur"][:, -1]
        hr = batch.get("h_rag", batch["h_cur"])[:, -1]
        hT = hT + self.text(torch.cat([hc, hr], -1))
        o = self.head(hT)
        return {"topo_prob": torch.sigmoid(o[..., :NUM_DISASTERS]),
                "level_logits": o[..., NUM_DISASTERS:NUM_DISASTERS + NUM_LEVELS],
                "cont": F.relu(o[..., NUM_DISASTERS + NUM_LEVELS:]),
                "phase_gate": torch.ones(B, N, device=x.device)}


# ==========================================================================
# 4) PDE-GCN：物理图上的扩散（有物理图但无相变门控与本征泛函）
# ==========================================================================
class PDEGCN(nn.Module):
    def __init__(self, n_features=15, adv_dim=3, text_dim=256, d_model=128,
                 hidden=128, dropout=0.1, n_steps=3):
        super().__init__()
        self.enc = _mlp(n_features, hidden, d_model, dropout)
        self.diff = nn.Linear(adv_dim * 2, 1)
        self.step = _mlp(d_model, hidden, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.text = _mlp(text_dim * 2, hidden, d_model, dropout)
        self.head = _mlp(d_model, hidden, NUM_DISASTERS + NUM_LEVELS + NUM_CONT, dropout)
        self.n_steps = n_steps

    def forward(self, batch):
        x = batch["x"]
        if batch.get("mask") is not None:
            x = x * batch["mask"]
        B, T, N, _ = x.shape
        U = self.enc(x[:, -1])
        adv, e, w = batch["adv"][:, -1], batch["edge_index"], batch["edge_w"]
        src, dst = e[0], e[1]
        for _ in range(self.n_steps):
            diff = F.softplus(self.diff(torch.cat(
                [adv[:, src], adv[:, dst] - adv[:, src]], dim=-1)))
            g = diff.squeeze(-1) * w.view(1, -1).expand(B, -1)
            deg = g.new_zeros((B, N)).scatter_add_(1, src.view(1, -1).expand(B, -1), g)
            msg = g.unsqueeze(-1) * (U[:, src] - U[:, dst])
            lap = -msg.new_zeros((B, N, U.shape[-1])).scatter_add_(
                1, src.view(1, -1, 1).expand(B, -1, U.shape[-1]), msg)
            U = self.norm(U + self.step(U) - lap / deg.clamp_min(1e-6).unsqueeze(-1))
        hc = batch["h_cur"][:, -1]
        hr = batch.get("h_rag", batch["h_cur"])[:, -1]
        U = U + self.text(torch.cat([hc, hr], -1))
        o = self.head(U)
        return {"topo_prob": torch.sigmoid(o[..., :NUM_DISASTERS]),
                "level_logits": o[..., NUM_DISASTERS:NUM_DISASTERS + NUM_LEVELS],
                "cont": F.relu(o[..., NUM_DISASTERS + NUM_LEVELS:]),
                "phase_gate": torch.ones(B, N, device=x.device)}


# ==========================================================================
# 5) RE-GCN 风格：时序知识图谱演化（关系感知 GRU 更新）
# ==========================================================================
class REGCNLite(nn.Module):
    def __init__(self, n_features=15, text_dim=256, d_model=128, hidden=128,
                 dropout=0.1, n_rel=4):
        super().__init__()
        self.enc = _mlp(n_features, hidden, d_model, dropout)
        self.rel = nn.Parameter(torch.randn(n_rel, d_model) * 0.1)
        self.w_rel = nn.Linear(d_model, d_model, bias=False)
        self.gru = nn.GRUCell(d_model, d_model)
        self.text = _mlp(text_dim * 2, hidden, d_model, dropout)
        self.head = _mlp(d_model, hidden, NUM_DISASTERS + NUM_LEVELS + NUM_CONT, dropout)

    def forward(self, batch):
        x = batch["x"]
        if batch.get("mask") is not None:
            x = x * batch["mask"]
        B, T, N, _ = x.shape
        h = self.enc(x)
        e, w = batch["edge_index"], batch["edge_w"]
        H = h[:, 0]
        for t in range(T):
            agg = H.new_zeros((B, N, H.shape[-1]))
            for r in range(self.rel.shape[0]):
                m = _scatter_msg(torch.tanh(self.w_rel(H) + self.rel[r]), e, w)
                agg = agg + m / self.rel.shape[0]
            H = self.gru((h[:, t] + agg).reshape(B * N, -1)).view(B, N, -1)
        hc = batch["h_cur"][:, -1]
        hr = batch.get("h_rag", batch["h_cur"])[:, -1]
        H = H + self.text(torch.cat([hc, hr], -1))
        o = self.head(H)
        return {"topo_prob": torch.sigmoid(o[..., :NUM_DISASTERS]),
                "level_logits": o[..., NUM_DISASTERS:NUM_DISASTERS + NUM_LEVELS],
                "cont": F.relu(o[..., NUM_DISASTERS + NUM_LEVELS:]),
                "phase_gate": torch.ones(B, N, device=x.device)}


BASELINE_REGISTRY = {
    "Stitched-GNN-LLM": StitchedGNNLLM,
    "GraphWaveNet": GraphWaveNetLite,
    "GRU-GCN": GRUGCN,
    "PDE-GCN": PDEGCN,
    "RE-GCN": REGCNLite,
}


def build_baseline(name: str, **kw):
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"未知基线 {name}，可选 {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)