"""PT-GNO：物理引导的相变图神经算子（唯一主模型，无模块缝合）。

数学结构（与 `CLAUDE.md` §1.2 / §1.4 严格对应）：

    dU_i(t) = -Σ_j K_adv(ΔP_ij, WIN_ij)(U_i - U_j)      # 水平气压-风场平流
              + Ψ_in(X^meteo_i,t)                        # 垂直致灾能量注入
              - γ_i(Θ) ⊙ U_i(t)                          # 防灾耗散
              + G_assim(U_i) dM_i(t)                     # 初/续/终报连续时间同化

    A^topo_i = σ_temp((||U_i||_2 - τ_i)/ε) · σ(W_cascade(Θ) z_i)
    Ŷ^loss_i = ReLU(head([U_i, gate])) · gate

其中 Θ = (τ, γ, W_cascade) 由 LLM 本构泛函 q_φ(Θ|T) 从实时文本 `h_cur` 与历史 RAG `h_rag`
推断（变分重参数化），**不是**特征拼接，也**不是**路由分支。

张量约定（区域级整窗，节点维共享同一算子）
----------------------------------------
- `x`        : (B, T, N, F)  气象势能场驱动（F = 15，见 `C.FEATURE_NAMES_15`）
- `edge_index`: (2, E)       有向边（i, j），由区域双池邻接图导出
- `U`        : (B, N, d)     连续气象势能场状态变量（由算子积分得到）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_DISASTERS = 14
NUM_LEVELS = 5
NUM_CONT = 6


# ==========================================================================
# 本构泛函：LLM 文本 -> 微分方程参数
# ==========================================================================
class LLMConstitutiveFunctional(nn.Module):
    """amortized variational constitutive functional q_φ(Θ | h_cur, h_rag)。

    从实时文本后验中采样潜变量 z，解码出三组本构参数：
      - τ  : 相变临界阈值（softplus + 基座，恒正）
      - γ  : 防灾耗散率（sigmoid，d_model 维）
      - W  : 14×14 级联转移矩阵（低秩增量叠加在可学习基座上）

    先验 p_LLM(Θ|T_RAG) 由 `h_rag` 条件化；消融 `w/o Assimilation & RAG Prior`
    把先验退化为标准正态。
    """

    def __init__(self, text_dim: int, hidden: int = 128, d_z: int = 24,
                 rank: int = 8, d_model: int = 128, prior_from_rag: bool = True):
        super().__init__()
        self.text_dim = text_dim
        self.d_z = d_z
        self.rank = rank
        self.d_model = d_model
        self.prior_from_rag = prior_from_rag

        self.post = nn.Sequential(
            nn.Linear(text_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_z * 2),
        )
        if prior_from_rag:
            self.prior = nn.Sequential(
                nn.Linear(text_dim, hidden), nn.GELU(),
                nn.Linear(hidden, d_z * 2),
            )
        self.tau_head = nn.Linear(d_z, 1)
        self.gamma_head = nn.Linear(d_z, d_model)
        self.w_head = nn.Linear(d_z, NUM_DISASTERS * rank + rank * NUM_DISASTERS)

        self.register_buffer("tau_base", torch.tensor(1.0))
        self.register_buffer("tau_scale", torch.tensor(0.5))

    def _decode(self, z: torch.Tensor):
        tau = F.softplus(self.tau_head(z)).squeeze(-1) * self.tau_scale + self.tau_base
        gamma = torch.sigmoid(self.gamma_head(z))
        w = self.w_head(z)
        d = NUM_DISASTERS * self.rank
        a = w[:, :d].view(-1, NUM_DISASTERS, self.rank)
        b = w[:, d:].view(-1, self.rank, NUM_DISASTERS)
        delta_w = torch.bmm(a, b) / (self.rank ** 0.5)          # (M,14,14)
        return tau, gamma, delta_w

    def forward(self, h_cur: torch.Tensor, h_rag: Optional[torch.Tensor] = None,
                sample: bool = True):
        # 输入按 (..., text_dim) 组织，内部展平处理
        lead = h_cur.shape[:-1]
        hc = h_cur.reshape(-1, h_cur.shape[-1])
        hr = (torch.zeros_like(hc) if h_rag is None else h_rag.reshape(-1, h_rag.shape[-1]))
        z_mu, z_lv = self.post(torch.cat([hc, hr], dim=-1)).chunk(2, dim=-1)
        z_lv = z_lv.clamp(-8.0, 8.0)
        z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_lv) if sample else z_mu
        tau, gamma, delta_w = self._decode(z)

        if self.prior_from_rag:
            p_mu, p_lv = self.prior(hr).chunk(2, dim=-1)
            p_lv = p_lv.clamp(-8.0, 8.0)
        else:  # 消融：先验退化为标准正态
            p_mu, p_lv = torch.zeros_like(z_mu), torch.zeros_like(z_lv)
        var_q, var_p = torch.exp(z_lv), torch.exp(p_lv)
        kl = 0.5 * ((z_mu - p_mu).pow(2) / var_p + var_q / var_p - 1.0 + p_lv - z_lv).sum(-1)
        # 复原前导维：tau (...,)，gamma (...,d)，delta_w (...,14,14)
        return {
            "tau": tau.view(*lead),
            "gamma": gamma.view(*lead, self.d_model),
            "delta_w": delta_w.view(*lead, NUM_DISASTERS, NUM_DISASTERS),
            "kl": kl.view(*lead),
        }


class StaticConstitutive(nn.Module):
    """消融 `w/o LLM Constitutive Functional`：Θ 退化为与样本无关的静态参数。"""

    def __init__(self, d_model: int = 128, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.d_model = d_model
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.gamma = nn.Parameter(torch.zeros(d_model))
        self.a = nn.Parameter(torch.randn(NUM_DISASTERS, rank) * 0.02)
        self.b = nn.Parameter(torch.randn(rank, NUM_DISASTERS) * 0.02)

    def forward(self, h_cur, h_rag=None, sample: bool = True):
        lead = h_cur.shape[:-1]
        m = int(torch.tensor(lead).prod().item()) if lead else 1
        device = h_cur.device
        tau = F.softplus(self.tau).expand(m)
        gamma = torch.sigmoid(self.gamma).unsqueeze(0).expand(m, -1)
        dw = (self.a @ self.b) / (self.rank ** 0.5)
        return {
            "tau": tau.view(*lead),
            "gamma": gamma.view(*lead, self.d_model),
            "delta_w": dw.unsqueeze(0).expand(m, -1, -1).view(*lead, NUM_DISASTERS, NUM_DISASTERS),
            "kl": torch.zeros(m, device=device).view(*lead),
        }


# ==========================================================================
# 平流-反应-相变算子的一步
# ==========================================================================
def _scatter_sum(src: torch.Tensor, index: torch.Tensor, n_out: int) -> torch.Tensor:
    out = src.new_zeros((*src.shape[:-2], n_out, src.shape[-1]))
    idx = index.view(1, -1, 1).expand(src.shape[0], -1, src.shape[-1])
    return out.scatter_add_(1, idx, src)

class PhysicsAdvectionReactionCell(nn.Module):
    """一步算子演化：水平平流 + 垂直注入 + 耗散 [+ 同化跳变]。

    平流核由**真实物理量**驱动（`C.ADV_KERNEL_FEATURES` = ΔPRS / WIN_S_Max / WIN_S_2mi_Avg）：

        k_ij = softplus(MLP([φ_i, φ_j, φ_j - φ_i])），φ = 物理量三元组

    再乘物理边权（训练期相关强度）并按出度归一。传导率依赖 i、j 各自气象，
    因此天然非对称——这是真实平流的性质，不是自由注意力。
    """

    def __init__(self, d_model: int, n_features: int, adv_dim: int = 3,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.enc = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_model))
        self.norm = nn.Identity()
        self.k_adv = nn.Sequential(nn.Linear(adv_dim * 4, hidden // 2), nn.GELU(),
                                   nn.Linear(hidden // 2, 1))
        self.psi_in = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                    nn.Linear(hidden, d_model))
        self.drop = nn.Dropout(dropout)
        self.log_dt = nn.Parameter(torch.zeros(1))
        # 势能幅值软饱和尺度：**不做 LayerNorm**，因为相变门控读取的正是 ||U||，
        # 逐节点标准化会把幅值信息抹平（所有节点门控恒等）。改用可学习的
        # 全局 RMS 上限做整体缩放，既防爆炸又保留节点间的相对幅值。
        self.rms_cap = nn.Parameter(torch.tensor(2.0))

    def advection(self, U, adv, edge_index, edge_w):
        """水平平流项：保守的各向异性输运（能量守恒）。

        无向邻接图（双池邻接已对称化）按**无向对**处理：每一对 (a,b) 只算一次，
        对 a 贡献 `-g·(U_a-U_b)`、对 b 贡献 `+g·(U_a-U_b)`，总和恒为 0。
        传导率 `g = k_ab · w_ab` 由物理量（|ΔP|、两端风速）决定，不含方向伪造。
        """
        src, dst = edge_index[0], edge_index[1]
        N = U.shape[1]
        a = torch.minimum(src, dst)
        b = torch.maximum(src, dst)
        key = a * N + b
        _, first = torch.unique(key, return_inverse=False), None
        # 每对无向边只保留首次出现，避免同一对重复计数
        order = torch.argsort(key, stable=True)
        k_sorted = key[order]
        keep_sorted = torch.ones_like(k_sorted, dtype=torch.bool)
        keep_sorted[1:] = k_sorted[1:] != k_sorted[:-1]
        keep = torch.zeros_like(key, dtype=torch.bool)
        keep[order[keep_sorted]] = True
        a, b = a[keep], b[keep]

        adv_a, adv_b = adv[:, a], adv[:, b]
        w = edge_w[keep].view(1, -1).clamp_min(0.0).expand(U.shape[0], -1)
        feat = torch.cat([adv_a, adv_b, (adv_b - adv_a).abs(),
                          adv_b.abs() - adv_a.abs()], dim=-1)
        g = F.softplus(self.k_adv(feat)).squeeze(-1) * w      # (B,E')
        U_a, U_b = U[:, a], U[:, b]
        msg = g.unsqueeze(-1) * (U_a - U_b)                   # (B,E',d)
        flux = -_scatter_sum(msg, b, N) + _scatter_sum(msg, a, N)
        return flux, g

    def forward(self, U, x_t, adv_t, edge_index, edge_w, gamma,
                use_advection: bool = True, assim_delta=None):
        dt = F.softplus(self.log_dt) * 0.5
        dU, _ = (self.advection(U, adv_t, edge_index, edge_w)
                if use_advection else (torch.zeros_like(U), None))
        dU = dU + torch.tanh(self.psi_in(x_t)) - gamma * U
        U = U + dt * self.drop(dU)
        # 平滑饱和：U *= cap·tanh(rms/cap)/rms，幅值趋近 cap 但不截断，
        # 严格保持节点间幅值次序（否则所有节点被压成同一范数，门控退化为常数）。
        cap = F.softplus(self.rms_cap) + 1e-3
        rms = U.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        U = U * (cap * torch.tanh(rms / cap) / rms)
        if assim_delta is not None:
            U = U + assim_delta
        return U


class GATAdvectionCell(PhysicsAdvectionReactionCell):
    """消融 `w/o Advection PDE`：自由注意力替代气压-风场物理核。"""

    def __init__(self, d_model: int, n_features: int, adv_dim: int = 3,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__(d_model, n_features, adv_dim, hidden, dropout)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def advection(self, U, adv, edge_index, edge_w):
        src, dst = edge_index[0], edge_index[1]
        q = self.q(U)[:, src]
        k = self.k(U)[:, dst]
        v = self.v(U)[:, dst]
        score = (q * k).sum(-1) / (q.shape[-1] ** 0.5)
        score = score.masked_fill(edge_w.view(1, -1) <= 0, -1e9)
        alpha = torch.softmax(score, dim=-1) * (edge_w.view(1, -1) > 0).float()
        deg = alpha.new_zeros((alpha.shape[0], U.shape[1])).scatter_add_(
            1, src.view(1, -1).expand(alpha.shape[0], -1), alpha).clamp_min(1e-6)
        msg = alpha.unsqueeze(-1) * (U[:, src] - v)
        return -_scatter_sum(msg, src, U.shape[1]) / deg.unsqueeze(-1), alpha


# ==========================================================================
# 朗道相变读出
# ==========================================================================
class LandauPhaseTransitionReadout(nn.Module):
    """可微朗道门控 + 多任务读出。

    `phase_gate = σ((||U||_2 - τ)/ε)`：亚临界（||U||<τ）时门控 ≈ 0，灾害图谱静默；
    突破阈值后 14 类多标签概率与 6 项连续损失同时被打开。
    """

    def __init__(self, d_model: int, eps: float = 0.2, hidden: int = 128,
                 use_phase_gate: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_phase_gate = use_phase_gate
        self.eps_raw = nn.Parameter(torch.tensor(float(eps)))
        self.type_proj = nn.Linear(d_model, NUM_DISASTERS)      # U -> 14 维灾种潜空间
        self.w_base = nn.Parameter(torch.zeros(NUM_DISASTERS, NUM_DISASTERS))
        self.b_topo = nn.Parameter(torch.zeros(NUM_DISASTERS))
        self.topo_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                       nn.Linear(hidden, NUM_DISASTERS))
        self.level_head = nn.Sequential(nn.Linear(d_model + 1, hidden), nn.GELU(),
                                        nn.Linear(hidden, NUM_LEVELS))
        self.cont_head = nn.Sequential(nn.Linear(d_model + 1, hidden), nn.GELU(),
                                       nn.Linear(hidden, NUM_CONT))
        # 连续损失在 log10(1+y) 空间取值 ~0.5~4；ReLU 读出若从 0 起步会长时间
        # 学不到尺度（R² 起初极端为负）。把末层偏置初始化到该先验均值附近。
        nn.init.constant_(self.cont_head[-1].bias, 1.5)
        with torch.no_grad():
            self.cont_head[-1].weight.mul_(0.1)
        self.drop = nn.Dropout(dropout)

    def phase_gate(self, U: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """朗道门控：σ((RMS(U) - τ)/ε)。

        用 RMS（`||U||_2/√d`）而非裸 L2 范数：`U` 经 LayerNorm 后裸范数恒 ≈ √d，
        门控将退化为常数；RMS 把临界阈值放回 O(1) 的可学习尺度，
        τ 的物理含义即"亚临界/超临界"的分界幅值。
        """
        if not self.use_phase_gate:
            return torch.ones(U.shape[:2], device=U.device)
        eps = F.softplus(self.eps_raw) + 0.02
        rms = U.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
        return torch.sigmoid((rms - tau) / eps)

    def forward(self, U, tau, delta_w):
        gate = self.phase_gate(U, tau)                              # (B,N)
        U_d = self.drop(U)
        z = self.type_proj(U_d)                                     # (B,N,14)
        cascade = self.w_base + delta_w                             # (B,N,14,14)
        # 级联项：14×14 转移矩阵作用在 14 维灾种潜空间上（表征"暴雨→洪涝→地质灾害"）
        logits = torch.einsum("bnij,bnj->bni", cascade, z) \
            + self.topo_head(U_d) + self.b_topo
        topo_prob = gate.unsqueeze(-1) * torch.sigmoid(logits)
        feat = torch.cat([U_d, gate.unsqueeze(-1)], dim=-1)
        level_logits = self.level_head(feat)
        cont = F.relu(self.cont_head(feat)) * gate.unsqueeze(-1)
        return {"topo_prob": topo_prob, "topo_logits": logits,
                "level_logits": level_logits, "cont": cont,
                "phase_gate": gate, "U": U}


# ==========================================================================
# 主模型
# ==========================================================================
@dataclass
class GNOConfig:
    n_features: int = 15
    adv_dim: int = 3
    text_dim: int = 256
    d_model: int = 128
    hidden: int = 128
    dropout: float = 0.1
    eps: float = 0.2
    use_phase_gate: bool = True
    use_advection_kernel: bool = True
    use_llm_functional: bool = True
    use_assimilation: bool = True
    prior_from_rag: bool = True

    @staticmethod
    def ablation(name: str) -> "GNOConfig":
        cfg = GNOConfig()
        if name == "w/o Phase-Transition Gate":
            cfg.use_phase_gate = False
        elif name == "w/o Advection PDE":
            cfg.use_advection_kernel = False
        elif name == "w/o LLM Constitutive Functional":
            cfg.use_llm_functional = False
        elif name == "w/o Assimilation & RAG Prior":
            cfg.use_assimilation = False
            cfg.prior_from_rag = False
        else:
            raise KeyError(f"未知消融配置: {name}")
        return cfg


class PT_GNO(nn.Module):
    """统一前向：连续场演化 → 相变门控 → 离散图谱与损失读出。"""

    def __init__(self, cfg: GNOConfig | None = None):
        super().__init__()
        self.cfg = cfg or GNOConfig()
        c = self.cfg
        cell_cls = PhysicsAdvectionReactionCell if c.use_advection_kernel else GATAdvectionCell
        self.cell = cell_cls(c.d_model, c.n_features, adv_dim=c.adv_dim,
                             hidden=c.hidden, dropout=c.dropout)
        self.constitutive = (LLMConstitutiveFunctional(
            c.text_dim, hidden=c.hidden, d_model=c.d_model, prior_from_rag=c.prior_from_rag)
            if c.use_llm_functional else StaticConstitutive(d_model=c.d_model))
        self.readout = LandauPhaseTransitionReadout(
            c.d_model, eps=c.eps, hidden=c.hidden,
            use_phase_gate=c.use_phase_gate, dropout=c.dropout)
        # 同化跳变算子 G_assim，仅在观测到达时刻 M(t)=1 生效（Task 3）
        self.assim = nn.Sequential(
            nn.Linear(c.d_model + c.text_dim, c.hidden), nn.GELU(),
            nn.Linear(c.hidden, c.d_model),
        )

    # ------------------------------------------------------------------
    def convective_params(self, batch) -> dict:
        """逐 (节点, 时刻) 推断本构参数 Θ=(τ, γ, W_cascade)。

        输入按 (B,T,N,D)，输出同形 —— 本征参数是**局地**的（每个区县、每一天各异），
        这正是"LLM 作为微分方程本构泛函"的含义。
        """
        h_cur = batch["h_cur"]
        h_rag = batch.get("h_rag") if self.cfg.prior_from_rag else None
        return self.constitutive(h_cur, h_rag, sample=self.training)

    def forward(self, batch: dict) -> dict:
        c = self.cfg
        x = batch["x"]                                  # (B,T,N,F)
        mask = batch.get("mask")
        if mask is not None:
            x = x * mask
        adv = batch["adv"]                              # (B,T,N,3)
        edge_index = batch["edge_index"]                # (2,E)
        edge_w = batch["edge_w"]                        # (E,)
        theta = self.convective_params(batch)
        assim_mask = batch.get("assim_mask") if c.use_assimilation else None

        B, T, N, _ = x.shape
        U = self.cell.enc(x[:, 0].reshape(B * N, -1)).view(B, N, -1)
        for t in range(T):
            assim_delta = None
            if assim_mask is not None:
                obs = assim_mask[:, t]                              # (B,N)
                if bool((obs > 0).any()):
                    h_post = batch["h_post"][:, t]
                    jump = self.assim(torch.cat([U, h_post], dim=-1))
                    assim_delta = jump * obs.unsqueeze(-1)
            U = self.cell(U, x[:, t], adv[:, t], edge_index, edge_w,
                          theta["gamma"][:, t], assim_delta=assim_delta)
        r = self.readout(U, theta["tau"][:, -1], theta["delta_w"][:, -1])
        r["kl"] = theta["kl"].mean()
        r["tau"] = theta["tau"][:, -1]
        return r


def build_pt_gno(ablation: str | None = None, **overrides) -> PT_GNO:
    cfg = GNOConfig() if ablation is None else GNOConfig.ablation(ablation)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return PT_GNO(cfg)