"""变分自由能训练器与多任务评估指标。

统一目标（`CLAUDE.md` §1.4）：

    L_FreeEnergy = L_topo (Focal BCE)
                 + λ_lvl · L_level (Weighted CE)
                 + λ_cont · L_cont (Masked Huber, log10 空间)
                 + β · KL(q_φ(Θ|T_≤t) || p(Θ|T_RAG))

Phase 2.5 校准（数值动力学修复）：
1. 连续损失头用 `softplus(head + cont_pos_mean)`，与相变门控**解耦**；
   门控通过 Straight-Through 保留梯度但不压缩正样本幅值 → 条件 R² 转正。
2. Focal BCE 用训练集 `pos_weight = clamp(sqrt(neg/pos), 1, 25)` 平衡 14 类长尾。
3. **废除 0.5 硬阈值**：在验证集上逐类网格搜索最大化 F1 的阈值 `best_th (14,)`，
   验证/测试统一用 `best_th`；同时报告「活跃发灾窗口」子域与全网格两套指标。

指标：Macro-F1、Micro-F1、AUPRC、MRR、Hits@3、log-MAE、log-RMSE、R²。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import common as C


# ==========================================================================
# 损失
# ==========================================================================
def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
              pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """多标签 Focal BCE（对 14 类逐元素，返回标量均值）。"""
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    return (bce * (1 - pt).pow(gamma)).mean()


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                class_weight: torch.Tensor | None = None) -> torch.Tensor:
    if mask.sum() < 1:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask > 0], target[mask > 0], weight=class_weight)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 delta: float = 1.0) -> torch.Tensor:
    """log10(1+y) 空间的 Huber，掩码为 (B,N)。"""
    if mask.sum() < 1:
        return pred.new_zeros(())
    diff = F.huber_loss(pred, target, reduction="none", delta=delta)     # (B,N,6)
    w = mask.unsqueeze(-1)
    return (diff * w).sum() / (w.sum() * pred.shape[-1]).clamp_min(1.0)


@dataclass
class LossConfig:
    w_topo: float = 1.0
    w_level: float = 0.5
    w_cont: float = 0.5
    beta_max: float = 0.05
    beta_warmup: int = 10
    focal_gamma: float = 2.0
    huber_delta: float = 1.0
    w_phase: float = 0.5      # 相变门控的相位监督权重（因子化似然）
    # 因子化似然的条件化类型监督：只在"确实有灾"的节点-日上监督 14 类判别。
    # 不这样做时，98% 以上的空样本主导类型头梯度，类型概率被整体压低，
    # 联合概率 gate·σ(logit) 的排序因此劣于纯 GNN 基线（实测 F1a 落后 0.035）。
    w_type: float = 0.5
    type_cond_only: bool = True


def free_energy_loss(out: dict, batch: dict, cfg: LossConfig, epoch: int,
                     class_weight=None) -> tuple[torch.Tensor, dict]:
    """变分自由能：拓扑 Focal BCE + 等级加权 CE + 条件连续 Huber + β·KL。

    连续损失使用 `out["cont"]`（Straight-Through 后的条件回归），
    不使用被亚临界门控压小的 `cont_full`。
    """
    # 基线若只给出概率（早期实现），退化为概率的 logit；主模型直接给 logits。
    topo_logits = out.get("topo_logits")
    if topo_logits is None:
        topo_logits = torch.logit(out["topo_prob"].float().clamp(1e-6, 1 - 1e-6))
    # 类型判别：因子化似然 P(A)=P(相变)·P(类型|相变)。
    # `type_cond_only` 时只在正样本节点-日上算 Focal BCE（条件似然），
    # 相位与"是否有灾"由门控单独承担；否则退回联合似然。
    pt = batch.get("phase_target")
    # 条件似然的前提是"检测由门控负责"。无门控消融/基线的相位由多标签输出导出，
    # 类型头必须同时承担检测，因此退回联合似然。
    cond_ok = bool(out.get("phase_learnable", True))
    if cfg.type_cond_only and cond_ok and pt is not None:
        # boolean indexing 后是一维张量，focal_bce 对任意形状逐元素算，直接传入即可
        m = pt.unsqueeze(-1).expand_as(topo_logits) > 0
        l_type = (focal_bce(topo_logits[m], batch["topo"][m], gamma=cfg.focal_gamma)
                  if bool(m.any()) else topo_logits.new_zeros(()))
    else:
        l_type = focal_bce(topo_logits, batch["topo"], gamma=cfg.focal_gamma,
                           pos_weight=batch.get("pos_weight"))
    l_lvl = weighted_ce(out["level_logits"], batch["level"], batch["level_mask"],
                        class_weight=class_weight)
    l_cnt = masked_huber(out["cont"], batch["cont"], batch["cont_mask"], cfg.huber_delta)
    # 相位监督：联合似然做因子化分解 P(A)=P(相变)·P(类型|相变)。
    # 若只监督联合项，门控在"几乎无正样本"时会退化为先验（恒 0），
    # 类型头随即失去梯度；直接用"该节点-日是否有灾"监督门控，
    # 等价于把**临界阈值 τ** 直接学成事件检测器。
    l_ph = torch.zeros((), device=l_type.device)
    if batch.get("phase_target") is not None:
        if out.get("phase_logit") is not None:      # PT-GNO：直接监督朗道门控
            phase_logit = out["phase_logit"]
        else:                                        # 基线：由多标签输出导出同构的相位
            p_any = 1.0 - (1.0 - out["topo_prob"]).prod(-1)
            phase_logit = torch.logit(p_any.clamp(1e-6, 1 - 1e-6))
        l_ph = F.binary_cross_entropy_with_logits(phase_logit, batch["phase_target"])
    beta = cfg.beta_max * min(1.0, epoch / max(cfg.beta_warmup, 1))
    kl = out.get("kl", torch.zeros((), device=l_type.device))
    total = (cfg.w_type * l_type + cfg.w_level * l_lvl + cfg.w_cont * l_cnt
             + cfg.w_phase * l_ph + beta * kl)
    return total, {"topo": float(l_type.detach()), "level": float(l_lvl.detach()),
                   "cont": float(l_cnt.detach()), "phase": float(l_ph.detach()),
                   "kl": float(kl.detach()), "beta": beta}


# ==========================================================================
# 阈值校准（Phase 2.5）
# ==========================================================================
def calibrate_thresholds(probs: np.ndarray, targets: np.ndarray,
                         lo: float = 0.05, hi: float = 0.85,
                         n_grid: int = 33) -> np.ndarray:
    """逐类网格搜索使单类 F1 最大的判定阈值。

    `probs/targets`: (n, 14)。返回 `(14,)` 阈值向量；某类在验证集中无正样本时
    回退 0.5（该类的 F1 也不参与 Macro 平均）。
    """
    n_cls = probs.shape[1]
    grid = np.linspace(lo, hi, n_grid)
    best = np.full(n_cls, 0.5, dtype=np.float32)
    y = targets > 0
    for c in range(n_cls):
        yc = y[:, c]
        if yc.sum() == 0:
            continue
        f1s = np.empty(len(grid), dtype=np.float64)
        for i, t in enumerate(grid):
            pred = probs[:, c] >= t
            tp = np.logical_and(pred, yc).sum()
            fp = np.logical_and(pred, ~yc).sum()
            fn = np.logical_and(~pred, yc).sum()
            f1s[i] = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        best[c] = float(grid[int(np.argmax(f1s))])
    return best


def topo_metrics(p: np.ndarray, y: np.ndarray, thresholds: np.ndarray | None = None,
                 domain: str = "all", k_hits: int = 3) -> dict:
    """14 类多标签指标；`domain="active"` 时只统计真有灾种的节点-日。"""
    y = y > 0
    if domain == "active":
        rows = y.sum(1) > 0
        p, y = p[rows], y[rows]
    if p.shape[0] == 0:
        return {"macro_f1": float("nan"), "micro_f1": float("nan"),
                "auprc": float("nan"), "mrr": float("nan"),
                "hits@3": float("nan"), "n_rows": 0, "n_pos": 0, "n_pred_pos": 0}
    th = (np.full(p.shape[1], 0.5, dtype=np.float32) if thresholds is None
          else np.asarray(thresholds, dtype=np.float32))
    pred = p >= th

    tp = np.logical_and(pred, y).sum(0)
    fp = np.logical_and(pred, ~y).sum(0)
    fn = np.logical_and(~pred, y).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    present = y.sum(0) > 0
    out = {
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-9)),
        "n_rows": int(p.shape[0]), "n_pos": int(y.sum()),
        "n_pred_pos": int(pred.sum()),
    }
    try:
        from sklearn.metrics import average_precision_score
        aps = [average_precision_score(y[:, c], p[:, c])
               for c in range(y.shape[1]) if y[:, c].sum() > 0]
        out["auprc"] = float(np.mean(aps)) if aps else float("nan")
    except Exception:                                   # noqa: BLE001
        out["auprc"] = float("nan")

    # 排序指标：只对真有灾种的样本统计（随机基线 = k/14）
    rows = y.sum(1) > 0
    if rows.any():
        order = np.argsort(-p[rows], axis=1)
        yr = y[rows]
        kk = min(k_hits, y.shape[1])
        out["hits@3"] = float(np.take_along_axis(
            yr.astype(np.int64), order[:, :kk], axis=1).max(1).mean())
        ranks = np.argmax(np.take_along_axis(yr.astype(np.int64), order, axis=1), axis=1) + 1
        out["mrr"] = float((1.0 / ranks).mean())
    else:
        out["hits@3"] = float("nan")
        out["mrr"] = float("nan")
    return out


def cont_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """条件连续损失指标（log10 空间）：MAE / RMSE / R²。"""
    m = mask.reshape(-1) > 0
    if not m.any():
        return {"log_mae": float("nan"), "log_rmse": float("nan"), "r2": float("nan"),
                "r2_n_valid": 0}
    cp = pred.reshape(-1, pred.shape[-1])[m]
    ct = true.reshape(-1, true.shape[-1])[m]
    ss_res = ((cp - ct) ** 2).sum(0)
    ss_tot = ((ct - ct.mean(0)) ** 2).sum(0)
    valid = ss_tot > 1e-3 * max(len(ct), 1)
    r2 = (float(np.clip((1 - ss_res[valid] / ss_tot[valid]).mean(), -10.0, 1.0))
          if valid.any() else float("nan"))
    return {"log_mae": float(np.abs(cp - ct).mean()),
            "log_rmse": float(np.sqrt(((cp - ct) ** 2).mean())),
            "r2": r2, "r2_n_valid": int(valid.sum())}


def level_metrics(logits: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    m = mask.reshape(-1) > 0
    if not m.any():
        return {"level_acc": float("nan"), "level_weighted_f1": float("nan")}
    lp = logits.reshape(-1, logits.shape[-1])[m].argmax(1)
    lt = target.reshape(-1)[m]
    f1s, ws = [], []
    for c in np.unique(lt):
        tp = ((lp == c) & (lt == c)).sum()
        fp = ((lp == c) & (lt != c)).sum()
        fn = ((lp != c) & (lt == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
        ws.append((lt == c).sum())
    return {"level_acc": float((lp == lt).mean()),
            "level_weighted_f1": float(np.average(f1s, weights=ws))}


def compute_metrics(topo_prob, topo_true, level_logits, level_true, level_mask,
                    cont, cont_true, cont_mask, k_hits: int = 3,
                    thresholds: np.ndarray | None = None) -> dict:
    """统一指标入口（张量输入，供单元测试与快速评估使用）。"""
    p = topo_prob.detach().cpu().numpy().reshape(-1, topo_prob.shape[-1])
    y = topo_true.detach().cpu().numpy().reshape(-1, topo_true.shape[-1])
    out = topo_metrics(p, y, thresholds=thresholds, domain="all", k_hits=k_hits)
    out.update(level_metrics(level_logits.detach().cpu().numpy(),
                             level_true.detach().cpu().numpy(),
                             level_mask.detach().cpu().numpy()))
    out.update(cont_metrics(cont.detach().cpu().numpy(),
                            cont_true.detach().cpu().numpy(),
                            cont_mask.detach().cpu().numpy()))
    return out


# ==========================================================================
# 训练循环
# ==========================================================================
@dataclass
class TrainConfig:
    region: str = "SW_Cascade"
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    tin: int = 7
    delta: int = 1
    stride: int = 2
    patience: int = 10
    seed: int = 42
    device: str = "cuda"
    out_dir: str = "outputs/runs"
    log_every: int = 20
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    amp: bool = True
    eval_test: bool = True
    val_stride: int = 8        # 验证集采样步长（早停判定用）
    # 训练期注入"报告到达"的概率（Task 3 同化门的唯一梯度来源；评估期恒为 0）
    assim_p: float = 0.3


def _split_range(bundle, split: str) -> tuple[int, int]:
    y0, y1 = C.SPLITS[split]
    years = bundle.dates.year
    return (int(np.nonzero(years >= y0)[0][0]),
            int(np.nonzero(years <= y1)[0][-1]))


def _topo_pos_weight(bundle, split: str = "train") -> torch.Tensor:
    """14 类 Focal BCE 正类权重：`clamp(sqrt(neg/pos), 1, 25)`（仅训练期统计）。"""
    lo, hi = _split_range(bundle, split)
    lab = np.asarray(bundle.labels[:, lo:hi + 1, :]).reshape(-1, bundle.labels.shape[-1])
    pos = lab.sum(0).astype(np.float64)
    neg = lab.shape[0] - pos
    w = np.sqrt(neg / (pos + 1.0))
    return torch.tensor(np.clip(w, 1.0, 25.0), dtype=torch.float32)


def _class_weight(bundle, split: str = "train") -> torch.Tensor:
    lo, hi = _split_range(bundle, split)
    lv = bundle.day_loss_level[:, lo:hi + 1].reshape(-1)
    cnt = np.bincount(lv[lv >= 0], minlength=len(C.LOSS_LEVEL_NAMES)).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (cnt * len(cnt))
    return torch.tensor(w, dtype=torch.float32)


def cont_pos_mean(bundle, split: str = "train") -> np.ndarray:
    """训练集**可信损失记录**的 `log10(1+y)` 均值向量（6,）。"""
    r = bundle.records
    m = (r["cont_loss_mask"] > 0) & (r["split"] == split)
    vals = []
    for c in C.CONTINUOUS_LOSS_TARGETS:
        v = np.asarray(r.loc[m, c], dtype=np.float64)
        vals.append(np.log10(1.0 + np.maximum(v, 0.0)))
    return np.stack(vals, axis=1).mean(0).astype(np.float32)


def _iter_batches(ds, bundle, tcfg, device, train_mode: bool, epoch: int,
                  assim_p: float = 0.0):
    from src.engine.data import build_sample, collate

    order = np.arange(len(ds))
    if train_mode:
        np.random.default_rng(tcfg.seed + epoch).shuffle(order)
    for s in range(0, len(order), tcfg.batch_size):
        idx = order[s:s + tcfg.batch_size]
        rng = np.random.default_rng(tcfg.seed + epoch * 100003 + s) if train_mode else None
        samples = [build_sample(bundle, int(ds[int(i)]), tcfg.tin, tcfg.delta,
                                assim_p=(assim_p if train_mode else 0.0), rng=rng)
                   for i in idx]
        yield collate(samples, bundle, device)


def _collect(model, ds, bundle, tcfg, device, train_mode: bool, epoch: int,
             pos_w, cw, loss_cfg, opt=None):
    """跑一遍数据集；返回 (平均损失字典, 预测数组字典)。"""
    model.train(train_mode)
    # 只有带同化门的模型才需要"报告到达"注入；基线/消融拿不到这个信号，
    # 强行注入只会白白多算一批索引（且它们根本不读 assim_mask）。
    assim_p = (tcfg.assim_p
               if bool(getattr(getattr(model, "cfg", None), "use_assimilation", False))
               else 0.0)
    losses: dict[str, list[float]] = {}
    collect = not train_mode          # 训练轮不需要预测数组，避免 40 MB/轮的无谓拷贝
    P, Y, LL, LT, LM, CC, CT, CM = [], [], [], [], [], [], [], []
    for batch in _iter_batches(ds, bundle, tcfg, device, train_mode, epoch,
                               assim_p=assim_p):
        batch["pos_weight"] = pos_w
        with torch.set_grad_enabled(train_mode):
            with torch.autocast("cuda", enabled=tcfg.amp and device == "cuda",
                                dtype=torch.float16):
                out = model(batch)
                loss, parts = free_energy_loss(out, batch, loss_cfg, epoch, class_weight=cw)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        parts["loss"] = float(loss.detach())
        for k, v in parts.items():
            losses.setdefault(k, []).append(v)
        # 展平节点维：整窗样本 (B,N,·) -> 节点-日 (B*N,·)，指标按节点-日统计
        if not collect:
            continue
        P.append(out["topo_prob"].detach().float().cpu().numpy().reshape(-1, C.NUM_DISASTER_TYPES))
        Y.append(batch["topo"].cpu().numpy().reshape(-1, C.NUM_DISASTER_TYPES))
        LL.append(out["level_logits"].detach().float().cpu().numpy().reshape(-1, len(C.LOSS_LEVEL_NAMES)))
        LT.append(batch["level"].cpu().numpy().reshape(-1))
        LM.append(batch["level_mask"].cpu().numpy().reshape(-1))
        CC.append(out["cont"].detach().float().cpu().numpy().reshape(-1, len(C.CONTINUOUS_LOSS_TARGETS)))
        CT.append(batch["cont"].cpu().numpy().reshape(-1, len(C.CONTINUOUS_LOSS_TARGETS)))
        CM.append(batch["cont_mask"].cpu().numpy().reshape(-1))

    cat = (lambda xs: np.concatenate(xs, axis=0)) if P else (lambda xs: np.zeros((0,)))
    avg = {k: float(np.mean(v)) for k, v in losses.items()}
    if not collect:
        return avg, {}
    preds = {"prob": cat(P), "topo": cat(Y), "level_logits": cat(LL),
             "level": cat(LT), "level_mask": cat(LM),
             "cont": cat(CC), "cont_true": cat(CT), "cont_mask": cat(CM)}
    return avg, preds


def evaluate_preds(preds: dict, thresholds: np.ndarray | None = None) -> dict:
    """把预测数组汇总为一套指标（全网格 + 活跃窗口 + 等级 + 连续）。"""
    out = {}
    for domain, suffix in (("all", ""), ("active", "_active")):
        m = topo_metrics(preds["prob"], preds["topo"], thresholds, domain=domain)
        for k in ("macro_f1", "micro_f1", "auprc", "mrr", "hits@3"):
            out[k + suffix] = m[k]
        out["n_rows" + suffix] = m["n_rows"]
        out["n_pos" + suffix] = m["n_pos"]
    out.update(level_metrics(preds["level_logits"], preds["level"], preds["level_mask"]))
    out.update(cont_metrics(preds["cont"], preds["cont_true"], preds["cont_mask"]))
    return out


def train(model_builder, tcfg: TrainConfig, loss_cfg: LossConfig | None = None,
          bundle_loader=None, verbose: bool = True) -> dict:
    from src.engine.data import WindowDataset, load_region

    loss_cfg = loss_cfg or LossConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = tcfg.device if (torch.cuda.is_available() or tcfg.device == "cpu") else "cpu"
    bundle_loader = bundle_loader or load_region
    bundle = bundle_loader(tcfg.region)

    tr = WindowDataset(bundle, "train", tcfg.tin, tcfg.delta, tcfg.stride)
    # 早停判定只需代表性子集：验证集按 val_stride 稀疏采样，
    # 否则每轮都要扫 2,696 个窗口（与训练同耗时），单 run 从 6 min 涨到 20 min。
    va = WindowDataset(bundle, "val", tcfg.tin, tcfg.delta,
                       max(tcfg.stride * 2, tcfg.val_stride))
    te = (WindowDataset(bundle, "test", tcfg.tin, tcfg.delta, max(tcfg.stride, 2))
          if tcfg.eval_test else None)
    rng = np.random.default_rng(tcfg.seed)
    if tcfg.max_train_samples and len(tr) > tcfg.max_train_samples:
        tr.days = np.sort(rng.choice(tr.days, tcfg.max_train_samples, replace=False))
    if tcfg.max_val_samples and len(va) > tcfg.max_val_samples:
        va.days = np.sort(rng.choice(va.days, tcfg.max_val_samples, replace=False))

    model = model_builder().to(device)
    if hasattr(model, "set_cont_pos_mean"):
        model.set_cont_pos_mean(torch.from_numpy(cont_pos_mean(bundle)).to(device))
    cw = _class_weight(bundle).to(device)
    pos_w = _topo_pos_weight(bundle).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.epochs, 1))

    label = getattr(model, "ablation_name", None) or type(model).__name__
    # checkpoint 必须按 seed 分目录：并行 seed 共享同一 best.pt 会互相覆盖，
    # 而最终测试又是**从磁盘重载** best 权重，导致不同 seed 评估到别人的模型。
    out_dir = Path(tcfg.out_dir) / f"{tcfg.region}_{label}" / f"s{tcfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = {"val_loss": float("inf"), "score": -1.0, "epoch": -1, "metrics": {},
            "thresholds": np.full(C.NUM_DISASTER_TYPES, 0.5, np.float32)}

    t0 = time.time()
    bad = 0
    for ep in range(1, tcfg.epochs + 1):
        tr_loss, _ = _collect(model, tr, bundle, tcfg, device, True, ep,
                              pos_w, cw, loss_cfg, opt=opt)
        with torch.no_grad():
            va_loss, va_pred = _collect(model, va, bundle, tcfg, device, False, ep,
                                        pos_w, cw, loss_cfg)
        sched.step()

        # Phase 2.5：验证集逐类阈值校准
        th = calibrate_thresholds(va_pred["prob"], va_pred["topo"])
        va_m = evaluate_preds(va_pred, thresholds=th)
        va_m.update(va_loss)
        history.append({"epoch": ep, "train": tr_loss, "val": va_m,
                        "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"[{tcfg.region}|{label}|s{tcfg.seed}] ep{ep:03d} "
                  f"tr={tr_loss['loss']:.4f} va={va_m['loss']:.4f} "
                  f"F1={va_m['macro_f1']:.3f} F1act={va_m['macro_f1_active']:.3f} "
                  f"AUPRC={va_m['auprc_active']:.3f} H@3={va_m['hits@3']:.3f} "
                  f"R2={va_m['r2']:.3f}", flush=True)

        # 模型选择：用**主指标**（活跃发灾窗口 Macro-F1）而不是总损失。
        # 总损失里等级 CE 与连续回归占主导，用它选模型会在 Task 1 指标还在上升时
        # 就把 checkpoint 锁死（实测 best epoch 会被压到第 7 轮）。
        score = va_m.get("macro_f1_active", 0.0)
        if score > best["score"] + 1e-4:
            best = {"val_loss": va_m["loss"], "score": score, "epoch": ep,
                    "metrics": va_m, "thresholds": th}
            torch.save({"model": model.state_dict(), "thresholds": th,
                        "epoch": ep, "val_metrics": va_m}, out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                if verbose:
                    print(f"[{tcfg.region}|{label}|s{tcfg.seed}] early stop @ ep{ep}",
                          flush=True)
                break

    # 测试集只在最终评估一次（用 best 的阈值）
    test_m = {}
    if te is not None and best["epoch"] > 0:
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            _, te_pred = _collect(model, te, bundle, tcfg, device, False, 0,
                                  pos_w, cw, loss_cfg)
        test_m = evaluate_preds(te_pred, thresholds=best["thresholds"])

    result = {"region": tcfg.region, "name": label, "seed": tcfg.seed,
              "ckpt": str(out_dir / "best.pt"),
              "config": asdict(tcfg), "loss_config": asdict(loss_cfg),
              "best": {"val_loss": best["val_loss"], "score": best["score"],
                       "epoch": best["epoch"], "metrics": best["metrics"]},
              "thresholds": np.asarray(best["thresholds"]).tolist(),
              "test": test_m, "history": history,
              "seconds": round(time.time() - t0, 1),
              "model_params": int(sum(p.numel() for p in model.parameters())),
              "n_train_windows": len(tr), "n_val_windows": len(va),
              "text": bundle.text_missing_reason}
    (out_dir / "train_log.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 释放显存与对 bundle 的引用，避免多 run 串行时内存持续上涨
    del bundle, tr, va, te, model, opt
    if device != "cpu":
        torch.cuda.empty_cache()
    return result


__all__ = ["LossConfig", "TrainConfig", "free_energy_loss", "compute_metrics",
           "focal_bce", "weighted_ce", "masked_huber", "train",
           "calibrate_thresholds", "topo_metrics", "cont_metrics", "level_metrics",
           "evaluate_preds", "cont_pos_mean"]