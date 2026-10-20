"""变分自由能训练器与多任务评估指标。

统一目标（`CLAUDE.md` §1.4）：

    L_FreeEnergy = L_topo (Focal BCE)
                 + λ_lvl · L_level (Weighted CE)
                 + λ_cont · L_cont (Masked Huber, log10 空间)
                 + β · KL(q_φ(Θ|T_≤t) || p(Θ|T_RAG))

- `β` 前 10 轮线性退火，避免早期 KL 塌缩；
- Focal BCE 处理 14 类极强长尾（沙尘暴/高温 个位数样本）；
- Huber 只在 `cont_mask == 1`（损失可信）的节点上计算。

指标：Macro-F1、Micro-F1、AUPRC、Hits@3、log-MAE、R²。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
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
    """log10(1+y) 空间的 Huber，掩码对齐为 (B,N)。"""
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


def free_energy_loss(out: dict, batch: dict, cfg: LossConfig, epoch: int,
                     class_weight=None) -> tuple[torch.Tensor, dict]:
    l_topo = focal_bce(out["topo_logits"], batch["topo"], gamma=cfg.focal_gamma,
                       pos_weight=batch.get("pos_weight"))
    l_lvl = weighted_ce(out["level_logits"], batch["level"], batch["level_mask"],
                        class_weight=class_weight)
    l_cnt = masked_huber(out["cont"], batch["cont"], batch["cont_mask"], cfg.huber_delta)
    beta = cfg.beta_max * min(1.0, epoch / max(cfg.beta_warmup, 1))
    kl = out.get("kl", torch.zeros((), device=l_topo.device))
    total = cfg.w_topo * l_topo + cfg.w_level * l_lvl + cfg.w_cont * l_cnt + beta * kl
    return total, {"topo": float(l_topo), "level": float(l_lvl),
                   "cont": float(l_cnt), "kl": float(kl), "beta": beta}


# ==========================================================================
# 指标
# ==========================================================================
def compute_metrics(topo_prob: torch.Tensor, topo_true: torch.Tensor,
                    level_logits: torch.Tensor, level_true: torch.Tensor,
                    level_mask: torch.Tensor, cont: torch.Tensor,
                    cont_true: torch.Tensor, cont_mask: torch.Tensor,
                    k_hits: int = 3) -> dict:
    """在整批 (B,N) 上计算多任务指标。"""
    out = {}
    p = topo_prob.reshape(-1, topo_prob.shape[-1]).detach().cpu().numpy()
    y = (topo_true.reshape(-1, topo_true.shape[-1]) > 0).detach().cpu().numpy()
    pred = (p >= 0.5).astype(np.float32)

    tp = (pred * y).sum(0)
    fp = (pred * (1 - y)).sum(0)
    fn = ((1 - pred) * y).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    present = y.sum(0) > 0
    out["macro_f1"] = float(f1[present].mean()) if present.any() else 0.0
    out["micro_f1"] = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-9))
    out["n_pos"] = int(y.sum())
    out["n_pred_pos"] = int(pred.sum())

    # AUPRC（逐类平均，仅在有正样本的类上）
    try:
        from sklearn.metrics import average_precision_score
        aps = []
        for c in range(y.shape[1]):
            if y[:, c].sum() > 0:
                aps.append(average_precision_score(y[:, c], p[:, c]))
        out["auprc"] = float(np.mean(aps)) if aps else 0.0
    except Exception:
        out["auprc"] = float("nan")

    # Hits@k：只对**真的有灾种**的样本统计（否则被海量空样本稀释，
    # 随机基线应约为 k/14 ≈ 0.214，不能拿全体均值充数）
    rows = y.sum(1) > 0
    if rows.any():
        kk = min(k_hits, y.shape[1])
        topk = np.argsort(-p[rows], axis=1)[:, :kk]
        hit = np.take_along_axis(y[rows].astype(np.int64), topk, axis=1).max(1)
        out["hits@3"] = float(hit.mean())
        out["hits@3_random"] = kk / y.shape[1]
    else:
        out["hits@3"] = float("nan")
        out["hits@3_random"] = float("nan")

    # 损失等级
    lm = level_mask.reshape(-1).detach().cpu().numpy() > 0
    if lm.any():
        lpred = level_logits.reshape(-1, level_logits.shape[-1]).detach().cpu().numpy()[lm]
        ltrue = level_true.reshape(-1).detach().cpu().numpy()[lm]
        out["level_acc"] = float((lpred.argmax(1) == ltrue).mean())
        # Weighted-F1
        f1s, ws = [], []
        for c in np.unique(ltrue):
            tp_c = ((lpred.argmax(1) == c) & (ltrue == c)).sum()
            fp_c = ((lpred.argmax(1) == c) & (ltrue != c)).sum()
            fn_c = ((lpred.argmax(1) != c) & (ltrue == c)).sum()
            f1s.append(2 * tp_c / max(2 * tp_c + fp_c + fn_c, 1e-9))
            ws.append((ltrue == c).sum())
        out["level_weighted_f1"] = float(np.average(f1s, weights=ws))
    else:
        out["level_acc"] = float("nan")
        out["level_weighted_f1"] = float("nan")

    # 连续损失（log10 空间）
    cm = cont_mask.reshape(-1).detach().cpu().numpy() > 0
    if cm.any():
        cp = cont.reshape(-1, cont.shape[-1]).detach().cpu().numpy()[cm]
        ct = cont_true.reshape(-1, cont_true.shape[-1]).detach().cpu().numpy()[cm]
        out["log_mae"] = float(np.abs(cp - ct).mean())
        # R² 只在**有方差**的目标列上统计：某列在批内几乎为常数时 ss_tot≈0，
        # 直接相除会给出 -1e9 级别、毫无信息量的伪结果。
        ss_res = ((cp - ct) ** 2).sum(0)
        ss_tot = ((ct - ct.mean(0)) ** 2).sum(0)
        valid = ss_tot > 1e-3 * max(len(ct), 1)
        out["r2"] = float(np.clip(
            (1 - ss_res[valid] / ss_tot[valid]).mean(), -10.0, 1.0)) if valid.any() else float("nan")
        out["r2_n_valid"] = int(valid.sum())
    else:
        out["log_mae"] = float("nan")
        out["r2"] = float("nan")
    return out


def merge_metrics(a: dict, b: dict, n_a: int, n_b: int) -> dict:
    """按样本数加权合并两批指标（近似，用于 epoch 级汇总）。"""
    tot = max(n_a + n_b, 1)
    w = {"f1_a": n_a / tot, "f1_b": n_b / tot}
    out = {}
    for k in a:
        if isinstance(a[k], (int, float)) and not (isinstance(a[k], float) and math.isnan(a[k])):
            out[k] = a[k] * w["f1_a"] + b.get(k, a[k]) * w["f1_b"]
        else:
            out[k] = a[k]
    return out


# ==========================================================================
# 训练循环
# ==========================================================================
@dataclass
class TrainConfig:
    region: str = "SW_Cascade"
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    tin: int = 7
    delta: int = 1
    stride: int = 3
    patience: int = 15
    seed: int = 42
    device: str = "cuda"
    out_dir: str = "outputs"
    log_every: int = 20
    max_train_samples: int | None = None
    max_val_samples: int | None = None


def _topo_pos_weight(bundle, split="train") -> torch.Tensor:
    """14 类多标签 Focal BCE 的正类权重（inverse sqrt 频率，仅训练期）。

    洪涝/暴雨占正样本 90%+，沙尘暴/高温近乎为空；用 `neg/pos` 的平方根
    在"压制长尾压倒"与"把空类推到全预测"之间取平衡。
    """
    y0, y1 = C.SPLITS[split]
    years = bundle.dates.year
    lo = int(np.nonzero(years >= y0)[0][0])
    hi = int(np.nonzero(years <= y1)[0][-1])
    lab = np.asarray(bundle.labels[:, lo:hi + 1, :]).reshape(-1, bundle.labels.shape[-1])
    pos = lab.sum(0).astype(np.float64)
    neg = lab.shape[0] - pos
    w = np.sqrt(neg / np.maximum(pos, 1.0))
    w = np.clip(w, 1.0, 20.0)
    return torch.tensor(w, dtype=torch.float32)


def _class_weight(bundle, split="train") -> torch.Tensor:
    """等级加权 CE 的类权重（inverse frequency，仅训练期）。"""
    y0, y1 = C.SPLITS[split]
    years = bundle.dates.year
    lo = int(np.nonzero(years >= y0)[0][0])
    hi = int(np.nonzero(years <= y1)[0][-1])
    lv = bundle.day_loss_level[:, lo:hi + 1].reshape(-1)
    cnt = np.bincount(lv[lv >= 0], minlength=len(C.LOSS_LEVEL_NAMES)).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (cnt * len(cnt))
    return torch.tensor(w, dtype=torch.float32)


def train(model_builder, tcfg: TrainConfig, loss_cfg: LossConfig | None = None,
          bundle_loader=None, verbose: bool = True) -> dict:
    from src.engine.data import (WindowDataset, build_sample, collate, load_region)

    loss_cfg = loss_cfg or LossConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = tcfg.device if torch.cuda.is_available() or tcfg.device == "cpu" else "cpu"
    bundle_loader = bundle_loader or load_region
    bundle = bundle_loader(tcfg.region)

    tr = WindowDataset(bundle, "train", tcfg.tin, tcfg.delta, tcfg.stride)
    va = WindowDataset(bundle, "val", tcfg.tin, tcfg.delta, max(tcfg.stride, 2))
    rng = np.random.default_rng(tcfg.seed)
    if tcfg.max_train_samples and len(tr) > tcfg.max_train_samples:
        tr.days = np.sort(rng.choice(tr.days, tcfg.max_train_samples, replace=False))
    if tcfg.max_val_samples and len(va) > tcfg.max_val_samples:
        va.days = np.sort(rng.choice(va.days, tcfg.max_val_samples, replace=False))

    model = model_builder().to(device)
    cw = _class_weight(bundle).to(device)
    pos_w = _topo_pos_weight(bundle).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.epochs, 1))

    out_dir = Path(tcfg.out_dir) / f"{tcfg.region}_{getattr(model, 'ablation_name', 'PT_GNO')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_loss": float("inf"), "epoch": -1, "metrics": {}}
    bad = 0

    def run_epoch(ds, train_mode: bool, epoch: int):
        order = np.arange(len(ds))
        if train_mode:
            np.random.default_rng(tcfg.seed + epoch).shuffle(order)
        agg, n_batches = None, 0
        model.train(train_mode)
        for s in range(0, len(order), tcfg.batch_size):
            idx = order[s: s + tcfg.batch_size]
            samples = [build_sample(bundle, int(ds[int(i)]), tcfg.tin, tcfg.delta) for i in idx]
            batch = collate(samples, bundle, device)
            batch["pos_weight"] = pos_w
            with torch.set_grad_enabled(train_mode):
                out = model(batch)
                loss, parts = free_energy_loss(out, batch, loss_cfg, epoch, class_weight=cw)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            parts["loss"] = float(loss.detach())
            m = compute_metrics(out["topo_prob"], batch["topo"], out["level_logits"],
                                batch["level"], batch["level_mask"], out["cont"],
                                batch["cont"], batch["cont_mask"])
            m.update(parts)
            if agg is None:
                agg = {}
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
            n_batches += 1
        # 逐批指标取**有限值**的均值：损失类指标在部分批上无正样本属正常，
        # 直接平均会把整轮结果污染成 NaN。
        out = {}
        for k, vals in (agg or {}).items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            out[k] = float(arr.mean()) if arr.size else float("nan")
        return out

    t0 = time.time()
    for ep in range(1, tcfg.epochs + 1):
        tr_m = run_epoch(tr, True, ep)
        va_m = run_epoch(va, False, ep)
        sched.step()
        row = {"epoch": ep, "train": tr_m, "val": va_m, "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        if verbose:
            print(f"[{tcfg.region}] ep{ep:03d} train_loss={tr_m['loss']:.4f} "
                  f"val_loss={va_m['loss']:.4f} val_macroF1={va_m['macro_f1']:.4f} "
                  f"val_auprc={va_m.get('auprc', float('nan')):.4f}", flush=True)
        if va_m["loss"] < best["val_loss"] - 1e-5:
            best = {"val_loss": va_m["loss"], "epoch": ep, "metrics": va_m}
            torch.save({"model": model.state_dict(),
                        "cfg": getattr(model, "cfg", None).__dict__
                        if getattr(model, "cfg", None) is not None else None},
                       out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                if verbose:
                    print(f"[{tcfg.region}] early stop @ ep{ep}", flush=True)
                break

    result = {"region": tcfg.region, "config": asdict(tcfg),
              "loss_config": asdict(loss_cfg), "best": best, "history": history,
              "seconds": round(time.time() - t0, 1),
              "model_params": int(sum(p.numel() for p in model.parameters())),
              "n_train_days": len(tr), "n_val_days": len(va),
              "text": bundle.text_missing_reason}
    (out_dir / "train_log.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


__all__ = ["LossConfig", "TrainConfig", "free_energy_loss", "compute_metrics",
           "focal_bce", "weighted_ce", "masked_huber", "train", "merge_metrics"]