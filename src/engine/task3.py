"""Task 3：初报 → 续报 → 终报 的连续时间数据同化评测。

物理对应 `CLAUDE.md` §1.2 的最后一项：

    dU_i(t) = ... + G_assim(U_i(t)) dM_i(t)

`M_i(t)` 是"观测到达"的计数过程。每收到一份报告，状态被跳变修正一次：

- **Stage 0**：仅灾前气象，`M(t) ≡ 0`（纯预报）；
- **Stage 1**：同化 `初报` 文本与状态；
- **Stage 2**：在同化 `初报` 的基础上再同化 `续报`。

评测指标：`log-MAE`、`RMSE`、`R²` 与相对 Stage 0 的误差衰减率
`ERR% = (MAE_0 - MAE_k) / MAE_0 × 100`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data import common as C
from src.engine.data import RegionBundle, build_sample, to_device


@dataclass
class AssimSample:
    node: int
    target_day: int              # 反事实预测日（终报当日）
    report_days: list[int]       # 各阶段报告日（已映射到日索引）
    report_text_idx: list[int]   # 各阶段报告对应的文本表行（-1 表示无）
    cont: np.ndarray             # (6,) log10(1+y) 目标
    level: int
    valid: bool
    valid_cols: np.ndarray | None = None   # 子集中真实存在的目标列索引


def build_assim_samples(bundle: RegionBundle, golden: pd.DataFrame,
                        tin: int = 7) -> list[AssimSample]:
    """把金标准事件链转成同化评测样本。

    只保留：节点存在于本区域、终报有可信连续损失、且报告日早于目标日的链。
    """
    node_of = dict(zip(bundle.nodes["cnty_id"], bundle.nodes["node_idx"]))
    day_of = pd.Series(np.arange(bundle.n_days), index=bundle.dates)
    cont_cols = list(C.CONTINUOUS_LOSS_TARGETS)

    out: list[AssimSample] = []
    g = golden.copy()
    g["stage_date"] = pd.to_datetime(g["stage_date"])
    for chain, sub in g.groupby("event_chain_id", sort=False):
        sub = sub.sort_values("stage_order")
        cid = int(sub["cnty_id"].iloc[0])
        if cid not in node_of:
            continue
        node = int(node_of[cid])
        final = sub[sub["report_status"] == "终报"]
        if final.empty:
            continue
        fin = final.iloc[-1]
        d_fin = day_of.get(pd.Timestamp(fin["stage_date"]).normalize())
        if d_fin is None or pd.isna(d_fin):
            continue
        d_fin = int(d_fin)
        if d_fin < tin:
            continue
        # `assimilation_golden_subset.parquet` 只有 `economic_loss`，
        # 其余 5 个连续目标在该子集里不存在。缺列填 0 会把 log10(1+y) 变成
        # "真值恒 0、模型预测 ~1" 的固定误差，使 log-MAE 虚高约 10 倍。
        # 因此：只对子集真实存在的目标建监督，并记录列索引供指标切片。
        present = [i for i, c in enumerate(cont_cols) if c in fin.index]
        cont = np.zeros(len(cont_cols), dtype=np.float32)
        for i in present:
            c = cont_cols[i]
            cont[i] = np.log10(1.0 + max(float(fin.get(c) or 0.0), 0.0))
        if not np.isfinite(cont).all() or float(fin.get("economic_loss", 0.0) or 0.0) <= 0:
            continue                       # 无损失目标的链不参与回归评测

        days, texts = [], []
        for status in C.REPORT_STAGES:      # 初报 -> 续报
            s = sub[sub["report_status"] == status]
            if s.empty:
                continue
            d = day_of.get(pd.Timestamp(s.iloc[0]["stage_date"]).normalize())
            if d is None or pd.isna(d):
                continue
            days.append(int(d))
            ti = bundle.text_index[node, int(d)]
            texts.append(int(ti))
        if not days:
            continue
        out.append(AssimSample(node=node, target_day=d_fin,
                               report_days=days, report_text_idx=texts,
                               cont=cont, level=int(fin.get("loss_level_id", 0) or 0),
                               valid=True,
                               valid_cols=np.asarray(present, dtype=np.int64)))
    return out


def _stage_mask_and_text(bundle: RegionBundle, s: AssimSample, stage: int,
                         tin: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 (T,N) 同化掩码与 (T,N,D) 注入文本。

    stage=0 无同化；stage=1 只注入初报；stage=2 注入初报+续报。
    报告日若早于窗口起点，则在窗口首日注入（状态已携带）。
    """
    T = tin
    N = bundle.n_nodes
    D = bundle.text_dim
    mask = np.zeros((T, N), dtype=np.float32)
    text = np.zeros((T, N, D), dtype=np.float32)
    if stage == 0:
        return mask, text
    # 输入窗口 = [target_day-tin, target_day-1]（`build_sample` 读 day=target_day-1，
    # 预测目标为 day+1 = target_day），因此窗口首日索引是 target_day - tin。
    t0 = s.target_day - tin
    n_use = min(stage, len(s.report_days))
    for k in range(n_use):
        d = s.report_days[k]
        step = int(np.clip(d - t0, 0, T - 1))
        mask[step, s.node] = 1.0
        ti = s.report_text_idx[k]
        if ti >= 0:
            text[step, s.node] = bundle.text_post[ti].numpy()
    return mask, text


def evaluate_stage(model, bundle: RegionBundle, samples: list[AssimSample],
                   stage: int, tin: int = 7, batch_size: int = 8,
                   device: str = "cuda") -> dict:
    """在给定同化阶段上评估连续损失与等级预测。"""
    model.eval()
    preds, trues, levels, level_true = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            # build_sample(day) 的预测目标是 day+1，故读 day=target_day-1
            samples_np = [build_sample(bundle, s.target_day - 1, tin, 1) for s in chunk]
            from src.engine.data import collate
            batch = collate(samples_np, bundle, device)
            T = tin
            m = np.zeros((len(chunk), T, bundle.n_nodes), dtype=np.float32)
            txt = np.zeros((len(chunk), T, bundle.n_nodes, bundle.text_dim), dtype=np.float32)
            for b, s in enumerate(chunk):
                mb, tb = _stage_mask_and_text(bundle, s, stage, tin)
                m[b], txt[b] = mb, tb
            batch["assim_mask"] = torch.from_numpy(m).to(device)
            # 逐时刻注入的同化文本必须是 `(B,T,N,D)`，键名为 `assim_text`；
            # `h_post` 是窗口锚点的 `(B,N,D)` 文本，两者维度不同不可混用。
            batch["assim_text"] = torch.from_numpy(txt).to(device)
            out = model(batch)
            c = out["cont"].detach().float().cpu().numpy()
            # 只取目标节点那一行
            for b, s in enumerate(chunk):
                preds.append(c[b, s.node])
                trues.append(s.cont)
                levels.append(s.level)
                level_true.append(batch["level"][b, s.node].item())

    preds = np.stack(preds)
    trues = np.stack(trues)
    levels = np.asarray(levels)
    # 只在实际有监督的目标列上算误差（子集缺列若计入，误差纯属人造）
    cols = next((s.valid_cols for s in samples if s.valid_cols is not None), None)
    if cols is not None and len(cols):
        preds, trues = preds[:, cols], trues[:, cols]
    err = preds - trues
    ss_res = (err ** 2).sum(0)
    ss_tot = ((trues - trues.mean(0)) ** 2).sum(0)
    valid = ss_tot > 1e-6
    return {
        "stage": stage,
        "n": int(len(preds)),
        "n_targets": int(trues.shape[1]),
        "log_mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float((1 - ss_res[valid] / ss_tot[valid]).mean()) if valid.any() else float("nan"),
        "per_target_log_mae": [float(np.abs(err[:, j]).mean()) for j in range(err.shape[1])],
        "level_acc": float((levels == level_true).mean()),
    }


def evaluate_assimilation(model, bundle, golden, tin: int = 7,
                          stages=(0, 1, 2), device: str = "cuda") -> dict:
    samples = build_assim_samples(bundle, golden, tin=tin)
    if not samples:
        return {"n_samples": 0, "stages": []}
    res = [evaluate_stage(model, bundle, samples, st, tin=tin, device=device)
           for st in stages]
    base = res[0]["log_mae"] if res else float("nan")
    for r in res:
        r["err_reduction_pct"] = (float((base - r["log_mae"]) / base * 100.0)
                                  if base and np.isfinite(base) and base > 0 else float("nan"))
    return {"n_samples": len(samples), "stages": res}


__all__ = ["AssimSample", "build_assim_samples", "evaluate_stage",
           "evaluate_assimilation"]