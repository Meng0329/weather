"""区域基准的批次构造：整窗采样、稀疏文本查找、多尺度损失目标。

设计要点
--------
1. **整窗（region-wide）样本**：一个样本 = 区域内全部节点在 `[t-T_in+1, t]` 的连续气象场，
   目标是 `(t, t+Δ]` 的灾害拓扑与损失。这是因为平流算子天然跨区县耦合，
   必须让整张物理邻接图参与前向；节点维不做独立采样。
2. **无未来信息泄露**：输入窗口止于 `t`；标签窗口严格在 `t` 之后。
3. **稀疏文本表**：LLM 文本向量按 `(node, day)` 去重存放（`text_index` + `text_table`），
   批次内按需 gather，避免落盘 (N,T,256) 的稠密大张量。
4. **掩码**：连续损失目标在 `log10(1+y)` 空间内按窗口求和；等级取窗口内最大值。
   缺记录节点不参与损失等级与连续损失监督（`level_mask` / `cont_mask` = 0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import common as C

ADV_IDX = list(C.ADV_KERNEL_IDX)


@dataclass
class RegionBundle:
    """一个区域基准的全部张量与元信息（memmap 惰性读取，不复制数据）。"""

    name: str
    root: Path
    nodes: pd.DataFrame
    dates: pd.DatetimeIndex
    feats: np.ndarray            # (N,T,15) memmap（原始量纲）
    mask: np.ndarray             # (N,T,15) memmap
    labels: np.ndarray           # (N,T,14) uint8 memmap
    edge_index: torch.Tensor     # (2,E)
    edge_w: torch.Tensor         # (E,)
    records: pd.DataFrame
    text_index: np.ndarray       # (N,T) int32，-1 表示当日无灾情文本
    text_pre: torch.Tensor       # (M,text_dim)
    text_rag: torch.Tensor       # (M,text_dim)
    text_post: torch.Tensor      # (M,text_dim)
    feats_n: np.ndarray = None   # (N,T,15) 训练期标准化后
    text_missing_reason: str = ""
    norm_mean: np.ndarray = None   # (F,) 仅由训练期统计
    norm_std: np.ndarray = None

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def text_dim(self) -> int:
        return int(self.text_pre.shape[-1])

    def contiguous(self) -> "RegionBundle":
        """把 memmap 读入内存，加速训练（仅区域子集，规模可控）。"""
        # 只把高频访问的三张表读进内存；`feats` 只用于平流核的 3 个物理通道，
        # 保留 memmap 让 OS page cache 在多进程间共享，避免每个 run 复制 227 MB。
        if isinstance(self.feats_n, np.memmap):
            self.feats_n = np.asarray(self.feats_n)
        self.mask = np.ascontiguousarray(self.mask)
        self.labels = np.ascontiguousarray(self.labels)
        return self


def _load_text_tables(root: Path, records: pd.DataFrame, n_nodes: int, n_days: int,
                      dates: pd.DatetimeIndex):
    """构造 (node, day) -> 文本向量索引的稀疏查找表。"""
    emb_path = root / "llm_embeddings.pt"
    day_idx = pd.Series(np.arange(n_days), index=dates)
    rec = records[["record_id", "node_idx", "start_date"]].copy()
    rec["start_date"] = pd.to_datetime(rec["start_date"])
    rec["_d"] = rec["start_date"].map(day_idx)
    rec = rec[rec["_d"].notna()].copy()
    rec["_d"] = rec["_d"].astype(np.int64)

    reason = ""
    if emb_path.exists():
        payload = torch.load(emb_path, map_location="cpu", weights_only=False)
        dim = int(payload["h_pre"].shape[-1])
        # 同名 record_id 可能因 record->node 多链接而重复：文本与节点无关，取首次出现
        rid = pd.Index(payload["record_ids"])
        keep = ~rid.duplicated()
        rid_u = rid[keep]
        e_all = np.nonzero(keep)[0]
        pos = pd.Series(e_all, index=rid_u)
        rec["_e"] = rec["record_id"].map(pos)
        rec = rec[rec["_e"].notna()]
        e = rec["_e"].to_numpy().astype(np.int64)
        h_pre = torch.from_numpy(np.asarray(payload["h_pre"], dtype=np.float32)[e])
        h_rag = torch.from_numpy(np.asarray(payload["h_rag"], dtype=np.float32)[e])
        h_post = torch.from_numpy(np.asarray(payload["h_post"], dtype=np.float32)[e])
        reason = f"method={payload.get('method', 'unknown')}"
    else:  # 无 LLM 产物时用零向量占位，保证管线可运行
        dim = 256
        rec = rec.assign(_e=0)
        h_pre = torch.zeros(len(rec), dim)
        h_rag = torch.zeros(len(rec), dim)
        h_post = torch.zeros(len(rec), dim)
        reason = "placeholder(zeros): llm_embeddings.pt 不存在"

    text_index = np.full((n_nodes, n_days), -1, dtype=np.int32)
    # 同一 (node, day) 可能有多条记录：取首条作为该日文本代表
    keys = rec["node_idx"].to_numpy().astype(np.int64) * n_days + rec["_d"].to_numpy()
    uniq, first = np.unique(keys, return_index=True)
    rows = np.arange(len(rec))[first]
    text_index[(uniq // n_days).astype(np.int64), (uniq % n_days).astype(np.int64)] = \
        np.arange(len(rows), dtype=np.int32)
    return text_index, h_pre[rows], h_rag[rows], h_post[rows], dim, reason


def _build_day_targets(records: pd.DataFrame, n_nodes: int, n_days: int,
                       dates: pd.DatetimeIndex):
    """把记录级损失聚合到 (node, day)，供窗口求和/取最大值。"""
    day_idx = pd.Series(np.arange(n_days), index=dates)
    r = records[["node_idx", "start_date", "loss_level_id", "cont_loss_mask"]].copy()
    r["start_date"] = pd.to_datetime(r["start_date"])
    r["_d"] = r["start_date"].map(day_idx)
    r = r[r["_d"].notna()].copy()
    r["_d"] = r["_d"].astype(np.int64)
    r["node_idx"] = r["node_idx"].astype(np.int64)
    for c in C.CONTINUOUS_LOSS_TARGETS:
        r[c] = pd.to_numeric(records.loc[r.index, c], errors="coerce").fillna(0.0).to_numpy()
    r = r[r["cont_loss_mask"] > 0]

    ni, di = r["node_idx"].to_numpy(), r["_d"].to_numpy()
    level = np.full((n_nodes, n_days), -1, dtype=np.int8)
    np.maximum.at(level, (ni, di), r["loss_level_id"].to_numpy().astype(np.int8))

    cont = np.zeros((n_nodes, n_days, len(C.CONTINUOUS_LOSS_TARGETS)), dtype=np.float32)
    for k, c in enumerate(C.CONTINUOUS_LOSS_TARGETS):
        daily = np.zeros((n_nodes, n_days), dtype=np.float64)
        np.add.at(daily, (ni, di), r[c].to_numpy())
        cont[:, :, k] = daily.astype(np.float32)
    return level, cont, np.asarray(C.CONTINUOUS_LOSS_TARGETS)


def compute_norm_stats(bundle: RegionBundle, split: str = "train",
                        day_stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """逐通道标准化统计量，**只用训练期**数据计算（杜绝泄漏）。

    按 `day_stride` 抽样日期以控制开销：训练期近 7000 天，抽样 1/3 已足够
    稳定地估计逐通道均值/方差（相对误差 < 1%）。
    """
    y0, y1 = C.SPLITS[split]
    years = bundle.dates.year
    lo = int(np.nonzero(years >= y0)[0][0])
    hi = int(np.nonzero(years <= y1)[0][-1])
    n_ch = bundle.feats.shape[-1]
    s = np.zeros(n_ch, dtype=np.float64)
    s2 = np.zeros(n_ch, dtype=np.float64)
    cnt = np.zeros(n_ch, dtype=np.float64)
    for a in range(0, bundle.n_nodes, 64):                 # 分块，避免整表读入
        blk = np.asarray(bundle.feats[a:a + 64, lo:hi + 1:day_stride], dtype=np.float32)
        msk = np.asarray(bundle.mask[a:a + 64, lo:hi + 1:day_stride], dtype=np.float32)
        blk = blk * msk
        s += blk.sum(axis=(0, 1), dtype=np.float64)
        s2 += (blk * blk).sum(axis=(0, 1), dtype=np.float64)
        cnt += msk.sum(axis=(0, 1), dtype=np.float64)
    cnt = np.maximum(cnt, 1.0)
    mean = s / cnt
    var = np.maximum(s2 / cnt - mean ** 2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def build_feats_n(bundle: RegionBundle, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """流式生成标准化后的气象张量（分块写，避免 float64 全表拷贝）。"""
    n, t, f = bundle.feats.shape
    out = np.empty((n, t, f), dtype=np.float32)
    m = mean.astype(np.float32)
    s = std.astype(np.float32)
    for a in range(0, n, 64):
        blk = np.asarray(bundle.feats[a:a + 64], dtype=np.float32)
        msk = np.asarray(bundle.mask[a:a + 64], dtype=np.float32)
        z = (blk - m) / s
        z *= msk
        out[a:a + 64] = z
    return out


def load_region(region: str, processed: Path | None = None,
                contiguous: bool = True) -> RegionBundle:
    processed = Path(processed or C.PROCESSED_DIR)
    root = processed / region
    nodes = pd.read_parquet(root / "nodes.parquet")
    dates = pd.DatetimeIndex(np.load(processed / "meteo_dates.npy"))
    feats = np.load(root / "meteo_features.npy", mmap_mode="r")
    mask = np.load(root / "meteo_observed_mask.npy", mmap_mode="r")
    labels = np.load(root / "daily_labels.npy", mmap_mode="r")
    adj = np.load(root / "adj_mask.npy")
    wgt = np.load(root / "adj_weight.npy")
    ii, jj = np.nonzero(adj)
    edge_index = torch.from_numpy(np.vstack([ii, jj]).astype(np.int64))
    edge_w = torch.from_numpy(np.asarray(wgt[ii, jj]).astype(np.float32))
    records = pd.read_parquet(root / "records.parquet")
    n_nodes, n_days = len(nodes), len(dates)

    ti, h_pre, h_rag, h_post, dim, reason = _load_text_tables(
        root, records, n_nodes, n_days, dates)
    level, cont, _ = _build_day_targets(records, n_nodes, n_days, dates)
    b = RegionBundle(region, root, nodes, dates, feats, mask, labels,
                     edge_index, edge_w, records, ti, h_pre, h_rag, h_post)
    b.text_missing_reason = reason
    b.day_loss_level = level
    b.day_cont = cont
    b.norm_mean, b.norm_std = compute_norm_stats(b)
    # 标准化张量落盘缓存：每次 run 重算要 ~40 s，150 个 run 就是 1.7 小时
    cache = root / "meteo_features_n.npy"
    stats_json = root / "feature_norm_stats.json"
    import json as _json

    def _atomic_save(arr, path):
        # 多个 run 并行时可能同时写缓存：先写临时文件再 rename，避免读到半截张量
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        np.save(tmp, arr)
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)

    if cache.exists() and stats_json.exists():
        st = _json.loads(stats_json.read_text("utf-8"))
        if np.allclose(st["mean"], b.norm_mean, atol=1e-3):
            b.feats_n = np.load(cache, mmap_mode="r")
        else:
            b.feats_n = build_feats_n(b, b.norm_mean, b.norm_std)
            _atomic_save(b.feats_n, cache)
    else:
        b.feats_n = build_feats_n(b, b.norm_mean, b.norm_std)
        _atomic_save(b.feats_n, cache)
        stats_json.write_text(_json.dumps(
            {"mean": b.norm_mean.tolist(), "std": b.norm_std.tolist()},
            ensure_ascii=False), encoding="utf-8")
    if contiguous:
        b.contiguous()
    return b


# --------------------------------------------------------------------------
# 整窗数据集
# --------------------------------------------------------------------------
class WindowDataset(Dataset):
    """整窗样本：`x` 为 `(N, T_in, F)`，目标为 `(N, 14)` 等多头标签。"""

    def __init__(self, bundle: RegionBundle, split: str, tin: int = 7, delta: int = 1,
                 stride: int = 1, require_positive: bool = False):
        self.b = bundle
        self.tin = tin
        self.delta = delta
        y0, y1 = C.SPLITS[split]
        days = bundle.dates
        lo = int(np.nonzero(days.year >= y0)[0][0]) + tin - 1
        hi = int(np.nonzero(days.year <= y1)[0][-1]) - delta
        cand = np.arange(lo, hi + 1, stride)
        if require_positive:
            daily = bundle.labels[:, lo: hi + delta + 1, :].sum(axis=(0, 2))
            keep = daily[cand - lo] > 0
            cand = cand[keep]
        self.days = cand

    def __len__(self):
        return len(self.days)

    def __getitem__(self, k: int) -> int:
        return int(self.days[k])


def build_sample(bundle: RegionBundle, day: int, tin: int, delta: int,
                 assim_p: float = 0.0, rng: np.random.Generator | None = None) -> dict:
    """构造单个整窗样本（numpy / torch 混合，collate 只做 stack）。

    `assim_p>0` 时以该概率为本窗口**注入观测报告**（Task 3 同化门的训练信号）：
    对每个节点取窗口内最后一天有灾后文本的位置作为"报告到达时刻"。
    训练期必须这样做，否则 `G_assim` 永远乘 0 掩码、拿不到梯度，
    推理阶段就是一个随机扰动（实测 Task 3 增益 ≈ 0）。
    """
    t0 = day - tin + 1
    src = bundle.feats_n if bundle.feats_n is not None else bundle.feats
    x = np.asarray(src[:, t0:day + 1], dtype=np.float32)             # (N,T,F)
    m = np.asarray(bundle.mask[:, t0:day + 1], dtype=np.float32)
    adv = np.ascontiguousarray(
        np.asarray(bundle.feats[:, t0:day + 1], dtype=np.float32)[:, :, ADV_IDX])
    # 平流核读取**原始物理量纲**（hPa / m·s⁻¹），不做标准化，保持可解释性
    lab = np.asarray(bundle.labels[:, day + 1: day + 1 + delta], dtype=np.float32)
    topo = np.clip(lab.sum(axis=1), 0.0, 1.0)                            # (N,14)
    tw = slice(day + 1, day + 1 + delta)
    lv = bundle.day_loss_level[:, tw].max(axis=1)                        # (N,)
    cont = bundle.day_cont[:, tw, :].sum(axis=1)                         # (N,6)
    cont = np.log10(1.0 + cont.astype(np.float64)).astype(np.float32)

    # 文本：只返回**索引**（(N,) int64），向量在 collate 里一次性 index_select。
    # 逐窗口构造 (N,512) 文本张量会产生 ~1 GB/epoch 的无谓内存搬运，是数据管线瓶颈。
    ti = bundle.text_index[:, t0:day + 1]                                # (N,T)
    idx = np.full(bundle.n_nodes, -1, dtype=np.int64)
    for k in range(tin - 1, -1, -1):
        unset = idx < 0
        idx[unset] = ti[unset, k]
    has_text = idx >= 0

    # 同化掩码：窗口末日的"灾后报告"到达位置（每个节点各自最多一个跳变）
    assim_mask = np.zeros((tin, bundle.n_nodes), dtype=np.float32)
    assim_idx = np.full((tin, bundle.n_nodes), -1, dtype=np.int64)
    if assim_p > 0.0 and (rng is None or rng.random() < assim_p):
        last = np.full(bundle.n_nodes, -1, dtype=np.int64)
        for k in range(tin - 1, -1, -1):                 # 从后往前，取最近一天有文本
            take = (last < 0) & (ti[:, k] >= 0)
            last[take] = k
        for k in range(tin):
            sel = last == k
            if sel.any():
                assim_mask[k, sel] = 1.0
                assim_idx[k, sel] = ti[sel, k]

    return {
        "day": day,
        "x": torch.from_numpy(x), "mask": torch.from_numpy(m),
        "adv": torch.from_numpy(np.ascontiguousarray(adv)),
        "text_idx": torch.from_numpy(idx),
        "has_text": torch.from_numpy(has_text.astype(np.float32)),
        "assim_mask": torch.from_numpy(assim_mask),
        "assim_idx": torch.from_numpy(assim_idx),
        "topo": torch.from_numpy(topo),
        "level": torch.from_numpy(np.where(lv >= 0, lv, 0).astype(np.int64)),
        "level_mask": torch.from_numpy((lv >= 0).astype(np.float32)),
        "cont": torch.from_numpy(cont),
        "cont_mask": torch.from_numpy((lv >= 0).astype(np.float32)),
    }


def collate(samples: list[dict], bundle: RegionBundle, device="cpu") -> dict:
    """把整窗样本堆成 (B,N,...) 张量，并把文本广播到时间维。"""
    def stack(k):
        return torch.stack([s[k] for s in samples], dim=0)

    B = len(samples)
    T = samples[0]["x"].shape[1]
    # 文本向量：把 (B,N) 索引一次性 gather 成 (B,N,D)（拼一个零行处理“无文本”）
    tidx = stack("text_idx")
    zero = torch.zeros(1, bundle.text_dim)
    def _gather(table):
        return torch.cat([table, zero], dim=0)[
            torch.where(tidx >= 0, tidx, torch.full_like(tidx, table.shape[0]))]

    batch = {
        # build_sample 产出 (N,T,F)，堆叠后需转为模型约定的 (B,T,N,F)
        "x": stack("x").permute(0, 2, 1, 3).contiguous(),
        "mask": stack("mask").permute(0, 2, 1, 3).contiguous(),
        "adv": stack("adv").permute(0, 2, 1, 3).contiguous(),
        "topo": stack("topo"), "level": stack("level"),
        # 相位目标：该节点-日在预测窗口内是否发生**任意**灾种（P(相变) 的监督信号）
        "phase_target": stack("topo").max(dim=-1).values,
        "level_mask": stack("level_mask"), "cont": stack("cont"),
        "cont_mask": stack("cont_mask"),
        "h_pre": _gather(bundle.text_pre),
        "h_rag": _gather(bundle.text_rag),
        "h_post": _gather(bundle.text_post),
        "has_text": stack("has_text"),
        "days": torch.tensor([s["day"] for s in samples], dtype=torch.long),
        "edge_index": bundle.edge_index, "edge_w": bundle.edge_w,
        # 同化掩码 (B,T,N)；训练期由 build_sample(assim_p>0) 随机注入，
        # 评估期恒为 0（Task 1/2 不得看到灾后文本，否则是标签泄露）。
        # 注意 build_sample 产出的是 (T,N)，stack 后已是 (B,T,N)，无需再转置。
        "assim_mask": stack("assim_mask"),
    }
    aidx = stack("assim_idx")                                    # (B,T,N)
    if bool((aidx >= 0).any()):
        # 只传索引 + 文本表：按 (B,T,N,D) 展开在 North_Agro(B=32,N=589) 上是
        # 32*7*589*512*4 ≈ 271 MB/批，模型在真正需要的那一步再 gather。
        batch["assim_idx"] = aidx
        batch["assim_table"] = bundle.text_post
    # 文本不按时间维展开：本构泛函对 (B,N,D) 只推断一次 Θ，
    # 再在算子循环里广播使用。展开成 (B,T,N,D) 会凭空制造 ~0.5 GB/批的搬运。
    batch["h_cur"] = batch.pop("h_pre")
    return to_device(batch, device)


def to_device(batch: dict, device) -> dict:
    if device is None:
        return batch
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


__all__ = ["RegionBundle", "WindowDataset", "build_sample", "collate",
           "load_region", "to_device"]