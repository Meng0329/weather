#!/usr/bin/env python3
"""Phase 1-B：三大区域基准数据集 + 空间邻接图 + Task 3 金标准子集。

前置：先运行 `clean_and_align.py`。

产物（默认 `datasets/processed/`）：
  <REGION>/nodes.parquet                区域节点字典（node_idx 0..N-1 -> 全局 cnty_id）
  <REGION>/adj_mask.npy                 (N, N) bool  有向平流图
  <REGION>/adj_weight.npy               (N, N) float32 边耦合强度（相关系数）
  <REGION>/meteo_features.npy           (N, T, 16) float32
  <REGION>/meteo_observed_mask.npy      (N, T, 16) bool
  <REGION>/records.parquet              记录级样本
  <REGION>/events.parquet               事件级样本（cnty_id × start_date）
  <REGION>/daily_labels.npy             (N, T, 14) uint8，生命期展开
  <REGION>/split_index.json
  assimilation_golden_subset.parquet     多阶段报告链（全局）
  benchmark_report.json

运行：
  python3 -I src/data/build_benchmarks.py
  python3 -I src/data/build_benchmarks.py --skip-adjacency   # 复用已有邻接
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import common as C  # noqa: E402

log = logging.getLogger("build_benchmarks")

LOSS_TARGET_COLS = [
    "economic_loss", "affected_population", "collapsed_houses", "damaged_houses",
    "affected_area", "disaster_area", "total_loss_area", "grain_loss",
    "deaths", "missing", "injuries", "relocated",
]


# ==========================================================================
# 工具
# ==========================================================================
def _jsonable(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, pd.Timestamp): return o.isoformat()
    raise TypeError(type(o))


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_jsonable)


def _split_tag(years: pd.Series) -> pd.Series:
    out = pd.Series("out_of_range", index=years.index, dtype=object)
    for name, (y0, y1) in C.SPLITS.items():
        out[(years >= y0) & (years <= y1)] = name
    return out


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """按行 z-score；NaN 先忽略，标准化后 NaN 置 0。"""
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    cnt = valid.sum(axis=1, keepdims=True)
    safe_cnt = np.maximum(cnt, 1)
    mean = np.where(valid, x, 0.0).sum(axis=1, keepdims=True) / safe_cnt
    dev = np.where(valid, x - mean, 0.0)
    std = np.sqrt((dev * dev).sum(axis=1, keepdims=True) / safe_cnt)
    std[std == 0] = 1.0
    z = dev / std
    z[~valid] = 0.0
    return z.astype(np.float32)


# ==========================================================================
# 空间邻接
# ==========================================================================
def build_adjacency(
    stations: pd.DataFrame,
    delta_prs: np.ndarray,
    tem_avg: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """同地市连通 + 同省/邻省物理 Top-K。返回 (adj, weight, stats)。"""
    n = len(stations)
    prov = stations["Province"].to_numpy()
    city_key = pd.factorize(
        stations["Province"].astype(str) + "|" + stations["City"].astype(str)
    )[0]

    adj = np.zeros((n, n), dtype=bool)
    wgt = np.zeros((n, n), dtype=np.float32)

    # ---- 1. 同地市完全连通（双向，权重 1.0）----
    city_groups = pd.Series(np.arange(n)).groupby(city_key).indices
    same_city_edges = 0
    for _, idx in city_groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        block = np.ones((len(idx), len(idx)), dtype=bool)
        np.fill_diagonal(block, False)
        adj[np.ix_(idx, idx)] = block
        wgt[np.ix_(idx, idx)] = np.maximum(wgt[np.ix_(idx, idx)], block.astype(np.float32))
        same_city_edges += int(block.sum())

    # ---- 2. 物理近邻 ----
    y0, y1 = C.SPLITS["train"]
    tr = np.nonzero((dates.year >= y0) & (dates.year <= y1))[0]
    n_tr = len(tr)
    zp = _zscore_rows(delta_prs[:, tr])
    zt = _zscore_rows(tem_avg[:, tr])

    score = np.full((n, n), -np.inf, dtype=np.float32)
    prov_nodes = {p: np.nonzero(prov == p)[0] for p in np.unique(prov)}
    seen: set[tuple[str, str]] = set()
    for pa, ia in prov_nodes.items():
        if len(ia) == 0:
            continue
        allowed = [pa] + [p for p in C.PROVINCE_ADJACENCY.get(pa, []) if p in prov_nodes]
        for pb in allowed:
            if pb not in prov_nodes:
                continue
            key = tuple(sorted((pa, pb)))
            if key in seen:
                continue
            seen.add(key)
            ib = prov_nodes[pb]
            if len(ib) == 0:
                continue
            block = (
                C.ADV_WEIGHT_PRS * (zp[ia] @ zp[ib].T)
                + C.ADV_WEIGHT_TEM * (zt[ia] @ zt[ib].T)
            ) / max(n_tr, 1)
            score[np.ix_(ia, ib)] = block
            if pa != pb:
                score[np.ix_(ib, ia)] = block.T
            else:
                np.fill_diagonal(score[np.ix_(ia, ib)], -np.inf)

    # ---- 3. 每节点取 Top-K 正相关出边 ----
    physical_edges = 0
    k = C.PHYSICAL_NEIGHBOR_K
    for i in range(n):
        row = score[i]
        cand = np.nonzero(np.isfinite(row) & (row > 0))[0]
        if len(cand) == 0:
            continue
        order = cand[np.argsort(row[cand])[::-1][:k]]
        for j in order:
            if i == j:
                continue
            if not adj[i, j]:
                physical_edges += 1
            adj[i, j] = True
            wgt[i, j] = max(wgt[i, j], float(row[j]))

    np.fill_diagonal(adj, False)
    np.fill_diagonal(wgt, 0.0)

    out_deg = adj.sum(axis=1)
    cross_prov = 0
    for i, j in zip(*np.nonzero(adj)):
        if prov[i] != prov[j]:
            cross_prov += 1

    stats = {
        "n_nodes": int(n),
        "n_edges": int(adj.sum()),
        "same_city_edges": int(same_city_edges),
        "physical_edges_added": int(physical_edges),
        "mean_out_degree": round(float(out_deg.mean()), 3),
        "median_out_degree": float(np.median(out_deg)),
        "max_out_degree": int(out_deg.max()),
        "isolated_nodes": int((out_deg == 0).sum()),
        "cross_province_edge_ratio": round(cross_prov / max(adj.sum(), 1), 4),
        "train_days_used": int(n_tr),
    }
    return adj, wgt, stats


# ==========================================================================
# 区域构建
# ==========================================================================
def build_region(
    name: str,
    cfg: dict,
    processed: Path,
    feats,
    mask,
    dates: pd.DatetimeIndex,
    stations: pd.DataFrame,
    disaster: pd.DataFrame,
    skip_adjacency: bool = False,
) -> dict:
    out = processed / name
    out.mkdir(parents=True, exist_ok=True)

    provinces = set(cfg["provinces"])
    sub_st = stations[stations["Province"].isin(provinces)].reset_index(drop=True)
    if len(sub_st) == 0:
        raise RuntimeError(f"{name}: 区域内无气象节点")

    global_ids = sub_st["cnty_id"].to_numpy()
    local_idx = {int(g): i for i, g in enumerate(global_ids)}

    nodes = sub_st.copy()
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    nodes.to_parquet(out / "nodes.parquet", index=False)

    # 气象张量切片
    f_sub = np.asarray(feats[global_ids])       # (N, T, 16)
    m_sub = np.asarray(mask[global_ids])
    np.save(out / "meteo_features.npy", f_sub)
    np.save(out / "meteo_observed_mask.npy", m_sub)

    # 邻接
    if skip_adjacency and (out / "adj_mask.npy").exists():
        adj = np.load(out / "adj_mask.npy")
        wgt = np.load(out / "adj_weight.npy")
        adj_stats = {"reused": True}
    else:
        delta = f_sub[:, :, C.FEATURE_NAMES.index("delta_prs_24h")]
        tem = f_sub[:, :, C.FEATURE_NAMES.index("TEM_Avg")]
        adj, wgt, adj_stats = build_adjacency(nodes, delta, tem, dates)
        np.save(out / "adj_mask.npy", adj)
        np.save(out / "adj_weight.npy", wgt)
        log.info(
            "  %-12s 邻接: %d 节点 / %d 边 / 平均出度 %.2f / 跨省边 %.1f%%",
            name, adj_stats["n_nodes"], adj_stats["n_edges"],
            adj_stats["mean_out_degree"], adj_stats["cross_province_edge_ratio"] * 100,
        )

    # ---- 灾情样本 ----
    rec = disaster[disaster["province"].isin(provinces)].copy()
    rec = rec[rec["cnty_id"] >= 0].copy()
    rec["node_idx"] = rec["cnty_id"].map(local_idx)
    n_before = len(rec)
    rec = rec[rec["node_idx"].notna()].copy()
    rec["node_idx"] = rec["node_idx"].astype(np.int64)
    n_dropped = n_before - len(rec)

    rec["split"] = _split_tag(rec["start_time"].dt.year)
    rec["type_idx"] = rec["disaster_type_14"].map(C.DISASTER_TYPE_TO_IDX)
    rec["start_date"] = rec["start_time"].dt.normalize()
    rec["end_date"] = rec["end_time"].dt.normalize()
    rec = rec[rec["split"] != "out_of_range"].copy()

    keep_cols = [
        "record_id", "node_idx", "cnty_id", "province", "city", "county",
        "disaster_type_14", "type_idx", "loss_level_id", "start_time", "end_time",
        "start_date", "end_date", "report_status", "report_time", "event_chain_id",
        "split", "align_level",
    ] + [c for c in LOSS_TARGET_COLS if c in rec.columns]
    records = rec[[c for c in keep_cols if c in rec.columns]].copy()
    records.to_parquet(out / "records.parquet", index=False)

    # ---- 事件级样本：同一节点同一天聚合 ----
    if len(records):
        agg = {"economic_loss": "sum"}
        for c in LOSS_TARGET_COLS:
            if c in records.columns and c != "economic_loss":
                agg[c] = "sum"
        agg["type_idx"] = lambda s: sorted(set(int(x) for x in s if pd.notna(x)))
        agg["record_id"] = "size"

        events = (
            records.groupby(["node_idx", "start_date"], as_index=False)
            .agg(agg)
            .rename(columns={"record_id": "n_records", "start_date": "date"})
        )
        events["split"] = _split_tag(events["date"].dt.year)
        events["n_types"] = events["type_idx"].apply(len)
        events["primary_type_idx"] = events["type_idx"].apply(
            lambda x: int(x[0]) if x else -1
        )
        events["type_idx"] = events["type_idx"].apply(
            lambda x: ",".join(str(i) for i in x)
        )
        events.to_parquet(out / "events.parquet", index=False)
    else:
        events = pd.DataFrame()
        events.to_parquet(out / "events.parquet", index=False)

    # ---- 生命期展开标签矩阵 (N, T, 14) ----
    labels = np.zeros((len(nodes), len(dates), C.NUM_DISASTER_TYPES), dtype=np.uint8)
    date_pos = {d.normalize(): i for i, d in enumerate(dates)}
    for node_i, sd, ed, ti in zip(
        records["node_idx"], records["start_date"], records["end_date"], records["type_idx"]
    ):
        if pd.isna(ti):
            continue
        a = date_pos.get(sd)
        b = date_pos.get(ed, a)
        if a is None:
            continue
        b = a if b is None else max(b, a)
        labels[int(node_i), a:b + 1, int(ti)] = 1
    np.save(out / "daily_labels.npy", labels)

    # ---- 统计 ----
    split_stats = {}
    for sp in ["train", "val", "test"]:
        r = records[records["split"] == sp]
        e = events[events["split"] == sp] if len(events) else events
        split_stats[sp] = {
            "n_records": int(len(r)),
            "n_events": int(len(e)),
            "n_nodes_with_events": int(r["node_idx"].nunique()),
            "class_counts": r["disaster_type_14"].value_counts().to_dict(),
            "label_positive_cells": int(labels[:, :, :].sum()) if sp == "train" else None,
        }
    # 去掉占位
    for sp in split_stats:
        split_stats[sp].pop("label_positive_cells", None)

    stats = {
        "region": name,
        "label": cfg["label"],
        "provinces": cfg["provinces"],
        "focus": cfg["focus"],
        "n_nodes": int(len(nodes)),
        "n_days": int(len(dates)),
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "n_feature_channels": int(f_sub.shape[2]),
        "disaster_records_dropped_no_node": int(n_dropped),
        "adjacency": adj_stats,
        "splits": split_stats,
        "positive_cells_total": int(labels.sum()),
    }
    _dump_json(out / "split_index.json", stats)
    return stats


# ==========================================================================
# Task 3：多阶段报告链
# ==========================================================================
def build_assimilation_subset(disaster: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """提取含 >=2 个不同报告阶段（初报/续报/终报）的事件链。"""
    d = disaster[disaster["report_status"].isin(C.REPORT_STAGES)].copy()
    d["stage_order"] = d["report_status"].map(C.REPORT_STAGE_ORDER)
    d["stage_date"] = d["report_time"].dt.normalize()

    g = d.groupby("event_chain_id")
    sizes = g.size()
    stage_counts = g["report_status"].nunique()
    has_all3 = g["report_status"].apply(lambda s: set(C.REPORT_STAGES) <= set(s.dropna()))
    multistage = stage_counts >= 2

    chains = pd.DataFrame({
        "n_reports": sizes,
        "n_stages": stage_counts,
        "has_all_3_stages": has_all3,
    }).reset_index()
    chains = chains.merge(
        d.groupby("event_chain_id").agg(
            disaster_type_14=("disaster_type_14", "first"),
            province=("province", "first"),
            county=("county", "first"),
            cnty_id=("cnty_id", "first"),
            start_time=("start_time", "min"),
            first_report_time=("report_time", "min"),
            last_report_time=("report_time", "max"),
            final_economic_loss=("economic_loss", "last"),
            final_loss_level=("loss_level_id", "last"),
        ).reset_index(),
        on="event_chain_id", how="left",
    )
    chains = chains.merge(
        d.pivot_table(
            index="event_chain_id", columns="report_status",
            values="economic_loss", aggfunc="last",
        ).add_prefix("loss_").reset_index(),
        on="event_chain_id", how="left",
    )

    golden_chains = chains[chains["n_stages"] >= 2].copy()
    golden_ids = set(golden_chains["event_chain_id"])
    golden_records = d[d["event_chain_id"].isin(golden_ids)].copy()
    golden_records = golden_records.sort_values(
        ["event_chain_id", "stage_order", "report_time"]
    )

    stats = {
        "n_chains_with_report_status": int(chains.shape[0]),
        "n_multistage_chains": int(multistage.sum()),
        "n_chains_all_3_stages": int(has_all3.sum()),
        "n_golden_records": int(len(golden_records)),
        "n_golden_chains": int(len(golden_chains)),
        "golden_stage_pattern": golden_chains["n_stages"].value_counts().sort_index().to_dict(),
        "golden_type_counts": golden_chains["disaster_type_14"].value_counts().to_dict(),
    }
    golden_chains["stage_pattern"] = golden_chains["n_stages"].astype(str) + "_stages"
    out = golden_records[[
        "record_id", "event_chain_id", "cnty_id", "province", "county",
        "disaster_type_14", "report_status", "stage_order", "stage_date",
        "report_time", "economic_loss", "loss_level_id",
    ]].copy()
    return out, stats


# ==========================================================================
# 数据统计文档
# ==========================================================================
def write_data_stats(
    processed: Path,
    region_stats: dict,
    assimil_stats: dict,
    cleaning: dict,
    availability: pd.DataFrame | None,
) -> Path:
    path = C.DOCS_PAPER_DIR / "data_stats.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    L = []
    L.append("# Phase 1 数据构建统计\n")
    L.append(f"> 自动生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}，"
             f"由 `src/data/build_benchmarks.py` 写入。请勿手工修改。\n")

    # --- 清洗 ---
    L.append("## 1. 灾情表清洗\n")
    if cleaning:
        ds = cleaning.get("disaster", {})
        L.append(f"- 输入记录：**{ds.get('n_records_input', 'NA'):,}**")
        L.append(f"- 输出记录：**{ds.get('n_records_output', 'NA'):,}**")
        ec = ds.get("economic_loss", {})
        L.append(f"- 经济损失量纲修正：**{ec.get('n_fixed', 'NA')}** 条"
                 f"（阈值 ≥ {C.ECONOMIC_LOSS_FIX_THRESHOLD:g} 万元，÷ {C.ECONOMIC_LOSS_FIX_DIVISOR:g}）")
        if ec.get("max_before") is not None:
            L.append(f"  - 修正前最大值：{ec['max_before']:,.1f} 万元")
            L.append(f"  - 修正后最大值：{ec['max_after']:,.1f} 万元")
        al = ds.get("alignment") or {}
        if al:
            L.append(f"- 区县对齐命中率：**{al.get('hit_rate', 0) * 100:.2f}%**"
                     f"（{al.get('hits', 0):,} / {al.get('total', 0):,}）")
            L.append("")
            L.append("| 匹配级别 | 记录数 |")
            L.append("|---|---:|")
            for k, v in al.get("by_level", {}).items():
                L.append(f"| `{k}` | {v:,} |")
        ch = ds.get("event_chains", {})
        if ch:
            L.append(f"\n- 事件链数：**{ch.get('n_chains', 0):,}**"
                     f"（多阶段链 {ch.get('n_multistage_chains', 0):,}）")
        L.append("")

    # --- 24->14 映射 ---
    L.append("## 2. 灾害类型 24 → 14 映射\n")
    mp_path = processed / "disaster_type_mapping.csv"
    if mp_path.exists():
        mp = pd.read_csv(mp_path)
        L.append("| 原始类型 | 记录数 | 动作 | 目标 14 类 | 说明 |")
        L.append("|---|---:|---|---|---|")
        for _, r in mp.iterrows():
            tgt = r["target_14"]
            tgt = "—" if (tgt is None or (isinstance(tgt, float) and np.isnan(tgt))
                          or str(tgt) == "nan") else str(tgt)
            L.append(f"| {r['disaster_type_std']} | {int(r['n_records']):,} | "
                     f"{r['action']} | {tgt} | {r['reason']} |")
        L.append("")

    # --- 特征可用率 ---
    if availability is not None and len(availability):
        L.append("## 3. 气象特征可用率（缺失率 = 缺测 + 无记录）\n")
        piv = availability.pivot(index="feature", columns="year", values="missing_rate")
        cols = [2000, 2005, 2010, 2015, 2018, 2019, 2021, 2022, 2024, 2025]
        cols = [c for c in cols if c in piv.columns]
        L.append("| 特征 | " + " | ".join(str(c) for c in cols) + " |")
        L.append("|---|" + "---:|" * len(cols))
        for feat in piv.index:
            vals = " | ".join(f"{piv.loc[feat, c] * 100:.1f}%" if c in piv.columns else "—"
                              for c in cols)
            L.append(f"| `{feat}` | {vals} |")
        L.append("")
        L.append("> ⚠️ `WIN_S_10mi_Avg` 在 2018–2025 全量缺测；`Lit`/`Thund`/`Tord` 已在 2016–2025 全缺并被剔除。\n")

    # --- 三大基准 ---
    L.append("## 4. 三大区域基准\n")
    L.append("| 数据集 | 节点数 | 边数 | 平均出度 | 跨省边占比 | 训练记录 | 验证记录 | 测试记录 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, st in region_stats.items():
        adj = st.get("adjacency", {})
        sp = st.get("splits", {})
        L.append(
            f"| **{st['label']}** | {st['n_nodes']} | {adj.get('n_edges', 'NA')} | "
            f"{adj.get('mean_out_degree', 'NA')} | "
            f"{adj.get('cross_province_edge_ratio', 0) * 100:.1f}% | "
            f"{sp.get('train', {}).get('n_records', 0):,} | "
            f"{sp.get('val', {}).get('n_records', 0):,} | "
            f"{sp.get('test', {}).get('n_records', 0):,} |"
        )
    L.append("")

    for name, st in region_stats.items():
        L.append(f"### {st['label']}\n")
        L.append(f"- 覆盖省份：{'、'.join(st['provinces'])}")
        L.append(f"- 节点 {st['n_nodes']}，时间 {st['date_start']} ~ {st['date_end']}")
        a = st.get("adjacency", {})
        L.append(f"- 边 {a.get('n_edges', 'NA')}，同地市边 {a.get('same_city_edges', 'NA')}，"
                 f"物理近邻新增 {a.get('physical_edges_added', 'NA')}，"
                 f"孤立节点 {a.get('isolated_nodes', 'NA')}")
        L.append(f"- 无法对齐而丢弃的灾情记录：{st.get('disaster_records_dropped_no_node', 0):,}")
        L.append("")
        L.append("| 划分 | 年份 | 记录数 | 事件数 | 有事件节点数 |")
        L.append("|---|---|---:|---:|---:|")
        for sp, (y0, y1) in C.SPLITS.items():
            s = st["splits"].get(sp, {})
            L.append(f"| {sp} | {y0}–{y1} | {s.get('n_records', 0):,} | "
                     f"{s.get('n_events', 0):,} | {s.get('n_nodes_with_events', 0):,} |")
        L.append("")
        L.append("**类型分布**\n")
        L.append("| 类型 | 全量 |")
        L.append("|---|---:|")
        allc: dict[str, int] = {}
        for sp in C.SPLITS:
            for k, v in st["splits"].get(sp, {}).get("class_counts", {}).items():
                allc[k] = allc.get(k, 0) + int(v)
        for k in C.DISASTER_TYPES_14:
            L.append(f"| {k} | {allc.get(k, 0):,} |")
        L.append("")

    # --- Task 3 ---
    L.append("## 5. Task 3 金标准同化子集\n")
    L.append(f"- 含报告状态的事件链：{assimil_stats.get('n_chains_with_report_status', 0):,}")
    L.append(f"- 多阶段链（≥2 种报告状态）：**{assimil_stats.get('n_multistage_chains', 0):,}**")
    L.append(f"- 含完整 初报+续报+终报 的链：**{assimil_stats.get('n_chains_all_3_stages', 0):,}**")
    L.append(f"- 金标准记录数：{assimil_stats.get('n_golden_records', 0):,}")
    L.append("")
    L.append("> ⚠️ `续报` 全量仅约 1.8%，Task 3 的有效样本以「初报→终报」两阶段为主，论文需披露。\n")

    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ==========================================================================
# 主流程
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1-B: 构建三大基准")
    ap.add_argument("--processed-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--regions", nargs="*", default=list(C.REGIONS))
    ap.add_argument("--skip-adjacency", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout, force=True,
    )
    t0 = time.time()
    p = args.processed_dir

    log.info("加载已处理数据 ...")
    feats = np.load(p / "meteo_features.npy", mmap_mode="r")
    mask = np.load(p / "meteo_observed_mask.npy", mmap_mode="r")
    dates = pd.DatetimeIndex(np.load(p / "meteo_dates.npy"))
    stations = pd.read_parquet(p / "station_index.parquet")
    disaster = pd.read_parquet(p / "disaster_cleaned.parquet")
    log.info("  meteo=%s, stations=%d, disaster=%d", feats.shape, len(stations), len(disaster))

    cleaning = {}
    cr_path = p / "cleaning_report.json"
    if cr_path.exists():
        cleaning = json.loads(cr_path.read_text(encoding="utf-8"))

    availability = None
    av_path = p / "meteo_feature_availability.csv"
    if av_path.exists():
        availability = pd.read_csv(av_path)

    region_stats = {}
    for name in args.regions:
        if name not in C.REGIONS:
            raise ValueError(f"未知区域: {name}")
        log.info("构建区域 %s ...", name)
        region_stats[name] = build_region(
            name, C.REGIONS[name], p, feats, mask, dates, stations, disaster,
            skip_adjacency=args.skip_adjacency,
        )

    log.info("构建 Task 3 同化子集 ...")
    golden, assimil_stats = build_assimilation_subset(disaster)
    golden.to_parquet(p / "assimilation_golden_subset.parquet", index=False)
    log.info(
        "  多阶段链 %d（其中三阶段 %d），金标准记录 %d",
        assimil_stats["n_multistage_chains"],
        assimil_stats["n_chains_all_3_stages"],
        assimil_stats["n_golden_records"],
    )

    report = {
        "generated_at": pd.Timestamp.now('UTC').isoformat(),
        "regions": region_stats,
        "assimilation": assimil_stats,
        "feature_names": C.FEATURE_NAMES,
        "splits": C.SPLITS,
    }
    _dump_json(p / "benchmark_report.json", report)

    stats_path = write_data_stats(p, region_stats, assimil_stats, cleaning, availability)
    log.info("写出 %s", stats_path)
    log.info("完成，用时 %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())