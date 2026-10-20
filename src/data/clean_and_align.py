#!/usr/bin/env python3
"""Phase 1-A：灾情表清洗/对齐 + 地面气象站日值清洗/插值/特征构造。

产物（默认 `datasets/processed/`）：
  disaster_cleaned.parquet       灾情主表（14 类、事件链、区县节点对齐）
  disaster_type_mapping.csv      24 -> 14 完整映射明细
  cleaning_report.json           清洗审计日志（量纲修正、丢弃、命中率）
  station_index.parquet          空间节点字典 (cnty_id, Province, City, Cnty)
  meteo_features.npy             (N, T, 16) float32，16 维物理特征
  meteo_observed_mask.npy        (N, T, 16) bool，True=该格由真实观测支撑
  meteo_dates.npy                (T,) datetime64[D]
  meteo_stats.json               气象侧审计日志（特殊值、重复、插值覆盖率）

运行：
  python3 -I -m src.data.clean_and_align
  python3 -I -m src.data.clean_and_align --skip-meteo   # 只跑灾情侧
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 允许 `python3 -I src/data/clean_and_align.py` 直接运行（隔离模式不读 cwd/PYTHONPATH）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data import common as C
from src.data.entity_align import StationIndex, alignment_report

log = logging.getLogger("clean_and_align")


# ==========================================================================
# 通用小工具
# ==========================================================================
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not serializable: {type(obj)}")


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_jsonable)


# ==========================================================================
# 一、灾情表
# ==========================================================================
def load_disaster_parts(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob(C.DISASTER_PART_GLOB))
    if not files:
        raise FileNotFoundError(f"未找到灾情分片: {raw_dir}/{C.DISASTER_PART_GLOB}")

    frames = []
    for f in files:
        t0 = time.time()
        df = pd.read_csv(
            f,
            encoding="utf-8-sig",
            usecols=C.DISASTER_USECOLS,
            low_memory=False,
        )
        # 决议 §6.1-2：隐私列已在 usecols 层面排除，这里再断言一次
        leaked = set(C.PRIVACY_COLS) & set(df.columns)
        assert not leaked, f"{f.name} 泄漏隐私列: {leaked}"
        frames.append(df)
        log.info("读取 %-42s %7d 行  %.1fs", f.name, len(df), time.time() - t0)

    out = pd.concat(frames, ignore_index=True)
    log.info("灾情合计 %d 行", len(out))
    return out


def fix_economic_loss(df: pd.DataFrame) -> dict:
    """决议 §6.1-3：>= 1e8 万元 -> ÷ 1e4。"""
    before = df["economic_loss"].copy()
    mask = before >= C.ECONOMIC_LOSS_FIX_THRESHOLD
    df.loc[mask, "economic_loss"] = before[mask] / C.ECONOMIC_LOSS_FIX_DIVISOR
    return {
        "rule": f"economic_loss >= {C.ECONOMIC_LOSS_FIX_THRESHOLD:g} -> / {C.ECONOMIC_LOSS_FIX_DIVISOR:g}",
        "n_fixed": int(mask.sum()),
        "max_before": float(np.nanmax(before.to_numpy())) if len(before) else None,
        "max_after": float(np.nanmax(df["economic_loss"].to_numpy())) if len(df) else None,
        "sum_before": float(np.nansum(before.to_numpy())),
        "sum_after": float(np.nansum(df["economic_loss"].to_numpy())),
    }


def map_disaster_types(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """决议 §6.3：24 -> 14 确定性映射，并输出完整明细表。"""
    raw_counts = df["disaster_type_std"].value_counts(dropna=False)
    df = df.copy()
    df["disaster_type_14"] = df["disaster_type_std"].map(C.DISASTER_TYPE_MAP_24_TO_14)

    mapping_rows = []
    for raw, n in raw_counts.items():
        key = "<空值>" if pd.isna(raw) else str(raw)
        target = C.DISASTER_TYPE_MAP_24_TO_14.get(key, None)
        if pd.isna(raw):
            action = "drop"
            reason = "原始类型为空"
        elif key not in C.DISASTER_TYPE_MAP_24_TO_14:
            action = "drop"
            reason = "未在映射表中定义（防御性丢弃）"
        elif target is None:
            action = "drop"
            reason = "占位符/非自然气象灾害/行政事件"
        elif target == key:
            action = "keep"
            reason = "标准类原样保留"
        else:
            action = "merge"
            reason = f"归并到 {target}"
        mapping_rows.append(
            {"disaster_type_std": key, "n_records": int(n), "target_14": target or "", "action": action, "reason": reason}
        )
    mapping_df = pd.DataFrame(mapping_rows).sort_values("n_records", ascending=False)

    kept = df[df["disaster_type_14"].notna()].copy()
    stats = {
        "n_input": int(len(df)),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(df) - len(kept)),
        "n_classes_kept": int(kept["disaster_type_14"].nunique()),
        "classes_present": sorted(kept["disaster_type_14"].unique().tolist()),
        "classes_missing": sorted(set(C.DISASTER_TYPES_14) - set(kept["disaster_type_14"].unique())),
        "class_counts": kept["disaster_type_14"].value_counts().to_dict(),
    }
    return kept, mapping_df, stats


def fill_end_time(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按灾害类型中位持续时长补全缺失 end_time。"""
    df = df.copy()
    for col in ("start_time", "end_time", "report_time"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    missing_before = int(df["end_time"].isna().sum())
    dur = (df["end_time"] - df["start_time"]).dt.total_seconds() / 86400.0
    median_dur = dur.groupby(df["disaster_type_14"]).transform("median")
    fallback = df["start_time"] + pd.to_timedelta(median_dur, unit="D")
    df["end_time"] = df["end_time"].fillna(fallback).fillna(df["start_time"] + pd.Timedelta(days=1))
    # 极端异常：end < start
    bad = df["end_time"] < df["start_time"]
    df.loc[bad, "end_time"] = df.loc[bad, "start_time"]
    return df, {
        "end_time_missing_before": missing_before,
        "end_time_missing_after": int(df["end_time"].isna().sum()),
        "end_before_start_fixed": int(bad.sum()),
    }


def build_event_chains(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """同一 (省, 区县, 14类) 下，起始间隔 <= 3 天的事件串成一条链。

    必须在区县对齐之后调用：优先用 `cnty_id` 作空间键，未命中时回退原始字符串。
    """
    df = df.copy()
    if "cnty_id" in df.columns:
        df["_chain_key"] = np.where(
            df["cnty_id"] >= 0,
            "N" + df["cnty_id"].astype(str),
            df["province"].fillna("") + "|" + df["county"].fillna("").astype(str),
        )
    else:
        df["_chain_key"] = (
            df["province"].fillna("") + "|" + df["city"].fillna("")
            + "|" + df["county"].fillna("").astype(str)
        )
    df = df.sort_values(
        ["_chain_key", "disaster_type_14", "start_time", "report_time"],
        kind="mergesort",
    )

    grp = df.groupby(["_chain_key", "disaster_type_14"], sort=False)
    prev_end = grp["end_time"].shift(1)
    prev_start = grp["start_time"].shift(1)
    gap_days = (df["start_time"] - prev_end).dt.total_seconds() / 86400.0
    # 与上一条重叠（gap<0）或间隔 <= 3 天 -> 同链；首条 / 起始时间缺失 -> 开新链
    is_new = (
        prev_start.isna()
        | df["start_time"].isna()
        | gap_days.isna()
        | (gap_days > C.EVENT_CHAIN_GAP_DAYS)
    )
    df["_new_chain"] = is_new.to_numpy()
    df["_chain_no"] = (
        df.groupby(["_chain_key", "disaster_type_14"], sort=False)["_new_chain"]
        .cumsum()
        .to_numpy()
    )
    # 生成全局唯一、可读的链 ID
    df["event_chain_id"] = (
        df["_chain_key"].astype(str) + "::" + df["disaster_type_14"].astype(str)
        + "::" + df["_chain_no"].astype(str)
    )

    df = df.drop(columns=["_chain_key", "_new_chain", "_chain_no"])
    df = df.sort_values(["start_time", "record_id"], kind="mergesort").reset_index(drop=True)

    sizes = df.groupby("event_chain_id").size()
    return df, {
        "n_chains": int(sizes.size),
        "chain_size_mean": round(float(sizes.mean()), 4),
        "chain_size_median": float(sizes.median()),
        "chain_size_max": int(sizes.max()),
        "n_multistage_chains": int((sizes > 1).sum()),
        "gap_threshold_days": C.EVENT_CHAIN_GAP_DAYS,
    }


def clean_disaster(
    raw_dir: Path, out_dir: Path, station_index: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = load_disaster_parts(raw_dir)
    n0 = len(df)

    econ_stats = fix_economic_loss(df)
    log.info("经济损失量纲修正: %d 条", econ_stats["n_fixed"])

    df, mapping_df, type_stats = map_disaster_types(df)
    log.info(
        "灾害类型映射: 保留 %d / 丢弃 %d，实有 %d 类，缺失 %s",
        type_stats["n_kept"], type_stats["n_dropped"],
        type_stats["n_classes_kept"], type_stats["classes_missing"],
    )

    df["loss_level_id"] = df["loss_level"].map(C.LOSS_LEVEL_MAP)
    level_stats = {
        "loss_level_missing": int(df["loss_level_id"].isna().sum()),
        "loss_level_counts": df["loss_level_id"].value_counts().sort_index().to_dict(),
        "inconsistent_none_loss": int(
            ((df["loss_level_id"] == 0) & (df["economic_loss"] > 0)).sum()
        ),
    }

    df, end_stats = fill_end_time(df)
    df["duration_days"] = (
        (df["end_time"] - df["start_time"]).dt.total_seconds() / 86400.0
    ).clip(lower=0)

    # 区县对齐必须先于事件链（链的空间键用 cnty_id）
    align_stats = None
    if station_index is not None:
        sidx = StationIndex(station_index)
        ids, levels, level_series = sidx.align_series(df)
        df["cnty_id"] = ids.astype(np.int64)
        df["align_level"] = level_series.to_numpy()
        align_stats = alignment_report(level_series)
        log.info(
            "区县对齐命中 %d/%d = %.2f%%",
            align_stats["hits"], align_stats["total"], align_stats["hit_rate"] * 100,
        )
        for k, v in align_stats["by_level"].items():
            log.info("  %-30s %d", k, v)
        miss = df[df["cnty_id"] < 0]
        if len(miss):
            align_stats["miss_samples"] = (
                miss[["province", "city", "county", "report_unit"]]
                .drop_duplicates().head(20).astype(str).to_dict("records")
            )
    else:
        df["cnty_id"] = np.int64(-1)
        df["align_level"] = "no_station_index"

    df, chain_stats = build_event_chains(df)

    stats = {
        "n_records_input": int(n0),
        "n_records_output": int(len(df)),
        "economic_loss": econ_stats,
        "disaster_type": type_stats,
        "loss_level": level_stats,
        "end_time": end_stats,
        "event_chains": chain_stats,
        "alignment": align_stats,
        "report_status": {
            "counts": {
                ("<空值>" if pd.isna(k) else str(k)): int(v)
                for k, v in df["report_status"].value_counts(dropna=False).items()
            },
            "report_time_fill_rate": round(float(df["report_time"].notna().mean()), 4),
        },
    }
    return df, stats, mapping_df


# ==========================================================================
# 二、气象表
# ==========================================================================
def _replace_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """特殊码解码。返回 (df, 计数)。

    顺序至关重要：先记录 trace 标志，再把特征值族转 NaN。
    实测发现 PRE_Time_2020 除 999990 外还有 999991 / 999992 两个微量变体，
    故按"特征值族"整体处理，而不是逐个精确值。
    """
    counts = {
        "trace_999990": 0, "trace_999991": 0, "trace_999992": 0,
        "999998": 0, "999999": 0, "family_other": 0,
        "residual_99xxxx": 0,
    }

    pre_raw = df["PRE_Time_2020"].to_numpy(dtype=np.float64, copy=True)
    trace = np.zeros(len(pre_raw), dtype=bool)
    for code in C.SENTINEL_TRACE_CODES:
        m = pre_raw == code
        counts[f"trace_{int(code)}"] += int(m.sum())
        trace |= m

    for col in C.CONTINUOUS_COLS:
        v = df[col].to_numpy(dtype=np.float64, copy=True)
        family = v >= C.SENTINEL_FAMILY_MIN
        if col == "PRE_Time_2020":
            family = family & ~trace
        counts["999999"] += int((v == C.SENTINEL_MISSING).sum())
        counts["999998"] += int((v == C.SENTINEL_NOT_OBSERVED).sum())
        counts["family_other"] += int((family & (v != C.SENTINEL_MISSING)
                                       & (v != C.SENTINEL_NOT_OBSERVED)).sum())
        if family.any():
            v[family] = np.nan
            df[col] = v

    # 微量降水 -> 0.05 mm（仅 PRE_Time_2020，实测确只出现在该列）
    v = df["PRE_Time_2020"].to_numpy(dtype=np.float64, copy=True)
    v[trace] = C.TRACE_PRECIP_MM
    df["PRE_Time_2020"] = v
    df["PRE_trace_flag"] = trace.astype(np.float32)

    # 二值天气现象同样存在 999999
    for col in C.BINARY_COLS:
        v = df[col].to_numpy(dtype=np.float64, copy=True)
        family = v >= C.SENTINEL_FAMILY_MIN
        counts["999999"] += int((v == C.SENTINEL_MISSING).sum())
        if family.any():
            v[family] = np.nan
            df[col] = v

    # 相态/极值编码区间（999000–999989）：全量仅 2 行真实出现，
    # 样本量不足以支撑相态建模，整体按非物理值剔除。
    for col in C.CONTINUOUS_COLS:
        v = df[col].to_numpy(dtype=np.float64, copy=True)
        phase = (v >= C.SENTINEL_PHASE_FAMILY_MIN) & (v < C.SENTINEL_FAMILY_MIN)
        if phase.any():
            counts["phase_family"] = counts.get("phase_family", 0) + int(phase.sum())
            v[phase] = np.nan
            df[col] = v

    # 物理合理域：剔除录入错误（越界样本全量仅 371 行）
    for col in C.CONTINUOUS_COLS:
        lo, hi = C.PHYSICAL_BOUNDS[col]
        v = df[col].to_numpy(dtype=np.float64, copy=True)
        bad = (v < lo) | (v > hi)
        if bad.any():
            counts["out_of_physical_bounds"] = (
                counts.get("out_of_physical_bounds", 0) + int(bad.sum())
            )
            v[bad] = np.nan
            df[col] = v

    # 防御性检查：应再无 99xxxx 家族码残留
    for col in C.CONTINUOUS_COLS:
        v = df[col].to_numpy(dtype=np.float64)
        counts["residual_99xxxx"] += int(np.nansum((v >= 990000) & (v <= 999999)))
    return df, counts


def load_meteo_year(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    n_raw = len(df)

    # 丢弃行政区划不全的行（如 City/Cnty 为空）
    empty = (
        df["Province"].isna() | (df["Province"].astype(str).str.strip() == "")
        | df["City"].isna() | (df["City"].astype(str).str.strip() == "")
        | df["Cnty"].isna() | (df["Cnty"].astype(str).str.strip() == "")
    )
    n_empty = int(empty.sum())
    df = df[~empty].copy()

    df, sentinel_counts = _replace_sentinels(df)

    agg_spec = {c: "mean" for c in C.CONTINUOUS_COLS}
    agg_spec["GaWIN"] = "max"
    agg_spec["Fog"] = "max"
    agg_spec["PRE_trace_flag"] = "max"

    keys = C.KEY_COLS + ["Year", "Mon", "Day"]
    n_before_agg = len(df)
    df = df.groupby(keys, as_index=False, sort=False).agg(agg_spec)
    n_after_agg = len(df)

    df["date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int), month=df["Mon"].astype(int), day=df["Day"].astype(int)),
        errors="coerce",
    )
    df = df[df["date"].notna()]

    stats = {
        "n_raw": n_raw,
        "n_empty_admin": n_empty,
        "n_duplicate_rows": n_before_agg - n_after_agg,
        "n_county_days": n_after_agg,
        "sentinels": sentinel_counts,
    }
    return df[keys + ["date"] + C.CONTINUOUS_COLS + C.BINARY_COLS + ["PRE_trace_flag"]], stats


def _interp_rows(arr: np.ndarray, max_gap: int = C.MAX_INTERP_GAP_DAYS):
    """沿 axis=1 对每行做线性插值。

    `observed_mask` 的定义：True = 该格是真实观测，或位于距最近真实观测
    <= `max_gap` 天的范围内（插值可信）。超出该窗口的格点数值仍被线性延拓
    填充（保证张量无缺失），但 mask 置 False，供下游显式识别"长期无观测"。

    例：`WIN_S_10mi_Avg` 在 2018-2025 全缺，其 mask 在该区间整体为 False。

    返回 (filled, observed)，均 shape (N, T)。
    """
    n, t = arr.shape
    filled = arr.copy()
    observed = np.zeros((n, t), dtype=bool)
    for i in range(n):
        row = arr[i]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        idx = np.nonzero(valid)[0]
        first, last = idx[0], idx[-1]
        filled[i] = np.interp(np.arange(t), idx, row[idx])

        # 计算每个时间点到最近有效观测的距离（向量化）
        pos = np.searchsorted(idx, np.arange(t))
        left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
        right = idx[np.clip(pos, 0, len(idx) - 1)]
        dist = np.minimum(np.abs(np.arange(t) - left), np.abs(right - np.arange(t)))
        observed[i] = dist <= max_gap
    return filled, observed


def build_meteo_tensor(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    files = sorted(raw_dir.glob(C.METEO_YEAR_GLOB))
    if not files:
        raise FileNotFoundError(f"未找到气象文件: {raw_dir}/{C.METEO_YEAR_GLOB}")

    frames = []
    year_stats = {}
    t_all = time.time()
    for f in files:
        t0 = time.time()
        df, st = load_meteo_year(f)
        frames.append(df)
        year_stats[f.stem] = st
        log.info(
            "气象 %-14s raw=%7d dup=%6d empty=%4d %5.1fs",
            f.name, st["n_raw"], st["n_duplicate_rows"], st["n_empty_admin"],
            time.time() - t0,
        )

    meteo = pd.concat(frames, ignore_index=True)
    del frames
    log.info("气象拼接完成: %d 行  %.1fs", len(meteo), time.time() - t_all)

    # 空间节点字典（稳定排序）
    station_index = (
        meteo[C.KEY_COLS].drop_duplicates()
        .sort_values(C.KEY_COLS)
        .reset_index(drop=True)
    )
    station_index.insert(0, "cnty_id", np.arange(len(station_index), dtype=np.int64))
    n_nodes = len(station_index)

    node_of = {
        (p, c, t): i
        for i, (p, c, t) in enumerate(
            zip(station_index["Province"], station_index["City"], station_index["Cnty"])
        )
    }
    meteo["cnty_id"] = [
        node_of[(p, c, t)]
        for p, c, t in zip(meteo["Province"], meteo["City"], meteo["Cnty"])
    ]

    dates = pd.date_range(C.DATE_START, C.DATE_END, freq="D")
    n_days = len(dates)
    start = dates[0]
    day_idx = (meteo["date"] - start).dt.days.to_numpy(dtype=np.int64)

    log.info("构造 %d 节点 × %d 天原始栅格 ...", n_nodes, n_days)
    raw = {
        c: np.full((n_nodes, n_days), np.nan, dtype=np.float64)
        for c in C.CONTINUOUS_COLS
    }
    for c in C.CONTINUOUS_COLS:
        raw[c][meteo["cnty_id"].to_numpy(), day_idx] = meteo[c].to_numpy(dtype=np.float64)

    binary = {}
    for c in C.BINARY_COLS + ["PRE_trace_flag"]:
        arr = np.full((n_nodes, n_days), np.nan, dtype=np.float64)
        arr[meteo["cnty_id"].to_numpy(), day_idx] = meteo[c].to_numpy(dtype=np.float64)
        binary[c] = arr
    del meteo

    # 逐列逐年可用率（缺失 = 原始缺测 或 该节点当日无记录）
    years = np.array([d.year for d in dates], dtype=np.int32)
    availability_rows = []
    for c in C.CONTINUOUS_COLS + C.BINARY_COLS:
        src = raw[c] if c in raw else binary[c]
        for y in range(int(years.min()), int(years.max()) + 1):
            mask_y = years == y
            availability_rows.append({
                "feature": c,
                "year": y,
                "missing_rate": round(float(np.isnan(src[:, mask_y]).mean()), 6),
            })
    availability = pd.DataFrame(availability_rows)

    # 插值
    log.info("逐节点线性插值 ...")
    interp = {}
    obs_mask = {}
    for c in C.CONTINUOUS_COLS:
        interp[c], obs_mask[c] = _interp_rows(raw[c])
        nan_after = int(np.isnan(interp[c]).sum())
        if nan_after:
            log.warning("  %s: 仍有 %d 个全空行（无任何观测）", c, nan_after)

    # 二值：NaN -> 0（缺测按未出现处理），并记录原始缺失
    binary_missing = {}
    binary_observed = {}
    for c in C.BINARY_COLS + ["PRE_trace_flag"]:
        binary_missing[c] = int(np.isnan(binary[c]).sum())
        binary_observed[c] = ~np.isnan(binary[c])
        binary[c] = np.nan_to_num(binary[c], nan=0.0)

    # 24 小时变压在插值后的 PRS 上计算
    prs = interp["PRS_Avg"]
    delta = np.full_like(prs, np.nan)
    delta[:, 1:] = prs[:, 1:] - prs[:, :-1]

    feats = np.empty((n_nodes, n_days, C.N_FEATURES), dtype=np.float32)
    for j, c in enumerate(C.CONTINUOUS_COLS):
        feats[:, :, j] = interp[c].astype(np.float32)

    feats[:, :, 9] = binary["GaWIN"].astype(np.float32)
    feats[:, :, 10] = binary["Fog"].astype(np.float32)
    feats[:, :, 11] = binary["PRE_trace_flag"].astype(np.float32)

    pre = interp["PRE_Time_2020"]
    tem_avg = interp["TEM_Avg"]
    rain_snow = (pre > 0).astype(np.float64) * C.sigmoid(-(tem_avg - 0.5))
    with np.errstate(invalid="ignore", divide="ignore"):
        gust = interp["WIN_S_Inst_Max"] / (interp["WIN_S_Max"] + 0.1)
    gust[~np.isfinite(gust)] = np.nan
    diurnal = interp["TEM_Max"] - interp["TEM_Min"]

    feats[:, :, 12] = rain_snow.astype(np.float32)
    feats[:, :, 13] = gust.astype(np.float32)
    feats[:, :, 14] = diurnal.astype(np.float32)
    feats[:, :, 15] = delta.astype(np.float32)

    # 派生特征的物理裁剪：长时段插值延拓会放大误差，
    # 24h 变压理论上界约 ±60 hPa；阵风系数与日较差同样有物理上界。
    feats[:, :, 13] = np.clip(feats[:, :, 13], 0.0, 20.0)
    feats[:, :, 14] = np.clip(feats[:, :, 14], 0.0, 100.0)
    feats[:, :, 15] = np.clip(feats[:, :, 15], -120.0, 120.0)

    # ---- 观测可信掩码：逐通道给出 True=由真实观测支撑 ----
    # 连续列直接用 _interp_rows 返回的 obs_mask（距最近观测 <= 7 天）
    mask = np.zeros(feats.shape, dtype=bool)
    for j, c in enumerate(C.CONTINUOUS_COLS):
        mask[:, :, j] = obs_mask[c]

    for j, c in enumerate(C.BINARY_COLS + ["PRE_trace_flag"], start=9):
        mask[:, :, j] = binary_observed[c]

    # 派生特征：所有输入通道都可信时才算可观测
    o_prs = obs_mask["PRS_Avg"]
    o_tem_avg = obs_mask["TEM_Avg"]
    o_tem_max = obs_mask["TEM_Max"]
    o_tem_min = obs_mask["TEM_Min"]
    o_pre = obs_mask["PRE_Time_2020"]
    o_wmax = obs_mask["WIN_S_Max"]
    o_winst = obs_mask["WIN_S_Inst_Max"]

    mask[:, :, 12] = o_pre & o_tem_avg                      # rain_snow_phase
    mask[:, :, 13] = o_winst & o_wmax                       # gust_factor
    mask[:, :, 14] = o_tem_max & o_tem_min                  # tem_diurnal_range
    delta_ok = o_prs.copy()
    delta_ok[:, 1:] &= o_prs[:, :-1]
    delta_ok[:, 0] = False
    mask[:, :, 15] = delta_ok                               # delta_prs_24h

    n_nan_before_fill = int((~mask).sum())
    # 无可观测值的格点用"该通道观测中位数"填充（比填 0 更中性；
    # 气压/气温填 0 会产生荒谬值）。mask 已标记为 False，下游应据其忽略。
    for j in range(C.N_FEATURES):
        chan = feats[:, :, j]
        if not np.isnan(chan).any():
            continue
        sample = chan[::10, ::10]
        med = np.nanmedian(sample)
        if not np.isfinite(med):
            med = 0.0
        chan[np.isnan(chan)] = med
        feats[:, :, j] = chan
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "n_nodes": int(n_nodes),
        "n_days": int(n_days),
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "n_tensor_cells": int(feats.size),
        "n_unobserved_cells": n_nan_before_fill,
        "unobserved_rate": round(n_nan_before_fill / feats.size, 6),
        "binary_missing_before_fill": binary_missing,
        "yearly": year_stats,
        "feature_names": C.FEATURE_NAMES,
        "dead_features_gt95pct_in_split": _dead_feature_report(availability),
    }
    return feats, mask, station_index, (dates, stats, availability)


def _dead_feature_report(availability: pd.DataFrame) -> dict:
    """标记在 val/test 区间缺测率 >= 95% 的列（不可作为跨期稳定特征）。"""
    out = {}
    for split, (y0, y1) in C.SPLITS.items():
        sub = availability[(availability["year"] >= y0) & (availability["year"] <= y1)]
        rates = sub.groupby("feature")["missing_rate"].mean()
        out[split] = {
            str(f): round(float(r), 4) for f, r in rates.items() if r >= 0.95
        }
    return out


# ==========================================================================
# 三、主流程
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1-A: 清洗与对齐")
    ap.add_argument("--raw-disaster-dir", type=Path, default=C.RAW_DISASTER_DIR)
    ap.add_argument("--raw-meteo-dir", type=Path, default=C.RAW_METEO_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--skip-meteo", action="store_true", help="只处理灾情表")
    ap.add_argument("--skip-disaster", action="store_true", help="只处理气象表")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    _setup_logging(args.log_level)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    report: dict = {"generated_at": pd.Timestamp.now('UTC').isoformat()}

    # 顺序固定：先建空间节点字典（气象），再清洗灾情并用其对齐
    station_index = None
    if not args.skip_meteo:
        log.info("=" * 60)
        log.info("气象表清洗与 16 维特征构造")
        log.info("=" * 60)
        feats, mask, station_index, (dates, mstats, availability) = build_meteo_tensor(
            args.raw_meteo_dir
        )

        np.save(args.out_dir / "meteo_features.npy", feats)
        np.save(args.out_dir / "meteo_observed_mask.npy", mask)
        np.save(args.out_dir / "meteo_dates.npy", dates.to_numpy(dtype="datetime64[D]"))
        station_index.to_parquet(args.out_dir / "station_index.parquet", index=False)
        availability.to_csv(
            args.out_dir / "meteo_feature_availability.csv", index=False, encoding="utf-8-sig"
        )
        log.info(
            "写出 meteo_features.npy shape=%s (%.2f GB)", feats.shape, feats.nbytes / 1e9
        )
        report["meteo"] = mstats
        report["station_index"] = {"n_nodes": int(len(station_index))}
        del feats, mask
    else:
        # --skip-meteo：复用磁盘上已有的节点字典，保证区县对齐仍可执行
        si_path = args.out_dir / "station_index.parquet"
        if si_path.exists():
            station_index = pd.read_parquet(si_path)
            log.info("复用已有节点字典: %d 个节点", len(station_index))
        else:
            log.warning("--skip-meteo 且无 station_index.parquet，跳过区县对齐")

    if not args.skip_disaster:
        log.info("=" * 60)
        log.info("灾情表清洗与区县对齐")
        log.info("=" * 60)
        disaster, dstats, mapping_df = clean_disaster(
            args.raw_disaster_dir, args.out_dir, station_index=station_index
        )

        out_path = args.out_dir / "disaster_cleaned.parquet"
        disaster.to_parquet(out_path, index=False)
        log.info("写出 %s (%d 行)", out_path, len(disaster))

        mapping_df.to_csv(
            args.out_dir / "disaster_type_mapping.csv", index=False, encoding="utf-8-sig"
        )
        report["disaster"] = dstats

        if station_index is not None:
            aligned = disaster[disaster["cnty_id"] >= 0]
            aligned.to_parquet(
                args.out_dir / "disaster_aligned.parquet", index=False
            )
            log.info("写出 disaster_aligned.parquet (%d 行)", len(aligned))

    # 合并既有报告，避免 --skip-meteo / --skip-disaster 分步运行互相覆盖
    report_path = args.out_dir / "cleaning_report.json"
    merged: dict = {}
    if report_path.exists():
        try:
            merged = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(report)
    _dump_json(report_path, merged)
    log.info("完成，用时 %.1fs -> %s", time.time() - t0, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())