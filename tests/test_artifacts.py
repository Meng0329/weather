"""Phase 1 产物集成测试：校验已落盘的数据集满足模型输入契约。

产物缺失时自动跳过（`pytest -m "not integration"` 可显式排除）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import common as C

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "datasets" / "processed"

pytestmark = pytest.mark.integration


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"产物不存在: {path.relative_to(REPO_ROOT)}")
    return path


@pytest.fixture(scope="module")
def station_index():
    return pd.read_parquet(_require(PROCESSED / "station_index.parquet"))


@pytest.fixture(scope="module")
def disaster():
    return pd.read_parquet(_require(PROCESSED / "disaster_cleaned.parquet"))


@pytest.fixture(scope="module")
def dates():
    return pd.DatetimeIndex(np.load(_require(PROCESSED / "meteo_dates.npy")))


@pytest.fixture(scope="module")
def report():
    return json.loads(_require(PROCESSED / "cleaning_report.json").read_text("utf-8"))


# --------------------------------------------------------------------------
# 气象张量
# --------------------------------------------------------------------------
def test_meteo_tensor_shape_and_no_nan(station_index, dates):
    feats = np.load(_require(PROCESSED / "meteo_features.npy"), mmap_mode="r")
    mask = np.load(_require(PROCESSED / "meteo_observed_mask.npy"), mmap_mode="r")
    assert feats.shape == (len(station_index), len(dates), C.N_FEATURES)
    assert mask.shape == feats.shape
    assert feats.dtype == np.float32
    # 分块检查无 NaN/Inf
    for s in range(0, feats.shape[0], 256):
        chunk = np.asarray(feats[s:s + 256])
        assert np.isfinite(chunk).all(), "16 维张量不允许 NaN/Inf"


def test_meteo_tensor_has_no_sentinel_codes(station_index):
    feats = np.load(_require(PROCESSED / "meteo_features.npy"), mmap_mode="r")
    for j, name in enumerate(C.CONTINUOUS_COLS):
        col = np.asarray(feats[:, :, j])
        assert col.max() < 990000, f"{name} 仍残留 99xxxx 特征值"
        assert col.min() > -990000, f"{name} 存在异常负值"


def test_observed_mask_marks_unavailable_features(station_index, dates):
    """WIN_S_10mi_Avg 在 2018-2025 全缺 -> mask 必须为 False。"""
    mask = np.load(_require(PROCESSED / "meteo_observed_mask.npy"), mmap_mode="r")
    j = C.FEATURE_NAMES.index("WIN_S_10mi_Avg")
    y0 = int(np.nonzero(dates.year == 2022)[0][0])
    y1 = int(np.nonzero(dates.year == 2025)[0][-1])
    block = np.asarray(mask[:, y0:y1 + 1, j])
    assert block.mean() < 0.05, "2018+ 的 WIN_S_10mi_Avg 应几乎完全不可用"


def test_station_index_is_unique_and_sorted(station_index):
    assert station_index["cnty_id"].is_unique
    assert station_index["cnty_id"].tolist() == list(range(len(station_index)))
    keys = list(zip(station_index["Province"], station_index["City"], station_index["Cnty"]))
    assert keys == sorted(keys)
    assert (station_index[["Province", "City", "Cnty"]].astype(str).apply(
        lambda s: s.str.strip().eq("")).any(axis=1) == False).all()


def test_feature_availability_report_marks_dead_column():
    df = pd.read_csv(_require(PROCESSED / "meteo_feature_availability.csv"))
    sub = df[(df["feature"] == "WIN_S_10mi_Avg") & (df["year"] >= 2018)]
    assert (sub["missing_rate"] > 0.99).all()


# --------------------------------------------------------------------------
# 灾情主表
# --------------------------------------------------------------------------
def test_disaster_has_no_privacy_columns(disaster):
    for col in C.PRIVACY_COLS:
        assert col not in disaster.columns


def test_disaster_types_are_within_14(disaster):
    assert set(disaster["disaster_type_14"]) <= set(C.DISASTER_TYPES_14)
    assert disaster["disaster_type_14"].nunique() == C.NUM_DISASTER_TYPES


def test_economic_loss_capped_after_unit_fix(disaster):
    assert disaster["economic_loss"].max() < C.ECONOMIC_LOSS_FIX_THRESHOLD


def test_disaster_record_count_matches_mapping(disaster):
    mp = pd.read_csv(_require(PROCESSED / "disaster_type_mapping.csv"))
    kept = mp[mp["action"] != "drop"]["n_records"].sum()
    assert len(disaster) == int(kept)


def test_event_chain_ids_are_populated_and_grouped(disaster):
    assert disaster["event_chain_id"].notna().all()
    assert disaster["event_chain_id"].nunique() > 0
    # 同一链内类型必须一致、空间键必须一致
    g = disaster.groupby("event_chain_id")
    assert (g["disaster_type_14"].nunique() == 1).all()
    assert (g["cnty_id"].nunique() == 1).all()


def test_alignment_rate_is_reported_and_reasonable(report):
    al = report["disaster"]["alignment"]
    assert al["total"] > 0
    assert 0.5 < al["hit_rate"] < 1.0
    assert set(al["by_level"]) >= {"L1_exact", "miss"}


def test_cnty_id_consistency_with_station_province(disaster, station_index):
    """对齐后的节点所属省份应与灾情记录省份一致（防止误挂）。"""
    prov_of = dict(zip(station_index["cnty_id"], station_index["Province"]))
    d = disaster[disaster["cnty_id"] >= 0]
    bad = 0
    for prov, cid in zip(d["province"].tolist(), d["cnty_id"].tolist()):
        p_node = prov_of[int(cid)]
        if isinstance(prov, str) and prov and prov != p_node:
            bad += 1
    assert bad / max(len(d), 1) < 0.02, f"省份不一致比例过高: {bad}/{len(d)}"


# --------------------------------------------------------------------------
# 三大区域基准
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench_report():
    return json.loads(_require(PROCESSED / "benchmark_report.json").read_text("utf-8"))


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_artifacts_exist(region):
    d = PROCESSED / region
    for f in ["nodes.parquet", "adj_mask.npy", "adj_weight.npy",
              "meteo_features.npy", "meteo_observed_mask.npy",
              "records.parquet", "events.parquet", "daily_labels.npy",
              "split_index.json"]:
        _require(d / f)


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_nodes_belong_to_region_provinces(region):
    nodes = pd.read_parquet(_require(PROCESSED / region / "nodes.parquet"))
    assert set(nodes["Province"]) <= set(C.REGIONS[region]["provinces"])
    assert nodes["node_idx"].tolist() == list(range(len(nodes)))


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_adjacency_is_valid(region):
    adj = np.load(_require(PROCESSED / region / "adj_mask.npy"))
    wgt = np.load(_require(PROCESSED / region / "adj_weight.npy"))
    nodes = pd.read_parquet(_require(PROCESSED / region / "nodes.parquet"))
    n = len(nodes)
    assert adj.shape == (n, n)
    assert wgt.shape == (n, n)
    assert adj.dtype == bool
    assert not np.any(np.diag(adj)), "邻接矩阵不得有自环"
    assert adj.sum() > 0
    # 同地市必须完全连通（不含对角自环）
    key = nodes["Province"].astype(str) + "|" + nodes["City"].astype(str)
    for _, idx in pd.Series(np.arange(n)).groupby(key.to_numpy()).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        block = adj[np.ix_(idx, idx)].copy()
        np.fill_diagonal(block, True)  # 自环已统一移除，比较时忽略
        assert block.all(), "同地市区县必须完全连通"


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_splits_are_chronological(region):
    rec = pd.read_parquet(_require(PROCESSED / region / "records.parquet"))
    assert set(rec["split"]) <= {"train", "val", "test"}
    years = rec["start_time"].dt.year
    assert years[rec["split"] == "train"].max() <= C.SPLITS["train"][1]
    assert years[rec["split"] == "val"].min() >= C.SPLITS["val"][0]
    assert years[rec["split"] == "val"].max() <= C.SPLITS["val"][1]
    assert years[rec["split"] == "test"].min() >= C.SPLITS["test"][0]
    assert years[rec["split"] == "test"].max() <= C.SPLITS["test"][1]


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_no_temporal_leakage_between_splits(region):
    """测试集的最早事件必须晚于训练集的最晚事件（严格时序切分）。"""
    rec = pd.read_parquet(_require(PROCESSED / region / "records.parquet"))
    tr = rec[rec["split"] == "train"]["start_time"].max()
    va = rec[rec["split"] == "val"]["start_time"].min()
    te = rec[rec["split"] == "test"]["start_time"].min()
    assert tr < va <= te


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_tensor_shapes(region, dates):
    d = PROCESSED / region
    nodes = pd.read_parquet(_require(d / "nodes.parquet"))
    feats = np.load(_require(d / "meteo_features.npy"), mmap_mode="r")
    labels = np.load(_require(d / "daily_labels.npy"), mmap_mode="r")
    assert feats.shape == (len(nodes), len(dates), C.N_FEATURES_MODEL)
    assert labels.shape == (len(nodes), len(dates), C.NUM_DISASTER_TYPES)
    assert labels.dtype == np.uint8


@pytest.mark.parametrize("region", list(C.REGIONS))
def test_region_records_are_within_region_and_aligned(region):
    rec = pd.read_parquet(_require(PROCESSED / region / "records.parquet"))
    nodes = pd.read_parquet(_require(PROCESSED / region / "nodes.parquet"))
    assert set(rec["province"]) <= set(C.REGIONS[region]["provinces"])
    assert set(rec["node_idx"]) <= set(nodes["node_idx"])
    assert rec["type_idx"].notna().all()
    assert rec["type_idx"].between(0, C.NUM_DISASTER_TYPES - 1).all()


# --------------------------------------------------------------------------
# Task 3 金标准子集
# --------------------------------------------------------------------------
def test_assimilation_golden_subset_exists_and_is_multistage():
    g = pd.read_parquet(_require(PROCESSED / "assimilation_golden_subset.parquet"))
    assert len(g) > 0
    assert set(g["report_status"]) <= set(C.REPORT_STAGES)
    sizes = g.groupby("event_chain_id").size()
    assert (sizes >= 2).all(), "金标准子集内每条链至少要有 2 条报告记录"
    assert g["stage_order"].notna().all()


def test_assimilation_golden_subset_has_report_times():
    g = pd.read_parquet(_require(PROCESSED / "assimilation_golden_subset.parquet"))
    assert g["report_time"].notna().mean() > 0.9


def test_benchmark_report_structure(bench_report):
    assert set(bench_report["regions"]) == set(C.REGIONS)
    for name, st in bench_report["regions"].items():
        assert st["n_nodes"] > 0
        assert st["n_days"] > 9000
        assert set(st["splits"]) == {"train", "val", "test"}
        assert st["splits"]["train"]["n_records"] > 0
        assert st["splits"]["test"]["n_records"] > 0
