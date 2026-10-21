"""Phase 1 数据管线的单元测试（纯逻辑，不依赖大数据产物）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import common as C
from src.data.entity_align import StationIndex, alignment_report, split_county_string


# --------------------------------------------------------------------------
# 特征契约
# --------------------------------------------------------------------------
def test_feature_contract_is_16_and_unique():
    assert len(C.FEATURE_NAMES) == 16
    assert len(set(C.FEATURE_NAMES)) == 16
    assert C.N_FEATURES == 16


def test_feature_order_matches_documented_contract():
    assert C.FEATURE_NAMES[:9] == C.CONTINUOUS_COLS
    assert C.FEATURE_NAMES[9:11] == C.BINARY_COLS
    assert C.FEATURE_NAMES[11] == "PRE_trace_flag"
    assert C.FEATURE_NAMES[12:] == [
        "rain_snow_phase", "gust_factor", "tem_diurnal_range", "delta_prs_24h"
    ]


def test_dropped_meteo_cols_are_not_features():
    # Lit/Thund/Tord 从未进入特征契约；WIN_S_10mi_Avg 只保留在 16 维原始落盘契约中，
    # 建模用的 15 维契约必须已剔除它（Phase 1.5 `docs/experiment_protocol.md` §3.2）
    for col in ("Lit", "Thund", "Tord"):
        assert col not in C.FEATURE_NAMES
        assert col not in C.FEATURE_NAMES_15
    assert "WIN_S_10mi_Avg" in C.FEATURE_NAMES
    assert "WIN_S_10mi_Avg" not in C.FEATURE_NAMES_15


def test_privacy_cols_not_in_disaster_usecols():
    assert not (set(C.PRIVACY_COLS) & set(C.DISASTER_USECOLS))
    assert "tabulator" not in C.DISASTER_USECOLS
    assert "contact" not in C.DISASTER_USECOLS


# --------------------------------------------------------------------------
# 灾害类型映射
# --------------------------------------------------------------------------
def test_type_map_targets_are_within_14_classes():
    for src, dst in C.DISASTER_TYPE_MAP_24_TO_14.items():
        if dst is not None:
            assert dst in C.DISASTER_TYPES_14, f"{src} -> {dst} 不在 14 类中"


def test_type_map_covers_all_24_observed_types():
    observed = {
        "暴雨", "冰雹", "大风", "干旱", "台风", "雪灾", "雷电", "地质灾害",
        "低温", "冻害", "未知", "雾霾", "森林草原火灾", "连阴雨", "高温",
        "其他", "沙尘暴", "气象有关的谣传和不当舆论", "大气污染", "干热风",
        "气象部门参与减轻事件", "凌汛", "风暴潮", "危险物质泄漏(气体、核应急)",
    }
    assert observed <= set(C.DISASTER_TYPE_MAP_24_TO_14), (
        observed - set(C.DISASTER_TYPE_MAP_24_TO_14)
    )


def test_dropped_types_are_non_disaster_or_placeholder():
    for t in ["未知", "其他", "气象部门参与减轻事件", "气象有关的谣传和不当舆论",
              "危险物质泄漏(气体、核应急)"]:
        assert C.DISASTER_TYPE_MAP_24_TO_14[t] is None


# --------------------------------------------------------------------------
# 派生特征计算
# --------------------------------------------------------------------------
def test_rain_snow_phase_is_higher_below_freezing():
    df = pd.DataFrame({
        "PRE_Time_2020": [5.0, 5.0, 5.0],
        "TEM_Avg": [-10.0, 0.5, 15.0],
        "TEM_Max": [0.0, 0.0, 0.0],
        "TEM_Min": [0.0, 0.0, 0.0],
        "WIN_S_Max": [2.0, 2.0, 2.0],
        "WIN_S_Inst_Max": [4.0, 4.0, 4.0],
        "WIN_S_2mi_Avg": [1.0, 1.0, 1.0],
        "WIN_S_10mi_Avg": [1.5, 1.5, 1.5],
        "PRS_Avg": [1000.0, 1000.0, 1000.0],
        "delta_prs_24h": [0.0, 0.0, 0.0],
        "GaWIN": [0.0, 0.0, 0.0],
        "Fog": [0.0, 0.0, 0.0],
        "PRE_trace_flag": [0.0, 0.0, 0.0],
    })
    out = C.build_16_features(df)
    assert out.shape == (3, 16)
    phase = out[:, C.FEATURE_NAMES.index("rain_snow_phase")]
    assert phase[0] > phase[1] > phase[2]
    assert 0.0 <= phase.min() and phase.max() <= 1.0


def test_rain_snow_phase_zero_without_precip():
    df = pd.DataFrame({
        "PRE_Time_2020": [0.0], "TEM_Avg": [-10.0],
        "TEM_Max": [0.0], "TEM_Min": [0.0],
        "WIN_S_Max": [1.0], "WIN_S_Inst_Max": [1.0],
        "WIN_S_2mi_Avg": [0.5], "WIN_S_10mi_Avg": [0.8],
        "PRS_Avg": [1000.0], "delta_prs_24h": [0.0],
        "GaWIN": [0.0], "Fog": [0.0], "PRE_trace_flag": [0.0],
    })
    out = C.build_16_features(df)
    assert out[0, C.FEATURE_NAMES.index("rain_snow_phase")] == 0.0


def test_gust_factor_and_diurnal_range():
    df = pd.DataFrame({
        "PRE_Time_2020": [0.0], "TEM_Avg": [10.0],
        "TEM_Max": [20.0], "TEM_Min": [5.0],
        "WIN_S_Max": [9.9], "WIN_S_Inst_Max": [20.0],
        "WIN_S_2mi_Avg": [5.0], "WIN_S_10mi_Avg": [7.0],
        "PRS_Avg": [1000.0], "delta_prs_24h": [0.0],
        "GaWIN": [0.0], "Fog": [0.0], "PRE_trace_flag": [0.0],
    })
    out = C.build_16_features(df)
    assert out[0, C.FEATURE_NAMES.index("gust_factor")] == pytest.approx(20.0 / 10.0)
    assert out[0, C.FEATURE_NAMES.index("tem_diurnal_range")] == pytest.approx(15.0)


# --------------------------------------------------------------------------
# 区县名归一化与层级剥离
# --------------------------------------------------------------------------
def test_norm_province():
    assert C.norm_province("广西壮族自治区") == "广西"
    assert C.norm_province("内蒙古自治区") == "内蒙古"
    assert C.norm_province("宁夏回族自治区") == "宁夏"
    assert C.norm_province("四川省") == "四川"


def test_county_stem():
    assert C.county_stem("中江县") == "中江"
    assert C.county_stem("徐汇区") == "徐汇"
    assert C.county_stem("尚志市") == "尚志"


def test_split_county_string_strips_hierarchy():
    p, c, t = split_county_string("四川省", "四川省德阳市", "四川省德阳市中江县")
    assert (p, c, t) == ("四川省", "德阳市", "中江县")


def test_split_county_string_handles_bare_names():
    p, c, t = split_county_string("重庆市", "重庆市", "忠县")
    assert t == "忠县"


def test_split_county_string_handles_autonomous_prefecture():
    p, c, t = split_county_string(
        "云南省", "云南省红河哈尼族彝族自治州", "云南省红河哈尼族彝族自治州河口瑶族自治县"
    )
    assert c == "红河哈尼族彝族自治州"
    assert t == "河口瑶族自治县"


def test_split_county_string_extracts_city_when_absent():
    _, c, t = split_county_string("河北省", "", "河北省石家庄市正定县")
    assert c == "石家庄市"
    assert t == "正定县"


@pytest.fixture
def station_index_fixture():
    stations = pd.DataFrame({
        "Province": ["四川省", "四川省", "四川省", "重庆市", "云南省"],
        "City": ["德阳市", "德阳市", "成都市", "重庆市", "红河哈尼族彝族自治州"],
        "Cnty": ["中江县", "旌阳区", "双流区", "忠县", "河口瑶族自治县"],
    })
    return StationIndex(stations)


def test_station_index_is_sorted_and_deterministic(station_index_fixture):
    idx = station_index_fixture
    keys = list(zip(idx.province, idx.city, idx.cnty))
    assert keys == sorted(keys)
    assert idx.n_nodes == 5


def test_align_exact_hierarchical_string(station_index_fixture):
    i, level = station_index_fixture.match("四川省", "四川省德阳市", "四川省德阳市中江县")
    assert level.startswith("L1")
    assert station_index_fixture.cnty[i] == "中江县"


def test_align_bare_county_name(station_index_fixture):
    i, level = station_index_fixture.match("重庆市", "重庆市", "忠县")
    assert level.startswith("L1")
    assert station_index_fixture.cnty[i] == "忠县"


def test_align_via_report_unit_fallback(station_index_fixture):
    i, level = station_index_fixture.match("四川省", "", "", "四川省成都市双流区")
    assert i >= 0
    assert station_index_fixture.cnty[i] == "双流区"


def test_align_miss_returns_minus_one(station_index_fixture):
    i, level = station_index_fixture.match("四川省", "四川省德阳市", "四川省德阳市不存在区")
    assert i == -1
    assert level == "miss"


def test_alignment_report_math():
    s = pd.Series(["L1_exact"] * 3 + ["miss"] * 1)
    r = alignment_report(s)
    assert r["total"] == 4
    assert r["hits"] == 3
    assert r["hit_rate"] == pytest.approx(0.75)


def test_align_series_batch(station_index_fixture):
    df = pd.DataFrame({
        "province": ["四川省", "重庆市", "云南省", "四川省"],
        "city": ["四川省德阳市", "重庆市", "云南省红河哈尼族彝族自治州", "四川省德阳市"],
        "county": ["四川省德阳市中江县", "重庆市忠县",
                   "云南省红河哈尼族彝族自治州河口瑶族自治县", "四川省德阳市不存在区"],
        "report_unit": ["", "", "", ""],
    })
    ids, levels, s = station_index_fixture.align_series(df)
    assert ids[0] >= 0 and ids[1] >= 0 and ids[2] >= 0
    assert ids[3] == -1
    assert alignment_report(s)["hits"] == 3


# --------------------------------------------------------------------------
# 区域/划分定义
# --------------------------------------------------------------------------
def test_regions_have_no_overlapping_provinces():
    seen = {}
    for name, cfg in C.REGIONS.items():
        for p in cfg["provinces"]:
            assert p not in seen, f"{p} 同时属于 {seen.get(p)} 和 {name}"
            seen[p] = name


def test_splits_are_contiguous_and_ordered():
    ranges = [C.SPLITS[k] for k in ("train", "val", "test")]
    assert ranges[0][1] < ranges[1][0]
    assert ranges[1][1] < ranges[2][0]
    assert ranges[0][0] == 2000
    assert ranges[2][1] == 2025


def test_province_adjacency_is_symmetric():
    for a, nbrs in C.PROVINCE_ADJACENCY.items():
        for b in nbrs:
            assert b in C.PROVINCE_ADJACENCY, f"{b} 未定义"
            assert a in C.PROVINCE_ADJACENCY[b], f"{a}-{b} 邻接不对称"


def test_province_adjacency_has_no_self_loops():
    for a, nbrs in C.PROVINCE_ADJACENCY.items():
        assert a not in nbrs


def test_benchmark_provinces_defined_in_adjacency():
    for cfg in C.REGIONS.values():
        for p in cfg["provinces"]:
            assert p in C.PROVINCE_ADJACENCY


# --------------------------------------------------------------------------
# 特征值族解码
# --------------------------------------------------------------------------
def test_sentinel_trace_codes_cover_observed_variants():
    assert 999990.0 in C.SENTINEL_TRACE_CODES
    assert 999991.0 in C.SENTINEL_TRACE_CODES   # 实测存在的微量变体
    assert 999992.0 in C.SENTINEL_TRACE_CODES
    assert C.SENTINEL_FAMILY_MIN == 999990.0


def test_replace_sentinels_handles_trace_variants_and_family():
    from src.data.clean_and_align import _replace_sentinels

    df = pd.DataFrame({
        "PRS_Avg": [1000.0, 999999.0, 999998.0, 999995.0],
        "WIN_S_Max": [3.0, 3.0, 3.0, 3.0],
        "WIN_S_Inst_Max": [5.0, 5.0, 5.0, 5.0],
        "WIN_S_2mi_Avg": [1.0, 1.0, 1.0, 1.0],
        "WIN_S_10mi_Avg": [2.0, 2.0, 2.0, 2.0],
        "TEM_Avg": [20.0, 20.0, 20.0, 20.0],
        "TEM_Max": [25.0, 25.0, 25.0, 25.0],
        "TEM_Min": [15.0, 15.0, 15.0, 15.0],
        "PRE_Time_2020": [0.0, 999990.0, 999991.0, 999992.0],
        "GaWIN": [0.0, 0.0, 0.0, 0.0],
        "Fog": [0.0, 0.0, 0.0, 0.0],
    })
    out, counts = _replace_sentinels(df)

    # 微量变体全部转为 0.05mm 并置标志
    assert out.loc[1:3, "PRE_Time_2020"].tolist() == [0.05, 0.05, 0.05]
    assert out.loc[1:3, "PRE_trace_flag"].tolist() == [1.0, 1.0, 1.0]
    assert out.loc[0, "PRE_trace_flag"] == 0.0
    # 缺测/不观测/未知家族码 -> NaN
    assert np.isnan(out.loc[1, "PRS_Avg"])
    assert np.isnan(out.loc[2, "PRS_Avg"])
    assert np.isnan(out.loc[3, "PRS_Avg"])
    assert out.loc[0, "PRS_Avg"] == 1000.0
    # 无残留
    for col in C.CONTINUOUS_COLS:
        assert out[col].max(skipna=True) < C.SENTINEL_FAMILY_MIN
    assert counts["residual_99xxxx"] == 0


def test_interp_rows_marks_long_gaps_unobserved():
    from src.data.clean_and_align import _interp_rows

    a = np.full((1, 30), np.nan)
    a[0, :3] = [1.0, 2.0, 3.0]
    a[0, 25] = 9.0
    filled, observed = _interp_rows(a, max_gap=7)
    assert np.isfinite(filled).all(), "为保证无缺失，长期空洞也应被延拓填充"
    assert observed[0, 0] and observed[0, 2]
    assert not observed[0, 15], "距最近观测 >7 天的格点必须标记为不可信"


def test_interp_rows_handles_all_nan_row():
    from src.data.clean_and_align import _interp_rows

    a = np.full((2, 10), np.nan)
    a[1, :] = np.arange(10.0)
    filled, observed = _interp_rows(a, max_gap=7)
    assert np.isnan(filled[0]).all()
    assert not observed[0].any()
    assert observed[1].all()


# --------------------------------------------------------------------------
# Phase 1.5 契约
# --------------------------------------------------------------------------
def test_model_feature_contract_is_15_dims():
    assert len(C.FEATURE_NAMES_15) == 15
    assert C.N_FEATURES_MODEL == 15
    assert "WIN_S_10mi_Avg" not in C.FEATURE_NAMES_15
    assert len(C.FEATURE_IDX_15) == 15
    # 索引必须指向原始 16 维张量中的正确位置
    for name, idx in zip(C.FEATURE_NAMES_15, C.FEATURE_IDX_15):
        assert C.FEATURE_NAMES[idx] == name


def test_adv_physical_cols_are_all_period_available():
    assert C.ADV_PHYSICAL_COLS == ["PRS_Avg", "WIN_S_Max", "WIN_S_2mi_Avg"]
    for c in C.ADV_PHYSICAL_COLS:
        assert c in C.FEATURE_NAMES_15


def test_flood_keywords_cover_raw_and_text():
    for k in ("洪涝", "洪水", "内涝", "渍涝", "山洪", "凌汛"):
        assert k in C.FLOOD_TYPE_KEYWORDS
    for k in ("洪涝", "山洪", "内涝", "渍涝", "洪水", "漫堤", "决口", "溃坝",
              "淹没农田", "积水受淹"):
        assert k in C.FLOOD_TEXT_KEYWORDS


def test_dual_pool_constants():
    assert C.K_IN_PROV == 4
    assert C.K_CROSS_PROV == 3


def test_flood_activation_rule_from_build_benchmarks():
    from src.data.build_benchmarks import add_flood_activation

    df = pd.DataFrame({
        "disaster_type_14": ["暴雨", "暴雨", "干旱", "台风"],
        "disaster_type": ["暴雨洪涝", "强降水", "干旱", "台风"],
        "disaster_subtype": [None, None, None, "风暴潮"],
        "impact_description": ["无", "无", "河道漫堤，洪水淹没农田", "无"],
    })
    out = add_flood_activation(df)
    assert out.loc[0, "label_types"] == ["暴雨", "洪涝"]
    assert out.loc[1, "label_types"] == ["暴雨"]
    assert out.loc[2, "label_types"] == ["洪涝", "干旱"]  # 按 14 类索引排序
    assert out.loc[3, "label_types"] == ["台风"]
    assert out["flood_activated"].tolist() == [True, False, True, False]


def test_text_recovery_recovers_unique_and_ambiguous():
    from src.data.entity_align import apply_text_recovery

    stations = pd.DataFrame({
        "cnty_id": [0, 1, 2],
        "Province": ["四川省", "四川省", "四川省"],
        "City": ["德阳市", "德阳市", "绵阳市"],
        "Cnty": ["中江县", "旌阳区", "涪城区"],
    })
    df = pd.DataFrame({
        "cnty_id": [-1, -1, -1, 0],
        "align_level": ["miss", "miss", "miss", "L1_exact"],
        "province": ["四川省", "四川省", "四川省", "四川省"],
        "city": ["四川省德阳市", "四川省德阳市", "四川省绵阳市", "四川省德阳市"],
        "county": [None, None, None, "中江县"],
        "report_unit": ["四川省德阳市中江县", "四川省德阳市中江县旌阳区", "四川省绵阳市", ""],
        "impact_description": ["", "", "", ""],
    })
    out, stats = apply_text_recovery(df, stations)
    assert stats["unique"] == 1
    assert stats["multi"] == 1
    assert stats["unresolved"] == 1
    assert out.loc[0, "cnty_id"] == 0
    assert out.loc[0, "align_level"] == "L4_text_unique"
    assert out.loc[1, "align_level"] == "L4_text_multi"
    assert out.loc[1, "cnty_id"] == -1
    assert out.loc[1, "cont_loss_mask"] == 0
    assert out.loc[2, "align_level"] == "miss"
    # 长度守恒：不得因索引问题凭空增行
    assert len(out) == len(df)
