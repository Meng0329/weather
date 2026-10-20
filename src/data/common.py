"""PT-GNO 数据管线共享常量与轻量工具。

本模块是 `clean_and_align.py` / `build_benchmarks.py` / 测试的唯一契约来源。
任何特征顺序、类型映射、区域定义的改动都必须先改这里。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
RAW_DISASTER_DIR = REPO_ROOT / "datasets" / "datasets"
RAW_METEO_DIR = REPO_ROOT / "datasets" / "雷达数据"
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
DOCS_PAPER_DIR = REPO_ROOT / "docs" / "paper"

DISASTER_PART_GLOB = "disaster_records_part*.csv"
METEO_YEAR_GLOB = "Year_*.csv"

# --------------------------------------------------------------------------
# 气象数据契约
# --------------------------------------------------------------------------
KEY_COLS = ["Province", "City", "Cnty"]

# 9 项连续要素（顺序即 16 维向量的前 9 维）
CONTINUOUS_COLS = [
    "PRS_Avg",
    "WIN_S_Max",
    "WIN_S_Inst_Max",
    "WIN_S_2mi_Avg",
    "WIN_S_10mi_Avg",
    "TEM_Avg",
    "TEM_Max",
    "TEM_Min",
    "PRE_Time_2020",
]

# 2 项全时段可用二值天气现象
BINARY_COLS = ["GaWIN", "Fog"]

# 仅存在于原始 20 列、但 2016-2025 全缺，按决议剔除
DROPPED_METEO_COLS = ["Lit", "Thund", "Tord"]

# 特殊特征值
SENTINEL_MISSING = 999999.0  # 缺测
SENTINEL_NOT_OBSERVED = 999998.0  # 不观测
SENTINEL_TRACE = 999990.0  # 微量降水（实测仅出现在 PRE_Time_2020）
# 实测还存在 999991 / 999992 两个微量变体（同为"微量降水"族）
SENTINEL_TRACE_CODES = (999990.0, 999991.0, 999992.0)
# 特征值家族下界：任何 >= 该值的观测都视为特征值而非物理量
SENTINEL_FAMILY_MIN = 999990.0
# 相态/极值编码区间的下界（9996xx ~ 9998xx）。
# 全量审计（23,165,849 行）显示该区间真实存在，但**仅 2 行**：
#   Year_2003 黑龙江七台河市 999800.1（雾露霜）、Year_2002 广东潮州市 999700.1（雨夹雪）。
# 样本量不足以支撑相态解耦建模，故整体按非物理值置 NaN。
SENTINEL_PHASE_FAMILY_MIN = 999000.0
TRACE_PRECIP_MM = 0.05

# 物理合理域（含物理极值，仅用于剔除录入错误，不做业务裁剪）
# 全量扫描发现的越界样本：PRS 1496.7、WIN_S_Max 204.8、WIN_S_2mi_Avg 249830.2、
# TEM_Max 76.8、TEM_Min -99.6、PRE 3059.7，合计 371 行。
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "PRS_Avg": (300.0, 1100.0),
    "WIN_S_Max": (0.0, 120.0),
    "WIN_S_Inst_Max": (0.0, 150.0),
    "WIN_S_2mi_Avg": (0.0, 120.0),
    "WIN_S_10mi_Avg": (0.0, 120.0),
    "TEM_Avg": (-70.0, 60.0),
    "TEM_Max": (-70.0, 60.0),
    "TEM_Min": (-70.0, 60.0),
    "PRE_Time_2020": (0.0, 2000.0),
}

# 插值可信窗口：距离最近真实观测超过该天数的格点，
# 数值仍被线性延拓填充（保证张量无缺失），但 observed_mask 置 False。
MAX_INTERP_GAP_DAYS = 7

# 16 维物理特征向量的固定顺序 —— 模型输入契约，不得重排
FEATURE_NAMES = [
    "PRS_Avg",
    "WIN_S_Max",
    "WIN_S_Inst_Max",
    "WIN_S_2mi_Avg",
    "WIN_S_10mi_Avg",
    "TEM_Avg",
    "TEM_Max",
    "TEM_Min",
    "PRE_Time_2020",
    "GaWIN",
    "Fog",
    "PRE_trace_flag",
    "rain_snow_phase",
    "gust_factor",
    "tem_diurnal_range",
    "delta_prs_24h",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 16, N_FEATURES

# --------------------------------------------------------------------------
# 灾情数据契约
# --------------------------------------------------------------------------
# 读取时立即丢弃的个人隐私列（决议 §6.1-2）
PRIVACY_COLS = ["tabulator", "contact"]

DISASTER_USECOLS = [
    "record_id",
    "disaster_type",
    "disaster_type_std",
    "disaster_subtype",
    "start_time",
    "end_time",
    "report_year",
    "province",
    "city",
    "county",
    "report_unit",
    "economic_loss",
    "loss_level",
    "deaths",
    "missing",
    "injuries",
    "affected_population",
    "relocated",
    "drinking_water_difficulty",
    "collapsed_houses",
    "damaged_houses",
    "affected_area",
    "disaster_area",
    "total_loss_area",
    "grain_loss",
    "schools_closed",
    "factories_stopped",
    "affected_crops",
    "impact_description",
    "weather_summary",
    "warning_info",
    "remarks",
    "report_status",
    "report_time",
    "search_text",
]

assert not set(DISASTER_USECOLS) & set(PRIVACY_COLS)

# 经济损失量纲修正阈值：>= 1e8 万元视为"元误填为万元"
ECONOMIC_LOSS_FIX_THRESHOLD = 1e8
ECONOMIC_LOSS_FIX_DIVISOR = 1e4

# 14 类标准灾害（data_dictionary.md 定义）
DISASTER_TYPES_14 = [
    "暴雨",
    "洪涝",
    "台风",
    "干旱",
    "高温",
    "低温",
    "雪灾",
    "大风",
    "冰雹",
    "雷电",
    "沙尘暴",
    "雾霾",
    "连阴雨",
    "地质灾害",
]
DISASTER_TYPE_TO_IDX = {t: i for i, t in enumerate(DISASTER_TYPES_14)}
NUM_DISASTER_TYPES = len(DISASTER_TYPES_14)
assert NUM_DISASTER_TYPES == 14

# 24 -> 14 确定性映射（决议 §6.3）。None 表示丢弃。
DISASTER_TYPE_MAP_24_TO_14 = {
    "暴雨": "暴雨",
    "冰雹": "冰雹",
    "大风": "大风",
    "干旱": "干旱",
    "台风": "台风",
    "雪灾": "雪灾",
    "雷电": "雷电",
    "地质灾害": "地质灾害",
    "低温": "低温",
    "冻害": "低温",  # 归并
    "未知": None,  # 丢弃：占位符
    "雾霾": "雾霾",
    "森林草原火灾": "干旱",  # 归并：干热驱动
    "连阴雨": "连阴雨",
    "高温": "高温",
    "其他": None,  # 丢弃：占位符
    "沙尘暴": "沙尘暴",
    "气象有关的谣传和不当舆论": None,  # 丢弃：非灾害
    "大气污染": "雾霾",  # ⚠️ 判断项，可一键改判为 None
    "干热风": "高温",  # 归并：热驱动
    "气象部门参与减轻事件": None,  # 丢弃：行政事件
    "凌汛": "洪涝",  # 归并
    "风暴潮": "台风",  # 归并
    "危险物质泄漏(气体、核应急)": None,  # 丢弃：非自然气象灾害
}

# 损失等级 -> 序号
LOSS_LEVEL_MAP = {"无损失": 0, "低": 1, "中": 2, "高": 3, "特大": 4}
LOSS_LEVEL_NAMES = ["无损失", "低", "中", "高", "特大"]

# 报告阶段（Task 3 数据同化）
REPORT_STAGES = ["初报", "续报", "终报"]
REPORT_STAGE_ORDER = {s: i for i, s in enumerate(REPORT_STAGES)}

# 事件链聚合：同一 (省, 区县, 灾害类型) 下，相邻报告起始间隔 <= 该天数视为同一条链
EVENT_CHAIN_GAP_DAYS = 3

# 连续损失回归目标（填充率 100% 的核心字段）
CONTINUOUS_LOSS_TARGETS = [
    "economic_loss",
    "affected_population",
    "collapsed_houses",
    "affected_area",
    "disaster_area",
    "total_loss_area",
]

# --------------------------------------------------------------------------
# 时间划分（严格时序，禁止未来信息泄露）
# --------------------------------------------------------------------------
SPLITS = {
    "train": (2000, 2018),
    "val": (2019, 2021),
    "test": (2022, 2025),
}
DATE_START = "2000-01-01"
DATE_END = "2025-12-31"

# --------------------------------------------------------------------------
# 三大区域基准
# --------------------------------------------------------------------------
REGIONS = {
    "SW_Cascade": {
        "label": "SW-Cascade",
        "provinces": ["云南省", "四川省", "贵州省", "重庆市"],
        "focus": "暴雨→洪涝→地质灾害链，人员伤亡与安置转移",
    },
    "SE_Typhoon": {
        "label": "SE-Typhoon",
        "provinces": ["广东省", "福建省", "浙江省", "海南省", "江苏省"],
        "focus": "台风、大风、暴雨，房屋倒损与社会停摆",
    },
    "North_Agro": {
        "label": "North-Agro",
        "provinces": ["河南省", "山东省", "河北省", "黑龙江省", "吉林省", "辽宁省"],
        "focus": "旱涝急转与强对流，农业三面积与粮食损失",
    },
}

# --------------------------------------------------------------------------
# 空间邻接（决议 §6.1-6）
# --------------------------------------------------------------------------
# 物理近邻候选池：同省 + 陆上邻省。省级邻接为稳定地理事实，非外部数据文件。
PROVINCE_ADJACENCY = {
    "北京市": ["天津市", "河北省"],
    "天津市": ["北京市", "河北省"],
    "河北省": ["北京市", "天津市", "山西省", "内蒙古自治区", "辽宁省", "山东省", "河南省"],
    "山西省": ["河北省", "内蒙古自治区", "陕西省", "河南省"],
    "内蒙古自治区": [
        "黑龙江省", "吉林省", "辽宁省", "河北省",
        "山西省", "陕西省", "宁夏回族自治区", "甘肃省",
    ],
    "辽宁省": ["吉林省", "内蒙古自治区", "河北省"],
    "吉林省": ["黑龙江省", "辽宁省", "内蒙古自治区"],
    "黑龙江省": ["吉林省", "内蒙古自治区"],
    "上海市": ["江苏省", "浙江省"],
    "江苏省": ["上海市", "浙江省", "安徽省", "山东省"],
    "浙江省": ["上海市", "江苏省", "安徽省", "江西省", "福建省"],
    "安徽省": ["江苏省", "浙江省", "江西省", "湖北省", "河南省", "山东省"],
    "福建省": ["浙江省", "江西省", "广东省"],
    "江西省": ["浙江省", "安徽省", "湖北省", "湖南省", "广东省", "福建省"],
    "山东省": ["河北省", "河南省", "安徽省", "江苏省"],
    "河南省": ["河北省", "山西省", "陕西省", "湖北省", "安徽省", "山东省"],
    "湖北省": ["河南省", "安徽省", "江西省", "湖南省", "重庆市", "陕西省"],
    "湖南省": ["湖北省", "江西省", "广东省", "广西壮族自治区", "贵州省", "重庆市"],
    "广东省": ["福建省", "江西省", "湖南省", "广西壮族自治区", "海南省"],
    "广西壮族自治区": ["广东省", "湖南省", "贵州省", "云南省"],
    "海南省": ["广东省"],
    "重庆市": ["四川省", "陕西省", "湖北省", "湖南省", "贵州省"],
    "四川省": [
        "重庆市", "贵州省", "云南省", "西藏自治区",
        "青海省", "甘肃省", "陕西省",
    ],
    "贵州省": ["重庆市", "四川省", "云南省", "广西壮族自治区", "湖南省"],
    "云南省": ["四川省", "贵州省", "广西壮族自治区", "西藏自治区"],
    "西藏自治区": ["新疆维吾尔自治区", "青海省", "四川省", "云南省"],
    "陕西省": [
        "内蒙古自治区", "山西省", "河南省", "湖北省",
        "重庆市", "四川省", "甘肃省", "宁夏回族自治区",
    ],
    "甘肃省": [
        "内蒙古自治区", "宁夏回族自治区", "陕西省",
        "四川省", "青海省", "新疆维吾尔自治区",
    ],
    "青海省": ["甘肃省", "新疆维吾尔自治区", "西藏自治区", "四川省"],
    "宁夏回族自治区": ["内蒙古自治区", "陕西省", "甘肃省"],
    "新疆维吾尔自治区": ["甘肃省", "青海省", "西藏自治区"],
    # 港澳台无陆上邻省数据，保留空表以免 KeyError
    "香港特别行政区": [],
    "澳门特别行政区": [],
    "台湾省": [],
}

PHYSICAL_NEIGHBOR_K = 6  # 每个节点的物理近邻出边数
ADV_WEIGHT_PRS = 0.5  # delta_prs_24h 相关系数权重
ADV_WEIGHT_TEM = 0.5  # TEM_Avg 相关系数权重

# --------------------------------------------------------------------------
# 派生特征工具
# --------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def build_16_features(df):
    """由已清洗/插值的连续列构造 16 维物理特征。

    参数
    ----
    df : pandas.DataFrame
        必须已含 `CONTINUOUS_COLS`、`BINARY_COLS` 与 `PRE_trace_flag`（0/1）。

    返回
    ----
    numpy.ndarray, shape (n, 16), dtype float32, 列顺序严格等于 `FEATURE_NAMES`。
    """
    pre = df["PRE_Time_2020"].to_numpy(dtype=np.float64)
    tem_avg = df["TEM_Avg"].to_numpy(dtype=np.float64)
    tem_max = df["TEM_Max"].to_numpy(dtype=np.float64)
    tem_min = df["TEM_Min"].to_numpy(dtype=np.float64)
    win_max = df["WIN_S_Max"].to_numpy(dtype=np.float64)
    win_inst = df["WIN_S_Inst_Max"].to_numpy(dtype=np.float64)

    # (a) 固液降水热力学相态：气温低于 0.5℃ 时降水更倾向固态
    rain_snow_phase = (pre > 0).astype(np.float64) * sigmoid(-(tem_avg - 0.5))
    # (b) 阵风脉动系数
    with np.errstate(invalid="ignore", divide="ignore"):
        gust_factor = win_inst / (win_max + 0.1)
    gust_factor[~np.isfinite(gust_factor)] = np.nan
    # (c) 日较差
    tem_diurnal_range = tem_max - tem_min
    # (d) 24 小时变压由上游步骤写入 df['delta_prs_24h']

    cols = [
        "PRS_Avg", "WIN_S_Max", "WIN_S_Inst_Max", "WIN_S_2mi_Avg",
        "WIN_S_10mi_Avg", "TEM_Avg", "TEM_Max", "TEM_Min",
        "PRE_Time_2020", "GaWIN", "Fog", "PRE_trace_flag",
    ]
    mat = [df[c].to_numpy(dtype=np.float64) for c in cols]
    mat += [
        rain_snow_phase,
        gust_factor,
        tem_diurnal_range,
        df["delta_prs_24h"].to_numpy(dtype=np.float64),
    ]
    return np.column_stack(mat).astype(np.float32)


# --------------------------------------------------------------------------
# 行政区划名归一化（层级剥离 + 词干匹配）
# --------------------------------------------------------------------------
PROVINCE_SUFFIXES = [
    "维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区",
    "自治区", "省", "市",
]

COUNTY_SUFFIXES = [
    "哈萨克自治县", "蒙古族自治县", "满族自治县", "回族自治县",
    "苗族自治县", "彝族自治县", "藏族自治县", "土家族自治县",
    "自治县", "自治旗", "联合旗", "林区", "特区",
    "县", "区", "市", "旗", "盟",
]


def norm_province(name: str) -> str:
    """省级名归一化：广西壮族自治区 -> 广西；内蒙古自治区 -> 内蒙古。"""
    if not isinstance(name, str) or not name:
        return ""
    s = name.strip()
    for suf in PROVINCE_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def county_stem(name: str) -> str:
    """区县名词干化：中江县 -> 中江；徐汇区 -> 徐汇。"""
    if not isinstance(name, str) or not name:
        return ""
    s = name.strip()
    for suf in COUNTY_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s
