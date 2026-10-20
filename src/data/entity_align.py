"""灾情区县 -> 气象站点单元 的层级剥离与词干匹配。

灾情表 `county` 是完整层级串（如 "四川省德阳市中江县"），
气象表 `(Province, City, Cnty)` 是裸名（如 "四川省 / 德阳市 / 中江县"）。
直接字符串 join 会大面积失配，因此分三级回退：

  L1 精确三元组：灾情 (province, city, county) 剥离前缀后与气象单元完全一致
  L2 省内词干匹配：(省, 县名词干) 唯一命中
  L3 报告单位兜底：用 `report_unit` 重复 L1/L2

未命中记录不丢弃，标记 `cnty_id = -1` 并计数，供数据统计表披露。
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from src.data.common import county_stem, norm_province


def _strip_prefix(full: str, prefix: str) -> str:
    """去掉 `full` 开头的 `prefix`（含原名与归一化后两种形式）。

    规则：
      - 原名命中且会剥成空串（直辖市场景，City="重庆市"、Province="重庆市"）→ 保留原值；
      - 原名未命中时才尝试归一化省名（"广西壮族自治区" -> "广西"），
        避免 "重庆市" 被归一化成 "广西" 式的误剥离（"重庆市" -> "市"）。
    """
    if not isinstance(full, str) or not isinstance(prefix, str) or not prefix:
        return full
    if full.startswith(prefix):
        return full[len(prefix):] if len(full) > len(prefix) else full
    np_ = norm_province(prefix)
    if np_ and np_ != prefix and full.startswith(np_) and len(full) > len(np_):
        return full[len(np_):]
    return full


def split_county_string(
    province: str, city: str, county: str
) -> tuple[str, str, str]:
    """把灾情/报告单位的层级串拆成 (省, 市, 县) 裸名。

    输入形如 ("四川省", "四川省德阳市", "四川省德阳市中江县")。
    若 `city`/`county` 本身已是裸名，剥离无效果，原样返回。
    """
    prov = province if isinstance(province, str) else ""
    city_raw = city if isinstance(city, str) else ""
    county_raw = county if isinstance(county, str) else ""

    city_rest = _strip_prefix(city_raw, prov)
    county_rest = _strip_prefix(county_raw, prov)
    # 再从区县串里剥掉市级前缀
    county_rest = _strip_prefix(county_rest, city_raw)
    county_rest = _strip_prefix(county_rest, city_rest)

    # 若 city 为空但 county 是层级串，尝试从 county 里切出中间部分
    if not city_rest and county_rest:
        m = re.match(r"^(.+?(?:市|自治州|地区|盟))(.+)$", county_rest)
        if m:
            city_rest, county_rest = m.group(1), m.group(2)
    return prov, city_rest, county_rest


class StationIndex:
    """气象站点单元索引。节点顺序即模型节点索引，必须稳定。"""

    def __init__(self, stations: pd.DataFrame):
        """stations: 含 Province / City / Cnty 三列、每行一个唯一单元的 DataFrame。"""
        st = stations[["Province", "City", "Cnty"]].drop_duplicates().reset_index(drop=True)
        st = st.sort_values(["Province", "City", "Cnty"]).reset_index(drop=True)
        self.stations = st
        self.province = st["Province"].to_numpy()
        self.city = st["City"].to_numpy()
        self.cnty = st["Cnty"].to_numpy()
        self.n_nodes = len(st)

        self._exact: dict[tuple[str, str, str], int] = {}
        self._by_prov_stem: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_prov_city_stem: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self._by_prov_stem_raw: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_stem: dict[str, list[int]] = defaultdict(list)
        self._by_city_stem: dict[tuple[str, str], list[int]] = defaultdict(list)

        for i, (p, c, t) in enumerate(zip(self.province, self.city, self.cnty)):
            pn, cn, tn = norm_province(p), str(c), str(t)
            stem = county_stem(tn)
            self._exact[(p, c, t)] = i
            self._exact.setdefault((pn, cn, tn), i)
            self._by_prov_stem[(pn, stem)].append(i)
            self._by_prov_city_stem[(pn, cn, stem)].append(i)
            self._by_prov_stem_raw[(p, stem)].append(i)
            self._by_stem[stem].append(i)
            self._by_city_stem[(county_stem(cn), stem)].append(i)

    # ------------------------------------------------------------------
    def match(
        self, province: str, city: str, county: str, report_unit: str | None = None
    ) -> tuple[int, str]:
        """返回 (节点索引, 匹配级别)。未命中返回 (-1, 'miss')。"""
        for field_value in (county, report_unit):
            if not isinstance(field_value, str) or not field_value.strip():
                continue
            p, c, t = split_county_string(province, city, field_value)
            if not t:
                continue
            pn = norm_province(p)

            # L1a：原始三元组精确命中
            key = (p, c, t)
            if key in self._exact:
                return self._exact[key], "L1_exact"

            # L1b：归一化后三元组精确命中
            key = (pn, c, t)
            if key in self._exact:
                return self._exact[key], "L1_exact_norm"

            stem = county_stem(t)

            # L2a：(省, 市, 词干) 唯一
            cands = self._by_prov_city_stem.get((pn, c, stem), [])
            if len(cands) == 1:
                return cands[0], "L2_prov_city_stem"

            # L2b：(省, 词干) 唯一
            cands = self._by_prov_stem.get((pn, stem), [])
            if len(cands) == 1:
                return cands[0], "L2_prov_stem"

            # L2c：原始省名 + 词干
            cands = self._by_prov_stem_raw.get((p, stem), [])
            if len(cands) == 1:
                return cands[0], "L2_prov_stem_raw"

            # L2d：多候选时按城市名再筛一次
            if cands:
                same_city = [i for i in cands if str(self.city[i]) == c]
                if len(same_city) == 1:
                    return same_city[0], "L2_prov_stem_city_tiebreak"

            # ---- L3 全局回退：省缺失/写错时，用全国唯一词干或城市词干消歧 ----
            cands = self._by_city_stem.get((county_stem(c), stem), []) if c else []
            if len(cands) == 1:
                return cands[0], "L3_city_stem"

            cands = self._by_stem.get(stem, [])
            if len(cands) == 1 and len(stem) >= 2:
                return cands[0], "L3_global_stem"

        return -1, "miss"

    # ------------------------------------------------------------------
    def align_series(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        """批量对齐。返回 (cnty_id, level_code, level_label)。"""
        ids = np.full(len(df), -1, dtype=np.int64)
        levels = np.empty(len(df), dtype=object)
        prov = df["province"].to_numpy() if "province" in df else np.array([""] * len(df))
        city = df["city"].to_numpy() if "city" in df else np.array([""] * len(df))
        cnty = df["county"].to_numpy() if "county" in df else np.array([""] * len(df))
        unit = df["report_unit"].to_numpy() if "report_unit" in df else np.array([""] * len(df))

        for i in range(len(df)):
            idx, lvl = self.match(prov[i], city[i], cnty[i], unit[i])
            ids[i] = idx
            levels[i] = lvl
        return ids, levels, pd.Series(levels)


def alignment_report(levels: pd.Series) -> dict:
    """汇总命中率。"""
    total = int(len(levels))
    counts = levels.value_counts().to_dict()
    hits = total - counts.get("miss", 0)
    return {
        "total": total,
        "hits": hits,
        "miss": counts.get("miss", 0),
        "hit_rate": round(hits / total, 6) if total else 0.0,
        "by_level": {str(k): int(v) for k, v in sorted(counts.items())},
    }