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


# --------------------------------------------------------------------------
# 市级记录文本二次下探（Phase 1.5 决议 `docs/experiment_protocol.md` §3.4）
# --------------------------------------------------------------------------
_TEXT_SEARCH_COLS = ("report_unit", "impact_description")
# 词干匹配的最小长度：过短的词干（如"城"）会大量误命中
_MIN_STEM_FOR_TEXT = 3


def build_city_node_map(stations: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, str, str]]]:
    """(省名归一, 地市裸名) -> [(cnty_id, 区县名, 词干), ...]。"""
    out: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    for cid, p, c, t in zip(
        stations["cnty_id"], stations["Province"], stations["City"], stations["Cnty"]
    ):
        pn = norm_province(p)
        for city_key in {str(c), county_stem(str(c))}:
            out[(pn, city_key)].append((int(cid), str(t), county_stem(str(t))))
    # 去重
    for k, v in out.items():
        out[k] = sorted(set(v))
    return out


def recover_county_from_text(
    province: str,
    city: str,
    report_unit: str,
    impact_description: str,
    city_map: dict,
) -> list[int]:
    """在 report_unit / impact_description 中检索该地市下辖的区县名。

    返回候选 cnty_id 列表（去重、保持稳定顺序）。
    匹配规则：完整区县名命中，或长度 >= 3 的词干命中。
    """
    _, city_bare, _ = split_county_string(province, city, "")
    pn = norm_province(province)
    candidates = city_map.get((pn, str(city_bare)), [])
    if not candidates:
        candidates = city_map.get((pn, county_stem(str(city_bare))), [])
    if not candidates:
        return []

    texts = [str(report_unit or ""), str(impact_description or "")]
    hits: list[int] = []
    for cid, cnty, stem in candidates:
        for txt in texts:
            if not txt:
                continue
            if cnty in txt or (len(stem) >= _MIN_STEM_FOR_TEXT and stem in txt):
                hits.append(cid)
                break
    # 稳定去重
    seen, out = set(), []
    for c in hits:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def apply_text_recovery(
    df: pd.DataFrame, stations: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """对未对齐记录做文本二次下探。

    写回三类结果：
      - 唯一命中：cnty_id = 该节点，align_level = "L4_text_unique"，cont_loss_mask = 1
      - 多命中  ：cnty_id = -1，cnty_candidates 记录候选，align_level = "L4_text_multi"
                  （仅激活拓扑标签，连续损失掩码为 0）
      - 仍无命中：align_level = "miss"
    """
    df = df.copy()
    if "cnty_candidates" not in df.columns:
        df["cnty_candidates"] = ""
    # 关键：上游过滤会留下带空洞的索引，必须重置，否则位置索引与标签不一致
    df = df.reset_index(drop=True)
    city_map = build_city_node_map(stations)

    miss_mask = df["cnty_id"] < 0
    stats = {"n_miss_before": int(miss_mask.sum()), "unique": 0, "multi": 0, "unresolved": 0}

    idxs = np.nonzero(miss_mask.to_numpy())[0]
    prov = df["province"].to_numpy()
    city = df["city"].to_numpy()
    unit = df["report_unit"].to_numpy()
    desc = df["impact_description"].to_numpy()

    # 用数组累积结果，最后整体写回，避免 .at / .iat 的逐元素开销与索引陷阱
    cand_str = df["cnty_candidates"].to_numpy().astype(object)
    new_id = df["cnty_id"].to_numpy().copy()
    new_level = df["align_level"].to_numpy().astype(object)

    for i in idxs:
        cands = recover_county_from_text(prov[i], city[i], unit[i], desc[i], city_map)
        if not cands:
            stats["unresolved"] += 1
            continue
        cand_str[i] = ",".join(str(c) for c in cands)
        if len(cands) == 1:
            new_id[i] = int(cands[0])
            new_level[i] = "L4_text_unique"
            stats["unique"] += 1
        else:
            new_level[i] = "L4_text_multi"
            stats["multi"] += 1

    df["cnty_candidates"] = cand_str
    df["cnty_id"] = new_id
    df["align_level"] = new_level
    df["cont_loss_mask"] = (df["cnty_id"] >= 0).astype(np.int8)
    # 唯一命中但无候选列（L1-L3 命中）也补上自身
    base = df["cnty_candidates"] == ""
    df.loc[base & (df["cnty_id"] >= 0), "cnty_candidates"] = (
        df.loc[base & (df["cnty_id"] >= 0), "cnty_id"].astype(str)
    )
    stats["n_aligned_after"] = int((df["cnty_id"] >= 0).sum())
    return df, stats