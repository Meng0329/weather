#!/usr/bin/env python3
"""一次性审计脚本：全量扫描气象原始文件，统计 99xxxx 特征值家族的真实分布。

之前的 grep 因 timeout 被截断，结论不可靠。本脚本逐文件读取、无截断，
输出每个要素列的特征值分桶计数（按年）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "datasets" / "雷达数据"
COLS = ["PRS_Avg", "WIN_S_Max", "WIN_S_Inst_Max", "WIN_S_2mi_Avg",
        "WIN_S_10mi_Avg", "TEM_Avg", "TEM_Max", "TEM_Min", "PRE_Time_2020"]

# 特征值分桶：按千分位前缀归类
BUCKETS = [
    ("9999xx_missing_family", 999990, 1000000),
    ("9998xx_phase_fog",      999800, 999900),
    ("9997xx_phase_rain_snow",999700, 999800),
    ("9996xx_phase_snowstorm",999600, 999700),
    ("9995xx",                999500, 999600),
    ("9990-9994xx",           999000, 999500),
    ("998xxx_exceed_upper",   998000, 999000),
    ("997xxx_exceed_lower",   997000, 998000),
    ("996xxx",                996000, 997000),
    ("995xxx",                995000, 996000),
    ("990-994xxx",            990000, 995000),
]


def bucket_of(v: float) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= v < hi:
            return name
    return None


def main() -> int:
    out: dict[str, dict[str, dict[str, int]]] = {c: {} for c in COLS}
    examples: dict[str, dict[str, list]] = {c: {} for c in COLS}
    total_rows = 0

    files = sorted(RAW.glob("Year_*.csv"))
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=COLS, low_memory=False)
        total_rows += len(df)
        for c in COLS:
            v = df[c].to_numpy(dtype=np.float64)
            sel = (v >= 990000) & (v < 1000000)
            if not sel.any():
                continue
            sub = v[sel]
            yr = out[c].setdefault(f.stem, {})
            for name, lo, hi in BUCKETS:
                k = int(((sub >= lo) & (sub < hi)).sum())
                if k:
                    yr[name] = yr.get(name, 0) + k
                    ex = examples[c].setdefault(name, [])
                    if len(ex) < 5:
                        ex.extend(np.round(sub[(sub >= lo) & (sub < hi)][: 5 - len(ex)], 3).tolist())
        print(f"[{i}/{len(files)}] {f.name} rows={len(df)}", flush=True)

    summary = {
        "total_rows": total_rows,
        "per_column": out,
        "examples": examples,
    }
    Path("/tmp/sentinel_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 特征值家族总览（全量）===")
    for c in COLS:
        agg: dict[str, int] = {}
        for yr, b in out[c].items():
            for k, v in b.items():
                agg[k] = agg.get(k, 0) + v
        print(f"\n{c}:")
        if not agg:
            print("   （无 99xxxx 特征值）")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"   {k:26s} {v:>10,}   e.g. {examples[c].get(k, [])[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
