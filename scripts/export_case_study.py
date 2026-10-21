"""导出可解释性案例：一次真实「暴雨→洪涝→地质灾害」级联的相变轨迹。

输出 `docs/paper/case_study_phase_portrait.json`，含灾害发生前后 10 天内：

- `norm_U(t)`   : 连续气象势能场的状态范数（RMS）
- `tau(t)`      : LLM 本构泛函推断的局地临界阈值
- `gamma(t)`    : 防灾耗散率（均值，跨 d 维）
- `phase_gate(t)`: 朗道门控值
- `topo_prob(t)`: 14 类灾害激活概率
- 该节点当日的真实灾种标签（用于对照）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import common as C                      # noqa: E402
from src.engine.data import (build_sample, collate,   # noqa: E402
                             load_region)
from src.models.pt_gno import build_pt_gno            # noqa: E402


def pick_case(bundle, records: pd.DataFrame):
    """在测试集里找一条真实的「暴雨 → 洪涝 → 地质灾害」级联。

    优先在 `daily_labels` 张量上直接检索三元链（暴雨+洪涝同日或相邻日激活，
    且 3 天内出现地质灾害）；找不到再退化为"暴雨+洪涝"两元链，
    最后退化为任意双灾种。返回 (node, event_day)。
    """
    y0, y1 = C.SPLITS["test"]
    years = bundle.dates.year
    lo = int(np.nonzero(years >= y0)[0][0]) + 7
    hi = int(np.nonzero(years <= y1)[0][-1]) - 4
    idx = {t: i for i, t in enumerate(C.DISASTER_TYPES_14)}
    a, b, c = idx["暴雨"], idx["洪涝"], idx["地质灾害"]
    # labels 是 (N, T, 14)，按天切片要在**轴 1**；写成 [lo:hi] 会切节点轴
    lab = np.asarray(bundle.labels[:, lo:hi, :])
    assert lab.shape[0] == bundle.n_nodes, lab.shape

    storm = lab[..., a] > 0
    flood = lab[..., b] > 0
    geo = lab[..., c] > 0
    # 3 天窗口内出现的地质灾害（含当日）
    geo3 = geo.copy()
    for k in (1, 2, 3):
        geo3[:, :-k] |= geo[:, k:]
    for require in ((storm & flood & geo3), (storm & flood), (storm & geo)):
        nodes, days = np.nonzero(require)
        if len(days):
            # 取级联最"密集"的节点（该节点在此窗口内命中的天数最多）
            uniq, cnt = np.unique(nodes, return_counts=True)
            best = int(uniq[int(np.argmax(cnt))])
            d = int(days[int(np.argmax(nodes == best))])
            return best, bundle.dates[lo + d]


def export(model, bundle, node: int, event_day: pd.Timestamp, window: int = 10,
           tin: int = 7, device: str = "cuda") -> dict:
    day_of = pd.Series(np.arange(bundle.n_days), index=bundle.dates)
    d0 = int(day_of.loc[event_day])
    lo, hi = max(tin - 1, d0 - window), min(bundle.n_days - 1, d0 + window)
    rows = []
    model.eval()
    with torch.no_grad():
        for d in range(lo, hi + 1):
            batch = collate([build_sample(bundle, d, tin, 1)], bundle, device)
            out = model(batch)
            i = node
            rows.append({
                "date": str(bundle.dates[d].date()),
                "rel_day": int(d - d0),
                "norm_U": float(out["U"][0, i].pow(2).mean().sqrt()),
                "tau": float(out["tau"][0, i]),
                "gamma": float(out["gamma"][0, i].mean()),
                "gamma_dim_mean": float(out["gamma"][0, i].mean()),
                "phase_gate": float(out["phase_gate"][0, i]),
                "topo_prob": {name: float(out["topo_prob"][0, i, k])
                              for k, name in enumerate(C.DISASTER_TYPES_14)},
                "label_active": [name for k, name in enumerate(C.DISASTER_TYPES_14)
                                 if bundle.labels[i, d, k] > 0],
                # build_sample(day=d) 的预测目标是 d+1，这里单列保存以免混淆
                "label_next": [name for k, name in enumerate(C.DISASTER_TYPES_14)
                               if d + 1 < bundle.n_days and bundle.labels[i, d + 1, k] > 0],
            })
    return {
        "region": bundle.name,
        "node_idx": int(node),
        "node": {k: str(v) for k, v in bundle.nodes.iloc[node][["Province", "City", "Cnty"]].items()},
        "event_date": str(event_day.date()),
        "window_days": window,
        "series": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SW_Cascade")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/paper/case_study_phase_portrait.json")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.ckpt is None:
        args.ckpt = str(Path("outputs/suite") / args.region / "PT_GNO" / "_ckpt"
                        / f"{args.region}_PT_GNO" / f"s{args.seed}" / "best.pt")
    bundle = load_region(args.region)
    node, day = pick_case(bundle, bundle.records)
    if node is None:
        print("未找到符合条件的级联案例")
        return 1
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = build_pt_gno(text_dim=bundle.text_dim).to(args.device)
    model.load_state_dict(ck["model"])
    payload = export(model, bundle, node, day, window=args.window, device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"案例已写出：{args.out}（{payload['node']} {payload['event_date']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())