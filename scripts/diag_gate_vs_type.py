"""诊断：联合概率 gate·σ(logits) 与条件概率 σ(logits) 的 Task 1 指标差异。

因子化似然把 P(A_c)=P(相变)·P(type=c|相变)，但当前 `topo_prob` 报的是**联合**概率。
本脚本在同一个 checkpoint 上分别用联合/条件概率、各自在验证集上校准阈值，
再在测试集上评测，用来判断"门控是否在压低类型判别指标"。

    python3 scripts/diag_gate_vs_type.py --region SW_Cascade --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import common as C                                   # noqa: E402
from src.engine.data import WindowDataset, load_region             # noqa: E402
from src.engine.trainer import (TrainConfig, calibrate_thresholds,  # noqa: E402
                                topo_metrics)
from src.models.pt_gno import build_pt_gno                         # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SW_Cascade")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tin", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/diag_gate_vs_type.json")
    args = ap.parse_args(argv)

    bundle = load_region(args.region)
    ck = (Path("outputs/suite") / args.region / "PT_GNO" / "_ckpt"
          / f"{args.region}_PT_GNO" / f"s{args.seed}" / "best.pt")
    state = torch.load(ck, map_location=args.device, weights_only=False)
    model = build_pt_gno(text_dim=bundle.text_dim).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    tcfg = TrainConfig(region=args.region, batch_size=32, stride=1, tin=args.tin,
                       delta=1, amp=True, seed=args.seed)
    va = WindowDataset(bundle, "val", args.tin, 1, max(args.tin, 8))
    te = WindowDataset(bundle, "test", args.tin, 1, 1)

    out = {"region": args.region, "seed": args.seed, "ckpt": str(ck), "variants": {}}

    # 同一 checkpoint、同一阈值校准流程，只改"报哪个概率"
    def run(split_ds):
        joint, cond = [], []
        model.eval()
        from src.engine.data import build_sample, collate
        with torch.no_grad():
            for i in range(len(split_ds)):
                b = collate([build_sample(bundle, int(split_ds[i]), args.tin, 1)],
                            bundle, args.device)
                o = model(b)
                joint.append(torch.sigmoid(o["topo_logits"]).float().cpu().numpy()
                             .reshape(-1, C.NUM_DISASTER_TYPES)
                             * o["phase_gate"].float().cpu().numpy().reshape(-1, 1))
                cond.append(torch.sigmoid(o["topo_logits"]).float().cpu().numpy()
                            .reshape(-1, C.NUM_DISASTER_TYPES))
                if i == 0:
                    Y = b["topo"].cpu().numpy().reshape(-1, C.NUM_DISASTER_TYPES)
                else:
                    Y = np.concatenate([Y, b["topo"].cpu().numpy()
                                        .reshape(-1, C.NUM_DISASTER_TYPES)], 0)
        return np.concatenate(joint, 0), np.concatenate(cond, 0), Y

    jv, cv, yv = run(va)
    jt, ct, yt = run(te)

    for name, vp, tp in (("joint_gate_x_type", jv, jt), ("conditional_type", cv, ct)):
        th = calibrate_thresholds(vp, yv)
        m_all = topo_metrics(tp, yt, th, domain="all")
        m_act = topo_metrics(tp, yt, th, domain="active")
        out["variants"][name] = {"macro_f1": m_all["macro_f1"],
                                 "auprc": m_all["auprc"],
                                 "macro_f1_active": m_act["macro_f1"],
                                 "auprc_active": m_act["auprc"],
                                 "hits@3_active": m_act["hits@3"]}
        print(f"{name:20s} F1act={m_act['macro_f1']:.3f} AUPRCact={m_act['auprc']:.3f} "
              f"H@3={m_act['hits@3']:.3f} | F1={m_all['macro_f1']:.3f} "
              f"AUPRC={m_all['auprc']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
