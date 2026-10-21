"""多步提前期评测：Δ ∈ {1,3,5,7} 天的性能衰减。

对同一 combo（主模型 / 缝合基线），用各自 Δ 重新训练会混入训练方差；
这里采用**单模型多提前期推断**：加载 Δ=1 训练的模型，把预测目标的跨度
分别设为 1/3/5/7 天重新评测（输入窗口不变），从而只度量"时间外推距离"
带来的衰减，而不是重新训练带来的噪声。

    python3 scripts/run_multihorizon.py --region SW_Cascade --deltas 1 3 5 7
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

from src.data import common as C                                  # noqa: E402
from src.engine.data import WindowDataset, load_region            # noqa: E402
from src.engine.trainer import (_class_weight, _collect, _topo_pos_weight,  # noqa: E402
                                LossConfig, TrainConfig, evaluate_preds)
from src.models.ablations import build_ablation                   # noqa: E402
from src.models.baselines import build_baseline                   # noqa: E402
from src.models.pt_gno import build_pt_gno                        # noqa: E402


def make(kind: str, text_dim: int):
    if kind == "main":
        return build_pt_gno(text_dim=text_dim)
    if kind.startswith("w/o"):
        return build_ablation(kind, text_dim=text_dim)
    return build_baseline(kind, text_dim=text_dim)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SW_Cascade")
    ap.add_argument("--models", nargs="*",
                    default=["PT_GNO", "Stitched-GNN-LLM"])
    ap.add_argument("--deltas", nargs="*", type=int, default=[1, 3, 5, 7])
    ap.add_argument("--tin", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-root", default="outputs/suite",
                    help="套件根目录；按 <root>/<region>/<model>/_ckpt/<region>_<model>/s<seed>/best.pt 解析")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/multihorizon.json")
    args = ap.parse_args(argv)

    bundle = load_region(args.region)
    res = []
    for name in args.models:
        kind = "main" if name == "PT_GNO" else name
        safe = name.replace("/", "_")
        base = Path(args.ckpt_root) / args.region / safe / "_ckpt"
        # 目录名用模型的 label（主模型=PT_GNO，基线=类名，如 SW_Cascade_StitchedGNNLLM），
        # 直接用注册名会找不到。
        cands = []
        if base.exists():
            cands += [p / f"s{args.seed}" / "best.pt" for p in base.iterdir() if p.is_dir()]
            cands += [p / "best.pt" for p in base.iterdir() if p.is_dir()]
        ck = next((c for c in cands if c.exists()), None)
        if ck is None:
            print(f"跳过 {name}：{base} 下无 best.pt")
            continue
        model = make(kind, bundle.text_dim)
        state = torch.load(ck, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(args.device).eval()
        # 阈值必须来自 checkpoint：Task 1 的 F1 是逐类校准阈值算的，
        # 用默认 0.5 会让长尾类全部判负（实测 F1act 恒为 0.000）。
        th = state.get("thresholds")
        th = None if th is None else np.asarray(th, dtype=np.float32)
        pw = _topo_pos_weight(bundle).to(args.device)
        cw = _class_weight(bundle).to(args.device)

        for delta in args.deltas:
            ds = WindowDataset(bundle, "test", args.tin, delta, max(delta, 2))
            tcfg = TrainConfig(region=args.region, batch_size=32, stride=1,
                               tin=args.tin, delta=delta, amp=True)
            with torch.no_grad():
                _, pred = _collect(model, ds, bundle, tcfg, args.device, False, 0,
                                   pw, cw, LossConfig())
            m = evaluate_preds(pred, thresholds=th)
            m.pop("prob", None)
            row = {"region": args.region, "model": name, "delta": delta,
                   "macro_f1_active": m["macro_f1_active"],
                   "auprc_active": m["auprc_active"],
                   "hits@3_active": m["hits@3_active"],
                   "mrr_active": m["mrr_active"],
                   "log_mae": m["log_mae"], "r2": m["r2"]}
            res.append(row)
            print(f"{name:20s} Δ={delta}: F1act={row['macro_f1_active']:.3f} "
                  f"AUPRC={row['auprc_active']:.3f} H@3={row['hits@3_active']:.3f} "
                  f"R2={row['r2']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())