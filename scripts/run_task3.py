"""Phase 3 主实验三：初报 → 续报 → 终报 连续时间数据同化。

对比 `PT-GNO`（有同化门）与 `w/o Assimilation & RAG Prior`（无同化门），
报告 Stage 0/1/2 的 log-MAE、RMSE、R² 与误差衰减率 ERR%。

    python3 scripts/run_task3.py --region SW_Cascade \
        --ckpt-main outputs/suite/_ckpt/SW_Cascade_PT_GNO/best.pt \
        --ckpt-wo   outputs/suite/_ckpt/SW_Cascade_w/o Assimilation & RAG Prior/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import common as C                       # noqa: E402
from src.engine.data import load_region                # noqa: E402
from src.engine.task3 import (build_assim_samples,     # noqa: E402
                              evaluate_assimilation)
from src.models.ablations import build_ablation        # noqa: E402
from src.models.pt_gno import build_pt_gno             # noqa: E402


def load_model(kind: str, ckpt: str, text_dim: int, device: str):
    if kind == "main":
        m = build_pt_gno(text_dim=text_dim)
    else:
        m = build_ablation(kind, text_dim=text_dim)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model"])
    return m.to(device)


def _resolve(root: str, region: str, name: str, seed: int = 42) -> str:
    safe = name.replace("/", "_")
    cands = [
        Path(root) / region / safe / "_ckpt" / f"{region}_{name}" / f"s{seed}" / "best.pt",
        Path(root) / region / safe / "_ckpt" / f"{region}_{name}" / "best.pt",
        Path(root) / "_ckpt" / f"{region}_{name}" / "best.pt",
        Path(root) / region / safe / "best.pt",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return str(cands[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SW_Cascade")
    ap.add_argument("--ckpt-main", default=None)
    ap.add_argument("--ckpt-wo", default=None)
    ap.add_argument("--ckpt-root", default="outputs/suite")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tin", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/task3")
    args = ap.parse_args(argv)

    WO_NAME = "w/o Assimilation & RAG Prior"
    args.ckpt_main = args.ckpt_main or _resolve(args.ckpt_root, args.region, "PT_GNO", args.seed)
    args.ckpt_wo = args.ckpt_wo or _resolve(args.ckpt_root, args.region, WO_NAME, args.seed)

    bundle = load_region(args.region)
    golden = pd.read_parquet(Path(C.PROCESSED_DIR) / "assimilation_golden_subset.parquet")
    samples = build_assim_samples(bundle, golden, tin=args.tin)
    print(f"{args.region}: 可评测同化样本 {len(samples)} 条", flush=True)

    out = {"region": args.region, "n_samples": len(samples), "models": {}}
    for name, kind, ckpt in (("PT-GNO", "main", args.ckpt_main),
                             (WO_NAME, WO_NAME, args.ckpt_wo)):
        if not Path(ckpt).exists():
            print(f"跳过 {name}：checkpoint 不存在 {ckpt}")
            continue
        model = load_model(kind, ckpt, bundle.text_dim, args.device)
        res = evaluate_assimilation(model, bundle, golden, tin=args.tin,
                                    device=args.device)
        out["models"][name] = res
        print(f"\n== {name} (n={res['n_samples']})")
        for st in res["stages"]:
            print(f"  Stage {st['stage']}: log-MAE={st['log_mae']:.4f} "
                  f"RMSE={st['rmse']:.4f} R2={st['r2']:.4f} "
                  f"level_acc={st['level_acc']:.3f} ERR={st['err_reduction_pct']:.2f}%")
        del model
        torch.cuda.empty_cache()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"task3_{args.region}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())