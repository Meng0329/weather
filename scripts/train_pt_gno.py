"""一键训练/评测入口（PT-GNO 主模型、消融、对照基线共用）。

用法示例：
    # 主模型 2 轮冒烟
    python3 scripts/train_pt_gno.py --region SW_Cascade --epochs 2 \
        --max-train-samples 64 --max-val-samples 24

    # 消融
    python3 scripts/train_pt_gno.py --ablation "w/o Phase-Transition Gate" --epochs 2

    # 基线（Stitched-GNN-LLM 等）
    python3 scripts/train_pt_gno.py --baseline "Stitched-GNN-LLM" --epochs 2

    # 全部 4 个消融 + 全部基线，各 2 轮
    python3 scripts/train_pt_gno.py --suite --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

# 多进程并行时 BLAS 线程会严重超订（每 epoch 从 6 s 退化到 30 s），
# 必须在导入 numpy/torch 之前把线程数钉死。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch                              # noqa: E402
torch.set_num_threads(1)

from src.data import common as C          # noqa: E402
from src.engine.data import load_region   # noqa: E402
from src.engine.trainer import LossConfig, TrainConfig, train  # noqa: E402
from src.models.ablations import ABLATION_NAMES, build_ablation  # noqa: E402
from src.models.baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from src.models.pt_gno import GNOConfig, PT_GNO  # noqa: E402


def make_builder(kind: str, name: str | None, text_dim: int, overrides: dict):
    """返回零参构造器，闭包捕获配置（trainer 内部只需 builder()）。"""
    if kind == "baseline":
        return lambda: build_baseline(name, text_dim=text_dim, **overrides)
    if kind == "ablation":
        return lambda: build_ablation(name, text_dim=text_dim, **overrides)
    return lambda: PT_GNO(GNOConfig(text_dim=text_dim, **overrides))


def run_one(kind: str, name: str | None, args, text_dim: int) -> dict:
    overrides = {}
    if args.d_model:
        overrides["d_model"] = args.d_model
        overrides["hidden"] = args.d_model
    builder = make_builder(kind, name, text_dim, overrides)
    tcfg = TrainConfig(
        region=args.region, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        tin=args.tin, delta=args.delta, stride=args.stride, seed=args.seed,
        device=args.device, out_dir=args.out_dir, log_every=args.log_every,
        patience=args.patience, amp=not args.no_amp, val_stride=args.val_stride,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    loss_cfg = LossConfig(beta_max=args.beta, beta_warmup=args.beta_warmup,
                          w_phase=args.w_phase, w_level=args.w_level,
                          w_cont=args.w_cont)
    label = name or "PT_GNO"
    print(f"\n=== [{kind}] {label} @ {args.region} ===", flush=True)
    res = train(builder, tcfg, loss_cfg)
    res["kind"] = kind
    res["name"] = label
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PT-GNO 训练与评测")
    ap.add_argument("--region", default="SW_Cascade", choices=list(C.REGIONS))
    ap.add_argument("--ablation", default=None, choices=ABLATION_NAMES)
    ap.add_argument("--baseline", default=None, choices=list(BASELINE_REGISTRY))
    ap.add_argument("--suite", action="store_true",
                    help="依次跑主模型 + 4 个消融 + 全部基线")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tin", type=int, default=7)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--beta-warmup", type=int, default=10)
    ap.add_argument("--w-phase", type=float, default=0.5)
    ap.add_argument("--w-level", type=float, default=0.5)
    ap.add_argument("--w-cont", type=float, default=1.0)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--val-stride", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/runs")
    ap.add_argument("--report", default="outputs/runs/summary.json")
    args = ap.parse_args(argv)

    text_dim = load_region(args.region, contiguous=False).text_dim
    print(f"[{args.region}] text_dim = {text_dim}", flush=True)

    jobs: list[tuple[str, str | None]] = []
    if args.suite:
        jobs.append(("main", None))
        jobs += [("ablation", n) for n in ABLATION_NAMES]
        jobs += [("baseline", n) for n in BASELINE_REGISTRY]
    elif args.baseline:
        jobs.append(("baseline", args.baseline))
    elif args.ablation:
        jobs.append(("ablation", args.ablation))
    else:
        jobs.append(("main", None))

    results = [run_one(k, n, args, text_dim) for k, n in jobs]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text("utf-8")) if out.exists() else []
    prev = [r for r in prev if not (r["region"] == args.region
                                    and r["name"] == results[0]["name"])]
    out.write_text(json.dumps(prev + results, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n================ 汇总 ================")
    for r in results:
        b = r["best"]["metrics"]
        print(f"{r['name']:<34} loss={r['best']['val_loss']:.4f} "
              f"macroF1={b.get('macro_f1', float('nan')):.4f} "
              f"AUPRC={b.get('auprc', float('nan')):.4f} "
              f"Hits@3={b.get('hits@3', float('nan')):.4f} "
              f"logMAE={b.get('log_mae', float('nan')):.4f} "
              f"R2={b.get('r2', float('nan')):.4f} "
              f"params={r['model_params']}")
    print(f"\n汇总写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
