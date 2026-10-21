"""Phase 3 全量实验驱动：3 区域 × 5 seeds × (主模型 + 4 消融 + 5 基线)。

每个 run 是独立子进程（独立 GPU、独立输出目录），失败不拖垮整个套件：

    python3 scripts/run_suite.py --regions SW_Cascade SE_Typhoon North_Agro \
        --seeds 42 2023 2024 2025 2026 --epochs 40 --jobs 6 --gpus 0 1

产物：`outputs/suite/<region>/<model>/seed<seed>.json`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import common as C          # noqa: E402
from src.models.ablations import ABLATION_NAMES  # noqa: E402
from src.models.baselines import BASELINE_REGISTRY  # noqa: E402

MAIN_NAME = "PT_GNO"


def build_jobs(regions, models, seeds):
    jobs = []
    for r in regions:
        for m in models:
            for s in seeds:
                jobs.append((r, m, s))
    return jobs


def run_one(job, args, gpu: str) -> dict:
    region, model, seed = job
    out_root = Path(args.out_dir) / region / model.replace("/", "_")
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"seed{seed}.json"
    if out_json.exists() and not args.overwrite:
        return {"job": job, "status": "skip"}

    cmd = [sys.executable, "scripts/train_pt_gno.py",
           "--region", region, "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size), "--stride", str(args.stride),
           "--delta", str(args.delta), "--seed", str(seed),
           "--device", "cuda",
           # checkpoint 目录必须带 seed：并行 seed 共享同一路径会互相覆盖，
           # 导致 Task3/案例导出加载到别的 seed 的权重。
           "--out-dir", str(out_root / "_ckpt"),
           "--patience", str(args.patience),
           "--val-stride", str(args.val_stride),
           "--report", str(out_json)]
    if model == MAIN_NAME:
        pass
    elif model in ABLATION_NAMES:
        cmd += ["--ablation", model]
    else:
        cmd += ["--baseline", model]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = "1"
    log = out_root / f"seed{seed}.log"
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return {"job": job, "status": "ok" if proc.returncode == 0 else f"fail({proc.returncode})",
            "seconds": round(time.time() - t0, 1), "gpu": gpu}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 全量实验套件")
    ap.add_argument("--regions", nargs="*", default=list(C.REGIONS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 2023, 2024, 2025, 2026])
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"默认 {MAIN_NAME} + 4 消融 + {list(BASELINE_REGISTRY)}")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gpus", nargs="*", default=["0", "1"])
    ap.add_argument("--out-dir", default="outputs/suite")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    models = args.models or ([MAIN_NAME] + list(ABLATION_NAMES) + list(BASELINE_REGISTRY))
    jobs = build_jobs(args.regions, models, args.seeds)
    print(f"共 {len(jobs)} 个 run，并发 {args.jobs}，GPU {args.gpus}", flush=True)

    def worker(i_job):
        i, job = i_job
        gpu = args.gpus[i % len(args.gpus)]
        try:
            r = run_one(job, args, gpu)
        except Exception as exc:                        # noqa: BLE001
            # 单个 run 的异常不允许打断整个套件（skip 分支没有 gpu 字段，
            # 早先正是 r['gpu'] 抛 KeyError 把 150 个 run 的套件整体拖死）
            r = {"job": job, "status": f"error:{type(exc).__name__}:{exc}", "gpu": gpu}
        print(f"[{i+1}/{len(jobs)}] {job} -> {r['status']} {r.get('seconds','')}s "
              f"gpu{r.get('gpu', '-')}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(worker, enumerate(jobs)))
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip")
    print(f"完成 {ok} / 跳过 {skip} / 失败 {len(results)-ok-skip}")
    (Path(args.out_dir) / "suite_status.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())