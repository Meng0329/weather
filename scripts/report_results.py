"""Phase 3 论文落盘：汇总全部 run → Mean±Std + t 检验 + LaTeX/Markdown 表 + 案例数据。

输入：`outputs/suite/<region>/<model>/seed<seed>.json`
输出：
  - `docs/paper/experimental_results.md`（Markdown 表 + 结论）
  - `docs/paper/experimental_results.tex`（IEEE booktabs 三线表）
  - `docs/paper/case_study_phase_portrait.json`（可解释性案例）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import common as C          # noqa: E402
from src.models.ablations import ABLATION_NAMES  # noqa: E402
from src.models.baselines import BASELINE_REGISTRY  # noqa: E402

MAIN_NAME = "PT_GNO"   # 与 run_suite 的目录名一致（否则显著性表为空）

TASK1_KEYS = ["macro_f1", "macro_f1_active", "auprc_active", "mrr_active",
              "hits@3_active", "micro_f1"]
TASK2_KEYS = ["level_acc", "level_weighted_f1", "log_mae", "log_rmse", "r2"]


# --------------------------------------------------------------------------
def load_runs(root: Path) -> dict[tuple[str, str, int], dict]:
    """读取全部 run 的 json。"""
    out = {}
    for region_dir in sorted(root.glob("*/")):
        region = region_dir.name
        if region not in C.REGIONS:
            continue
        for model_dir in sorted(region_dir.glob("*/")):
            model = model_dir.name
            for f in sorted(model_dir.glob("seed*.json")):
                seed = int(f.stem.replace("seed", ""))
                try:
                    d = json.loads(f.read_text("utf-8"))
                except Exception:                       # noqa: BLE001
                    continue
                # train_pt_gno 会把结果追加成列表；取本 seed 的最后一条
                if isinstance(d, list):
                    pick = [x for x in d if int(x.get("seed", -1)) == seed]
                    d = (pick or d)[-1]
                out[(region, model, seed)] = d
    return out


def _mean_std(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def _fmt(m: float, s: float, n: int, digits: int = 3) -> str:
    if n == 0:
        return "--"
    if n > 1 and s > 0:
        return f"{m:.{digits}f}±{s:.{digits}f}"
    return f"{m:.{digits}f}"


def _stars(p: float) -> str:
    if not (isinstance(p, float) and math.isfinite(p)):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def paired_t(main: list[float], other: list[float]) -> float:
    """双侧配对 t 检验的 p 值（主模型 vs 其他，按 seed 配对）。"""
    try:
        from scipy import stats
    except Exception:                                   # noqa: BLE001
        return float("nan")
    a = np.asarray([x for x in main if math.isfinite(x)], dtype=np.float64)
    b = np.asarray([x for x in other if math.isfinite(x)], dtype=np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.allclose(a, b):
        return 1.0
    r = stats.ttest_rel(a, b)
    return float(r.pvalue)


def summarize(runs: dict, split: str = "test") -> dict:
    """按 (region, model) 汇总 Mean±Std，并算主模型对每个对照的配对 t 检验。"""
    summary: dict[str, dict[str, dict]] = {}
    for region in C.REGIONS:
        summary[region] = {}
        models = sorted({m for (r, m, s) in runs if r == region})
        main_vals = {}
        for model in models:
            seeds = [s for (r, m, s) in runs if r == region and m == model]
            rec = {}
            for key in TASK1_KEYS + TASK2_KEYS:
                vals = []
                for s in seeds:
                    d = runs[(region, model, s)]
                    src = d.get(split) or d.get("best", {}).get("metrics", {})
                    v = src.get(key, float("nan"))
                    if isinstance(v, dict):     # 已聚合过的 {"mean":...} 形式
                        v = v.get("mean", float("nan"))
                    vals.append(float(v))
                m_, s_, n_ = _mean_std(vals)
                rec[key] = {"mean": m_, "std": s_, "n": n_, "fmt": _fmt(m_, s_, n_)}
            rec["_seeds"] = seeds
            rec["_raw"] = {}
            for k in TASK1_KEYS + TASK2_KEYS:
                vals = []
                for s in seeds:
                    d = runs[(region, model, s)]
                    src = d.get(split) or d.get("best", {}).get("metrics", {})
                    v = src.get(k, float("nan"))
                    if isinstance(v, dict):
                        v = v.get("mean", float("nan"))
                    vals.append(float(v))
                rec["_raw"][k] = vals
            summary[region][model] = rec
            if model == MAIN_NAME:
                main_vals = rec["_raw"]
        for model in models:
            if model == MAIN_NAME or not main_vals:
                continue
            other = summary[region][model]["_raw"]
            for key in ("macro_f1_active", "auprc_active", "hits@3_active", "r2"):
                if key in main_vals and key in other:
                    p = paired_t(main_vals[key], other[key])
                    summary[region][model].setdefault("_p", {})[key] = p
    return summary


# --------------------------------------------------------------------------
def markdown_table(summary: dict, region: str, keys: list[str]) -> str:
    rows = ["| 模型 | " + " | ".join(_pretty(k) for k in keys) + " |",
            "|" + "---|" * (len(keys) + 1)]
    for model, rec in sorted(summary[region].items(),
                             key=lambda kv: -_sort_score(kv[1])):
        cells = [rec[k]["fmt"] if k in rec else "--" for k in keys]
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def latex_table(summary: dict, keys: list[str], caption: str, label: str) -> str:
    header = " & ".join(_pretty_latex(k) for k in keys)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{" + caption + r"}", r"\label{" + label + r"}",
             r"\small", r"\begin{tabular}{l" + "r" * len(keys) + "}",
             r"\toprule", header + r" \\", r"\midrule"]
    models = list(summary[list(summary)[0]].keys())
    for region in summary:
        lines.append(r"\multicolumn{" + str(len(keys) + 1) + r"}{l}{\textbf{" + region + r"}} \\")
        for model in sorted(summary[region].keys(),
                            key=lambda m: -_sort_score(summary[region][m])):
            cells = [summary[region][model][k]["fmt"] if k in summary[region][model]
                     else "--" for k in keys]
            lines.append(r"\quad " + model + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def _pretty(k: str) -> str:
    return {
        "macro_f1": "Macro-F1", "macro_f1_active": "Macro-F1 (Active)",
        "micro_f1": "Micro-F1", "auprc_active": "AUPRC (Active)",
        "mrr_active": "MRR (Active)", "hits@3_active": "Hits@3 (Active)",
        "level_acc": "Level Acc", "level_weighted_f1": "Level W-F1",
        "log_mae": "log-MAE", "log_rmse": "log-RMSE", "r2": "R²",
    }.get(k, k)


def _pretty_latex(k: str) -> str:
    return {
        "macro_f1": "Macro-F1", "macro_f1_active": "Macro-F1$_{act}$",
        "micro_f1": "Micro-F1", "auprc_active": "AUPRC$_{act}$",
        "mrr_active": "MRR$_{act}$", "hits@3_active": "Hits@3$_{act}$",
        "level_acc": "Lvl-Acc", "level_weighted_f1": "Lvl-WF1",
        "log_mae": "log-MAE", "log_rmse": "log-RMSE", "r2": "$R^2$",
    }.get(k, k)


def _sort_score(rec: dict) -> float:
    v = rec.get("macro_f1_active", {}).get("mean", float("nan"))
    return v if isinstance(v, float) and math.isfinite(v) else -9.9


def write_report(runs: dict, out_md: Path) -> dict:
    summary = summarize(runs, split="test")
    md = ["# PT-GNO 实验结果（Phase 3 全量）", "",
          f"- 区域：{', '.join(summary)}", f"- 模型数：{len({m for r in summary.values() for m in r})}",
          f"- seeds：{sorted({s for (_, _, s) in runs})}", ""]
    for title, keys in (("Task 1：相变与级联预测（测试集）", TASK1_KEYS),
                        ("Task 2：损失等级与连续回归（测试集）", TASK2_KEYS)):
        md += [f"## {title}", ""]
        for region in summary:
            md += [f"### {region}", "", markdown_table(summary, region, keys), ""]
    md += ["## 显著性检验（主模型 vs 各对照，配对 t 检验，测试集）", "",
           "| 区域 | 对照 | ΔMacro-F1(act) | p | sig |", "|---|---|---:|---:|:--:|"]
    for region in summary:
        rec = summary[region].get(MAIN_NAME)
        if not rec:
            continue
        for model, r2 in summary[region].items():
            if model == MAIN_NAME:
                continue
            p = r2.get("_p", {}).get("macro_f1_active", float("nan"))
            d = (rec["macro_f1_active"]["mean"] - r2["macro_f1_active"]["mean"])
            md.append(f"| {region} | {model} | {d:+.3f} | "
                      f"{'--' if not math.isfinite(p) else f'{p:.3g}'} | {_stars(p)} |")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def _task3_section(root: Path) -> list[str]:
    files = sorted(root.glob("task3_*.json"))
    if not files:
        return []
    md = ["## Task 3：初报 → 续报 → 终报 连续时间同化（测试期多阶段事件链）", ""]
    for f in files:
        d = json.loads(f.read_text("utf-8"))
        md += [f"### {d.get('region')}（可评测链 {d.get('n_samples', 0)} 条）", "",
               "| 模型 | Stage | log-MAE | RMSE | R² | Level-Acc | ERR% |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for name, res in (d.get("models") or {}).items():
            for st in res.get("stages", []):
                md.append(f"| {name} | S{st['stage']} | {st['log_mae']:.4f} | "
                          f"{st['rmse']:.4f} | {st['r2']:.4f} | "
                          f"{st['level_acc']:.3f} | {st['err_reduction_pct']:+.2f} |")
        md.append("")
    return md


def _multihorizon_section(root: Path) -> list[str]:
    f = root / "multihorizon.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text("utf-8"))
    if not rows:
        return []
    md = ["## 多步提前期衰减（单模型多提前期推断，Δ ∈ {1,3,5,7}）", "",
          "| 区域 | 模型 | Δ=1 | Δ=3 | Δ=5 | Δ=7 |", "|---|---|---:|---:|---:|---:|"]
    keys = ["macro_f1_active", "auprc_active", "hits@3_active"]
    for region in sorted({r["region"] for r in rows}):
        for model in sorted({r["model"] for r in rows if r["region"] == region}):
            sub = {r["delta"]: r for r in rows if r["region"] == region and r["model"] == model}
            for k in keys:
                cells = []
                for d in (1, 3, 5, 7):
                    v = sub.get(d, {}).get(k, float("nan"))
                    cells.append(f"{v:.3f}" if isinstance(v, (int, float)) and math.isfinite(v) else "--")
                md.append(f"| {region} | {model} · {_pretty(k)} | " + " | ".join(cells) + " |")
    md.append("")
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 结果汇总")
    ap.add_argument("--root", default="outputs/suite")
    ap.add_argument("--out-md", default="docs/paper/experimental_results.md")
    ap.add_argument("--out-tex", default="docs/paper/experimental_results.tex")
    ap.add_argument("--task3-root", default="outputs/task3")
    args = ap.parse_args(argv)

    runs = load_runs(Path(args.root))
    if not runs:
        print(f"未找到 run：{args.root}")
        return 1
    summary = write_report(runs, Path(args.out_md))
    extra = _task3_section(Path(args.task3_root)) + _multihorizon_section(Path(args.task3_root).parent)
    if extra:
        with Path(args.out_md).open("a", encoding="utf-8") as fh:
            fh.write("\n".join(extra) + "\n")
    tex = ["% PT-GNO 实验表（由 scripts/report_results.py 生成）", "",
           latex_table(summary, TASK1_KEYS, "Task 1: Phase-transition prediction on the test split.",
                       "tab:task1"), "",
           latex_table(summary, TASK2_KEYS, "Task 2: Loss level classification and continuous regression.",
                       "tab:task2"), ""]
    Path(args.out_tex).write_text("\n".join(tex), encoding="utf-8")
    print(f"已写出 {args.out_md} 与 {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())