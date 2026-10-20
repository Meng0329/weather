"""PT-GNO 的 4 个受控消融变体（只切换数学式中的一项，其余完全共享）。

与 `CLAUDE.md` §6.5 一一对应：

1. `w/o Phase-Transition Gate`      → 门控恒为 1（`phase_gate` 不再依赖 ||U||-τ）
2. `w/o Advection PDE`              → 平流核退化为自由注意力（丢失气压-风场物理结构）
3. `w/o LLM Constitutive Functional` → τ / γ / W_cascade 变为与样本无关的静态参数
4. `w/o Assimilation & RAG Prior`   → 移除同化跳变 M(t) 与 RAG 条件先验（KL 退化为标准正态）
"""

from __future__ import annotations

from src.models.baselines import count_params  # noqa: F401
from src.models.pt_gno import GNOConfig, PT_GNO, build_pt_gno  # noqa: F401

ABLATION_NAMES = [
    "w/o Phase-Transition Gate",
    "w/o Advection PDE",
    "w/o LLM Constitutive Functional",
    "w/o Assimilation & RAG Prior",
]


def build_ablation(name: str, **overrides) -> PT_GNO:
    """按名称构造消融模型；`overrides` 可覆盖其余配置（如 d_model / text_dim）。"""
    model = build_pt_gno(ablation=name, **overrides)
    model.ablation_name = name
    return model


def build_all_ablations(**overrides) -> dict[str, PT_GNO]:
    return {n: build_ablation(n, **overrides) for n in ABLATION_NAMES}


__all__ = ["ABLATION_NAMES", "GNOConfig", "PT_GNO", "build_ablation",
           "build_all_ablations", "build_pt_gno", "count_params"]