"""Phase 2 模型单元测试：算子结构、消融开关、损失与指标。

全部为纯逻辑/小张量测试，不需要真实数据产物（秒级可跑）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.engine.trainer import (LossConfig, compute_metrics, focal_bce,
                                free_energy_loss, masked_huber, weighted_ce)
from src.models.ablations import ABLATION_NAMES, build_ablation, build_all_ablations
from src.models.baselines import BASELINE_REGISTRY, build_baseline, count_params
from src.models.pt_gno import (GNOConfig, LLMConstitutiveFunctional, PT_GNO,
                               PhysicsAdvectionReactionCell, StaticConstitutive,
                               LandauPhaseTransitionReadout, build_pt_gno)

N = 6           # 节点数
T = 4           # 窗口长度
F = 15          # 特征维度
D = 32          # 文本维度


def _fake_batch(B=2, text_dim=D):
    torch.manual_seed(0)
    x = torch.randn(B, T, N, F)
    mask = torch.ones(B, T, N, F)
    adv = torch.randn(B, T, N, 3)
    edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 0, 4, 0, 5]], dtype=torch.long)
    edge_w = torch.rand(edge_index.shape[1]) + 0.1
    return {
        "x": x, "mask": mask, "adv": adv,
        "edge_index": edge_index, "edge_w": edge_w,
        # 文本按 (B,N,D)：本构泛函对每个节点推断一次 Θ，不随窗口时间步重复
        "h_cur": torch.randn(B, N, text_dim),
        "h_rag": torch.randn(B, N, text_dim),
        "h_post": torch.randn(B, N, text_dim),
        "assim_mask": torch.zeros(B, T, N),
    }


# --------------------------------------------------------------------------
# 1. 算子结构
# --------------------------------------------------------------------------
def test_pt_gno_forward_shapes():
    b = _fake_batch()
    m = build_pt_gno(text_dim=D)
    out = m(b)
    assert out["topo_prob"].shape == (2, N, 14)
    assert out["level_logits"].shape == (2, N, 5)
    assert out["cont"].shape == (2, N, 6)
    assert out["phase_gate"].shape == (2, N)
    assert torch.isfinite(out["topo_prob"]).all()
    assert torch.isfinite(out["cont"]).all()


def test_topology_is_multilabel_not_softmax():
    """A_topo 必须是 14 类**多标签**概率（同一节点可同时激活多类）。"""
    b = _fake_batch()
    m = build_pt_gno(text_dim=D)
    out = m(b)
    p = out["topo_prob"][0]
    assert ((p >= 0) & (p <= 1)).all()
    # softmax 的行和恒为 1；多标签的行和可 > 1
    assert (p.sum(-1) > 1.0 + 1e-6).any(), "多标签读出不得被归一化成 softmax"


def test_multilabel_can_activate_two_classes_at_once():
    """构造强激活势能后，同一节点必须能同时点亮两个灾种。"""
    ro = LandauPhaseTransitionReadout(d_model=8)
    U = torch.zeros(1, 1, 8)
    U[0, 0, 0] = 50.0                      # 远高于阈值 -> 门控 ≈ 1
    with torch.no_grad():
        ro.topo_head[-1].bias[:2] = 50.0   # 前两类 logit 极高
    out = ro(U, torch.full((1, 1), 0.1), torch.zeros(1, 1, 14, 14))
    assert (out["topo_prob"][0, 0] > 0.9).sum() >= 2


def test_phase_gate_in_unit_interval_and_monotone():
    ro = LandauPhaseTransitionReadout(d_model=8, eps=0.2)
    U = torch.zeros(1, 3, 8)
    U[0, 0] = 0.01
    U[0, 1] = 1.0
    U[0, 2] = 10.0
    tau = torch.full((1, 3), 1.0)
    g = ro.phase_gate(U, tau)
    assert ((g >= 0) & (g <= 1)).all()
    assert g[0, 0] < g[0, 1] < g[0, 2], "门控必须随势能幅值单调递增"


def test_cont_prediction_is_gated_by_phase():
    """亚临界时**全网格推断**的连续损失必须置零；条件回归头不受门控压缩。

    Phase 2.5：`cont`（条件回归，供损失与条件指标）保持量级，
    `cont_full`（无条件推断）在 gate < 阈值时置零。
    """
    ro = LandauPhaseTransitionReadout(d_model=8)
    U = torch.zeros(1, 1, 8)
    tau = torch.full((1, 1), 1e6)          # 阈值极高 -> 亚临界
    out = ro(U, tau, torch.zeros(1, 1, 14, 14))
    assert float(out["cont_full"].abs().max()) < 1e-6, "亚临界全网格推断必须静默"
    assert float(out["topo_prob"].abs().max()) < 1e-6
    assert float(out["cont_cond"].min()) > 0.0, "条件回归头不应被门控压成 0"


def test_cont_cond_is_centered_on_prior_mean():
    """正样本条件回归的初值应落在 log10 目标量级，而不是 0 附近。"""
    ro = LandauPhaseTransitionReadout(d_model=8)
    ro.cont_pos_mean.copy_(torch.full((6,), 2.0))
    out = ro(torch.randn(2, 5, 8), torch.zeros(2, 5), torch.zeros(2, 5, 14, 14))
    assert float(out["cont_cond"].mean()) > 1.5


def test_advection_kernel_depends_on_physical_inputs():
    """平流核必须由物理量驱动：物理输入改变 -> 传导率改变。"""
    cell = PhysicsAdvectionReactionCell(d_model=8, n_features=F)
    e = torch.tensor([[0], [1]])
    w = torch.ones(1)
    U = torch.randn(1, N, 8)
    # 物理上只有**梯度**驱动平流：均匀平移不该改变传导率，空间对比才会
    a1 = torch.zeros(1, N, 3)
    a2 = torch.arange(N, dtype=torch.float32).view(1, N, 1).repeat(1, 1, 3)
    f1, k1 = cell.advection(U, a1, e, w)
    f2, k2 = cell.advection(U, a2, e, w)
    assert not torch.allclose(k1, k2, atol=1e-6), "传导率不得与物理量无关"
    assert not torch.allclose(f1, f2)
    # 等幅异分布：改变空间对比结构时传导率必须改变（物理梯度敏感性）
    a3 = torch.zeros_like(a2)
    a3[0, 0] = 3.0
    _, k3 = cell.advection(U, a3, e, w)
    assert not torch.allclose(k2, k3, atol=1e-6), "传导率须对物理梯度敏感"


def test_advection_is_mass_conserving_when_no_source():
    """无注入无耗散时，平流项应守恒（节点势能之和不变）。"""
    cell = PhysicsAdvectionReactionCell(d_model=4, n_features=F)
    # 环形图 0->1->2->3->4->5->0：每节点必须同时存在出入边才可能守恒
    e = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    w = torch.ones(6)
    U = torch.randn(1, N, 4)
    adv = torch.randn(1, N, 3)
    flux, _ = cell.advection(U, adv, e, w)
    assert torch.allclose(flux.sum(dim=1), torch.zeros(1, 4), atol=1e-4)


# --------------------------------------------------------------------------
# 2. 本构泛函（变分）
# --------------------------------------------------------------------------
def test_constitutive_kl_is_zero_when_prior_matches_posterior():
    f = LLMConstitutiveFunctional(text_dim=D, hidden=16, d_z=8)
    h = torch.randn(3, D)
    with torch.no_grad():
        f.post[-1].weight.zero_()
        f.post[-1].bias.zero_()
        f.prior[-1].weight.zero_()
        f.prior[-1].bias.zero_()
    out = f(h, h, sample=False)
    assert float(out["kl"].abs().max()) < 1e-4


def test_constitutive_tau_is_positive_and_gamma_in_unit_interval():
    f = LLMConstitutiveFunctional(text_dim=D, hidden=16, d_z=8)
    out = f(torch.randn(5, D), torch.randn(5, D))
    assert (out["tau"] > 0).all()
    assert ((out["gamma"] >= 0) & (out["gamma"] <= 1)).all()
    assert out["delta_w"].shape == (5, 14, 14)


def test_static_constitutive_is_sample_independent():
    f = StaticConstitutive(d_model=8)
    out = f(torch.randn(4, D))
    assert float(out["tau"].std()) == 0.0
    assert float(out["gamma"].std()) == 0.0


def test_static_constitutive_has_no_kl():
    f = StaticConstitutive(d_model=8)
    out = f(torch.randn(4, D))
    assert float(out["kl"].abs().max()) == 0.0


# --------------------------------------------------------------------------
# 3. 消融开关
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ABLATION_NAMES)
def test_each_ablation_toggles_exactly_one_component(name):
    cfg = GNOConfig.ablation(name)
    full = GNOConfig()
    diffs = [k for k in ("use_phase_gate", "use_advection_kernel",
                         "use_llm_functional", "use_assimilation", "prior_from_rag")
             if getattr(cfg, k) != getattr(full, k)]
    if name == "w/o Assimilation & RAG Prior":
        assert set(diffs) == {"use_assimilation", "prior_from_rag"}
    else:
        assert len(diffs) == 1, f"{name} 应只改一项，实际改了 {diffs}"


@pytest.mark.parametrize("name", ABLATION_NAMES)
def test_ablation_forward_runs(name):
    m = build_ablation(name, text_dim=D)
    out = m(_fake_batch())
    assert out["topo_prob"].shape == (2, N, 14)
    assert torch.isfinite(out["topo_prob"]).all()


def test_wo_phase_gate_disables_gating():
    m = build_ablation("w/o Phase-Transition Gate", text_dim=D)
    out = m(_fake_batch())
    assert torch.allclose(out["phase_gate"], torch.ones_like(out["phase_gate"]))


def test_wo_llm_constitutive_uses_static_parameters():
    m = build_ablation("w/o LLM Constitutive Functional", text_dim=D)
    assert isinstance(m.constitutive, StaticConstitutive)


def test_wo_advection_uses_gat_cell():
    from src.models.pt_gno import GATAdvectionCell
    m = build_ablation("w/o Advection PDE", text_dim=D)
    assert isinstance(m.cell, GATAdvectionCell)


def test_wo_assimilation_removes_jump():
    m = build_ablation("w/o Assimilation & RAG Prior", text_dim=D)
    assert m.cfg.use_assimilation is False
    assert m.cfg.prior_from_rag is False


# --------------------------------------------------------------------------
# 4. 损失与指标
# --------------------------------------------------------------------------
def test_focal_bce_downweights_easy_examples():
    easy = torch.full((1, 1, 14), 5.0)
    easy_t = torch.ones(1, 1, 14)
    hard = torch.full((1, 1, 14), -5.0)
    hard_t = torch.ones(1, 1, 14)
    assert float(focal_bce(easy, easy_t)) < float(focal_bce(hard, hard_t))


def test_weighted_ce_respects_mask():
    logits = torch.randn(4, 5)
    tgt = torch.randint(0, 5, (4,))
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    v_masked = float(weighted_ce(logits, tgt, mask))
    v_all = float(weighted_ce(logits, tgt, torch.ones(4)))
    assert v_masked != v_all


def test_masked_huber_ignores_masked_rows():
    pred = torch.zeros(2, 3, 6)
    true = torch.ones(2, 3, 6)
    m1 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert float(masked_huber(pred, true, m1)) > 0
    assert float(masked_huber(pred, true, torch.zeros(2, 3))) == 0.0


def test_free_energy_loss_terms_present():
    b = _fake_batch()
    b["topo"] = (torch.rand(2, N, 14) > 0.7).float()
    b["level"] = torch.randint(0, 5, (2, N))
    b["level_mask"] = torch.ones(2, N)
    b["cont"] = torch.rand(2, N, 6)
    b["cont_mask"] = torch.ones(2, N)
    m = build_pt_gno(text_dim=D)
    out = m(b)
    total, parts = free_energy_loss(out, b, LossConfig(), epoch=1)
    assert set(parts) >= {"topo", "level", "cont", "kl", "beta"}
    assert torch.isfinite(total)
    assert parts["beta"] < LossConfig().beta_max + 1e-9


def test_kl_annealing_schedule():
    b = _fake_batch()
    b["topo"] = torch.zeros(2, N, 14)
    b["level"] = torch.zeros(2, N, dtype=torch.long)
    b["level_mask"] = torch.zeros(2, N)
    b["cont"] = torch.zeros(2, N, 6)
    b["cont_mask"] = torch.zeros(2, N)
    m = build_pt_gno(text_dim=D)
    out = m(b)
    _, p1 = free_energy_loss(out, b, LossConfig(), epoch=1)
    _, p20 = free_energy_loss(out, b, LossConfig(), epoch=20)
    assert p1["beta"] < p20["beta"]


def test_compute_metrics_perfect_prediction():
    topo_true = torch.zeros(1, 4, 14)
    topo_true[0, 0, 0] = 1
    topo_true[0, 1, 3] = 1
    topo_prob = topo_true.clone()
    level_true = torch.tensor([[0, 1, 2, 3]])
    level_logits = torch.full((1, 4, 5), -10.0)
    level_logits[0, torch.arange(4), level_true[0]] = 10.0
    cont_true = torch.rand(1, 4, 6)
    m = compute_metrics(topo_prob, topo_true, level_logits, level_true,
                        torch.ones(1, 4), cont_true, cont_true, torch.ones(1, 4))
    assert m["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert m["micro_f1"] == pytest.approx(1.0, abs=1e-6)
    # Hits@3 按全类 14 排序：正类概率 1 与大量并列 0，命中率非必为 1，只校验可计算
    assert 0.0 <= m["hits@3"] <= 1.0
    assert m["level_acc"] == pytest.approx(1.0, abs=1e-6)
    assert m["log_mae"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_handles_empty_mask():
    z = torch.zeros(1, 3, 14)
    m = compute_metrics(z, z, torch.randn(1, 3, 5), torch.zeros(1, 3, dtype=torch.long),
                        torch.zeros(1, 3), torch.zeros(1, 3, 6), torch.zeros(1, 3, 6),
                        torch.zeros(1, 3))
    assert math.isnan(m["level_acc"])
    assert math.isnan(m["log_mae"])


# --------------------------------------------------------------------------
# 5. 基线
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BASELINE_REGISTRY))
def test_baseline_forward_and_backward(name):
    b = _fake_batch()
    m = build_baseline(name, text_dim=D)
    out = m(b)
    assert out["topo_prob"].shape == (2, N, 14)
    assert out["level_logits"].shape == (2, N, 5)
    assert out["cont"].shape == (2, N, 6)
    out["topo_prob"].sum().backward()


def test_stitched_baseline_has_no_kl_and_no_gate():
    """缝合基线没有本构泛函，也没有相变门控。"""
    m = build_baseline("Stitched-GNN-LLM", text_dim=D)
    out = m(_fake_batch())
    assert "kl" not in out
    assert torch.allclose(out["phase_gate"], torch.ones_like(out["phase_gate"]))


def test_pt_gno_and_stitched_have_comparable_parameters():
    """缝合基线必须与主模型参数量同量级（回应审稿人）。"""
    main = count_params(build_pt_gno(text_dim=D))
    stitched = count_params(build_baseline("Stitched-GNN-LLM", text_dim=D))
    ratio = stitched / main
    assert 0.2 <= ratio <= 5.0, f"参数量比 {ratio:.2f} 偏离过大"


def test_pt_gno_forward_is_not_cat_fusion():
    """红线：主模型不得出现 `torch.cat([h_gnn, h_llm])` 式末端拼接。"""
    import inspect

    from src.models import pt_gno
    src = inspect.getsource(pt_gno)
    # 允许 MLP 输入端的特征拼接（如本构泛函接收 [h_cur, h_rag]）；
    # 禁止的是把 GNN 分支与 LLM 分支输出拼接成新特征张量的"缝合"写法。
    forbidden = ["h_gnn", "h_llm", "cat([h_gnn", "cat([h_llm"]
    hits = [f for f in forbidden if f in src]
    assert not hits, f"PT-GNO 出现分支缝合痕迹: {hits}"


# --------------------------------------------------------------------------
# 6. 相变门控的结构性要求（回归测试：曾因 LayerNorm 抹平 ||U|| 而退化为常数）
# --------------------------------------------------------------------------
def test_gate_varies_across_nodes_with_different_drivers():
    """不同驱动强度的节点必须得到不同门控值（不得整批同值）。"""
    m = build_ablation("w/o Assimilation & RAG Prior", text_dim=D)
    m.eval()
    b = _fake_batch(B=1)
    b["x"] = torch.randn(1, T, N, F) * 3.0
    b["x"][0, :, 0] *= 0.01                     # 让不同节点驱动强度相差百倍
    out = m(b)
    g = out["phase_gate"][0]
    assert float(g.max() - g.min()) > 1e-4, "门控对节点幅值不敏感 = 相变机制失效"
