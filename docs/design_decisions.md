# 架构与数值设计决议（PT-GNO）

> 这些是**实现层**的不可回退决定：改任何一条都要同步改代码、测试与论文表述。

## 1. Phase 2 关键设计决议（论文须如实表述）

1. **15 维建模契约**：`C.FEATURE_NAMES_15` + `C.FEATURE_IDX_15`；落盘张量仍 16 维，
   区域基准按索引切出 15 维（`<REGION>/feature_contract.json` 记录通道名）。
2. **平流核物理输入 3 维**：`C.ADV_KERNEL_IDX = [delta_prs_24h, WIN_S_Max, WIN_S_2mi_Avg]`
   （用 **24h 变压**而非绝对气压，才是"气压梯度驱动平流"）。
3. **平流项保守化**：无向邻接图按**无向对**处理（`min/max` 规范化 + 每对只算一次），
   对 a 贡献 `-g(U_a-U_b)`、对 b 贡献 `+g(U_a-U_b)`，能量严格守恒。
   传导率 `g` 只依赖物理对（`|ΔP|`、两端风速），不伪造风向。
4. **禁止 LayerNorm 作用于状态变量 U**：相变门控读的正是 `RMS(U)`，
   逐节点标准化会把幅值抹平、使门控退化为常数。改用可学习全局 RMS 上限的
   **平滑饱和** `U *= cap·tanh(rms/cap)/rms`，既防爆炸又保留节点间幅值次序。
5. **门控用 RMS 而非裸 L2**：`phase_gate = σ((RMS(U) - τ)/ε)`，τ 由本构泛函给出
   （`softplus` + 基座，恒正），保证"亚临界静默 / 超临界激活"的物理可解释性。
6. **`A_topo` 是 14 类多标签**（逐类 sigmoid，行和可 > 1），不是 14 类互斥 softmax；
   级联项为 `W_cascade(Θ) @ z`，`W_cascade` 是 14×14 **节点级**低秩增量 + 静态基座。
7. **文本两级泄露防护**：`h_cur` 只编码灾前可见文本（`weather_summary` / `warning_info` /
   `remarks`）；`h_post`（`impact_description`）**仅**在 Task 3 观测到达时刻经
   `assim_mask` 注入；`h_rag` 只在训练集语料内检索。
8. **四类损失可加权求和**：`L_FreeEnergy = L_topo + 0.5·L_level + 0.5·L_cont + β·KL`，
   β 前 10 轮线性退火；Focal BCE 用 `pos_weight = clip(sqrt(neg/pos), 1, 20)` 平衡长尾。
9. **指标口径**：Hits@3 只在真有灾种的样本上统计（随机基线 = 3/14 ≈ 0.214），
   不得用全体样本均值充数（会被海量空样本稀释成假高分）。

## 2. Phase 2.5 数值动力学校准（2026-10-21）

冒烟测试暴露了三个数值问题，已定位并修复（全部有对应单元测试）：

### 条件连续损失与相变门控解耦

- **问题**：`cont = ReLU(head) * phase_gate`。正样本（`cont_mask=1`）的
  `log10(1+y)` 均值在 1.5~3.5，而初期 `phase_gate≈0.15`，把正样本预测整体缩小
  近一个数量级 → 条件 R² 长期为负（冒烟实测 −1.4）。
- **修复**：`cont_cond = softplus(cont_head(feat) + cont_pos_mean)`，
  其中 `cont_pos_mean` 是训练集可信损失记录的 `log10(1+y)` 逐目标均值（buffer）。
  训练与条件评估用 `cont_cond`；**Straight-Through** 保留门控梯度：
  `cont_st = cont_cond * (gate - gate.detach() + 1.0)`。
- 全网格无条件推断用 `cont_full = cont_cond * 1[gate ≥ 阈值]`（亚临界静默）。
- 所有基线共用同一 `cont_pos_mean` 解码方式（`ContMeanMixin`），保证公平。

### 废除 0.5 硬阈值，改为验证集逐类校准

- `calibrate_thresholds(val_probs, val_targets)`：逐类在 `[0.05, 0.85]` 上取
  33 点网格，最大化单类 F1，得到 `best_th (14,)`；无正样本的类回退 0.5。
- 验证集与测试集统一使用 `best_th`；`best_th` 随 checkpoint 落盘。
- Focal BCE 的 `pos_weight = clamp(sqrt(neg/(pos+1)), 1, 25)`（训练期统计）。

### 因子化相位监督（相变门控不再塌缩）

- **问题**：只监督联合项 `P(A)=gate·σ(logits)` 时，正样本占比仅 0.4%，
  门控学到的最优解是"恒关闭"（实测 gate 均值 0.23、最小 0），
  类型头因此长期拿不到梯度。
- **修复**：把联合似然因子化 `P(A)=P(相变)·P(类型|相变)`，直接监督相位：
  `L_phase = BCE(logit(gate), 1[该节点-日有任何灾种])`，权重 `w_phase=0.5`。
  这与"临界阈值 τ 由数据学成事件检测器"的物理叙事一致；基线用其多标签输出
  导出的 `1-∏(1-p_i)` 作同构相位，保证对照公平。

### 模型选择与评估口径

- **不再按总损失选模型**：总损失被等级 CE 与连续回归主导，会把 checkpoint
  锁死在 Task 1 指标仍在上升的第 7 轮。改为按**活跃发灾窗口 Macro-F1** 选择。
- 同时报告两套指标：全网格（`macro_f1` 等）与活跃域（`*_active`），
  后者只在"该节点-日至少有一个灾种"的样本上统计；Hits@3/MRR 一律在活跃域上算。

### 工程性能修正（多 run 并行必需）

- `load_region` 每次重算标准化张量要 ~40 s：改为落盘缓存
  `<REGION>/meteo_features_n.npy` + `feature_norm_stats.json`（7 s）。
- 文本向量不再按 `(B,T,N,D)` 展开（每批 ~0.5 GB 无谓搬运），改为
  `(B,N,D)` 一次 `index_select`，本构泛函每节点推断一次 Θ。
- 训练轮不累积预测数组；`feats` 保持 memmap 让 page cache 跨进程共享；
  结束后 `del` + `empty_cache()`。
- 多进程并行时必须 `OMP_NUM_THREADS=1`，否则 BLAS 线程超订使每 epoch
  从 6 s 退化到 30 s。

---
