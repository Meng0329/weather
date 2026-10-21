# CLAUDE.md — PT-GNO 项目契约（入口 / 索引）

> 本文件是唯一项目契约的**入口**。契约按主题拆分到 `docs/`，改动前先读对应分册；
> 任何实现、实验、写作冲突时以本文件 + 对应分册为准，其次 `docs/idea.md`，
> 与 `docs/复现方案.md` 的示例代码冲突时以 §1「第一性原理架构」为准。
>
> 三条铁律：**一个状态张量、一组微分方程、一个变分自由能目标**。不许模块缝合。

---

## 0. 项目定位

- **论文题目**：PT-GNO: Physics-Informed Phase-Transition Graph Neural Operator for Unifying
  Continuous Meteorology and Discrete Disaster Knowledge Evolution
- **中文**：物理引导的相变图神经算子：统一连续气象场与离散灾害知识演化的变分动力学框架
- **目标期刊**：IEEE TKDE / Information Fusion（中科院一区 Top）
- **一句话主张**：连续气象势能场与离散灾害知识图谱不是两个模态，而是同一时空状态张量在
  临界阈值处发生拓扑相变前后的两种相态。

---

## 1. 第一性原理统一架构（唯一真理源，严禁模块缝合）

架构必须在数学上自洽：一个状态张量、一组微分方程、一个变分自由能目标。**禁止**把 GNN、
PDE 损失项、LLM 特征拼接成流水线（这是论文要批判的对象，也是 `Stitched-GNN-LLM` 基线）。

### 1.1 本体统一（Ontology Unification）

- **下层连续场**：区县节点 $i$ 的连续地面气象势能场 $\mathbf{U}_i(t) \in \mathbb{R}^d$，
  由日值气象要素驱动，处于亚临界态。
- **上层离散图谱**：当势能范数突破局地临界阈值 $\tau_{i,t}$ 时，通过**可微朗道相变门控**
  涌现出离散灾害时序知识图谱 $\mathbf{A}_{i,t}$：14 类标准灾害 `disaster_type_std`、
  灾害亚型、级联边、承灾体冲击。
- **数学表述**：连续场与离散 TKG 是同一状态张量的"连续幅值"与"离散支撑集"，不是双模态融合。

### 1.2 算子统一（小模型即机理，Operator Unification）

小模型直接实现为神经微分算子，**不是**"网络 + 外部 PDE 损失项"：

$$
\mathrm{d}\mathbf{U}_i(t) =
\underbrace{-\sum_{j\in\mathcal{N}(i)} \mathcal{K}_{\text{adv}}\big(\Delta\mathrm{PRS}_{ij,t},\mathbf{WIN}_{ij,t}\big)\big(\mathbf{U}_i(t)-\mathbf{U}_j(t)\big)}_{\text{跨区县气压-风场平流扩散（水平 PDE）}}
+ \underbrace{\Psi_{\text{in}}\big(\mathbf{X}^{\text{meteo}}_{i,t}\big)}_{\text{垂直致灾能量注入}}
- \underbrace{\gamma_{i,t}(\Theta)\odot\mathbf{U}_i(t)}_{\text{防灾耗散}}
+ \underbrace{\mathbf{G}_{\text{assim}}\big(\mathbf{U}_i(t)\big)\mathrm{d}\mathbf{M}_i(t)}_{\text{初报/续报/终报连续时间同化}}
$$

相变与图谱涌现：

$$
\mathbf{A}^{\text{topo}}_{i,t}=\sigma_{\text{temp}}\!\Big(\frac{\lVert\mathbf{U}_i(t)\rVert_2-\tau_{i,t}}{\epsilon}\Big)\cdot
\mathrm{Softmax}\big(\mathbf{W}_{\text{cascade}}(\Theta)\mathbf{U}_i(t)\big),
\qquad
\hat{\mathbf{Y}}^{\text{loss}}_{i,t}=\mathcal{R}_{\text{readout}}\big(\mathbf{A}^{\text{topo}}_{i,t},\mathbf{U}_i(t)\big)
$$

- $\lVert\mathbf{U}\rVert<\tau$ 时门控趋 0，图谱静默；突破后瞬时激活灾害节点与损失读出。
- 平流核由**真实物理量**（气压梯度、风速）驱动，不是自由注意力。

### 1.3 大小模型统一（无路由，Amortized Variational Constitutive Functional）

- LLM **不是**分类器、不是路由分支、不是特征拼接块，而是微分方程的本构泛函：
  $q_\phi(\Theta_{i,t}\mid\mathcal{T}_{i,\le t})$。
- 从 `weather_summary`、`warning_info`、`impact_description`、`remarks` 与历史 RAG
  `search_text` 中推断三组局地本构参数：
  - 临界阈值 $\tau_{i,t}$（恒正）
  - 防灾耗散率 $\gamma_{i,t}$（与 `relocated`、预警级别相关）
  - 级联转移矩阵 $\mathbf{W}_{\text{cascade}}$（14×14）
- **离线蒸馏优先**：冻结开源基座（Qwen2.5-7B-Instruct / DeepSeek-R1-Distill-Qwen-7B，
  `temperature=0.0`）批量提取语义向量并缓存，训练轻量变分投影网络 $q_\phi$；
  推理阶段不依赖实时大模型调用。

### 1.4 统一目标函数（唯一 Loss）

$$
\min_{\theta,\phi}\ \mathcal{L}_{\text{FreeEnergy}}=
\mathbb{E}_{q_\phi(\Theta\mid\mathcal{T}_{\le t})}\big[-\log p_\theta(\mathbf{A}^{\text{topo}}_{1:T},\mathbf{Y}^{\text{loss}}_{1:T}\mid\mathbf{U}_{1:T}(\Theta))\big]
+\beta\,\mathrm{KL}\big(q_\phi(\Theta\mid\mathcal{T}_{\le t})\,\Vert\,p_{\text{LLM}}(\Theta\mid\mathcal{T}_{\text{RAG}})\big)
$$

- 似然分解为三项：拓扑相变 Focal BCE、损失等级加权 CE、连续损失 $\log_{10}(1+y)$ Huber。
- KL 项连接实时文本后验与 RAG 历史先验；$\beta$ 前 10 轮线性退火。

### 1.5 架构红线（评审必查）

1. **禁止**硬路由（router / gating between LLM and GNN branches）。
2. **禁止**末端 `torch.cat([h_gnn, h_llm])` 式拼接融合。
3. **禁止**以外部 `/ PDE residual loss` 代替算子内部物理结构；物理必须在内核前向传播中。
4. **禁止**为提升指标而引入与相变叙事无关的模块；每个模块必须能对应上式中的一项。
5. 所有消融必须能映射回上式的某一项（门控 / 平流核 / 本构泛函 / 同化项）。

---

## 2. 目录结构与文档索引

### 2.1 目录结构

```
src/
  data/        # 清洗、99xxxx 特征值处理、区县对齐、三大区域数据集切分
  llm/         # 离线语义提取与历史 RAG 先验缓存
  models/      # PT-GNO 主模型；Stitched-GNN-LLM 缝合对比基线；其他 SOTA 基线
  engine/      # 变分自由能 ELBO 训练器；多步相变与损失评估；初报-续报-终报同化评测
scripts/       # 一键实验脚本（数据构建 / 训练 / 评测 / 消融 / 出表出图）
docs/paper/    # 论文各章节 Markdown/LaTeX 源码与实验结果表格汇总
```

- `docs/` 现有：`idea.md`（立意与章节规划）、`复现方案.md`（数据流程与参考实现）。
- 数据只读；任何清洗产物写入 `outputs/` 或 `data_processed/`（已被 `.gitignore` 忽略），
  **不要**修改 `datasets/` 下原始文件。
- 大文件不入 Git（见 `.gitignore`）。

### 2.2 文档索引（细节都在这里，本文件不重复）

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| `docs/data_audit.md` | 原始数据规模、**全部已验证的数据陷阱**（99xxxx 家族、逐年缺测、字段填充率、经济损失异常、重复站点、区县不可 join）、16 维特征与 24→14 映射全文 | 改 `src/data/**`、写数据统计表、审稿人质询数据口径 |
| `docs/experiment_protocol.md` | 实验协议（区域基准/任务指标/基线/复现纪律）与阻塞项决议 6.1–6.9 | 跑实验、做消融、定超参、写实验设置章节 |
| `docs/design_decisions.md` | Phase 2 算子设计决议 + Phase 2.5 数值动力学校准（门控、平流、条件回归、阈值校准、相位监督） | 改 `src/models/**`、`src/engine/trainer.py` |
| `docs/phase_log.md` | 各 Phase 交付流水与**实测数字**（数据规模、对齐率、图统计、冒烟指标） | 追溯"这个数字哪来的"、写实验章节 |
| `docs/paper/` | 论文立意（`idea.md`）、实验结果表（`experimental_results.md/.tex`）、案例数据 | 写作 |
| `docs/复现方案.md` | 数据流程参考实现（**非契约**） | 快速上手 |

代码入口：`src/data/`（清洗与基准）、`src/llm/`（离线语义与 RAG 先验）、
`src/models/`（PT-GNO / 基线 / 消融）、`src/engine/`（训练、指标、Task 3）、
`scripts/`（一键脚本）、`tests/`（单元 + 产物测试）。

---

## 3. 硬约束速查（违反即返工）

### 3.1 数据

- **时间划分固定**：Train 2000–2018 / Val 2019–2021 / Test 2022–2025；
  测试集只在最终评估跑一次，归一化统计量只从训练集算。
- **隐私字段**：读灾情表时立即丢弃 `tabulator`、`contact`，不得进入任何特征或论文表格。
- **数据只读**：不修改 `datasets/`；产物落 `datasets/processed/` 或 `outputs/`（均 gitignore）。
- **经济损失量纲**：`economic_loss >= 1e8`（万元）的记录 **÷1e4** 修正。
- **缺失值**：`999999/999998` → NaN；`999000 ≤ v < 999990` 视为非物理值剔除；
  `PRE_Time_2020 ∈ {999990,999991,999992}` → 0.05 mm + `PRE_trace_flag`。
  **必须先剔除再按区县求均值**。
- **禁止跨期伪特征**：`Lit`/`Thund`/`Tord`（2016–2025 全缺）、`WIN_S_10mi_Avg`
  （2018–2025 全缺）不得进入建模特征。

### 3.2 特征契约

- 建模输入固定 **15 维**（`C.FEATURE_NAMES_15` / `C.FEATURE_IDX_15`）；
  落盘张量仍 16 维，区域基准按索引切出。
- 平流核物理输入固定 **3 维**：`[ΔPRS_24h, WIN_S_Max, WIN_S_2mi_Avg]`（`C.ADV_KERNEL_IDX`），
  且读**原始量纲**（hPa / m·s⁻¹），不做标准化。
- 完整 16 维定义与逐列缺测率：`docs/data_audit.md` §3.1。

### 3.3 标签契约

- `A_topo` 是 **14 类多标签**（逐类 sigmoid），不是互斥 softmax。
- 24 → 14 确定性映射表见 `docs/data_audit.md` §3.2；丢弃 `未知/其他/谣传/行政事件/危险物质泄漏`。
- **洪涝多标签共激活**：原始类型或子类型/描述命中 `洪涝|洪水|内涝|渍涝|山洪|凌汛` 时，
  与 `暴雨` **同时**激活（详见 `docs/experiment_protocol.md` §3.3）。
- 级联正样本：`暴雨→洪涝`、`暴雨→洪涝→地质灾害` 三元链。

### 3.4 基准规模（实测，见 `docs/phase_log.md`）

| 区域 | 节点 | 边 | 跨省边占比 |
|---|---:|---:|---:|
| SW-Cascade | 398 | 5,388 | 22.2% |
| SE-Typhoon | 353 | 3,587 | 29.5% |
| North-Agro | 589 | 7,084 | 24.9% |

---

## 4. 实验协议速查

- **基线五族**：ST-GNN / TKG / 物理 / LLM 增强 / **Stitched-GNN-LLM（缝合基线，同参数量）**。
- **消融四条**：`w/o Phase-Transition Gate`、`w/o Advection PDE`、
  `w/o LLM Constitutive Functional`、`w/o Assimilation & RAG Prior`。
  每条必须能映射回 §1.2 方程中的某一项。
- **5 seeds**：`{42, 2023, 2024, 2025, 2026}`，报告 Mean ± Std + 配对 t 检验。
- **指标口径**：Task 1 同时报全网格与**活跃域**（`*_active`）；
  Hits@3/MRR 只在真有灾种的样本上统计（随机基线 3/14 ≈ 0.214）。
- **模型选择**：按验证集活跃域 Macro-F1，不按总损失；逐类阈值在验证集校准后用于测试集。
- 完整协议与超参：`docs/experiment_protocol.md` §1。

---

## 5. 工程约定

- Python 3 + PyTorch；数据处理 pandas/pyarrow；脚本入口放 `scripts/`。
- 任何从 `datasets/` 读数据的临时脚本用 `python3 -I` 运行，避免被同名模块劫持。
- 日志、checkpoint、预测结果写入 `outputs/`（已 gitignore）；大文件不入 Git。
- 每个实验脚本必须可通过命令行参数复现完整配置，并把配置快照写入输出目录。
- **并行 run 的硬性要求**：`OMP_NUM_THREADS=1`（否则 BLAS 线程超订）；
  checkpoint 必须按 seed 分目录；磁盘缓存必须原子写（`tmp` + `os.replace`）。
- 新增数据事实先写 `docs/data_audit.md`，再改代码；新增设计决议先写
  `docs/design_decisions.md`，再改代码。

---

## 6. 交付状态

见 `docs/phase_log.md`（按 Phase 追加，只记实测数字，不记契约）。
