# CLAUDE.md — PT-GNO 项目持久记忆

> 本文件是本仓库的唯一项目契约。任何实现、实验、写作冲突时以本文件 + `docs/idea.md` 为准；
> 与 `docs/复现方案.md` 中的示例代码冲突时，以"第一性原理架构"与"数据契约"章节为准。

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

## 2. 目录结构规范

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

---

## 3. 数据契约（本地已验证，2026-10-20）

### 3.1 灾情直报数据

| 路径 | 规模 | 说明 |
|---|---|---|
| `datasets/datasets/disaster_records_unified.csv` | 305 MB, 151,591 行 × 39 列 | UTF-8-BOM，主表 |
| `datasets/datasets/disaster_records_part1_2000-2010.csv` | 28,144 行 | 训练 |
| `datasets/datasets/disaster_records_part2_2011-2014.csv` | 36,417 行 | 训练 |
| `datasets/datasets/disaster_records_part3_2015-2018.csv` | 34,302 行 | 训练 |
| `datasets/datasets/disaster_records_part4_2019-2021.csv` | 27,739 行 | 验证 |
| `datasets/datasets/disaster_records_part5_2022-2025.csv` | 24,989 行 | 测试 |

- 时间划分固定：**Train 2000–2018 = 98,863 / Val 2019–2021 = 27,739 / Test 2022–2025 = 24,989**。
- 关键字段：`disaster_type_std`、`start_time`、`end_time`、`province/city/county/report_unit`、
  损失字段、文本字段、`report_status`、`report_time`、`search_text`。

### 3.2 地面气象站日值数据

| 路径 | 规模 | 说明 |
|---|---|---|
| `datasets/雷达数据/Year_2000.csv` … `Year_2025.csv` | 26 文件，23,158,575 行 | 逐日，无坐标 |

- 列（20）：`City, Cnty, Province, Year, Mon, Day, PRS_Avg, WIN_S_Max, WIN_S_Inst_Max,
  WIN_S_2mi_Avg, WIN_S_10mi_Avg, TEM_Avg, TEM_Max, TEM_Min, PRE_Time_2020, Lit, GaWIN, Thund, Tord, Fog`
- 覆盖 34 个省级单位、359 个地市、2,414 个 `(Province, City, Cnty)` 站点单元、
  2,336 个不同区县名。
- `说明.docx` 仅给出上述 20 列的中文名与单位；**"特征值说明"章节不是规则表**。

### 3.3 缺失 / 不存在的数据文件（必须补齐或改设计）

- `datasets/datasets/disaster_records_unified.parquet` — **不存在**（数据字典声称存在）。
  当前只有 CSV 形态的 `disaster_records_unified.csv`；`复现方案.md` Step 1 中
  "读取 `disaster_records_unified.parquet`" 的代码路径需改为 CSV，或先自行构建 parquet。
- 气象站点经纬度 / 行政区边界 shapefile — **不存在**，但 80 km 邻接图与平流核需要。
  → 备选方案：用 `City`/`Cnty` 的行政区划层级构建**层级邻接图**（同城 + 同省），
    或按公开县级中心经纬度表离线构建（本仓库无此表，需外部补齐）。
- 行政区划代码映射表（县级）— **不存在**。
- `99xxxx` 特征值原始规则说明 — **不存在**（见 §4.1）。
- `datasets/雷达数据/说明.docx` — 仅 20 列字段名与单位表；"特征值说明"章节经解包后
  是 **base64 → UTF-16LE → base64 → zlib** 的三层嵌套，内容为一个 EndNote `.els`
  参考文献库二进制，**不含任何特征值规则**。不要从中寻找编码语义。

### 3.4 主表与分片一致性

- `disaster_records_unified.csv` 与 5 个分片行数一致（28,144 + 36,417 + 34,302 +
  27,739 + 24,989 = **151,591**）。
- 建议：训练/验证/测试一律直接读分片 CSV（`usecols` 按需裁剪），避免整表加载 305 MB。
  UTF-8-BOM 需用 `encoding='utf-8-sig'` 读，否则首列名会带 `﻿`。

---

## 4. 已验证的数据陷阱（实现前必须处理，勿按文档默认值写死）

### 4.1 `99xxxx` 特征值家族的真实分布（全量无截断审计，2026-10-20 修正）

> 早期用 `grep` + `timeout` 扫描被截断，曾误判"相态码完全不存在"。以下为
> `scripts/audit_sentinel_codes.py` 对 **26 个文件 / 23,165,849 行**的无截断全量审计结果。

| 码族 | 出现次数 | 说明 |
|---|---:|---|
| `999999`（缺测） | 25,529,199 | 唯一大量存在的特征值 |
| `999990` / `999991` / `999992`（微量降水） | 1,685,278 | 仅出现在 `PRE_Time_2020`；999991/999992 是微量变体 |
| `999998`（不观测） | 0 | 实测不存在 |
| `999700.x`（雨夹雪相态） | **1** | Year_2002 广东潮州市 |
| `999800.x`（雾露霜相态） | **1** | Year_2003 黑龙江七台河市 |
| `9996xx` / `998xxx` / `997xxx` | 0 | 实测不存在 |

- **结论修正**：相态码不是"不存在"，而是**全库仅 2 行**（占比 8.6e-8）。
  因此 `idea.md` 的"降水物理相态解耦"在统计上**无建模价值**，
  但仍应在论文中如实表述为"相态编码存在但极度稀疏（2/23M），不构成可用特征"。
- 处理：`999000 ≤ v < 999990` 一律按非物理值剔除（置 NaN 后插值）；
  `999990/999991/999992` 在 `PRE_Time_2020` 中解码为 `0.05 mm` + `PRE_trace_flag`。
- 另发现 **371 行物理不合理值**（PRS 1496.7、WIN_S_Max 204.8、WIN_S_2mi_Avg 249830.2、
  TEM_Max 76.8、TEM_Min −99.6、PRE 3059.7），已按 `PHYSICAL_BOUNDS` 剔除后再聚合。
  **顺序要求**：必须先剔除再按区县求均值，否则异常值会污染区县均值。

### 4.2 天气现象列（Lit/GaWIN/Thund/Tord/Fog）分年覆盖极不稳定 ⚠️

全量按年统计 `=999999`（缺测）行数：

| 年份区间 | `Lit` 闪电 | `Thund` 雷暴 | `Tord` 龙卷风 | `GaWIN` 大风 | `Fog` 雾 |
|---|---|---|---|---|---|
| 2000–2011 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| 2012 | 2.6% | 2.6% | 2.6% | 0.01% | 0.01% |
| 2015 | 5.0% | 5.0% | 5.0% | 4.5% | 4.5% |
| **2016–2025** | **100%** | **100%** | **100%** | 2.4%–38% | 2.4%–38% |

- 结论：**`Lit`、`Thund`、`Tord` 在 2016–2025 全量缺测**，即验证集（2019–2021）与
  测试集（2022–2025）中这三列**完全不可用**；训练集也只有 2000–2015 可用。
- 处理要求：这三列必须在 `src/data/` 中显式标记为"按年份分段的可用特征"，
  并按缺失掩码输入；**禁止**把 `999999` 当 0 或当 1 参与训练，
  否则模型在验证/测试阶段会学到"闪电/雷暴/龙卷风恒不发生"的伪规律。
- 审稿风险：`idea.md` 把"雷暴、龙卷风、闪电"列为致灾能量注入项，需在论文中
  如实披露其在测试期的不可用性，或改用其他代理变量。

### 4.3 报告时序字段填充率低于文档声称

- `report_status` 与 `report_time` 的全量填充率**恰好都是 60.63%**（两者掩码一致，
  数据字典称 ~90%）→ Task 3 的可用样本上限就是这 60.63%。
- 状态分布：`初报` **45,225**、`终报` **43,895**、`续报` **2,790（1.8%）**、缺失 60,681。
- 分年续报量：part1=1、part2=11、part3=1,179、part4=760、part5=839。
- 以 `(province, county, disaster_type_std)` 聚类，同时含初报与终报的组仅 **4,252 / 12,464**，
  覆盖 89,655 条记录 → Task 3 可做，但必须自建事件链定义并在论文中报告样本量。
- 真正的"初报→续报→终报"三阶段完整链会显著少于 4,252，需在构建后实测。

### 4.4 灾害类型不是 14 类

- `disaster_type_std` 实际有 **24 个取值**，包括非气象灾种与占位符：
  `冻害、凌汛、风暴潮、其他、未知、森林草原火灾、大气污染、干热风、
  危险物质泄漏(气体、核应急)、气象有关的谣传和不当舆论、气象部门参与减轻事件` 等。
- 处理要求：在 `src/data/` 中显式定义 **24 → 14 的确定性映射表**，并记录被合并/丢弃的类；
  不得在模型里假定 `num_disasters=14` 就自动成立。

### 4.5 区县名不可直接 join

- 灾情表 `county` 是**完整层级串**，如 `四川省德阳市中江县`、`重庆市忠县`。
- 气象表 `Cnty` 是**裸区县名**，如 `尚志市`。
- 处理要求：解析 `province/city/county` 层级、统一"市/县/区/旗"后缀、建立行政区划代码主键；
  仅做字符串相等连接会大面积失配。

### 4.6 经济损失异常值不止 12 条

- `economic_loss ≥ 1e8` 万元：**59 条**；`≥ 1e9` 万元：**12 条**；最大值 18,200,000,500。
- `复现方案.md` 的"`≥1e8` 除以 1e4"只是启发式，**不能直接当定论**。
- 处理要求：结合 `loss_level` 与 `impact_description` 中的金额文本做量纲审计，
  逐条判定并保留审计日志；论文需报告清洗前后分布。

### 4.7 气象表存在重复站点-日

- 2024 年：848,078 个唯一 `(City, Cnty, Province, Y-M-D)` 组中存在 **46,782 个重复组**。
- 处理要求：先确定去重规则（同值去重 / 取均值 / 取首条），并验证是否跨年一致性；
  重复率需在论文数据统计表中披露。

### 4.8 其他

- 2000–2007 年仅 17 条灾情记录，训练集实际有效起点约为 2008 年。
- `data_quality_report.json` 的 `source_files` 为空，不能用于溯源。
- 省份分布极不均衡：云南 19,437 条 vs 上海 254 条；三大区域基准之间不可直接横向比较绝对量。

### 4.9 字段语义与数据字典不符 ⚠️

全量实测填充率（n=151,591）：

| 字段 | 实测填充率 | 数据字典声称 | 实测语义 |
|---|---|---|---|
| `impact_description` | 84.58% | 84.6% | ✓ |
| `weather_summary` | 83.59% | 83.6% | ✓ |
| `warning_info` | 62.59% | 62.6% | ✓ |
| `remarks` | 15.29% | 15.3% | ✓ |
| `disaster_subtype` | **65.26%** | ~50% | 需重估 |
| `county` | **78.74%** | ~70% | 需重估 |
| `city` | **93.87%** | ~85% | 需重估 |
| `end_time` | **79.31%** | ~80% | ✓ |
| `affected_crops` | **45.40%** | ~30% | **不是作物名列表**，而是自由文本，常与 `impact_description` 内容重复（完全相同的占 14.3%） |
| `tabulator` | **70.26%** | ~30% | 含个人信息，应丢弃 |
| `contact` | **68.61%** | ~30% | 含个人信息，应丢弃 |
| `data_source` | 55.05% | ~60% | ✓ |

- `affected_crops` 不能直接作为"承灾体-作物"实体来源，需用 NER/规则从文本抽取，
  或改从 `impact_description` 抽取。
- `tabulator` / `contact` 属个人可联系信息，**不得**进入任何模型特征或论文数据表。

---

## 5. 实验协议（固定）

### 5.1 三大区域基准（按 `province` 过滤）

| 数据集 | 省份 | 核心现象 |
|---|---|---|
| **SW-Cascade** | 云南、四川、贵州、重庆 | 暴雨→洪涝→地质灾害链；伤亡与转移 |
| **SE-Typhoon** | 广东、福建、浙江、海南、江苏 | 台风、大风、暴雨；房屋倒损与社会停摆 |
| **North-Agro** | 河南、山东、河北、黑龙江、吉林、辽宁 | 干旱/高温/连阴雨 vs 冰雹/雷电/低温/雪灾；农业三面积 |

### 5.2 任务与指标

- **Task 1** 相变与级联预测（$\Delta t\in\{1,3,5,7\}$）：Macro-F1、Micro-F1、AUPRC、MRR、Hits@1/3/5。
- **Task 2** 多维损失联合预测：`loss_level` 5 分类（Accuracy、Weighted-F1）+
  `economic_loss`、`affected_population`、`disaster_area`、`total_loss_area` 等
  （log 空间 MAE、RMSE、MAPE、$R^2$）。
- **Task 3** 初报→续报→终报同化：Error Reduction Rate、Assimilation Gain。

### 5.3 基线与消融

- 基线五族：ST-GNN（GraphWaveNet/AGCRN/DSTAGNN/GinAR）、TKG（RE-GCN/TiRGN/HisRES/CENET）、
  物理（PINN/PDE-GCN/ClimODE/FNO）、LLM 增强（Time-LLM/OFA/GraphGPT/LLM4TS）、
  **Stitched-GNN-LLM（缝合基线，同参数量 ~1.8M，回应审稿人）**。
- 消融：w/o Phase Gate、w/o Advection PDE、w/o LLM Constitutive、
  **w/o Assimilation & RAG Prior**（原 99xxxx 消融已作废，见 §6.5）。

### 5.4 复现纪律

- 5 个 seed：`{42, 2023, 2024, 2025, 2026}`，报告 Mean ± Std + 双侧配对 t 检验。
- 默认超参：$T_{in}=7$、$d=64$、$\epsilon=0.2$、$\beta=0.05$、AdamW lr $1e-3$、
  weight_decay $1e-4$、batch 16、max 100 epochs、early-stop patience 15。
- 测试集只在最终评估运行一次；所有归一化统计量只从训练集计算。

---

## 6. 阻塞项决议（规划者已裁定，2026-10-20）

以下为正式决议，**取代**上文的候选方案。实现必须严格遵循本节。

### 6.1 决议清单

| # | 阻塞项 | 决议 |
|---|---|---|
| 1 | 路径与格式 | 读 `datasets/datasets/disaster_records_part{1..5}*.csv`（或 `unified.csv`）；读 `datasets/雷达数据/Year_2000..2025.csv`；产物统一落 `datasets/processed/` 的 `.parquet` |
| 2 | 隐私字段 | 读取灾情表时**立即丢弃** `tabulator`、`contact` |
| 3 | 经济损失量纲 | 全部 `economic_loss >= 1e8`（万元）记录 **÷ 1e4** 做 元→万元 修正 |
| 4 | 气象特征 | 固定 **16 维无缺失**物理向量（见 §6.2）；剔除 2016–2025 全缺的 `Lit`/`Thund`/`Tord` |
| 5 | 灾害类型 | **24 → 14 确定性映射**（见 §6.3），过滤非自然气象灾害与"未知"/空值，日志输出完整映射明细 |
| 6 | 区县对齐与邻接 | 以气象 `(Province, City, Cnty)` 为节点字典；层级剥离+词干匹配对齐灾情区县；邻接 = 同地市连通 **+** 同省/邻省内按训练集 `delta_prs_24h` 与 `TEM_Avg` 皮尔逊相关 Top-6 物理近邻 |
| 7 | 消融 4 | 更新为 **`w/o Assimilation & RAG Prior`**（移除初/续报同化门 + 历史 RAG 先验 KL） |

### 6.2 16 维物理特征向量（固定顺序，模型输入契约）

| # | 特征 | 定义 |
|---|---|---|
| 1–9 | `PRS_Avg`, `WIN_S_Max`, `WIN_S_Inst_Max`, `WIN_S_2mi_Avg`, `WIN_S_10mi_Avg`, `TEM_Avg`, `TEM_Max`, `TEM_Min`, `PRE_Time_2020` | 连续要素。`999999`/`999998` → NaN 后按站点线性插值；`PRE_Time_2020` 的 `999990` → 0.05 mm |
| 10–11 | `GaWIN`, `Fog` | 全时段可用的二值天气现象 |
| 12 | `PRE_trace_flag` | `(PRE_Time_2020_raw == 999990)` |
| 13 | `rain_snow_phase` | `(PRE_Time_2020 > 0) * sigmoid(-(TEM_Avg - 0.5))` — 固液降水热力学相态 |
| 14 | `gust_factor` | `WIN_S_Inst_Max / (WIN_S_Max + 0.1)` — 阵风脉动系数 |
| 15 | `tem_diurnal_range` | `TEM_Max - TEM_Min` — 日较差 |
| 16 | `delta_prs_24h` | `PRS_Avg(t) - PRS_Avg(t-1)` — 24 小时变压 |

**⚠️ 决议修正（2026-10-20 实测）**：`WIN_S_10mi_Avg` 在 **2018–2025 全量 100% 缺测**
（2000–2004 缺测 68%–100%，2005–2017 可用 1.5%–15%）。
即验证集与测试集该列**完全不可用**，训练集早期也不可用。
因此第 5 维从"连续要素"降级为**分段可用特征**：
- `clean_and_align.py` 仍计算该列，但会同时输出 `meteo_feature_availability.csv`
  逐列逐年缺测率；
- 若某列在验证/测试区间缺测率 ≥ 95%，在张量中该通道按 `observed_mask=False` 标记，
  并**在建模阶段从 `feat_dim` 中剔除**（最终有效通道为 **15 维**）。
- 论文必须披露：`WIN_S_10mi_Avg` 不能作为跨期稳定特征，只能作为训练期内特征
  （若启用），否则它在 val/test 上恒等于插值填充的常量，构成伪信号。

同理 `Lit`/`Thund`/`Tord` 在 2016–2025 全缺（§4.2），已剔除。

### 6.3 24 → 14 灾害类型映射（确定性，完整表）

| 实测 `disaster_type_std` | 记录数 | → 14 类目标 |
|---|---|---|
| 暴雨 | 70,158 | 暴雨 |
| 冰雹 | 21,270 | 冰雹 |
| 大风 | 17,368 | 大风 |
| 干旱 | 10,161 | 干旱 |
| 台风 | 5,514 | 台风 |
| 雪灾 | 4,980 | 雪灾 |
| 雷电 | 4,662 | 雷电 |
| 地质灾害 | 3,324 | 地质灾害 |
| 低温 | 3,078 | 低温 |
| 冻害 | 2,849 | **低温** |
| 未知 | 2,052 | **丢弃** |
| 雾霾 | 1,226 | 雾霾 |
| 森林草原火灾 | 1,124 | **干旱** |
| 连阴雨 | 988 | 连阴雨 |
| 高温 | 786 | 高温 |
| 其他 | 763 | **丢弃** |
| 沙尘暴 | 453 | 沙尘暴 |
| 气象有关的谣传和不当舆论 | 387 | **丢弃** |
| 大气污染 | 216 | **雾霾**（判断项，⚠️ 可一键改判为丢弃） |
| 干热风 | 99 | **高温** |
| 气象部门参与减轻事件 | 78 | **丢弃** |
| 凌汛 | 36 | **洪涝** |
| 风暴潮 | 15 | **台风** |
| 危险物质泄漏(气体、核应急) | 4 | **丢弃** |

- 保留 14 类：暴雨、洪涝、台风、干旱、高温、低温、雪灾、大风、冰雹、雷电、沙尘暴、雾霾、连阴雨、地质灾害。
- **⚠️ 已知副作用**：`洪涝` 在实测 24 类中**不存在**，归并后仅剩 `凌汛` 贡献的 36 条（占保留量的 0.02%）。
  原始 `disaster_type` 中的"洪涝"已被标准化为 `暴雨`（70,158 条 `暴雨洪涝` → `暴雨`）。
  这直接削弱 SW-Cascade 的"暴雨→洪涝→地质灾害"三级级联监督信号，论文中必须披露，
  并考虑从 `disaster_subtype` / `impact_description` 重新抽取洪涝节点（属后续决策）。
- 映射表与丢弃明细由 `src/data/clean_and_align.py` 落盘为
  `datasets/processed/disaster_type_mapping.csv` 与 `datasets/processed/cleaning_report.json`。

### 6.4 区县重复站点-日的性质修正（§4.7 更新）

- 实测 6.06% 的行属于 `(Province, City, Cnty, Year, Mon, Day)` 重复键，
  且 **98.4% 的重复组内取值互不相同**（例：上海市徐汇区 2024-01-01 有 3 条，
  `PRS_Avg` = 1019.2 / 1019.6 / 1028.7 hPa）。
- 结论：这些**不是重复记录，而是同一行政区内多个不同物理观测站**（海拔/微气候差异）。
- 决议：按**区县均值**聚合（`mean`），与"区县连续气象势能场"的本体一致；
  特殊值必须**先转 NaN 再聚合**，否则 `999999` 会污染均值。
  二值列 `GaWIN`/`Fog` 与 `PRE_trace_flag` 用 `max` 聚合（任一站观测到即置 1）。
- `Lit`/`Thund`/`Tord` 已剔除（§6.2），不再受此影响。

### 6.5 消融实验清单（更新后）

1. `w/o Phase-Transition Gate`
2. `w/o Advection PDE`
3. `w/o LLM Constitutive Functional`
4. **`w/o Assimilation & RAG Prior`**（原"w/o 99xxxx 编码消融"已作废——§4.1 证明该编码不存在）

---

## 7. Phase 1 交付状态（2026-10-20 已完成并实测）

### 8.1 代码与产物

| 文件 | 作用 |
|---|---|
| `src/data/common.py` | 共享契约：16 维特征顺序、24→14 映射表、省级邻接表、物理合理域 |
| `src/data/entity_align.py` | 区县层级剥离与词干匹配（L1/L2/L3 三级回退） |
| `src/data/clean_and_align.py` | 灾情清洗 + 气象去重/解码/插值/特征构造 |
| `src/data/build_benchmarks.py` | 三大基准、空间邻接图、Task 3 金标准子集、`docs/paper/data_stats.md` |
| `scripts/audit_sentinel_codes.py` | 99xxxx 特征值家族全量审计（无 timeout 截断） |
| `tests/test_data_pipeline.py` | 32 项单元测试（纯逻辑） |
| `tests/test_artifacts.py` | 36 项产物集成测试 |

产物目录 `datasets/processed/`（已被 `.gitignore` 排除，不入库）：
`meteo_features.npy (2408, 9497, 16) float32`、`meteo_observed_mask.npy`、
`station_index.parquet`、`disaster_cleaned.parquet`（148,307 行）、
`disaster_type_mapping.csv`、`cleaning_report.json`、
`assimilation_golden_subset.parquet`、`<REGION>/{nodes,adj_mask,adj_weight,records,events,daily_labels}`。

### 8.2 实测关键数字

- 灾情：输入 151,591 → 保留 **148,307**（丢弃 3,284：未知 2,052 / 其他 763 / 谣传 387 /
  行政事件 78 / 危险物质泄漏 4）；**59 条**经济损失 ÷1e4 修正；**14 类全部存在**。
- 区县对齐命中率 **72.54%**（107,582/148,307）：
  L1 精确 96,216 / L2 省+市+词干 5,938 / L2 省+词干 3,573 / L3 市+词干 651 / L3 全局词干 1,204。
  未命中 40,725 条中 **28,675 条 `county` 为空**（仅到市级，无法下沉到区县）。
- 气象：2,408 节点 × 9,497 天（2000-01-01 ~ 2025-12-31）× 16 维；
  特征值族缺测 25.5M、微量降水 1.69M、相态码仅 2 行、物理不合理值 371 行；
  不可观测格点占比 **5.1%**（按通道输出 `meteo_observed_mask.npy`）。
- 三大基准（节点/边/训练/验证/测试 事件数）：
  - **SW-Cascade** 398 / 4,686 / 13,348 / 3,617 / 4,566
  - **SE-Typhoon** 353 / 3,021 / 3,764 / 643 / 720
  - **North-Agro** 589 / 6,013 / 6,057 / 1,645 / 952
- Task 3 金标准：多阶段链 **3,863**（初报+终报 3,617，含完整三阶段 **246**），
  记录 17,067（初报 8,864 / 终报 7,150 / 续报 1,053）。

### 8.3 Phase 1 暴露、需规划者裁决的遗留问题

1. **`洪涝` 类几乎为空**：全库仅 36 条（来自 `凌汛`），SW-Cascade 内 12 条、SE-Typhoon 0 条。
   "暴雨→洪涝→地质灾害"三级级联缺少中间节点监督。需决定：是否从 `disaster_subtype` /
   `impact_description` 文本重新抽取洪涝节点（原始 `暴雨洪涝` 已被标准化为 `暴雨`，70,158 条）。
2. **`WIN_S_10mi_Avg` 应从建模特征中剔除**：2018–2025 全缺，val/test 恒为掩码填充值。
   建议有效特征维度定为 **15**。
3. **区县对齐上限 ~72.5%**：剩余缺口主要是只有市级、无区县的记录（19.3% 全库）。
   若论文需要更高的对齐率，需引入区县实体链接或按市级聚合的降级方案。
4. **邻接图跨省边占比仅 3.3%–5.4%**：Top-K 物理近邻绝大多数落在同省内，
   跨省台风通道偏弱。可考虑把 `PHYSICAL_NEIGHBOR_K` 提高或改用跨省候选池加权。

---

## 8. 工程约定

- Python 3 + PyTorch；数据处理用 pandas/pyarrow；脚本入口放 `scripts/`。
- 任何从 `datasets/` 读数据的临时 Python 脚本用 `python3 -I` 运行，
  避免被数据目录中的同名模块劫持。
- 日志、checkpoint、预测结果写入 `outputs/`（已 gitignore）。
- 每个实验脚本必须可通过命令行参数复现完整配置，并把配置快照写入输出目录。
- 新增数据事实先写回本文件 §3/§4，再改代码。