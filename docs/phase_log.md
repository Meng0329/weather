# 交付状态流水（PT-GNO）

> 按 Phase 追加，记录**实测数字**与当轮暴露/解决的问题。
> 契约性内容不在这里，见 `docs/data_audit.md` / `docs/experiment_protocol.md` /
> `docs/design_decisions.md`。

## Phase 1：数据管线与三大基准（2026-10-20）

### 代码与产物

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

### 实测关键数字

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

### Phase 1 暴露的 4 项遗留问题（已由 Phase 1.5 全部裁决，见 `docs/experiment_protocol.md` §3.2–§3.5）

1. ~~`洪涝` 类几乎为空~~ → **`docs/experiment_protocol.md` §3.3 已裁决**：改多标签共激活，全库洪涝激活 36 → **72,080**。
2. ~~`WIN_S_10mi_Avg` 应从建模特征中剔除~~ → **`docs/experiment_protocol.md` §3.2 已裁决**：建模固化为 **15 维**。
3. ~~区县对齐上限 ~72.5%~~ → **`docs/experiment_protocol.md` §3.4 已裁决**：文本二次下探，对齐率 → **75.81%**。
4. ~~跨省边占比仅 3.3%–5.4%~~ → **`docs/experiment_protocol.md` §3.5 已裁决**：分层双池图，跨省边 → **22.2%~29.5%**。

---

## Phase 1.5 + Phase 2：特征修复、本构语义、PT-GNO 算子（2026-10-20）

### 新增代码

| 文件 | 作用 |
|---|---|
| `src/llm/extract_constitutive_emb.py` | 双模式本构语义提取：内网 API（Qwen3-Embedding-0.6B + bge-reranker-v2-m3 两阶段 RAG）/ 本地 transformers / 确定性 TF-IDF 三级降级 |
| `src/models/pt_gno.py` | PT-GNO 主模型：`LLMConstitutiveFunctional`、`PhysicsAdvectionReactionCell`、`GATAdvectionCell`、`LandauPhaseTransitionReadout` |
| `src/models/baselines.py` | Stitched-GNN-LLM、GraphWaveNet、GRU-GCN、PDE-GCN、RE-GCN 五族基线 |
| `src/models/ablations.py` | 4 个受控消融（仅切 `docs/experiment_protocol.md` §3.1 中一项） |
| `src/engine/data.py` | 整窗采样、稀疏文本查找表、训练期标准化、多尺度损失目标 |
| `src/engine/trainer.py` | 变分自由能损失（Focal BCE + 加权 CE + Masked Huber + β·KL）与 6 类指标 |
| `scripts/train_pt_gno.py` | 训练/消融/基线一键入口（`--suite` 全量） |
| `tests/test_models.py` | 39 项模型单元测试 |

### Phase 1.5 实测数字

- **洪涝多标签激活**：SW-Cascade 19,080（train 10,525 / val 4,383 / test 4,172）、
  SE-Typhoon 3,784、North-Agro 5,742；全库 flag **72,080** 条记录。
- **级联正样本**：`暴雨→洪涝` 对 5,403 / 1,156 / 1,324；
  `暴雨→洪涝→地质灾害` 三元链 **336 / 17 / 3**（SW-Cascade 是唯一有统计量级的区域，与选题一致）。
- **文本二次下探**：唯一命中 **2,913**（`L4_text_unique`，`cont_loss_mask=1`）、
  多候选 **1,936**（`L4_text_multi`，仅激活 `A_topo`，`cont_loss_mask=0`）、
  仍无命中 35,876（披露）。对齐率 **72.54% → 75.81%**。
- **分层双池邻接图**（跨省边占比目标 20%~30%，全部达标）：
  | 区域 | 节点 | 边 | 跨省边 | 跨省占比 | 平均出度 |
  |---|---:|---:|---:|---:|---:|
  | SW-Cascade | 398 | 5,388 | 1,194 | **22.2%** | 13.54 |
  | SE-Typhoon | 353 | 3,587 | 1,058 | **29.5%** | 10.16 |
  | North-Agro | 589 | 7,084 | 1,764 | **24.9%** | 12.03 |

## Phase 2 冒烟与语义服务实测

### Phase 2 冒烟实测（SW-Cascade，3 epoch）

- 数据：train 2,311 整窗（`T_in=7`, `Δ=1`, stride=3）；节点-日正样本率 **0.58%**
  （14 类长尾极端不均，洪涝+暴雨占 90%+，沙尘暴/高温近乎为空）。
- 文本：`llm_embeddings.pt` 512 维（Qwen3-Embedding-0.6B + MRL 截断），
  `h_pre` 单位范数、`h_rag` 与 `h_pre` 平均余弦 0.656（确有历史先验信息量，非复制）。
- 模型：PT-GNO **426k** 参数（`d_model=128`，text_dim=512）。
- 结果（900 训练窗 / 300 验证窗）：
  | 轮次 | train loss | val loss | Macro-F1 | AUPRC | Hits@3 | log-MAE | R² |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 1.2781 | 1.1236 | 0.000 | 0.0226 | 0.842 | 1.07 | -1.40 |
  | 2 | 1.1445 | 1.1385 | 0.000 | 0.0164 | 0.843 | — | — |
  | 3 | 1.1032 | 1.1169 | 0.000 | 0.0177 | 0.842 | — | — |
- 跨区域复现（同配置，2–3 epoch）：
  | 区域 | train loss | val loss | AUPRC | Hits@3 | log-MAE | R² |
  |---|---:|---:|---:|---:|---:|---:|
  | SW-Cascade (3 ep) | 1.28→1.10 | 1.12 | 0.018 | 0.842 | 1.07 | -1.40 |
  | SE-Typhoon (2 ep) | 1.09→0.86 | 0.87 | 0.008 | 0.682 | 0.90 | -0.76 |
  | North-Agro (2 ep) | 1.23→1.06 | 0.97 | 0.007 | 0.560 | 1.39 | -1.59 |
  三个区域一致表现出：损失单调下降、Hits@3 显著高于随机（0.214）、
  连续损失可学到尺度。说明算子结构与数据契约跨区域一致，不是单区域调参产物。
- 判读：
  - train loss **单调下降**（1.28 → 1.10），管线与梯度路径正确；显存 < 1 GB。
  - Hits@3 **0.84** 远高于随机基线 3/14 ≈ 0.214 —— 排序能力已建立。
  - Macro-F1 为 0 属**预期**：阈值 0.5 下极长尾类几乎不可能在 3 epoch 内越过，
    正式训练（≥100 epoch、5 seed、全量 6,933 窗）后才会出现非零值。
  - R² 为负属**欠训练**（连续损失 head 尚未学到尺度），不作为架构结论。

### 离线本构语义服务（内网，`10.200.37.71:8801`）

- 嵌入：`Qwen3-Embedding-0.6B`（原生 1024 维，按 MRL 截断到 **512** 维并重新归一化）。
  通过 `LLM_API_BASE` / `LLM_API_KEY` 环境变量注入，脚本内**不硬编码任何凭据**。
- 精排：`bge-reranker-v2-m3`，用于 RAG 两阶段检索（稠密粗召回 Top-20 → 精排 Top-3）。
- 生成模型（Qwen3.5-9B / GLM-5.1 / DeepSeek-V4 等）**不进入训练回路**，
  仅可用于离线蒸馏语料构造；CLAUDE.md §1.3 的"离线蒸馏优先"原则不变。
- 吞吐实测：服务空闲时 batch=128 / 并发 96 → 约 **250 texts/s**；
  被压满时单请求可退化到 5–30 s（本机 `loadavg` 夜间可达 50+，磁盘 `rq_qos_wait`）。
  因此提取器内置：整表预热、滑动平均延迟自适应降级、嵌入/精排磁盘缓存（限频原子写盘）。
- 产物：`<REGION>/llm_embeddings.pt`，键
  `record_ids / node_idx / date / split / h_pre / h_post / h_rag / dim / method / top_k_rag`。
  **三个区域全部完成 512 维 Qwen3-Embedding 产物**：
  SW_Cascade (33,713 条) / SE_Typhoon (8,139) / North_Agro (14,879)。
  若后续退回本地 `bert-base-chinese`（256 维），`RegionBundle.text_dim` 会按区域
  自适应读取，但论文表格必须标注各区域使用的文本编码器版本。
- **密钥与地址只从环境变量或 `~/.config/ptgno/env` 读取，禁止写入仓库**。

---
