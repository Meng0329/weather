# 实验协议与阻塞项决议（PT-GNO）

> CLAUDE.md 只保留速查表；本文件是协议与决议全文。
> 实现层/数值层决议见 `docs/design_decisions.md`。

## 1. 实验协议（固定）

### 三大区域基准（按 `province` 过滤）

| 数据集 | 省份 | 核心现象 |
|---|---|---|
| **SW-Cascade** | 云南、四川、贵州、重庆 | 暴雨→洪涝→地质灾害链；伤亡与转移 |
| **SE-Typhoon** | 广东、福建、浙江、海南、江苏 | 台风、大风、暴雨；房屋倒损与社会停摆 |
| **North-Agro** | 河南、山东、河北、黑龙江、吉林、辽宁 | 干旱/高温/连阴雨 vs 冰雹/雷电/低温/雪灾；农业三面积 |

### 任务与指标

- **Task 1** 相变与级联预测（$\Delta t\in\{1,3,5,7\}$）：Macro-F1、Micro-F1、AUPRC、MRR、Hits@1/3/5。
- **Task 2** 多维损失联合预测：`loss_level` 5 分类（Accuracy、Weighted-F1）+
  `economic_loss`、`affected_population`、`disaster_area`、`total_loss_area` 等
  （log 空间 MAE、RMSE、MAPE、$R^2$）。
- **Task 3** 初报→续报→终报同化：Error Reduction Rate、Assimilation Gain。

### 基线与消融

- 基线五族：ST-GNN（GraphWaveNet/AGCRN/DSTAGNN/GinAR）、TKG（RE-GCN/TiRGN/HisRES/CENET）、
  物理（PINN/PDE-GCN/ClimODE/FNO）、LLM 增强（Time-LLM/OFA/GraphGPT/LLM4TS）、
  **Stitched-GNN-LLM（缝合基线，同参数量 ~1.8M，回应审稿人）**。
- 消融：w/o Phase Gate、w/o Advection PDE、w/o LLM Constitutive、
  **w/o Assimilation & RAG Prior**（原 99xxxx 消融已作废，见 `docs/experiment_protocol.md` §3.1）。

### 复现纪律

- 5 个 seed：`{42, 2023, 2024, 2025, 2026}`，报告 Mean ± Std + 双侧配对 t 检验。
- 默认超参：$T_{in}=7$、$d=64$、$\epsilon=0.2$、$\beta=0.05$、AdamW lr $1e-3$、
  weight_decay $1e-4$、batch 16、max 100 epochs、early-stop patience 15。
- 测试集只在最终评估运行一次；所有归一化统计量只从训练集计算。

---

## 2. 阻塞项决议（规划者已裁定，2026-10-20）

以下为正式决议，**取代**上文的候选方案。实现必须严格遵循本节。

### 决议清单

| # | 阻塞项 | 决议 |
|---|---|---|
| 1 | 路径与格式 | 读 `datasets/datasets/disaster_records_part{1..5}*.csv`（或 `unified.csv`）；读 `datasets/雷达数据/Year_2000..2025.csv`；产物统一落 `datasets/processed/` 的 `.parquet` |
| 2 | 隐私字段 | 读取灾情表时**立即丢弃** `tabulator`、`contact` |
| 3 | 经济损失量纲 | 全部 `economic_loss >= 1e8`（万元）记录 **÷ 1e4** 做 元→万元 修正 |
| 4 | 气象特征 | 固定 **16 维无缺失**物理向量（见 `docs/data_audit.md` §3.1）；剔除 2016–2025 全缺的 `Lit`/`Thund`/`Tord` |
| 5 | 灾害类型 | **24 → 14 确定性映射**（见 `docs/data_audit.md` §3.2），过滤非自然气象灾害与"未知"/空值，日志输出完整映射明细 |
| 6 | 区县对齐与邻接 | 以气象 `(Province, City, Cnty)` 为节点字典；层级剥离+词干匹配对齐灾情区县；邻接 = 同地市连通 **+** 同省/邻省内按训练集 `delta_prs_24h` 与 `TEM_Avg` 皮尔逊相关 Top-6 物理近邻 |
| 7 | 消融 4 | 更新为 **`w/o Assimilation & RAG Prior`**（移除初/续报同化门 + 历史 RAG 先验 KL） |

## 3. Phase 1.5 决议（特征、洪涝多标签、文本下探、双池图）

### 消融实验清单（更新后）

1. `w/o Phase-Transition Gate`
2. `w/o Advection PDE`
3. `w/o LLM Constitutive Functional`
4. **`w/o Assimilation & RAG Prior`**（原"w/o 99xxxx 编码消融"已作废——`docs/data_audit.md` §2.1 证明该编码不存在）

### Phase 1.5 决议：剔除 WIN_S_10mi_Avg，确立 15 维契约

- 建模输入固定为 **15 维**：`FEATURE_NAMES_15 = FEATURE_NAMES - {WIN_S_10mi_Avg}`，
  索引映射由 `C.FEATURE_IDX_15` 给出。
- 落盘的 `datasets/processed/meteo_features.npy` 仍为 16 维（不重算 3 GB 张量），
  区域基准构建时按 `FEATURE_IDX_15` 索引出 15 维并写入 `<REGION>/meteo_features.npy`
  （`feature_contract.json` 记录通道名）。
- **平流核物理输入固定为 3 维**：`[ΔPRS, WIN_S_Max, WIN_S_2mi_Avg]`
  （`C.ADV_PHYSICAL_COLS`），三者全时段无缺失。

### Phase 1.5 决议：洪涝多标签相变激活

`A_topo` 是 14 类灾害的**多标签二值向量**，不是互斥单分类。除主类外，附加规则：

- 原始 `disaster_type` 命中 `洪涝|洪水|内涝|渍涝|山洪|凌汛` → 激活 `洪涝`
  （含 70,158 条 `暴雨洪涝`，使其**同时**激活 `暴雨` 与 `洪涝`）；
- `disaster_subtype` 或 `impact_description` 命中
  `洪涝|山洪|内涝|渍涝|洪水|漫堤|决口|溃坝|淹没农田|积水受淹` → 激活 `洪涝`。
- 级联正样本：`暴雨洪涝` 且 `duration_days >= 2` 记一个 `暴雨→洪涝` 对；
  同区县内洪涝激活后 `[t, t+3]` 天出现 `地质灾害` 记一个 `暴雨→洪涝→地质灾害` 三元链。
- 实测：全库洪涝激活 **72,080** 条记录（原单标签口径仅 36 条）。

### Phase 1.5 决议：市级记录文本二次下探（Text-Guided County Recovery）

对 `county` 为空但 `province`/`city` 已知的记录，在该地市下辖 `Cnty` 名集合中检索
`report_unit` 与 `impact_description`（完整区县名命中，或长度 ≥3 的词干命中）：

- **唯一命中** → 完整纳入该区县监督（`align_level = L4_text_unique`，`cont_loss_mask = 1`）；
- **多候选** → 仅在候选区县上激活 `A_topo`，`align_level = L4_text_multi`，
  **`cont_loss_mask = 0`**（避免均摊损失造成空间污染）；
- **仍无命中** → 剔除并在 `data_stats.md` 披露。

记录展开为 record→node 链接（`records.parquet`），`n_records_unique` 与
`n_record_node_links` 分别统计。

### Phase 1.5 决议：分层双池空间平流图

- 池 1（同域）：同地市完全连通 + 同省内相关系数 **Top-4**（`C.K_IN_PROV`）；
- 池 2（跨域）：相邻省份（限区域子集内）相关系数 **Top-3**（`C.K_CROSS_PROV`）；
- 仍只用训练期（2000–2018）序列计算相关，只保留正相关边。
- 目标跨省边占比 20%~30%（Phase 1 双池前仅 3.3%~5.4%）。

---
