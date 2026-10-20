### 一、 顶刊核心立意与第一性原理创新方案（Idea Blueprint）

#### 1. 论文题目与核心定位

* **英文标题**：**PT-GNO: Physics-Informed Phase-Transition Graph Neural Operator for Unifying Continuous Meteorology and Discrete Disaster Knowledge Evolution**
* **中文对照**：物理引导的相变图神经算子：统一连续气象场与离散灾害知识演化的变分动力学框架
* **目标期刊**：*IEEE Transactions on Knowledge and Data Engineering (TKDE)* 或 *Information Fusion*（中科院一区 Top）

---

#### 2. 第一性原理下的三大核心创新点（拒绝模块缝合）

**创新点一：连续物理场与离散时序知识图谱的“拓扑相变”本体统一（Ontology Unification）**

* **痛点**：现有方法将连续地面气象数值序列与离散灾害知识图谱视为异构双模态进行生硬的特征拼接，忽略了“由天致灾”的物理突变本质。
* **第一性原理解法**：借鉴非平衡态统计物理中的**朗道相变理论（Landau Phase Transition Theory）**，将每日连续观测的区县气象数据（平均气压 `PRS_Avg`、最大/极大风速 `WIN_S_Max`/`WIN_S_Inst_Max`、平均/最高/最低气温 `TEM_Avg`/`TEM_Max`/`TEM_Min`、20-20时降水量 `PRE_Time_2020` 等） 定义为**底层连续热力学势能场（亚临界态）**；将离散突发的灾害事件（14类标准灾害 `disaster_type_std`、灾害亚型 `disaster_subtype`） 及受灾承灾体（受损农作物 `affected_crops`、倒损房屋 `collapsed_houses`/`damaged_houses`、停课学校 `schools_closed`、停产工厂 `factories_stopped`） 定义为势能突破临界表面时涌现出的**上层离散拓扑相变态（超临界态）**。连续场与离散时序知识图谱（TKG）在数学上被统一为同一时空状态张量的“连续幅值”与“离散拓扑支撑集”。



**创新点二：图消息传递与大气-致灾微分方程的“算子级”统一（Operator Unification: 小模型即机理）**

* **痛点**：传统“机理+AI”方法通常在神经网络外额外挂载一个物理偏微分方程损失项（$\mathcal{L}_{\text{PDE}}$），导致梯度冲突与伪物理拟合。
* **第一性原理解法**：从流形微分几何出发，证明图神经网络的空间聚合算子本质上是偏微分方程（PDE）的离散有限差分算子。我们直接将**水平大气平流-扩散定律**与**垂直致灾能量蓄满-释放定律**编码为图神经算子（PT-GNO）的内部前向传播结构：
* **横向空间平流核（Horizontal Advection PDE）**：直接由相邻区县（`Cnty`）之间的气压梯度 $\Delta\text{PRS}_{\text{Avg}}$ 与多时间尺度风速（`WIN_S_Max`, `WIN_S_2mi_Avg`, `WIN_S_10mi_Avg`）驱动跨区县的水汽与极温传输。


* **纵向致灾积分核（Vertical Dose-Response ODE）**：将降水（`PRE_Time_2020`）、极温（`TEM_Max`/`TEM_Min`）及强对流天气现象（闪电 `Lit`、大风 `GaWIN`、雷暴 `Thund`、龙卷风 `Tord`、雾 `Fog`）作为致灾能量注入项，与防灾耗散项进行常微分动态积分。





**创新点三：大模型作为微分方程的“变分本构泛函”与连续时间数据同化（Amortized Functional Unification: 无路由大小模型协同）**

* **痛点**：同样的气象强度在不同地区造成的灾情截然不同（同雨不同灾），因为控制物理微分方程的**局地脆弱性阈值**与**防灾耗散率**隐藏在非结构化文本中，传统物理模型无法量化；而采用“大小模型硬路由（Routing）”又会导致梯度截断与推理割裂。
* **第一性原理解法**：在统一变分贝叶斯框架下，将大模型（LLM）定义为微分方程的**摊销变分本构泛函（Amortized Variational Constitutive Functional）** $q_{\phi}(\boldsymbol{\Theta}_{i,t} \mid \mathcal{T}_{i,\leq t})$。
* **离线先验蒸馏 + 在线参数生成**：LLM 负责解析非结构化灾情文本（天气过程概述 `weather_summary`、预警发布情况 `warning_info`、灾害影响描述 `impact_description`、备注 `remarks`） 与检索增强文本（`search_text`），直接输出小模型微分方程所需的**三维物理本构参数**——临界致灾阈值 $\boldsymbol{\tau}_{i,t}$、防灾耗散系数 $\boldsymbol{\gamma}_{i,t}$（结合安置转移人数 `relocated`） 以及灾害链级联转移张量 $\mathbf{W}_{\text{cascade}}$。


* **伴随“初报-续报-终报”的贝叶斯数据同化**：利用报告状态（`report_status`：初报/续报/终报）与上报时间（`report_time`），当灾中动态接收到初报或续报时，通过受控微分方程（Neural CDE）的跳跃校准项对微分轨迹进行实时后验修正，实现从早期预警到终报多维损失的连续同化推演。





---

#### 3. 统一数学建模方程（Methodology 核心公式）

定义第 $t$ 天区县节点 $i$ 的潜能状态为 $\mathbf{U}_i(t) \in \mathbb{R}^d$，上层时序知识图谱的激活拓扑与损失属性张量为 $\mathbf{A}_i(t)$。全模型仅由以下**耦合相变微分方程组**与**单一变分自由能目标函数**完全定义：

**（1）LLM 参数化的平流-反应-同化微分方程（小模型主干）**：


$$\mathrm{d}\mathbf{U}_i(t) = \Big[\underbrace{-\sum_{j \in \mathcal{N}(i)} \mathcal{K}_{\text{adv}}\big(\Delta\text{PRS}_{ij,t}, \mathbf{WIN}_{ij,t}\big)\big(\mathbf{U}_i(t) - \mathbf{U}_j(t)\big)}_{\text{跨区县气压-风场平流扩散项}} + \underbrace{\Psi_{\text{in}}\big(\mathbf{X}_{i,t}^{\text{meteo}}, \mathbf{I}_{i,t}^{99\text{code}}\big)}_{\text{气象要素与特殊相态注入项}} - \underbrace{\boldsymbol{\gamma}_{i,t}(\boldsymbol{\Theta}_{i,t}) \odot \mathbf{U}_i(t)}_{\text{LLM本构防灾耗散项}}\Big]\mathrm{d}t + \underbrace{\mathbf{G}_{\text{assim}}\big(\mathbf{U}_i(t)\big)\,\mathrm{d}\mathbf{M}_{i}(t)}_{\text{初报/续报数据同化校准项}}$$

* 其中 $\mathbf{X}_{i,t}^{\text{meteo}}$ 包含常规气象要素，$\mathbf{I}_{i,t}^{99\text{code}}$ 编码《说明.docx》中 `99xxxx` 特殊物理相态与超量程极值特征，$\mathbf{M}_i(t)$ 为沿上报时间（`report_time`）触发的初报/续报观测计数过程。



**（2）可微朗道相变与灾害图谱涌现方程（TKG 生成与级联）**：


$$\mathbf{A}_{i,t}^{\text{topo}} = \sigma_{\text{temp}}\Big(\frac{\Vert{}\mathbf{U}_i(t)\Vert{}_2 - \boldsymbol{\tau}_{i,t}(\boldsymbol{\Theta}_{i,t})}{\epsilon}\Big) \cdot \text{Softmax}\big(\mathbf{W}_{\text{cascade}}(\boldsymbol{\Theta}_{i,t}) \mathbf{U}_i(t)\big), \quad \hat{\mathbf{Y}}_{i,t}^{\text{loss}} = \mathcal{R}_{\text{readout}}\big(\mathbf{A}_{i,t}^{\text{topo}}, \mathbf{U}_i(t)\big)$$

* 当累积势能 $\Vert{}\mathbf{U}_i(t)\Vert{}_2 < \boldsymbol{\tau}_{i,t}$ 时，相变门控 $\sigma_{\text{temp}} \to 0$，图谱保持无灾静默；一旦突破临界值，瞬间激活灾害类型节点（`disaster_type_std`、`disaster_subtype`）、衍生灾害边及多维损失读出 $\hat{\mathbf{Y}}_{i,t}^{\text{loss}}$（损失等级 `loss_level`、经济损失 `economic_loss`、受灾/成灾/绝收面积等）。



**（3）统一变分自由能损失函数（Unique Objective Function）**：


$$\min_{\theta, \phi} \mathcal{L}_{\text{FreeEnergy}} = \mathbb{E}_{q_{\phi}(\boldsymbol{\Theta} \mid \mathcal{T}_{\leq t})}\Big[-\log p_{\theta}\big(\mathbf{A}_{1:T}^{\text{topo}}, \mathbf{Y}_{1:T}^{\text{loss}} \;\big\vert{}\; \mathbf{U}_{1:T}(\boldsymbol{\Theta})\big)\Big] + \beta \, \mathrm{KL}\Big(q_{\phi}(\boldsymbol{\Theta} \mid \mathcal{T}_{\leq t}) \;\Big\Vert{}\; p_{\text{LLM}}\big(\boldsymbol{\Theta} \mid \mathcal{T}_{\text{RAG}}(\text{search\_text})\big)\Big)$$

---

### 二、 支撑顶刊的实验设置方案（Experimental Setup）

#### 1. 异构数据工程与物理编码预处理（Data Engineering）

| 处理维度 | 涉及原始字段 | 顶刊级规范化处理方法 |
| --- | --- | --- |
| **行政时空网格对齐** | 气象表：`Province`, `City`, `Cnty`, `Year`, `Mon`, `Day`<br><br>

<br>灾情表：`province`(100%), `city`(~85%), `county`(~70%), `report_unit`(100%), `start_time`, `end_time`<br> | 利用 LLM 配合行政区划代码树，从 100% 填充的 `report_unit` 与 `impact_description` 中补全约 30% 缺失的 `county`，将全国区县统一映射为标准空间图节点 $\mathcal{V}$，时间轴按日对齐并将 `start_time` 至 `end_time` 展开为连续生命期窗口。

 |
| **特殊气象编码（`99xxxx`）物理相态解耦** | `PRE_Time_2020`, `WIN_S_Max`, `PRS_Avg`, `TEM_Max`/`Min` 等及特征值规则表

 | **拒绝粗暴删除或填零**：① 将 `999990`（微量降水）转化为数值 $0.05\text{ mm}$ + 微量指示位；② 将 `9998xx.x`（纯雾露霜）、`9997xx.x`（雨和雪）、`9996xx.x`（雨夹雪和雪暴）拆解为真实降水量 `xx.x` mm + **3维降水物理相态向量**（精准赋能雪灾、冷冻、连阴雨预测）；③ 将千位为 `8`（如 `998xxx.x` 超上限正极值）和千位为 `7`（如 `997xxx.x` 超下限负极值）拆解为截断观测值 `xxx.x` + **仪器破极值指示器**；④ `999999`（缺测）与 `999998`（不观测）通过邻域气压-风场平流核进行物理插值。

 |
| **灾情极值校验与多阶段序列重组** | `record_id`, `economic_loss`, `loss_level`, `report_status`(~90%), `report_time`(~90%)

 | ① 针对数据说明中标记的“12条经济损失超10亿万元”异常记录，结合 `loss_level` 与 `impact_description` 文本金额进行量纲对齐修正；② 按时空重叠度与灾害类型将同一事件下的 `初报`、`续报`、`终报` 串联为动态追踪事件链。

 |

---

#### 2. 三大国家级区域基准数据集构建（3 Benchmark Datasets）

从 2000–2025 年共 151,591 条记录中，按气候带与致灾机理切分出 3 个独立评测基准（按 2000–2018 训练集、2019–2021 验证集、2022–2025 测试集进行严格时序划分）：

1. **Dataset I: SW-Cascade（西南山地暴雨-地质灾害链数据集）**
* **空间范围**：云南、四川、贵州、重庆等西南区县。


* **核心特征**：聚焦局地强降水累积引发的 `暴雨` $\rightarrow$ `洪涝` $\rightarrow$ `地质灾害`（滑坡、泥石流、崩塌）跨天相变与级联演化，以及人员伤亡（`deaths`, `missing`, `injuries`）与安置转移（`relocated`）。




2. **Dataset II: SE-Typhoon（东南沿海台风-风暴潮复合冲击数据集）**
* **空间范围**：广东、福建、浙江、海南、江苏等沿海区县。


* **核心特征**：检验强气压梯度（`PRS_Avg`）与极大风速（`WIN_S_Inst_Max`）驱动的跨区县高速平流算子性能，重点预测 `台风`、`大风`、`暴雨`、`洪涝` 对房屋倒损（`collapsed_houses`, `damaged_houses`）及社会停摆（`schools_closed`, `factories_stopped`）的冲击。




3. **Dataset III: North-Agro（北方农业旱涝急转与强对流数据集）**
* **空间范围**：河南、山东、河北、黑龙江等北方农业主产区县。


* **核心特征**：包含慢速能量累积型灾害（`干旱`、`高温`、`连阴雨`）与突发强对流/寒潮灾害（`冰雹`、`雷电`、`低温`、`雪灾`），重点评估农业受损三面积（受灾面积 `affected_area`、成灾面积 `disaster_area`、绝收面积 `total_loss_area`）、粮食损失（`grain_loss`）及作物实体（`affected_crops`）预测精度。





---

#### 3. 预测任务与评价指标体系（Tasks & Metrics）

* **Task 1：时序知识图谱拓扑相变与灾害链预测（TKG Phase-Transition & Cascade Link Prediction）**
* **任务定义**：预测未来 $\Delta t \in \{1, 3, 5, 7\}$ 天内各区县是否发生致灾相变（Zero-to-One 节点涌现）、激活的具体灾害类型（14类 `disaster_type_std` 与 `disaster_subtype`）及次生灾害边连接。


* **评价指标**：Macro-F1, Micro-F1, AUPRC（针对稀疏不平衡灾害事件）, MRR（平均倒数排名）, Hits@1/3/5。


* **Task 2：多维致灾损失联合预测（Multi-Scale Disaster Impact Prediction）**
* **任务定义**：联合预测灾害节点的损失等级 `loss_level`（无损失/低/中/高/特大 5分类），以及经济损失 `economic_loss`、受灾总人数 `affected_population`、农业成灾/绝收面积 `disaster_area`/`total_loss_area` 等连续数值指标。


* **评价指标**：分级准确率（Accuracy, Weighted-F1）、连续数值对数空间下的 MAE, RMSE, MAPE 及决定系数 $R^2$。


* **Task 3：“初报 $\rightarrow$ 续报 $\rightarrow$ 终报”动态数据同化评测（Dynamic Report Assimilation）**
* **任务定义**：对比“仅靠气象+初报”、“同化第1次续报后”、“同化第2次续报后”对最终 `终报` 实际损失的预测收敛轨迹。


* **评价指标**：多阶段误差衰减率（Error Reduction Rate, ERR%）与同化校准增益（Assimilation Gain）。



---

#### 4. 五大类对比基线设置（12+ SOTA Baselines）

1. **时空图神经网络（Spatio-Temporal GNNs）**：GraphWaveNet, AGCRN, DSTAGNN, GinAR（仅将气象与灾情拼接为多维节点序列，无物理机理与文本语义）。
2. **时序知识图谱推理模型（Temporal KG Reasoning）**：RE-GCN, TiRGN, HisRES, CENET（仅在离散灾害三元组上推理，无法处理底层连续气象场的物理蓄能相变）。
3. **物理引导神经网络与神经算子（Physics-Informed Models）**：PINN, PDE-GCN, ClimODE, FNO（具备平流/扩散物理偏微分先验，但缺乏大模型对文本脆弱性与防灾响应的动态本构调制）。
4. **大模型赋能时序/图学习方法（LLM-Enhanced Graph/TS）**：Time-LLM, OFA (One Fits All), GraphGPT, LLM4TS（直接用大模型或文本嵌入做预测，缺乏大气平流与能量守恒机理约束）。
5. **模块缝合对比基线（Stitched Baseline — 专用于回应审稿人）**：构建一个 `GNN + 外部ODE Loss + LLM向量拼接` 的标准缝合模型（命名为 *Stitched-GNN-LLM*），在相同参数量下对比我们的统一相变算子 **PT-GNO**，从实验上直接证明“第一性原理统一建模”在精度、收敛速度与物理一致性上全面碾压“模块缝合”。

---

#### 5. 消融实验、参数敏感性与顶刊可视化设计

* **核心消融实验（Ablation Study）**：
1. `w/o Phase-Transition Gate`：移除朗道相变门控，退化为线性连续映射，验证对极端突发灾害（如 `冰雹`、`龙卷风`、`地质灾害`） 的虚警率与漏报率变化。


2. `w/o Advection PDE`：移除气压差（`PRS_Avg`）与风速（`WIN_S_Max`）驱动的跨区县平流核，替换为普通 GAT 注意力，观察在 `SE-Typhoon` 台风移动路径上的性能崩塌。


3. `w/o LLM Constitutive Functional`：将脆弱性阈值 $\boldsymbol{\tau}$ 和耗散率 $\boldsymbol{\gamma}$ 设为可学习的静态常数，不读取 `warning_info`、`impact_description` 与 `search_text`，证明其在“同雨不同灾”样本上的严重误判。


4. `w/o 99xxxx Physical Phase Encoding`：将气象表中的 `9997xx.x`、`9996xx.x`、`998xxx.x` 等特殊编码简单按缺失值或普通数值处理，验证对 `雪灾`、`低温`、`大风` 预测精度的下降幅度。




* **顶刊必杀可视化实验（Visualizations & Case Studies）**：
1. **相变势能面三维演化图（3D Phase Portrait Trajectory）**：展示某次真实特大暴雨诱发山体滑坡事件中，区县势能轨迹 $\mathbf{U}(t)$ 如何在连续降水（`PRE_Time_2020`） 驱动下逼近并刺穿 LLM 推断的临界阈值曲面 $\boldsymbol{\tau}(t)$，瞬间触发上层 TKG 灾害链生长的物理过程。


2. **跨区县平流向量场与灾害链传播地图（Spatial Advection & Cascade Map）**：在 GIS 地图上叠加模型学到的气压-风场传输流线 与真实次生灾害扩散路径。


3. **“同雨不同灾”反事实推演（Counterfactual Analysis）**：固定某区县的暴雨气象输入，人为修改 `warning_info`（如将“未发布预警”改为“提前发布红色预警并转移安置 5000 人 `relocated`”），展示 LLM 如何调高耗散率 $\boldsymbol{\gamma}_{i,t}$ 并成功压制最终成灾面积与伤亡预测值。





---

### 三、 顶刊论文完整章节框架与图表规划（Paper Outline）

#### 章节结构设计（IEEE TKDE 标准 12–14 页双栏排版）

* **Abstract**（约 200–250 词）：点明连续气象观测与离散灾害知识演化的割裂痛点 $\rightarrow$ 从非平衡相变与算子理论的第一性原理出发提出统一框架 **PT-GNO** $\rightarrow$ 简述平流-反应-相变微分算子与 LLM 变分本构泛函的嵌套机制 $\rightarrow$ 汇报在 25 年（2000–2025）15.1 万条灾情记录构建的三大国家级基准上的 SOTA 提升幅度。


* **Section 1: Introduction**（约 1.5 页）
* **1.1 研究背景**：极端天气频发背景下，融合地面气象观测网 与多源灾情直报历史库 进行跨时空灾害链与损失预测的重大价值。


* **1.2 现有范式的三大鸿沟（Research Gaps）**：① 连续物理场与离散事件图谱的表征鸿沟；② 黑盒神经网络与大气/致灾动力学定律的机理鸿沟；③ 非结构化灾情语境（预警、脆弱性文本） 与微分方程本构参数之间的协同鸿沟（批判现有“模块缝合”与“硬路由”缺陷）。


* **1.3 第一性原理破局与本文贡献（Contributions）**：总结理论统一、算子创新、变分同化机制及三大国家级数据集开源贡献。


* **Section 2: Related Work**（约 1 页）
* 2.1 时序知识图谱构建与推理（Temporal Knowledge Graph Reasoning）；
* 2.2 物理引导的时空图学习与神经微分算子（Physics-Informed Spatio-Temporal Graph Learning）；
* 2.3 大小模型协同与地球科学大模型应用（LLM-Small Model Collaboration in Earth & Disaster Sciences）。


* **Section 3: Preliminaries and Unified Problem Formulation**（约 1.5 页）
* 3.1 连续气象物理底图与特殊编码相态空间定义（含 `99xxxx` 物理相态形式化）；


* 3.2 离散灾害演进时序知识图谱（Emergent Disaster TKG）定义；


* 3.3 连续-离散耦合相空间与双重预测问题形式化（拓扑涌现预测 + 多阶段同化损失预测）。




* **Section 4: Methodology: Variational Phase-Transition Graph Dynamics**（核心章节，约 3.5–4 页）
* **4.1 总体变分自由能框架推导（Variational Free Energy Principle Overview）**：从第一性原理推导出唯一目标函数 ELBO。
* **4.2 大模型摊销变分本构泛函（LLM as Amortized Constitutive Functional）**：详述如何结合 RAG（`search_text`）与实时灾情文本（`weather_summary`, `warning_info`, `impact_description`） 推断局地阈值 $\boldsymbol{\tau}$、耗散率 $\boldsymbol{\gamma}$ 与级联矩阵 $\mathbf{W}_{\text{cascade}}$。


* **4.3 物理引导的平流-反应-相变图神经算子（Advection-Reaction-Phase-Transition GNO）**：详述水平气压-风场非对称拉普拉斯算子、垂直能量积分器与可微朗道相变门控。


* **4.4 面向初报-续报-终报的连续时间神经数据同化（Continuous-Time Neural Data Assimilation）**：基于 Neural CDE 的多阶段观测跳跃更新机制。


* **4.5 训练与高效推理算法流程（Algorithm Pseudo-code & Complexity Analysis）**：证明推理阶段可通过先验蒸馏脱离实时大模型高频调用的算力瓶颈。


* **Section 5: Experiments**（约 4–4.5 页）
* 5.1 实验设置：三大区域基准数据集（SW-Cascade, SE-Typhoon, North-Agro）统计表、12个基线模型、评估指标与超参数配置；


* 5.2 主实验一：时序知识图谱灾害节点涌现与级联边预测结果对比（Task 1）；


* 5.3 主实验二：多维灾情损失等级与连续指标联合预测结果对比（Task 2）；


* 5.4 动态数据同化实验：初报 $\rightarrow$ 续报 $\rightarrow$ 终报误差收敛曲线分析（Task 3）；


* 5.5 消融实验与抗“缝合怪”对比分析（Ablation & Unified vs. Stitched Analysis）；
* 5.6 物理可解释性与反事实案例研究（Phase Portrait, Advection Field & Counterfactual Case Studies）。


* **Section 6: Conclusion and Future Work**（约 0.5 页）

---

#### 论文核心图表规划清单（Figures & Tables Blueprint）

| 编号 | 类型 | 图表主题与核心展示内容 |
| --- | --- | --- |
| **Fig. 1** | 动机对比图 | **第一性原理概念图**：左半部分展示传统“GNN + 外部机理公式 + LLM路由”的生硬缝合缺陷；右半部分展示本文“连续气象场蓄能 $\xrightarrow{\text{LLM本构调制}}$ 突破临界相变曲面 $\rightarrow$ 涌现离散灾害TKG”的物理大一统过程。

 |
| **Fig. 2** | 架构总览图 | **PT-GNO 变分动力学统一架构图**：清晰画出水平气压-风场平流核、垂直致灾积分器、LLM 变分本构泛函 $q_\phi(\boldsymbol{\Theta}\mid\mathcal{T})$、可微朗道相变算子与初/续/终报数据同化闭环。

 |
| **Table 1** | 数据集统计表 | 三大国家级基准（SW-Cascade, SE-Typhoon, North-Agro）的区县节点数、时间跨度（2000–2025）、气象观测步数、灾害事件数、图谱三元组数及初/续/终报序列占比。

 |
| **Table 2** | 主实验表 I | 在三大数据集上与 12 个 SOTA 基线关于 **灾害图谱相变与级联预测（Task 1）** 的 Macro-F1, AUPRC, MRR, Hits@K 全面对比（包含显著性检验 $p\text{-value}$）。 |
| **Table 3** | 主实验表 II | 在三大数据集上关于 **损失等级 `loss_level` 分类与多维连续损失指标（`economic_loss`, `disaster_area` 等）回归（Task 2）** 的 MAE, RMSE, $R^2$ 对比。

 |
| **Fig. 3** | 同化收敛曲线 | **初报 $\rightarrow$ 续报 $\rightarrow$ 终报动态同化实验图（Task 3）**：展示随着早期报告文本与转移人数等信息的注入，PT-GNO 的预测不确定性阴影带如何快速收窄并逼近真实终报值。

 |
| **Table 4 / Fig. 4** | 消融与对比图 | 各核心物理与变分组件的消融结果，重点高亮 **PT-GNO（统一算子）vs. Stitched-GNN-LLM（缝合基线）** 在极端罕见灾害样本上的性能差距与推理耗时对比。 |
| **Fig. 5 & Fig. 6** | 可解释性案例 | **Fig. 5**：真实暴雨-洪涝-滑坡灾害链的三维相变势能面与临界曲面跨越轨迹可视化；**Fig. 6**：“同雨不同灾”与预警干预（`warning_info`, `relocated`）的反事实推演热力图。

 |
