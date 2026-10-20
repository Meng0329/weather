# 气象灾害数据字典

## 概述

- **总记录数**: 151,591 条
- **时间范围**: 2000-2025年
- **数据来源**: 气象局灾情直报数据库

## 字段说明

### 基础信息字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `record_id` | string | 唯一记录ID，格式 REC_XXXXXXXX | 100% |
| `source_file` | string | 来源文件名 | 100% |
| `disaster_type` | string | 原始灾害类型 | 98.6% |
| `disaster_type_std` | string | 标准化灾害类型 | 98.6% |
| `disaster_subtype` | string | 灾害亚型 | ~50% |

### 时间字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `start_time` | datetime | 灾害开始时间 (YYYY-MM-DD HH:MM:SS) | 100% |
| `end_time` | datetime | 灾害结束时间 | ~80% |
| `report_year` | int | 上报年份 | 100% |

### 地理位置字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `province` | string | 省级行政区 | 100% |
| `city` | string | 市级行政区 | ~85% |
| `county` | string | 县级行政区 | ~70% |
| `report_unit` | string | 完整上报单位名称 | 100% |

### 经济损失字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `economic_loss` | float | 经济损失（万元）| 100% |
| `loss_level` | string | 损失等级（无损失/低/中/高/特大）| 100% |

### 人员伤亡字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `deaths` | int | 死亡人数 | 100% |
| `missing` | int | 失踪人数 | 100% |
| `injuries` | int | 受伤人数 | 100% |
| `affected_population` | int | 受灾总人数 | 100% |
| `relocated` | int | 安置转移人数 | 100% |
| `drinking_water_difficulty` | int | 饮水困难人数 | 100% |

### 房屋损毁字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `collapsed_houses` | int | 倒塌房屋数（间）| 100% |
| `damaged_houses` | int | 损坏房屋数（间）| 100% |

### 农业损失字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `affected_area` | float | 受灾面积（公顷）| 100% |
| `disaster_area` | float | 成灾面积（公顷）| 100% |
| `total_loss_area` | float | 绝收面积（公顷）| 100% |
| `grain_loss` | float | 损失粮食（吨）| 100% |
| `affected_crops` | string | 受灾作物名称 | ~30% |

### 社会影响字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `schools_closed` | int | 停课学校数 | 100% |
| `factories_stopped` | int | 停产工厂数 | 100% |

### 文本描述字段（LLM分析核心）

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `impact_description` | text | 灾害影响描述 | 84.6% |
| `weather_summary` | text | 天气过程概述 | 83.6% |
| `warning_info` | text | 预警发布情况 | 62.6% |
| `remarks` | text | 备注 | 15.3% |

### 元数据字段

| 字段名 | 类型 | 说明 | 填充率 |
|--------|------|------|--------|
| `data_source` | string | 数据采集来源 | ~60% |
| `report_status` | string | 报告状态（初报/续报/终报）| ~90% |
| `report_time` | datetime | 上报时间 | ~90% |
| `tabulator` | string | 制表人 | ~30% |
| `contact` | string | 联系方式 | ~30% |

### 检索增强字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `search_text` | text | 合并所有关键信息的检索文本，用于向量检索 |

## 标准化灾害类型

| 标准类型 | 包含的原始类型 |
|----------|---------------|
| 暴雨 | 暴雨、大暴雨、特大暴雨、强降水、降雨、暴雨洪涝 |
| 洪涝 | 洪涝、洪水、内涝、渍涝 |
| 台风 | 台风、热带气旋、热带风暴 |
| 干旱 | 干旱、旱灾、旱情、冬春旱 |
| 高温 | 高温、热浪、酷暑 |
| 低温 | 低温、冷冻、寒潮、霜冻 |
| 雪灾 | 雪灾、暴雪、大雪、冰冻 |
| 大风 | 大风、狂风、龙卷风、雷雨大风 |
| 冰雹 | 冰雹、雹灾 |
| 雷电 | 雷电、雷击 |
| 沙尘暴 | 沙尘暴、扬沙、浮尘 |
| 雾霾 | 雾霾、大雾、霾 |
| 连阴雨 | 连阴雨、阴雨 |
| 地质灾害 | 滑坡、泥石流、崩塌、地面塌陷 |

## 损失等级标准

| 等级 | 经济损失（万元）|
|------|----------------|
| 无损失 | 0 |
| 低 | < 100 |
| 中 | 100 - 1,000 |
| 高 | 1,000 - 10,000 |
| 特大 | > 10,000 |

## 数据文件

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `disaster_records_unified.csv` | CSV | 统一主表（UTF-8-BOM编码）|
| `disaster_records_unified.parquet` | Parquet | 列式存储格式，适合大数据分析 |
| `data_quality_report.json` | JSON | 数据质量报告 |

### 按时间划分的数据文件

| 文件名 | 时间范围 | 记录数 | 文件大小 |
|--------|----------|--------|----------|
| `disaster_records_part1_2000-2010.csv` | 2000-2010 | 28,144 | ~31MB |
| `disaster_records_part2_2011-2014.csv` | 2011-2014 | 36,417 | ~67MB |
| `disaster_records_part3_2015-2018.csv` | 2015-2018 | 34,302 | ~71MB |
| `disaster_records_part4_2019-2021.csv` | 2019-2021 | 27,739 | ~64MB |
| `disaster_records_part5_2022-2025.csv` | 2022-2025 | 24,989 | ~61MB |

## 使用示例

### Python读取数据

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('datasets/cleaned/disaster_records_unified.csv', low_memory=False)

# 或读取Parquet（更快）
df = pd.read_parquet('datasets/cleaned/disaster_records_unified.parquet')

# 按省份筛选
yunnan_df = df[df['province'] == '云南省']

# 按灾害类型筛选
rainstorm_df = df[df['disaster_type_std'] == '暴雨']

# 按时间范围筛选
df_2020_2024 = df[(df['report_year'] >= 2020) & (df['report_year'] <= 2024)]
```

### 全文检索示例

```python
# 检索包含特定关键词的记录
results = df[df['search_text'].str.contains('工业园区', na=False)]
```

## 注意事项

1. **经济损失异常值**: 有12条记录经济损失超过10亿万元，已标记但保留原始值
2. **地点提取**: 省市县信息从"上报单位"字段自动提取，部分可能不完整
3. **年份0问题**: 已修复，所有记录都有有效的年份信息
4. **文本字段**: 2010+的数据实际上包含文本描述字段，与之前的理解不同

## 更新日志

- 2026-03-01: 初始数据清洗完成，151,591条记录
