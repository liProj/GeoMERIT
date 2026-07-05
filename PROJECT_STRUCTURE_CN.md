# GeoMERIT 工程结构与代码简介

本文配套代码仓库用于支撑以下投稿论文：

**GeoMERIT: Missingness-Aware and Geological-Cost-Guided Lithology Prediction across Field Wells**

该仓库是论文投稿与同行评审用的轻量发布版，包含核心方法代码、实验脚本、配置文件、轻量结果摘要、论文图表和 LaTeX 资产。仓库不包含 FORCE 2020 原始 LAS 数据、完整特征表、大型 OOF 概率矩阵、模型缓存或服务器环境凭据。

## 目录结构

```text
GeoMERIT-GitHub-Release
|-- geomerit/              # 核心方法代码
|-- scripts/               # 实验运行与画图脚本
|-- configs/               # 数据、特征、模型和惩罚矩阵配置
|-- results/               # 轻量实验结果摘要
|-- paper/                 # 论文 LaTeX、图表、数据和作图脚本
|-- requirements.txt       # Python 依赖
|-- README.md              # GitHub 首页说明
`-- PROJECT_STRUCTURE_CN.md # 中文工程说明
```

## 核心方法代码

| 文件 | 作用 |
|---|---|
| `geomerit/io_las.py` | 读取 FORCE 2020 LAS 文件和 NPD 辅助 Excel 表 |
| `geomerit/features.py` | 构造缺失感知、异常感知、滑动窗口上下文和梯度特征 |
| `geomerit/labels.py` | 12 类岩性标签映射、地质粗类映射和长尾类定义 |
| `geomerit/weights.py` | 类别权重、边界权重和标签置信度权重 |
| `geomerit/models.py` | LightGBM/XGBoost/CatBoost 融合、粗到细分类和尾类专家 |
| `geomerit/decode.py` | logit adjustment、Bayes-risk geological-cost 解码和门控策略 |
| `geomerit/metrics.py` | Weighted F1、Macro F1、Boundary F1、Penalty 和 Tail F1 |
| `geomerit/cv.py` | 按井分组的 GroupKFold 验证工具 |

## 实验脚本

| 脚本 | 作用 |
|---|---|
| `scripts/00_build_dataset.py` | 从原始 LAS 和 Excel 文件构建特征表 |
| `scripts/01_train.py` | 运行 10 折 GroupKFold 训练 |
| `scripts/02_predict_decode.py` | 使用地质惩罚矩阵进行 Bayes-risk 解码并评估 |
| `scripts/03_ablation.py` | 消融实验入口 |
| `scripts/04_georacs_oof.py` | GeoRACS/OOF 后处理与诊断实验 |
| `scripts/04_make_figures.py` | 基础论文图表生成 |
| `scripts/05_make_paper_figures.py` | 论文风格图表生成 |

## 投稿代码说明

论文中可写：

> The source code, configuration files, lightweight result summaries, and figure-generation scripts are publicly available at: `https://github.com/liProj/GeoMERIT`.

也可以写成更正式的投稿表述：

> We provide a paper-reference implementation of GeoMERIT, including the core Python package, experiment entry points, model configurations, decoding settings, lightweight result summaries, and manuscript figure-generation scripts.

## 未上传的大文件

以下内容没有放入 GitHub 仓库：

- FORCE 2020 原始 LAS 数据；
- `feature_table.parquet` 完整特征表；
- 完整 `decode_report.csv` 逐行 OOF 预测；
- `.npy` 概率矩阵、检索先验和 stacking 输出；
- 远程服务器完整快照；
- 任何 API key、服务器密码或本地环境凭据。
