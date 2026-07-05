# GeoMERIT 工程结构与代码简介

本仓库是 GeoMERIT 论文的 GitHub 引用版代码包，用于支撑论文中的方法实现、实验复现和图表生成。仓库只包含代码、配置、轻量结果摘要和论文图表资产；不包含 FORCE 2020 原始 LAS 数据、大型特征表、完整 OOF 概率矩阵和模型缓存。

## 目录结构

```text
GeoMERIT-GitHub-Release
├── geomerit/              # 核心方法代码
├── scripts/               # 实验运行脚本
├── configs/               # 数据、特征、模型、惩罚矩阵配置
├── results/               # 轻量实验结果摘要
├── paper/                 # 最新论文 LaTeX、图表和作图脚本
├── requirements.txt       # Python 依赖
├── README.md              # GitHub 首页说明
└── CITATION.cff           # GitHub 引用元数据
```

## 核心代码模块

| 文件 | 作用 |
|---|---|
| `geomerit/io_las.py` | 读取 FORCE 2020 LAS 文件和 NPD 辅助 Excel 表 |
| `geomerit/features.py` | 构造缺失感知、异常感知、窗口上下文和梯度特征 |
| `geomerit/labels.py` | 12 类岩性标签映射、地质粗类映射、长尾类定义 |
| `geomerit/weights.py` | 类别权重、边界权重、标签置信度权重 |
| `geomerit/models.py` | LightGBM/XGBoost/CatBoost 融合、粗到细分类、尾类专家 |
| `geomerit/decode.py` | logit adjustment、Bayes-risk penalty 解码、门控策略 |
| `geomerit/metrics.py` | Weighted F1、Macro F1、Boundary F1、Penalty、Tail F1 |
| `geomerit/cv.py` | 按井分组的 GroupKFold 验证工具 |

## 实验脚本

| 脚本 | 作用 |
|---|---|
| `scripts/00_build_dataset.py` | 从原始 LAS 和 Excel 文件构建特征表 |
| `scripts/01_train.py` | 运行 10 折 GroupKFold 训练 |
| `scripts/02_predict_decode.py` | 使用惩罚矩阵进行 Bayes-risk 解码并评估 |
| `scripts/03_ablation.py` | 消融实验入口 |
| `scripts/04_georacs_oof.py` | GeoRACS/OOF 后处理与诊断实验 |
| `scripts/04_make_figures.py` | 基础论文图表生成 |
| `scripts/05_make_paper_figures.py` | 论文风格图表生成 |

## 论文引用建议

论文中可写：

> The source code, configuration files, lightweight result summaries, and figure-generation scripts are publicly available at: `https://github.com/<your-username>/GeoMERIT`.

如果需要更正式，可以写：

> We release the GeoMERIT implementation as a paper-reference repository, including the core Python package, experiment entry points, model configurations, decoding settings, lightweight result summaries, and manuscript figure-generation scripts.

## 不上传的大文件

以下内容没有放入 GitHub 仓库：

- FORCE 2020 原始 LAS 数据；
- `feature_table.parquet` 完整特征表；
- 完整 `decode_report.csv` 逐行 OOF 预测；
- `.npy` 概率矩阵、检索先验、stacking 输出；
- 远程服务器完整快照；
- 任何 API key、服务器密码或本地环境凭据。

