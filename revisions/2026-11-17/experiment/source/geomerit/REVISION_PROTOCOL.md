> Historical source documentation. For the completed 80-fold release, effective scope and current runnable commands, use [the publication instructions](../../PROVENANCE.md). Earlier entry points or proposed variants below are retained only as historical context.

# GeoMERIT 2026-11-03 修订实验

本文件描述新流程，不把历史结果视为新实验结果。配置在 `configs/revision_20261103.yaml`，新入口为 `scripts/10_revision_train.py`。

## 已实施的约束

- 保留历史 118 口井的十个外层折划分；从原始 LAS 重建 1,431,383 个有标签样本，以来源文件和原始行号标识样本。逐井逐类别核对样本数。
- 每个外层训练集内部做三个按井划分的交叉验证。外层测试标签不进入模型拟合、早停、专家启用或参数选择。
- 基模型预算和结构事先固定；没有宣称全部参数都经过内层优化。内层仅选择集成权重、层次混合和决策参数。
- 类别字典和坐标中心只从相应训练井拟合。目标井全井归一化和居中窗口属于离线测井解释，不能用于宣称实时在线性能。
- 20 条测井曲线用于缺失率计算；坐标、套管、GROUP、FORMATION 属于单独的元数据权限。
- 缺失类别显式映射回统一 12 类，后验为零；专家是否启用只取决于训练正负例支持数。决策动作空间仍为 12 类。
- RFE 是根据论文公式重建的实现；窗口半径和标准化约定记录在配置中，不冒充原作者代码或精确复现。
- 训练时只保留外层最终模型，内层保留预测、特征状态、划分、参数和日志；这样可复查选择过程并控制磁盘占用。
- 源码和配置冻结为哈希。中断重启只复用完全匹配且校验通过的检查点；磁盘不足 15 GiB 时停止，不删除其他数据。

## 运行与输出

先运行 `python tests/test_revision_protocol.py`，再运行小规模流程：

```
python scripts/10_revision_train.py --work /path/to/experiment --smoke
```

正式主实验：

```
python scripts/10_revision_train.py --work /path/to/experiment --variants full rfe_xgb raw_xgb
```

随后可用相同冻结配置依次运行配置中列出的表示消融、权重消融和元数据权限变体。不能更改外层评分后再重新选择所谓最佳配置。新代码变更应使用新的实验目录并解释原因。

每个外层折包含：内层井划分、逐行内层预测、训练先验、完整决策搜索记录、冻结参数、外层模型、最终后验、逐行预测、折级指标和完整性标记。正式运行目录与 smoke 目录分离；smoke 数值不可用于论文。

## 回答的科学问题及边界

1. 完整 GeoMERIT 与同折 RFE 的 argmax 和先验调整＋Bayes 版本比较，区分结构增益和单纯更换解码器。
2. 表示与权重组件分别移除并重新训练；无缺失编码保留相同的异常值编码，无类别权重保留边界权重。
3. 层次、专家、先验和 argmax 的条件干预复用相同预测并冻结其他已选参数，明确称为条件消融，不替代重新优化的独立基线。
4. 元数据四种权限变体已配置；地层注释是否可用于目标场景需由应用前提限定。
5. 自然缺失分层将报告井数、样本数和类别分布。空间分区留出、从原始观测重新遮蔽并重建特征的实验尚未实施，不得写成完成。

统计报告使用相同井和样本标识配对；不将外部论文的一个总体数值作为配对折级检验对象。按井 bootstrap 的不确定性以已拟合程序为条件，不能替代重训不确定性；折间训练集重叠也必须说明。论文结果在全折汇总完成前保留“待回填”。

历史入口 `01_train.py`、`02_predict_decode.py`、`03_ablation.py` 保留用于追溯。它们存在历史外层选择问题，不可作为修订实验入口。
