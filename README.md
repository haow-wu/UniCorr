# UniCorr

在训练时没见过的蛋白上，根据野生型复合物和突变注释，预测单点突变引起的蛋白–配体结合自由能变化 ΔΔG。

正值表示结合变弱。主结果只使用实验 Kd 或 Ki。评价单位是整条 UniProt：同一个蛋白的全部分子只进入训练、验证或测试之一。

## 使用时要提供的输入

- 野生型蛋白–配体复合物，PDB 里同时有蛋白坐标和配体坐标
- 配体 SMILES
- 突变注释，例如 E255K
- 野生型序列。突变序列由野生型序列和突变注释得到

不读取突变体晶体，也不为每条突变单独做分子动力学、自由能微扰或 AlphaFold。

氢键、靠近程度、碰撞和极性对数几率这四项校正，是在同一个蛋白的一组突变内部做标准化。只提交一条突变时，这四项为 0，预测会靠近训练集均值。

## 模型

1. 同一个复合物内的特征和标签先减去该复合物的均值。岭回归惩罚为 3×10⁵，只拟合突变之间的差。
2. 组内的差再乘以距离门 exp(−d/10)。d 是突变位点到配体的重原子距离。
3. 加上 ThermoMPNN 在该位点的稳定性残差，惩罚 1000，修正限制在 ±1 kcal/mol。
4. 在每个 UniProt 内部，对玻尔兹曼氢键、靠近程度和碰撞做标准化，样本数为 n 时乘以 n/(n+4)，权重分别是 +1、+1 和 −0.5。
5. 再加上极性对数几率 llr_x_polar 的蛋白内标准化分数，权重 0.25。这一项不再做 n/(n+4) 收缩。

验证时的选择分数是两个数里较差的一个：五折平均 Pearson，以及去掉最抬高相关的那个验证蛋白之后的 Pearson。llr_x_polar 权重 0.25 的验证分数是 0.369。

## 已保存的 Kd/Ki 结果

`results/` 里的数字来自冻结后的 `predictions.csv`，由 `score_metrics.py` 计算。571 条单点突变，102 个 UniProt，五折，种子 42。训练、验证、测试蛋白的交集为空，见 `results/overlap_check.txt`。

| | Pearson | Spearman | RMSE（kcal/mol） |
|---|---:|---:|---:|
| 折平均 | 0.340 | 0.308 | 1.390 |
| 去掉 P00374 之后的折平均 Pearson | 0.301 | | |

五折 Pearson 为 0.283、0.296、0.266、0.439、0.417。同一划分上的静态基线（MdrDB 的 MUT 列，极端随机树）折平均 Pearson 为 0.020。

这是按蛋白留出的结果。随机划分会让同一蛋白出现在训练集和测试集，不能和这个数字直接比较。

## 代码

- `unicorr/`：队列、划分、特征接入、岭回归、蛋白内校正和指标
- `src/`：在野生型主链上搭建侧链，并计算旋转异构体和玻尔兹曼氢键
- `results/`：冻结的划分、预测和指标

入口是 `unicorr/train_specsel.py`。它在验证蛋白上确认 llr_x_polar 权重 0.25 之后写出测试预测。`unicorr/freeze_result.py` 把该结果核对后写成正式的 `predictions.csv`。

脚本里的数据和权重路径写的是：

- MdrDB：`/root/autodl-tmp/project/data`
- 项目与缓存：`/root/autodl-tmp/project4`

## 数据和第三方权重

仓库不包含原始数据和模型权重。

- MdrDB：Yang 等，Communications Chemistry，2023。https://doi.org/10.1038/s42004-023-00920-7
- Platinum：Pires、Blundell、Ascher，Nucleic Acids Research，2015。用于确认测定类型是 KD 还是 Ki
- ESM-2：Lin 等，Science，2023。https://doi.org/10.1126/science.ade2574
- PSICHIC：Koh 等，Nature Machine Intelligence，2024。https://doi.org/10.1038/s42256-024-00847-1
- ThermoMPNN：Dieckhaus 等，PNAS，2024。https://doi.org/10.1073/pnas.2314853121
- ProteinMPNN：Dauparas 等，Science，2022。ThermoMPNN 使用它的结构编码器

第三方代码和权重保留各自的许可证。本仓库只包含 UniCorr 的配方、特征脚本和已保存的评价结果。
