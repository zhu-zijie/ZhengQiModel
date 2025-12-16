# ZhengQiModel - 蒸汽工业蒸汽质量预测模型

## 📋 项目简介

本项目是一个机器学习回归预测模型，用于预测工业蒸汽的质量。通过分析多个工业过程参数特征，构建和优化各种机器学习算法，实现高精度的蒸汽质量预测。

## 🎯 项目目标

- 建立蒸汽质量与工业过程参数的关系模型
- 对比评估多种机器学习算法的预测性能
- 通过特征工程和模型优化提高预测精度
- 提供可靠的蒸汽质量预测方案

## 📊 数据集

- **训练集**：zhengqi_train.txt - 包含带标签的历史数据
- **测试集**：zhengqi_test.txt - 用于模型验证的测试数据
- **特征数量**：38个工业过程参数（V0-V37）
- **目标变量**：蒸汽质量值

## 📁 项目结构

```
ZhengQiModel/
├── README.md                      # 项目文档
├── zhengqi1.ipynb                 # 数据探索与预处理
└── zhengqi2.ipynb                 # 模型训练与评估
```

## 🔧 主要工作流程

### zhengqi1.ipynb - 数据分析与预处理
- **数据加载**：读取训练和测试数据
- **数据探索**：统计描述、分布分析
- **异常检测**：通过箱线图识别离群点
- **数据清洗**：处理缺失值和异常值
- **特征工程**：方差膨胀因子(VIF)、主成分分析(PCA)等特征选择和降维

### zhengqi2.ipynb - 模型构建与评估
- **数据准备**：加载原始数据和降维数据
- **多个回归模型**：
  - 线性回归 (Linear Regression)
  - 岭回归 (Ridge Regression)
  - Lasso回归 (Lasso Regression)
  - K近邻回归 (KNN)
  - 决策树回归 (Decision Tree)
  - 随机森林回归 (Random Forest)
  - 梯度提升回归 (Gradient Boosting)
  - 支持向量机回归 (SVR)
  - XGBoost
  - LightGBM
- **模型评估**：学习曲线分析、交叉验证
- **性能对比**：MSE等指标对比

## 🛠️ 技术栈

| 类别 | 工具库 |
|------|--------|
| 数据处理 | Pandas, NumPy |
| 数据可视化 | Matplotlib, Seaborn |
| 机器学习 | Scikit-learn |
| 梯度提升 | XGBoost, LightGBM |
| 统计分析 | SciPy, Statsmodels |

## 📈 主要分析方法

1. **统计分析**
   - 描述性统计：均值、标准差、分位数等
   - 异常值检测：箱线图分析

2. **特征工程**
   - 方差膨胀因子(VIF)：检测多重共线性
   - 主成分分析(PCA)：数据降维

3. **模型评估**
   - 均方误差(MSE)：定量评估预测精度
   - 学习曲线：诊断模型欠拟合/过拟合
   - 交叉验证：提高模型泛化能力估计

## 🚀 使用说明

1. 将数据文件放在指定路径：
   ```
   /Users/zhuzijie/Downloads/zhengqi/zhengqi_train.txt
   /Users/zhuzijie/Downloads/zhengqi/zhengqi_test.txt
   ```

2. 运行 `zhengqi1.ipynb`：
   - 执行数据探索和预处理
   - 生成处理后的数据文件：`processed_zhengqi_data.csv`、`train_data_pca.npz`、`test_data_pca.npz`

3. 运行 `zhengqi2.ipynb`：
   - 加载处理后的数据
   - 训练多个回归模型
   - 对比评估模型性能
   - 选择最优模型进行预测

## 📊 预期输出

- **可视化结果**：特征分布图、箱线图、学习曲线等
- **模型性能**：各模型的MSE和R²分数
- **预测结果**：测试集的预测值

## 👤 项目作者

zhu-zijie)

**更新时间**：2025年12月