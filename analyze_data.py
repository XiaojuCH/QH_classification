import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('样品特征数据集.csv', header=None)

print(f"数据集形状: {df.shape}")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n前5行数据:\n{df.head()}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n标签列（最后一列）的唯一值:\n{df.iloc[:, -1].unique()}")
print(f"\n标签分布:\n{df.iloc[:, -1].value_counts().sort_index()}")
print(f"\n数据统计信息:\n{df.describe()}")
print(f"\n缺失值统计:\n{df.isnull().sum().sum()}")
