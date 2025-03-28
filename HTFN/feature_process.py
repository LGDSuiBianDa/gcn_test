import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from utile import *

# 数据预处理
# 1.时间补全 填充
# 2.归一化（feature scaler、 label scaler) :注意训练集和测试集归一化的一致性，可以手动归一化
# 3.数据集划分
