import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import os
def load_libsvm_data(data_path='data/heart_scale.txt'):
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return None, None
    data, target = load_svmlight_file(data_path)
    X = data.toarray()
    y = target.reshape(-1, 1)
    # 核心：标准化使特征量级统一，是生成直线的关键
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return  X, y