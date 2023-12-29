# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# 加载波士顿房屋数据集
boston = load_boston()
print('数据量：{}'.format(len(boston.data)))   # 数据量：506
print('特征数：{}'.format(len(boston.data[0])))   # 特征数：13

print(boston.feature_names)

# 数据准备
X = boston.data
y = boston.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型实例
model = LinearRegression()

# 模型训练
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = np.mean((y_pred - y_test) ** 2)
print("均方误差：", mse)

# 结果可视化
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('true_price')
plt.ylabel('predict_price')
plt.title('linear')
plt.show()