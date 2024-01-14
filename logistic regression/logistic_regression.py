import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = load_breast_cancer()

print(data.feature_names)

# 特征选择
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
target = 'target'

# 划分特征和目标变量
X = pd.DataFrame(data.data, columns=data.feature_names)[features]
y = data.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 创建逻辑回归模型
model = LogisticRegression()

# 模型训练
model.fit(X_train, y_train)

# 估计特征的权重
weights = model.coef_
intercept = model.intercept_


# 预测测试数据集
y_pred = model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


# 新样本特征
new_sample = np.array([15.12, 15.16, 97.67, 712.2, 0.0898]).reshape(1, -1)

# 特征缩放
new_sample_scaled = scaler.transform(new_sample)

# 预测新样本的分类
prediction = model.predict(new_sample_scaled)

print("Prediction:", prediction)