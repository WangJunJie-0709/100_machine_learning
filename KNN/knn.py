import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 步骤1：加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 步骤2：分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 步骤3：特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # 计算新样本与已知样本之间的距离
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # 选择最近的K个样本
        k_indices = np.argsort(distances)[:self.K]
        # 进行类别投票
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 进行预测
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


# 步骤4-7：计算步骤和预测
knn = KNN(K=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 步骤8：评估算法的性能
accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuracy:', accuracy)