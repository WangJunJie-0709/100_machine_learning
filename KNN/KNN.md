# 线性回归

这篇博客将介绍KNN——最近邻算法。老样子，这篇博客将分为以下两个部分：数学原理介绍和代码实战。由于这个算法比较简单，这篇博客的篇幅也会比较少。



## 数学原理

利用训练数据集对特征向量空间进行划分。**KNN算法的核心思想是在一个含未知样本的空间，可以根据样本最近的k个样本的数据类型来确定未知样本的数据类型。**

该算法涉及的3个主要因素是：**k值选择，距离度量，分类决策**

- 通用步骤

  - 计算距离（常用欧几里得距离或马氏距离）
    $$
    d(x,y)=\sqrt{\sum^n_{i=1}(x_i-y_i)^2}
    $$

  - 升序排列

  - 取前K个

    在应用中，k值一般取比较小的值，并采用交叉验证法进行调优。K的选取：

    - K太大：导致分类模糊。相当于用较大的领域中的训练实例进行预测，减少测试误差，增大训练误差。K值增大意味着整体模型变简单，**容易欠拟合**。
    - K太小：受个例影响，波动较大。相当于用较小的领域中的训练实例进行预测，训练误差会减小，只有与输入实例较近的训练实例才会对预测结果起作用，测试误差会增大。K值减小意味着整体模型变复杂，**容易过拟合。**

    如何选取K：

    - 经验
    - 均方根误差

  - 加权平均



## 代码实战

导入鸢尾花数据集

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 步骤1：加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 步骤2：分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤3：特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

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
knn = KNN(K=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 步骤8：评估算法的性能
accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuracy:', accuracy)
```
