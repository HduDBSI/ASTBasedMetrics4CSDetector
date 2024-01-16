import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# 构造训练数据
X_train = torch.randn(100, 5)  # 100个样本，5个特征
y_train = torch.randint(0, 3, (100, ))  # 3个类别

# 转换为numpy数组
X_train = X_train.numpy()
y_train = y_train.numpy()

# 训练决策树
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 构造测试数据
X_test = torch.randn(10, 5)  # 10个样本，5个特征
y_test = torch.randint(0, 3, (10, ))  # 3个类别

# 转换为numpy数组
X_test = X_test.numpy()
y_test = y_test.numpy()

# 预测
y_pred = clf.predict(X_test)

# 计算p, r, f1, acc
p = precision_score(y_test, y_pred, average='macro')
r = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)

# 计算auc
y_scores = clf.predict_proba(X_test)
auc = roc_auc_score(y_test, y_scores, multi_class='ovr')

# 输出指标
print('Precision: {:.4f}'.format(p))
print('Recall: {:.4f}'.format(r))
print('F1: {:.4f}'.format(f1))
print('Accuracy: {:.4f}'.format(acc))
print('AUC: {:.4f}'.format(auc))