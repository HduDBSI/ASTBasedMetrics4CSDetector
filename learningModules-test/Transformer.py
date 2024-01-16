from tkinter import Y
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


class MyTransformer(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyTransformer, self).__init__()
        self.embedding = torch.nn.Linear(1, hidden)
        self.transformer1_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        #self.transformer3_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)


        self.transformer1_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        #self.transformer3_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)

        self.fc = torch.nn.Linear(hidden*2, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = torch.unsqueeze(x, -1)
        y = torch.unsqueeze(y, -1)
        x = self.embedding(F.normalize(x, dim=1))
        x = x.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第一维
        x = self.transformer1_l(x)
        x = self.transformer2_l(x)
        #x = self.transformer3_l(x)
        x = x.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第二维
        x = x[:, -1, :]  # 取最后一个时间步的输出，作为整个序列的输出

        y = self.embedding(y)
        y = y.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第一维
        y = self.transformer1_r(y)
        y = self.transformer2_r(y)
        #y = self.transformer3_r(y)
        y = y.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第二维
        y = y[:, -1, :]  # 取最后一个时间步的输出，作为整个序列的输出

        fusion = self.fc(torch.cat([x.squeeze(), y.squeeze()], dim=-1))
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

if __name__ == '__main__':
    # 构造训练数据
    X_train = torch.randn(100, 36)  # 100个样本，每个样本36个时间步，每个时间步1个特征
    Z_train = torch.randn(100, 128)
    y_train = torch.randint(0, 3, (100, ))  # 3个类别

    # 构建GRU模型
    model = MyTransformer(36,128,3)

    # 训练模型
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in tqdm(range(3)):
        optimizer.zero_grad()
        outputs = model(X_train, Z_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 构造测试数据
    X_test = torch.randn(10, 36)  # 10个样本，每个样本36个时间步，每个时间步1个特征
    Z_train = torch.randn(10, 128)
    y_test = torch.randint(0, 3, (10, ))  # 3个类别

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test, Z_train)
        y_scores = y_pred.numpy()

    # 计算p, r, f1, acc
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    p = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)

    # 计算auc
    y_prob = torch.softmax(torch.tensor(y_scores), dim=1).numpy()
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    # 输出指标
    print('Precision: {:.4f}'.format(p))
    print('Recall: {:.4f}'.format(r))
    print('F1: {:.4f}'.format(f1))
    print('Accuracy: {:.4f}'.format(acc))
    print('AUC: {:.4f}'.format(auc))