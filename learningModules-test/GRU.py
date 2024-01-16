from tkinter import Y
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# 构建GRU模型
# class MyGRU(torch.nn.Module):
#     def __init__(self, metricSize, hidden, num_classes):
#         super(MyGRU, self).__init__()
#         self.gru_l = torch.nn.GRU(1, hidden, batch_first=True)
#         self.gru_r = torch.nn.GRU(1, hidden, batch_first=True)

#         self.fc = torch.nn.Linear(hidden*2, num_classes)
#         self.softmax = torch.nn.Softmax(dim=-1)

#     def forward(self, x, y):
#         x = F.normalize(x, dim=1)
#         x = torch.unsqueeze(x, -1)
#         y = torch.unsqueeze(y, -1)
#         _, x = self.gru_l(x)
#         _, y = self.gru_r(y)

#         fusion = self.fc(torch.cat([x.squeeze(), y.squeeze()], dim=-1))
#         out = self.softmax(fusion)
#         #print('out',out.shape)
#         return out

class MyGRU(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyGRU, self).__init__()
        self.gru_l_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_l_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        #self.gru_l_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        self.gru_r_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_r_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        #self.gru_r_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        self.fc = torch.nn.Linear(hidden*2, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = torch.unsqueeze(x, -1)
        y = torch.unsqueeze(y, -1)
        
        _, x = self.gru_l_1(x)
        x = x.permute(1, 0, 2)
        _, x = self.gru_l_2(x)
        # x = x.permute(1, 0, 2)
        # _, x = self.gru_l_3(x)

        _, y = self.gru_r_1(y)
        y = y.permute(1, 0, 2)
        _, y = self.gru_r_2(y)
        # y = y.permute(1, 0, 2)
        # _, y = self.gru_r_3(y)

        fusion = self.fc(torch.cat([x.squeeze(), y.squeeze()], dim=-1))
        out = self.softmax(fusion)
        return out

if __name__ == '__main__':
    # 构造训练数据
    X_train = torch.randn(100, 36)  # 100个样本，每个样本36个时间步，每个时间步1个特征
    Z_train = torch.randn(100, 128)
    y_train = torch.randint(0, 3, (100, ))  # 3个类别

    # 构建GRU模型
    model = MyGRU(36,128,3)

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