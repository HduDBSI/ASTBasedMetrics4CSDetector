import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

class MyCNN(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyCNN, self).__init__()
        self.hidden = hidden
        self.metricSize = metricSize
        self.convl1 = torch.nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.convl2 = torch.nn.Conv1d(hidden, hidden*2, kernel_size=3, padding=1)
        self.convl3 = torch.nn.Conv1d(hidden*2, hidden, kernel_size=3, padding=1)
        self.fcl = torch.nn.Linear(hidden * metricSize, hidden)

        self.convr1 = torch.nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.convr2 = torch.nn.Conv1d(hidden, hidden*2, kernel_size=3, padding=1)
        self.convr3 = torch.nn.Conv1d(hidden*2, hidden, kernel_size=3, padding=1)
        self.fcr = torch.nn.Linear(hidden * hidden, hidden)

        self.fc_fision = torch.nn.Linear(hidden*2, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y): # batch_metric, batch_codeEmbedding
        # batch_metric
        x = F.normalize(x, dim=1)
        x = torch.unsqueeze(x, 1)
        x = self.relu(self.convl1(x))
        x = self.relu(self.convl2(x))
        x = self.relu(self.convl3(x))
        x = x.view(-1, self.hidden * self.metricSize)
        x = self.fcl(x)

        # batch_codeEmbedding
        y = torch.unsqueeze(y, 1)
        y = self.relu(self.convr1(y))
        y = self.relu(self.convr2(y))
        y = self.relu(self.convr3(y))
        y = y.view(-1, self.hidden * self.hidden)
        y = self.fcr(y)

        #print('x',x.shape)
        fusion = self.fc_fision(torch.cat([x, y], dim=-1))
        #print('fusion',fusion.shape)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

if __name__ == '__main__':
    # 构造训练数据
    X_train = torch.randn(100, 36)  # 100个样本，36个特征
    Z_train = torch.randn(100, 128)
    y_train = torch.randint(0, 3, (100, ))  # 3个类别
    # 构建DNN模型
    model = MyCNN(36,128,3)

    # 训练模型
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in tqdm(range(30)):
        optimizer.zero_grad()
        outputs = model(X_train, Z_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 构造测试数据
    X_test = torch.randn(10, 36)  # 10个样本，36个特征
    Z_test = torch.randn(10, 128)
    y_test = torch.randint(0, 3, (10, ))  # 3个类别

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test, Z_test)
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