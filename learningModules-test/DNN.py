import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


device = "cuda" if torch.cuda.is_available() else "cpu"
class SimpleAttention(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout=0.1, concat=True):
        super(SimpleAttention, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)).to(device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_dim, 1)).to(device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def forward(self, input): 
        
        e = torch.tanh(torch.matmul(input, self.W))
        #print("e", e.shape)
        att = F.softmax(torch.matmul(e, self.a),dim=0)
        #print("att", att.shape)
        ouput = torch.matmul(att.permute(0,2,1), e).squeeze(1)
        #print("simgleAttOut", ouput.shape)
        return ouput  

# 构建DNN模型
class MyDNN(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyDNN, self).__init__()
        self.fcl1 = torch.nn.Linear(metricSize, hidden)
        self.fcl2 = torch.nn.Linear(hidden, hidden*2)
        self.fcl3 = torch.nn.Linear(hidden*2, hidden)

        self.fcr1 = torch.nn.Linear(hidden, hidden)
        self.fcr2 = torch.nn.Linear(hidden, hidden*2)
        self.fcr3 = torch.nn.Linear(hidden*2, hidden)

        self.att = SimpleAttention(hidden, 2*hidden)
        self.fc_fision = torch.nn.Linear(hidden*2, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = self.relu(self.fcl1(x))
        x = self.relu(self.fcl2(x))
        x = self.relu(self.fcl3(x))
        
        y = self.relu(self.fcr1(y))
        y = self.relu(self.fcr2(y))
        y = self.relu(self.fcr3(y))

        #print('x',x.shape)
        fusion = torch.stack([x, y],dim=1)
        #print('fusion',fusion.shape)
        fusion = self.att(fusion)
        # print('fusion',fusion.shape)
        # exit()
        #fusion = self.fc_fision(torch.cat([x, y], dim=-1))
        fusion = self.fc_fision(fusion)
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
    model = MyDNN(36,128,3)

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