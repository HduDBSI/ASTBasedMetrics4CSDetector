import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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



#################################### Metrics & Semantics #######################################
class MyCNN_fusion(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyCNN_fusion, self).__init__()
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

        #self.att = SimpleAttention(hidden, 2*hidden)
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
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

# 构建DNN模型
class MyDNN_fusion(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyDNN_fusion, self).__init__()
        self.fcl1 = torch.nn.Linear(metricSize, hidden)
        self.fcl2 = torch.nn.Linear(hidden, hidden*2)
        self.fcl3 = torch.nn.Linear(hidden*2, hidden)

        self.fcr1 = torch.nn.Linear(hidden, hidden)
        self.fcr2 = torch.nn.Linear(hidden, hidden*2)
        self.fcr3 = torch.nn.Linear(hidden*2, hidden)

        #self.att = SimpleAttention(hidden, 2*hidden)
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
        fusion = torch.cat([x, y],dim=-1)
        #print('fusion',fusion.shape)
        #exit()
        #fusion = self.att(fusion)
        # print('fusion',fusion.shape)
        # exit()
        #fusion = self.fc_fision(torch.cat([x, y], dim=-1))
        fusion = self.fc_fision(fusion)
        #print('fusion',fusion.shape)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

class MyGRU_fusion(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyGRU_fusion, self).__init__()
        self.gru_l_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_l_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        #self.gru_l_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        self.gru_r_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_r_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        #self.gru_r_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        #self.att = SimpleAttention(hidden, 2*hidden)
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
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyLSTM_fusion(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyLSTM_fusion, self).__init__()
        self.lstm_l = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        self.lstm_r = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        #self.att = SimpleAttention(hidden, 2*hidden)
        self.fc = torch.nn.Linear(hidden*2, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = torch.unsqueeze(x, -1)
        y = torch.unsqueeze(y, -1)
        _, (h_l, _) = self.lstm_l(x)
        _, (h_r, _) = self.lstm_r(y)
        fusion = self.fc(torch.cat([h_l[-1].squeeze(), h_r[-1].squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyTransformer_fusion(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyTransformer_fusion, self).__init__()
        self.embedding = torch.nn.Linear(1, hidden)
        self.transformer1_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        #self.transformer3_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)


        self.transformer1_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        #self.transformer3_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)

        #self.att = SimpleAttention(hidden, 2*hidden)
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
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out



#################################### Metrics Only #######################################
class MyDNN_metrics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyDNN_metrics, self).__init__()
        self.fcl1 = torch.nn.Linear(metricSize, hidden)
        self.fcl2 = torch.nn.Linear(hidden, hidden*2)
        self.fcl3 = torch.nn.Linear(hidden*2, hidden)

        self.fc_fision = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = self.relu(self.fcl1(x))
        x = self.relu(self.fcl2(x))
        x = self.relu(self.fcl3(x))

        #print('x',x.shape)
        fusion = self.fc_fision(torch.cat([x], dim=-1))
        #print('fusion',fusion.shape)
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

class MyCNN_metrics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyCNN_metrics, self).__init__()
        self.hidden = hidden
        self.metricSize = metricSize
        self.convl1 = torch.nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.convl2 = torch.nn.Conv1d(hidden, hidden*2, kernel_size=3, padding=1)
        self.convl3 = torch.nn.Conv1d(hidden*2, hidden, kernel_size=3, padding=1)
        self.fcl = torch.nn.Linear(hidden * metricSize, hidden)

        self.fc_fision = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
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

        #print('x',x.shape)
        fusion = self.fc_fision(torch.cat([x], dim=-1))
        #print('fusion',fusion.shape)
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

class MyGRU_metrics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyGRU_metrics, self).__init__()
        self.gru_l_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_l_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.gru_l_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        
        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = torch.unsqueeze(x, -1)
        
        _, x = self.gru_l_1(x)
        x = x.permute(1, 0, 2)
        _, x = self.gru_l_2(x)
        x = x.permute(1, 0, 2)
        _, x = self.gru_l_3(x)

        fusion = self.fc(torch.cat([x.squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyLSTM_metrics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyLSTM_metrics, self).__init__()
        self.lstm_l = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        self.lstm_r = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        
        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        x = torch.unsqueeze(x, -1)
        _, (h_l, _) = self.lstm_l(x)
        fusion = self.fc(torch.cat([h_l[-1].squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyTransformer_metrics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyTransformer_metrics, self).__init__()
        self.embedding = torch.nn.Linear(1, hidden)
        self.transformer1_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer3_l = torch.nn.TransformerEncoderLayer(hidden, nhead=8)

        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
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

        fusion = self.fc(torch.cat([x.squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out






#################################### Semantics Only #######################################
class MyDNN_semantics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyDNN_semantics, self).__init__()

        self.fcr1 = torch.nn.Linear(hidden, hidden)
        self.fcr2 = torch.nn.Linear(hidden, hidden*2)
        self.fcr3 = torch.nn.Linear(hidden*2, hidden)

        self.fc_fision = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        
        y = self.relu(self.fcr1(y))
        y = self.relu(self.fcr2(y))
        y = self.relu(self.fcr3(y))

        #print('x',x.shape)
        fusion = self.fc_fision(torch.cat([y], dim=-1))
        #fusion = self.att(fusion)
        #print('fusion',fusion.shape)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

class MyCNN_semantics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyCNN_semantics, self).__init__()
        self.hidden = hidden
        self.convr1 = torch.nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.convr2 = torch.nn.Conv1d(hidden, hidden*2, kernel_size=3, padding=1)
        self.convr3 = torch.nn.Conv1d(hidden*2, hidden, kernel_size=3, padding=1)
        self.fcr = torch.nn.Linear(hidden * hidden, hidden)

        self.fc_fision = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y): # batch_metric, batch_codeEmbedding
     
        # batch_codeEmbedding
        y = torch.unsqueeze(y, 1)
        y = self.relu(self.convr1(y))
        y = self.relu(self.convr2(y))
        y = self.relu(self.convr3(y))
        y = y.view(-1, self.hidden * self.hidden)
        y = self.fcr(y)

        #print('x',x.shape)
        fusion = self.fc_fision(torch.cat([y], dim=-1))
        #print('fusion',fusion.shape)
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out

class MyGRU_semantics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyGRU_semantics, self).__init__()
        self.gru_r_1 = torch.nn.GRU(1, hidden, batch_first=True)
        self.gru_r_2 = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.gru_r_3 = torch.nn.GRU(hidden, hidden, batch_first=True)

        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        y = torch.unsqueeze(y, -1)
        

        _, y = self.gru_r_1(y)
        y = y.permute(1, 0, 2)
        _, y = self.gru_r_2(y)
        y = y.permute(1, 0, 2)
        _, y = self.gru_r_3(y)

        fusion = self.fc(torch.cat([y.squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyLSTM_semantics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyLSTM_semantics, self).__init__()
        self.lstm_r = torch.nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        y = torch.unsqueeze(y, -1)
        _, (h_r, _) = self.lstm_r(y)
        fusion = self.fc(torch.cat([h_r[-1].squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        return out

class MyTransformer_semantics(torch.nn.Module):
    def __init__(self, metricSize, hidden, num_classes):
        super(MyTransformer_semantics, self).__init__()
        self.embedding = torch.nn.Linear(1, hidden)

        self.transformer1_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer2_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)
        self.transformer3_r = torch.nn.TransformerEncoderLayer(hidden, nhead=8)

        self.fc = torch.nn.Linear(hidden, num_classes)
        #self.att = SimpleAttention(hidden, hidden)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        y = torch.unsqueeze(y, -1)

        y = self.embedding(y)
        y = y.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第一维
        y = self.transformer1_r(y)
        y = self.transformer2_r(y)
        y = self.transformer3_r(y)
        y = y.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第二维
        y = y[:, -1, :]  # 取最后一个时间步的输出，作为整个序列的输出

        fusion = self.fc(torch.cat([y.squeeze()], dim=-1))
        #fusion = self.att(fusion)
        out = self.softmax(fusion)
        #print('out',out.shape)
        return out