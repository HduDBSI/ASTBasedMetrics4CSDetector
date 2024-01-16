

from mimetypes import common_types
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import random
import json
import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from deepLearningModuleX5 import *

from focalloss import FocalLossMulti, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_attention_heads', type=int, default=8)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = FocalLoss().to(device)

def getDictByJson(jsonPath):
    with open(jsonPath, 'r') as f:
        mydata = f.read()
    mydict = json.loads(mydata)
    return mydict

def getTypeSplitList(type_list, fold_num, fold_idx):
    fold_size = len(type_list)//fold_num
    start_index = (fold_idx-1) * fold_size
    end_index = fold_idx * fold_size
    train_list = type_list[:start_index] + type_list[end_index:]
    test_list = type_list[start_index:end_index]
    return train_list, test_list

def getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx): # e.g. 分为5折（1，2，3，4，5）
        fold_size = len(label_list)//fold_num
        #print("data_list",len(label_list))
        print("fold_size",fold_size)
        #应当将每一类五等分，确保划分均匀
        none_item = []
        minor_item = []
        major_item = []
        critical_item = []
        for item in label_list:
            deg = int(item.rstrip('\n').split()[2])
            if deg == 0:
                none_item.append(item)
            if deg == 1:
                minor_item.append(item)
            if deg == 2:
                major_item.append(item)
            if deg == 3:
                critical_item.append(item)

        none_train, none_test = getTypeSplitList(none_item, fold_num, fold_idx)
        minor_train, minor_test = getTypeSplitList(minor_item, fold_num, fold_idx)
        major_train, major_test = getTypeSplitList(major_item, fold_num, fold_idx)
        critical_train, critical_test = getTypeSplitList(critical_item, fold_num, fold_idx)

        test_label_list = none_test + minor_test + major_test + critical_test
        train_label_list = none_train + minor_train + major_train + critical_train

        # print('test_label_list',len(test_label_list))
        # print('train_label_list',len(train_label_list))
        # print('none_item',len(none_item))
        # print('minor_item',len(minor_item))
        # print('major_item',len(major_item))
        # print('critical_item',len(critical_item))
        # exit()
        return train_label_list, test_label_list

def showRitiaOfPosNeg(train_label_list, test_label_list):
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0
    for item in train_label_list:
        label = int(item.split()[1])
        if label == 1:
            train_pos+=1
        else:
            train_neg+=1

    for item in test_label_list:
        label = int(item.split()[1])
        if label == 1:
            test_pos+=1
        else:
            test_neg+=1
    print('showRitiaOfPosNeg:', train_pos/train_neg, test_pos/test_neg)

def Undersampling(trainlist):
    print('trainlist',len(trainlist))
    random.seed(int(time.time()))
    random.shuffle(trainlist)
    pos = 0
    neg = 0
    posSamples = []
    negSamples = []
    selectSamples = []
    for sample in trainlist:
        if sample.split()[1] == '0':
            neg+=1
            negSamples.append(sample)
        else:
            pos+=1
            posSamples.append(sample)
    print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        selectSamples = negSamples + posSamples[:neg]
    elif neg>pos:
        selectSamples = negSamples[:pos] + posSamples
    random.seed(int(time.time()))
    random.shuffle(selectSamples)
    pos = 0
    neg = 0
    for item in selectSamples:
        if item.split()[1] == '0':
            neg+=1
        else:
            pos+=1
    print('after sampling(pos:neg): ',pos,':',neg)
    return selectSamples

def getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics, allSemanticEdges, allCodeEmbDict, line_list):
    batchData = []
    batch_metric = []
    batch_codeEmbedding = []
    batch_label = []
    batch_severity_label = []
    for line in line_list:
        try:
            info = line.rstrip('\n').split()
            #print('info',info)
            codeName = info[0] + ".java"
            label = int(info[1])
            severity_label = int(info[2])

            if our_RQ == 1 or our_RQ == 2:
                metrics = allCommonMetrics[info[0]] + allCommitMetrics[info[0]] + allStructuralMetrics[info[0]] + finalSelectedSynMetrics[info[0]] + allSemanticEdges[info[0]]
            elif our_RQ == 3:
                metrics = allCommonMetrics[info[0]] + allCommitMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[0]: # metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']
                metrics = allCommonMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[1]:
                metrics = allCommitMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[2]:
                metrics = allStructuralMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[3]:
                metrics = finalSelectedSynMetrics[info[0]]
            elif our_RQ == 4 and metric_type == metricType[4]:
                metrics = allSemanticEdges[info[0]]

            codeEmbedding = allCodeEmbDict[codeName]
            batch_metric.append(metrics)
            batch_codeEmbedding.append(codeEmbedding)
            batch_label.append(label)
            batch_severity_label.append(severity_label)
            batchData.append([batch_metric, batch_codeEmbedding, batch_label, batch_severity_label])
        except:
            pass
    return batchData

def getBatch(line_list, batch_size, batch_index, our_RQ, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    batchData = getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics, allSemanticEdges , allCodeEmbDict, line_list[start_line:end_line])
    return batchData

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, binaryClassifaction):
    # 预测
    model.eval()

    batchData = getBatch(testlist, 100000, 0, our_RQ, device)
    batch_metric, batch_codeEmbedding, batch_label, batch_severity_label = batchData[0]

    x = torch.as_tensor(batch_metric).to(device)
    y = torch.as_tensor(batch_codeEmbedding).to(device)
    batch_label = torch.LongTensor(batch_label).to(device)
    batch_severity_label = torch.LongTensor(batch_severity_label).to(device)

    with torch.no_grad():
        y_pred = model(x, y)
        y_scores = y_pred.cpu().numpy()
    if binaryClassifaction:
        y_test = batch_label.cpu()
    else:
        y_test = batch_severity_label.cpu()
    # 计算p, r, f1, acc
    y_pred = np.argmax(y_pred.cpu().numpy(), axis=1)
    p = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
    r = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    # 计算auc
    if binaryClassifaction:
        y_prob = torch.softmax(torch.tensor(y_scores), dim=1).numpy()[:, 1]
        #print('y_prob',y_prob.shape)
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = torch.softmax(torch.tensor(y_scores), dim=1).numpy()
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    # 输出指标
    print('Precision: {:.4f}'.format(p))
    print('Recall: {:.4f}'.format(r))
    print('F1: {:.4f}'.format(f1))
    print('Accuracy: {:.4f}'.format(acc))
    print('AUC: {:.4f}'.format(auc))

    p = float(format(p, '.4f'))
    r = float(format(r, '.4f'))
    f1 = float(format(f1, '.4f'))
    acc = float(format(acc, '.4f'))
    auc = float(format(auc, '.4f'))
    print("\n p, r, f1, acc, auc:", p, r, f1, acc, auc)
    return p, r, f1, acc, auc

        


def train(model, train_label_list, test_label_list, test_result, fold_idx, our_RQ, binaryClassifaction):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    fold_index_record = open(test_result, 'a')
    fold_index_record.write("\n\n-------fold_index_record:" + str(fold_idx) + "--------\n")
    fold_index_record.close()

    f1_max = 0
    acc_max = 0
    auc_max = 0
    iterations = 0
    epochs = trange(args.epochs, leave=True, desc = "Epoch")
    for epoch in epochs:
        #print(epoch)
        totalloss=0.0
        main_index=0.0
        count = 0
        right = 0
        acc = 0
        
        trainlist = Undersampling(train_label_list)
        #random.shuffle(train_label_list)
        #trainlist = train_label_list
        model.train()
        for batch_index in tqdm(range(int(len(trainlist)/args.batch_size))):
            optimizer.zero_grad()
            batchData = getBatch(trainlist, args.batch_size, batch_index, our_RQ, device)
            batch_metric, batch_codeEmbedding, batch_label, batch_severity_label = batchData[0]

            x = torch.as_tensor(batch_metric).to(device)
            y = torch.as_tensor(batch_codeEmbedding).to(device)
            batch_label = F.one_hot(torch.LongTensor(batch_label), 2).to(device)
            batch_severity_label = F.one_hot(torch.LongTensor(batch_severity_label), 4).to(device)
            
            output = model(x, y)
            #print('output',output.shape,output)
            #print('batch_label',batch_label.shape,batch_label)
            if binaryClassifaction: # binary classifacation
                batchloss = criterion(output, batch_label.float())
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(batch_label, dim=1)))
            else: # multi classifacation
                batchloss = criterion(output, batch_severity_label.float())
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(batch_severity_label, dim=1)))
            
            count += len(batch_metric)
            #print("batchloss",batchloss)
            acc = right*1.0/count
            batchloss.backward(retain_graph = True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch_metric)
            loss = totalloss/main_index
            epochs.set_description("Epoch (Loss=%.6g) (Acc = %.6g)" % (round(loss,5) , acc))
            iterations += 1
            
        
        p, r, f1, acc, auc = test(test_label_list, epoch, binaryClassifaction)

        if f1>f1_max and auc>auc_max:
            p_max = p
            r_max = r
            f1_max = f1
            acc_max = acc
            auc_max = auc
            test_p_r_f1 = open(test_result, 'a')
            test_p_r_f1.write('epoch'+str(epoch) +" "+ str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(acc) +" "+ str(auc)+"\n")
            test_p_r_f1.close()

    return p_max, r_max, f1_max, acc_max, auc_max



if __name__ == '__main__':

    smellType = ['data_class','blob','feature_envy','long_method']
    metricType = ['common', 'commit', 'structure', 'syntax', 'semantics'][2:]
    #metricType = ['syntax', 'semantics']

    # javaMetricsPP metrics ## 16 / 4 common metrics  +  19 commit metrics
    allCommonMetricsPath = "LearningAlgorithms/dataset/allCommonMetrics.json"
    allCommitMetricsPath = "LearningAlgorithms/dataset/allCommitMetrics.json"
    allCommonMetrics = getDictByJson(allCommonMetricsPath)
    allCommitMetrics = getDictByJson(allCommitMetricsPath)

    # AST-based metrics ## 10 structureal metrics + 20 syntax metrics + 9 semantics metrics
    allStructuralMetricsPath = "LearningAlgorithms/dataset/allStructuralMetrics.json"
    finalSelectedSynMetricsPath = "LearningAlgorithms/dataset/finalSelectedSynMetrics.json"
    allSemanticEdgesPath = "LearningAlgorithms/dataset/allSemanticEdges.json"
    allStructuralMetrics = getDictByJson(allStructuralMetricsPath)
    finalSelectedSynMetrics = getDictByJson(finalSelectedSynMetricsPath)
    allSemanticEdges = getDictByJson(allSemanticEdgesPath)

    # Code2Vec  ## 128d code embedding
    codeEmbDictPath = "LearningAlgorithms/dataset/allCodeVectors.json"
    allCodeEmbDict = getDictByJson(codeEmbDictPath)

    #our_RQ = 3
    #binaryClassifaction = True
    #binaryClassifaction = False

    # when RQ4
    #metric_type = metricType[0]
    our_RQ = 4
    for metric_type in metricType:
        for binaryClassifaction in [True, False]:
            if our_RQ == 1:  #  (commonMetrics + commitMetrics + structuralMetrics + syntaxMetrics + semanticMetrics) && (codeVector)  as  model input
                modelList = [MyDNN_fusion, MyCNN_fusion, MyGRU_fusion, MyLSTM_fusion, MyTransformer_fusion]
            elif our_RQ == 2:  #  (commonMetrics + commitMetrics + structuralMetrics + syntaxMetrics + semanticMetrics)  or (codeVector)  as  model input
                modelList = [MyDNN_metrics, MyCNN_metrics, MyGRU_metrics, MyLSTM_metrics, MyTransformer_metrics, MyDNN_semantics, MyCNN_semantics, MyGRU_semantics, MyLSTM_semantics, MyTransformer_semantics]
            elif our_RQ == 3:  #  (commonMetrics + commitMetrics) && (codeVector)  as  model input
                modelList = [MyDNN_fusion, MyCNN_fusion, MyGRU_fusion, MyLSTM_fusion, MyTransformer_fusion]
            elif our_RQ == 4:  #  commonMetrics  or commitMetrics  or structuralMetrics  or syntaxMetrics  or semanticMetrics  as  model input
                #modelList = [MyDNN_metrics, MyCNN_metrics, MyGRU_metrics, MyLSTM_metrics, MyTransformer_metrics]
                modelList = [MyDNN_metrics, MyCNN_metrics]


            for curModel in modelList:
                print('model:', str(curModel))
                for smell in smellType:
                    print('smell:', smell)
                    ############## save experimental results ###########
                    if binaryClassifaction:
                        current_task = 'binary'
                        num_classes = 2
                    else:
                        current_task = 'severity'
                        num_classes = 4
                    
                    if our_RQ == 4:
                        test_result = "LearningAlgorithms/results/result_RQ_" + str(our_RQ) + '__' + str(curModel.__name__) + '_' + smell + '_' + current_task + '_'  + metric_type + '.txt'
                    else:
                        test_result = "LearningAlgorithms/results/result_RQ_" + str(our_RQ) + '__' + str(curModel.__name__) + '_' + smell + '_' + current_task + '.txt'
                    ####################################################
                    
                    labelPath = "LearningAlgorithms/dataset/labels_"+ smell +".txt"
                    with open(labelPath) as f:
                        label_list = f.readlines()

                    print("\n -----------------------DataInfo------------------------")
                    fold_num = 5
                    seed = 666
                    print(smell)
                    print("seed =",seed)
                    print("fold_num =",fold_num)
                    random.seed(seed)
                    random.shuffle(label_list)

                    p_sum = 0
                    r_sum = 0
                    f1_sum = 0
                    acc_sum = 0
                    auc_sum = 0

                    for fold_idx in range(1,6):
                        print(' fold_idx:',fold_idx)
                        #数据集划分，按项目名称划分，进行5折/3折交叉验证
                        train_label_list, test_label_list = getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx)
                        #print("train_label_list",len(train_label_list))
                        #print("test_label_list",len(test_label_list))
                        showRitiaOfPosNeg(train_label_list, test_label_list)
                        #continue
                        # exit()
                        if our_RQ == 1 or our_RQ == 2:
                            if smell == "data_class" or smell == "blob":
                                metricSize = 74
                            elif smell == "feature_envy" or smell == "long_method":
                                metricSize = 62
                        elif our_RQ == 3:
                            if smell == "data_class" or smell == "blob":
                                metricSize = 35
                            elif smell == "feature_envy" or smell == "long_method":
                                metricSize = 23
                        elif our_RQ == 4 and metric_type == metricType[0]: # metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']
                            if smell == "data_class" or smell == "blob":
                                metricSize = 16
                            elif smell == "feature_envy" or smell == "long_method":
                                metricSize = 4
                        elif our_RQ == 4 and metric_type == metricType[1]:
                            metricSize = 19
                        elif our_RQ == 4 and metric_type == metricType[2]:
                            metricSize = 10
                        elif our_RQ == 4 and metric_type == metricType[3]:
                            metricSize = 20
                        elif our_RQ == 4 and metric_type == metricType[4]:
                            metricSize = 9

                        model = curModel(metricSize, args.hidden, num_classes).to(device)
                        p, r, f1, acc, auc = train(model, train_label_list, test_label_list, test_result, fold_idx, our_RQ, binaryClassifaction)
                        p_sum+=p
                        r_sum+=r
                        f1_sum+=f1
                        acc_sum+=acc
                        auc_sum+=auc

                    avg_p = p_sum / fold_num
                    avg_r = r_sum / fold_num
                    avg_f1 = f1_sum / fold_num
                    avg_acc = acc_sum / fold_num
                    avg_auc = auc_sum / fold_num

                    test_p_r_f1 = open(test_result, 'a')
                    test_p_r_f1.write("\n\navg_values: "+ str(avg_p) +" "+ str(avg_r) +" "+ str(avg_f1) +" "+ str(avg_acc) +" "+ str(avg_auc)+"\n")
                    test_p_r_f1.close()