
import torch
import numpy as np
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=10000, formatter={'float': '{:0.9f}'.format})
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import random
import json
import time

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

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

def getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics, allSemanticEdges, allCodeEmbDict, line_list):
    batchData = []
    batch_label = []
    feature_list = []
    codeEmb_list = []
    metric_list = []
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
                #metrics = allStructuralMetrics[info[0]] + finalSelectedSynMetrics[info[0]] + allSemanticEdges[info[0]]
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
            metrics = F.normalize(torch.as_tensor(metrics).view(1,-1), dim=1).tolist()[0]
            #feature_list.append(codeEmbedding + metrics)
            codeEmb_list.append(codeEmbedding)
            metric_list.append(metrics)
            batch_label.append(label)
            batch_severity_label.append(severity_label)
            batchData.append([codeEmb_list, metric_list, batch_label, batch_severity_label])
            #batchData.append([feature_list, batch_label, batch_severity_label])
        except:
            pass
    return batchData

def getBatch(line_list, batch_size, batch_index, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    batchData = getBatchList(our_RQ, allCommonMetrics, allCommitMetrics, allStructuralMetrics, finalSelectedSynMetrics, allSemanticEdges, allCodeEmbDict, line_list[start_line:end_line])
    return batchData

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def train(curClf, train_label_list, test_label_list, test_result, fold_idx, binaryClassifaction):
    
    fold_index_record = open(test_result, 'a')
    fold_index_record.write("\n\n-------fold_index_record:" + str(fold_idx) + "--------\n")
    fold_index_record.close()

    if curClf == 'svm':
        # SVM参数
        KERNEL = 'linear'
        clf = svm.SVC(kernel=KERNEL, probability=True)
    elif curClf == 'XGBClassifier':
        # XGBoost参数
        N_ESTIMATORS = 100
        MAX_DEPTH = 3
        LEARNING_RATE = 0.1
        # 实例化分类器对象
        clf = XGBClassifier(use_label_encoder=False, eval_metric='error', n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE)
    elif curClf == 'RandomForestClassifier':
        # 随机森林参数
        NUM_TREES = 10
        MAX_DEPTH = 10
        # 训练随机森林
        clf = RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEPTH)
    elif curClf == 'GaussianNB':
        # 训练朴素贝叶斯分类器
        clf = GaussianNB()
    elif curClf == 'DecisionTreeClassifier':
        # 训练决策树
        clf = DecisionTreeClassifier()

    batchData = getBatch(train_label_list, 100000, 0, device)
    #features, batch_label, batch_severity_label = batchData[0]

    codeEmb_list, metric_list, batch_label, batch_severity_label = batchData[0]
    # metric_list = pca.fit_transform(np.array(metric_list)).tolist()
    if our_RQ == 1 or our_RQ == 3:
        #codeEmb_list = pca.fit_transform(np.array(codeEmb_list)).tolist()
        features = [i+j for i,j in zip(codeEmb_list,metric_list)]
    elif our_RQ == 2 and inputInfo == 'metrics':
        features = metric_list
    elif our_RQ == 2 and inputInfo == 'semantics':
        features = codeEmb_list
    elif our_RQ == 4:
        features = metric_list
    
    # 转换为numpy数组
    X_train = torch.as_tensor(features).cuda()
    if binaryClassifaction:
        y_train = torch.LongTensor(batch_label).cuda()
    else:
        y_train = torch.LongTensor(batch_severity_label).cuda()
    
    # 训练分类器
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    

    # 预测
    batchData = getBatch(test_label_list, 100000, 0, device)
    #features, batch_label, batch_severity_label = batchData[0]
    
    codeEmb_list, metric_list, batch_label, batch_severity_label = batchData[0]
    # metric_list = pca.fit_transform(np.array(metric_list)).tolist()
    if our_RQ == 1 or our_RQ == 3:
        #codeEmb_list = pca.fit_transform(np.array(codeEmb_list)).tolist()
        features = [i+j for i,j in zip(codeEmb_list,metric_list)]
    elif our_RQ == 2 and inputInfo == 'metrics':
        features = metric_list
    elif our_RQ == 2 and inputInfo == 'semantics':
        features = codeEmb_list
    elif our_RQ == 4:
        features = metric_list

    # 转换为numpy数组
    X_test = torch.as_tensor(features).cuda().cpu().numpy()
    if binaryClassifaction:
        y_test = torch.LongTensor(batch_label).cuda().cpu().numpy()
    else:
        y_test = torch.LongTensor(batch_severity_label).cuda().cpu().numpy()

    # 预测
    y_pred = clf.predict(X_test)
    
    # 计算p, r, f1, acc
    p = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
    r = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    if binaryClassifaction:
        # 计算auc
        y_scores = clf.predict_proba(X_test)[:, 1]  # 二分类
        auc = roc_auc_score(y_test, y_scores)
    else:
        # 计算auc
        y_scores = clf.predict_proba(X_test)  # 多分类
        auc = roc_auc_score(y_test, y_scores, multi_class='ovr')

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
    
    test_p_r_f1 = open(test_result, 'a')
    test_p_r_f1.write(str(curClf) +" "+ str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(acc) +" "+ str(auc)+"\n")
    test_p_r_f1.close()

    return p, r, f1, acc, auc


if __name__ == '__main__':


    smellType = ['data_class','blob','feature_envy','long_method']

    clfs = ['svm', 'XGBClassifier', 'RandomForestClassifier', 'GaussianNB', 'DecisionTreeClassifier']

    metricType = ['common', 'commit', 'structure', 'syntax', 'semantics']

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

    # when RQ == 1 or 3
    #our_RQ = 3
    #binaryClassifaction = True
    #binaryClassifaction = False

    # when RQ == 2
    our_RQ = 2
    #binaryClassifaction = True
    binaryClassifaction = False
    #
    inputInfo = 'metrics'
    #inputInfo = 'semantics'

    # when RQ4
    metric_type = metricType[0]
    #####################################################################################################################################
    if our_RQ > 1:
        clfs = ['XGBClassifier', 'RandomForestClassifier']

    for curClf in clfs:
        print('curClf:', str(curClf))
        for smell in smellType:
            print('smell:', smell)
            labelPath = "LearningAlgorithms/dataset/labels_"+ smell +".txt"
            ############## save experimental results ###########
            if binaryClassifaction:
                current_task = 'binary'
                #num_classes = 2
            else:
                current_task = 'severity'
                #num_classes = 4
            if our_RQ == 1 or our_RQ == 3:
                test_result = "LearningAlgorithms/results/result_RQ_" + str(our_RQ) + '__' + str(curClf) + '_' + smell + '_' + current_task + '.txt'
            elif our_RQ == 2:
                test_result = "LearningAlgorithms/results/result_RQ_" + str(our_RQ) + '__' + str(curClf) + '_' + smell + '_' + current_task + '_' + inputInfo + '.txt'
            elif our_RQ == 4:
                test_result = "LearningAlgorithms/results/result_RQ_" + str(our_RQ) + '__' + str(curClf) + '_' + smell + '_' + current_task + '_' + metric_type + '.txt'
            ####################################################
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
                        #metricSize = 39
                    elif smell == "feature_envy" or smell == "long_method":
                        metricSize = 62
                        #metricSize = 39
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

                #pca = PCA(n_components=metricSize)
                p, r, f1, acc, auc = train(curClf, train_label_list, test_label_list, test_result, fold_idx, binaryClassifaction)
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