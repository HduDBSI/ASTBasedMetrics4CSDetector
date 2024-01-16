
import os
import numpy as np
import json
from astTools import *
from tqdm import tqdm
#from func_timeout import func_set_timeout, FunctionTimedOut

def getCodeGraphDataByPath(codePath):
    pureAST = code2AST(codePath) #得到AST需要的数据，递归各节点遍历出一棵树 tree
    newtree, nodelist = getNodeList(pureAST)
    ifcount,whilecount,forcount,blockcount,docount,switchcount,alltokens,vocabdict = astStaticCollection(pureAST)
    h,x,vocabdict,edge_index,edge_attr = getFA_AST(newtree, vocabdict)
    return nodelist, edge_attr


if __name__ == '__main__':
    datasetPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/"
    sourceCodePath = datasetPath + 'sourceCode/'
    synMetricsSavePath = datasetPath + 'allSynMetrics.json'
    semanticEdgeSavePath = datasetPath + 'allSemanticEdges.json'
    fanailSelectedSynMetricsSavePath = datasetPath + 'fanailSelectedSynMetrics.json'


    selected_node = ['ReferenceType', 'VariableDeclarator', 'MemberReference', 'Literal', 'MethodDeclaration', 'BlockStatement', 'ReturnStatement', 'set', 'str', 'FieldDeclaration', 'FormalParameter', 'StatementExpression', 'Assignment', 'This', 'MethodInvocation', 'LocalVariableDeclaration', 'BasicType', 'IfStatement', 'BinaryOperation', 'ClassCreator']

    fanailSelectedSynMetrics = {}
    except_num = 0
    for root, dirs, files in os.walk(sourceCodePath):
        for file in tqdm(files):
            # 判断文件是否以".java"结尾
            if file.endswith(".java"):
                # 打印文件路径
                codaPath = os.path.join(root, file)
                fileName = os.path.join(file).split('.')[0]
                #print(codaPath)
                try:
                    typeDict = {}
                    nodelist,_ = getCodeGraphDataByPath(codaPath)
                    for node in nodelist:
                        node_name = str(node.__class__.__name__)
                        flag = fileName.split('__')[0]
                        if node_name in selected_node and node_name not in typeDict:
                            typeDict[node_name] = 1
                        elif node_name in typeDict:
                            typeDict[node_name] += 1
                    type_count_list = []
                    for t in selected_node:
                        if t in typeDict.keys():
                            type_count_list.append(typeDict[t])
                        else:
                            type_count_list.append(0)
                    fanailSelectedSynMetrics[fileName] = type_count_list
                except:
                    except_num += 1
    print('except_num', except_num)
    print('allSynMetrics', len(fanailSelectedSynMetrics))
    # 将其保存到本地
    fanailSelectedSynMetricsDictFile = open(fanailSelectedSynMetricsSavePath, "w")
    json.dump(fanailSelectedSynMetrics,fanailSelectedSynMetricsDictFile)
    fanailSelectedSynMetricsDictFile.close()

    exit()
    # exampleCode = sourceCodePath + "class__blob__critical__0__10147__73aadbad04fc9ec8b14faa2191c4c9545f3396a4__ConstraintBasePanel__44__505.java"
    # ifcount,whilecount,forcount,blockcount,docount,switchcount = getCodeGraphDataByPath(exampleCode)
    # print(ifcount,whilecount,forcount,blockcount,docount,switchcount)

    #遍历文件夹下的所有文件
    # typeDict = {}
    # for root, dirs, files in os.walk(sourceCodePath):
    #     for file in tqdm(files):
    #         # 判断文件是否以".java"结尾
    #         if file.endswith(".java"):
    #             # 打印文件路径
    #             codaPath = os.path.join(root, file)
    #             fileName = os.path.join(file).split('.')[0]
    #             #print(codaPath)
    #             try:
    #                 nodelist,_ = getCodeGraphDataByPath(codaPath)
    #                 for node in nodelist:
    #                     node_name = str(node.__class__.__name__)
    #                     if node_name not in typeDict:
    #                         typeDict[node_name] = 1
    #                     else:
    #                         typeDict[node_name] += 1
    #             except:
    #                 pass
    # print(typeDict)
    typeDict = {'ClassDeclaration': 9551, 'set': 159392, 'str': 2801348, 'FieldDeclaration': 41553, 'ReferenceType': 411099, 'VariableDeclarator': 132624, 'ClassCreator': 47618, 'MemberReference': 502991, 'Literal': 241536, 'ConstructorDeclaration': 8893, 'Annotation': 37652, 'FormalParameter': 97559, 'StatementExpression': 201865, 'SuperConstructorInvocation': 2927, 'Assignment': 74071, 'This': 43625, 'MethodInvocation': 373025, 'LocalVariableDeclaration': 78461, 'BinaryOperation': 150750, 'MethodDeclaration': 85264, 'BasicType': 75109, 'Cast': 16408, 'IfStatement': 74590, 'BlockStatement': 89731, 'ReturnStatement': 75257, 'WhileStatement': 2689, 'ForStatement': 10822, 'ForControl': 4096, 'VariableDeclaration': 10490, 'TryStatement': 9459, 'CatchClause': 8746, 'CatchClauseParameter': 8746, 'TypeArgument': 54312, 'EnhancedForControl': 6726, 'ElementValuePair': 6647, 'ClassReference': 7417, 'ThrowStatement': 11996, 'TryResource': 678, 'TernaryExpression': 5770, 'TypeParameter': 2284, 'SuperMethodInvocation': 3186, 'ContinueStatement': 1585, 'SwitchStatement': 1816, 'SwitchStatementCase': 7807, 'DoStatement': 305, 'BreakStatement': 5496, 'InterfaceDeclaration': 594, 'ElementArrayValue': 738, 'ArrayCreator': 3148, 'ArraySelector': 11110, 'ConstantDeclaration': 796, 'ExplicitConstructorInvocation': 925, 'ArrayInitializer': 1676, 'AssertStatement': 986, 'MethodReference': 772, 'LambdaExpression': 1822, 'SynchronizedStatement': 945, 'EnumDeclaration': 316, 'EnumBody': 316, 'EnumConstantDeclaration': 1644, 'InferredFormalParameter': 264, 'bool': 838, 'Statement': 42, 'VoidClassReference': 22, 'InnerClassCreator': 14, 'AnnotationDeclaration': 108, 'SuperMemberReference': 24}
    type_list = ['ClassDeclaration', 'set', 'str', 'FieldDeclaration', 'ReferenceType', 'VariableDeclarator', 'ClassCreator', 'MemberReference', 'Literal', 'ConstructorDeclaration', 'Annotation', 'FormalParameter', 'StatementExpression', 'SuperConstructorInvocation', 'Assignment', 'This', 'MethodInvocation', 'LocalVariableDeclaration', 'BinaryOperation', 'MethodDeclaration', 'BasicType', 'Cast', 'IfStatement', 'BlockStatement', 'ReturnStatement', 'WhileStatement', 'ForStatement', 'ForControl', 'VariableDeclaration', 'TryStatement', 'CatchClause', 'CatchClauseParameter', 'TypeArgument', 'EnhancedForControl', 'ElementValuePair', 'ClassReference', 'ThrowStatement', 'TryResource', 'TernaryExpression', 'TypeParameter', 'SuperMethodInvocation', 'ContinueStatement', 'SwitchStatement', 'SwitchStatementCase', 'DoStatement', 'BreakStatement', 'InterfaceDeclaration', 'ElementArrayValue', 'ArrayCreator', 'ArraySelector', 'ConstantDeclaration', 'ExplicitConstructorInvocation', 'ArrayInitializer', 'AssertStatement', 'MethodReference', 'LambdaExpression', 'SynchronizedStatement', 'EnumDeclaration', 'EnumBody', 'EnumConstantDeclaration', 'InferredFormalParameter', 'bool', 'Statement', 'VoidClassReference', 'InnerClassCreator', 'AnnotationDeclaration', 'SuperMemberReference']
    print('type_list',len(type_list))

    # 为每个代码片段生成其对应的节点类型树目统计
    allSynMetrics = {}
    except_num = 0
    for root, dirs, files in os.walk(sourceCodePath):
        for file in tqdm(files):
            # 判断文件是否以".java"结尾
            if file.endswith(".java"):
                # 打印文件路径
                codaPath = os.path.join(root, file)
                fileName = os.path.join(file).split('.')[0]
                #print(codaPath)
                try:
                    typeDict = {}
                    nodelist,_ = getCodeGraphDataByPath(codaPath)
                    for node in nodelist:
                        node_name = str(node.__class__.__name__)
                        flag = fileName.split('__')[0]
                        if node_name not in typeDict:
                            typeDict[node_name] = 1
                            if flag == "function" and ('ClassDeclaration' == node_name or 'ClassCreator' == node_name):
                                typeDict[node_name] = 0
                                print(node_name)
                        else:
                            typeDict[node_name] += 1
                            if flag == "function" and ('ClassDeclaration' == node_name or 'ClassCreator' == node_name):
                                typeDict[node_name] = 0
                                print(node_name)
                    type_count_list = []
                    for t in type_list:
                        if t in typeDict.keys():
                            type_count_list.append(typeDict[t])
                        else:
                            type_count_list.append(0)
                    allSynMetrics[fileName] = type_count_list
                except:
                    except_num += 1
    print('except_num', except_num)
    print('allSynMetrics', len(allSynMetrics))
    # 将其保存到本地
    synMetricsDictFile = open(synMetricsSavePath, "w")
    json.dump(allSynMetrics,synMetricsDictFile)
    synMetricsDictFile.close()












    exit()
    
    # 遍历文件夹下的所有文件
    edges = ['childFather','Nextsib','Nexttoken','Nextuse','If','Ifelse','While','For','Nextstmt']
    edges = [0,1,2,4,6,7,8,9,10]
    allSemanticEdgeDict = {}
    except_num = 0
    for root, dirs, files in os.walk(sourceCodePath):
        for file in tqdm(files):
            # 判断文件是否以".java"结尾
            if file.endswith(".java"):
                # 打印文件路径
                codaPath = os.path.join(root, file)
                fileName = os.path.join(file).split('.')[0]
                #print(codaPath)
                try:
                    typeDict = {}
                    _, edgelist = getCodeGraphDataByPath(codaPath)
                    for node in edgelist:
                        if node not in typeDict:
                            typeDict[node] = 1
                        else:
                            typeDict[node] += 1
                    edge_type_list = []
                    for k in edges:
                        if k in typeDict.keys():
                            edge_type_list.append(typeDict[k])
                        else:
                            edge_type_list.append(0)
                    allSemanticEdgeDict[fileName] = edge_type_list
                except:
                    except_num += 1

    print('except_num',except_num)

    # 将其保存到本地
    semanticEdgeDictFile = open(semanticEdgeSavePath, "w")
    json.dump(allSemanticEdgeDict,semanticEdgeDictFile)
    semanticEdgeDictFile.close()

    # print(typeDict)
    # edges={'childFather':0,'Nextsib':1,'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}
    # {0: 12035638, 1: 2808573, 12: 2808573, 6: 149180, 7: 38360, 8: 5378, 9: 21644, 10: 82847, 11: 82847, 2: 2808573, 3: 2808573, 4: 343593, 5: 343593}