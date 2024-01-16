# PCA
# PCA
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np 
import json

datasetPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/"
synMetricsSavePath = datasetPath + 'allSynMetrics.json'
type_list = ['ClassDeclaration', 'set', 'str', 'FieldDeclaration', 'ReferenceType', 'VariableDeclarator', 'ClassCreator', 'MemberReference', 'Literal', 'ConstructorDeclaration', 'Annotation', 'FormalParameter', 'StatementExpression', 'SuperConstructorInvocation', 'Assignment', 'This', 'MethodInvocation', 'LocalVariableDeclaration', 'BinaryOperation', 'MethodDeclaration', 'BasicType', 'Cast', 'IfStatement', 'BlockStatement', 'ReturnStatement', 'WhileStatement', 'ForStatement', 'ForControl', 'VariableDeclaration', 'TryStatement', 'CatchClause', 'CatchClauseParameter', 'TypeArgument', 'EnhancedForControl', 'ElementValuePair', 'ClassReference', 'ThrowStatement', 'TryResource', 'TernaryExpression', 'TypeParameter', 'SuperMethodInvocation', 'ContinueStatement', 'SwitchStatement', 'SwitchStatementCase', 'DoStatement', 'BreakStatement', 'InterfaceDeclaration', 'ElementArrayValue', 'ArrayCreator', 'ArraySelector', 'ConstantDeclaration', 'ExplicitConstructorInvocation', 'ArrayInitializer', 'AssertStatement', 'MethodReference', 'LambdaExpression', 'SynchronizedStatement', 'EnumDeclaration', 'EnumBody', 'EnumConstantDeclaration', 'InferredFormalParameter', 'bool', 'Statement', 'VoidClassReference', 'InnerClassCreator', 'AnnotationDeclaration', 'SuperMemberReference']
with open(synMetricsSavePath, 'r') as f:
    synData = f.read()
    jsonData = json.loads(synData)
    X = []
    y = []
    for k in jsonData.keys():
        X.append(jsonData[k])
        label = k.split('__')[2]
        if label == 'none':
            y.append(0)
        elif label == 'minor':
            y.append(1)
        elif label == 'major':
            y.append(2)
        elif label == 'critical':
            y.append(3)

        #break
X = pd.DataFrame(np.array(X))
y = np.array(y)
print('X',X)
print('y',y)

# 进行主成分分析
# PCA降维
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

# 输出筛选后的特征在原始特征中的index
print("PCA选出的特征index为：", list(pca.components_.argsort()[::-1][0][:20]))