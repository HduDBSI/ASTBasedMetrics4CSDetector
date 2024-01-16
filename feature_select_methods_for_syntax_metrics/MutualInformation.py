#互信息
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
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
#print('X',X)
#print('y',y)

# 计算特征与目标变量的互信息
scores = mutual_info_classif(X, y)

# 输出特征得分排序
print(pd.Series(scores, index=X.columns).sort_values(ascending=False))

# 筛选得分最高的 top K 特征，以 top 2 为例
selected_features = pd.Series(scores, index=X.columns).nlargest(20).index.tolist()

# 输出筛选后的特征在原始特征中的序号
print([X.columns.get_loc(feature) for feature in selected_features])

print("选中的特征序号为：", list(selected_features))
for i in list(selected_features):
    print(type_list[i])