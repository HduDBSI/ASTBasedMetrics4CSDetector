
feature_list = ['ClassDeclaration', 'set', 'str', 'FieldDeclaration', 'ReferenceType', 'VariableDeclarator', 'ClassCreator', 'MemberReference', 'Literal', 'ConstructorDeclaration', 'Annotation', 'FormalParameter', 'StatementExpression', 'SuperConstructorInvocation', 'Assignment', 'This', 'MethodInvocation', 'LocalVariableDeclaration', 'BinaryOperation', 'MethodDeclaration', 'BasicType', 'Cast', 'IfStatement', 'BlockStatement', 'ReturnStatement', 'WhileStatement', 'ForStatement', 'ForControl', 'VariableDeclaration', 'TryStatement', 'CatchClause', 'CatchClauseParameter', 'TypeArgument', 'EnhancedForControl', 'ElementValuePair', 'ClassReference', 'ThrowStatement', 'TryResource', 'TernaryExpression', 'TypeParameter', 'SuperMethodInvocation', 'ContinueStatement', 'SwitchStatement', 'SwitchStatementCase', 'DoStatement', 'BreakStatement', 'InterfaceDeclaration', 'ElementArrayValue', 'ArrayCreator', 'ArraySelector', 'ConstantDeclaration', 'ExplicitConstructorInvocation', 'ArrayInitializer', 'AssertStatement', 'MethodReference', 'LambdaExpression', 'SynchronizedStatement', 'EnumDeclaration', 'EnumBody', 'EnumConstantDeclaration', 'InferredFormalParameter', 'bool', 'Statement', 'VoidClassReference', 'InnerClassCreator', 'AnnotationDeclaration', 'SuperMemberReference']

s1 = [1, 19, 5, 24, 2, 3, 7, 16, 12, 14, 17, 4, 23, 11, 22, 10, 8, 6, 20, 15]
s2 = [2, 8, 7, 18, 4, 16, 12, 23, 5, 1, 22, 17, 11, 20, 14, 32, 15, 19, 24, 6]
s3 = [3, 36, 7, 19, 24, 20, 8, 5, 16, 4, 32, 30, 31, 29, 12, 23, 18, 25, 11, 50]
s4 = [2, 7, 5, 16, 8, 12, 14, 4, 22, 17, 23, 1, 18, 24, 3, 19, 11, 20, 26, 15]
s5 = [19, 5, 1, 7, 8, 34, 0, 2, 3, 4, 9, 10, 13, 14, 15, 17, 21, 22, 23, 24]



# 合并所有特征集合
S_all = set(s1 + s2 + s3 + s4 + s5)
print('S_all', len(S_all))
# 计算每个特征的得票数和排名
vote_count = {}
for f in S_all:
    vote_count[f] = s1.count(f) + s2.count(f) + s3.count(f) + s4.count(f) + s5.count(f)

print(vote_count)
k = 20
sorted_dict = dict(sorted(vote_count.items(), key=lambda item: item[1], reverse=True)[:k])
print('sorted_dict', sorted_dict)  # {'e': 50, 'd': 40, 'c': 30}
selected_node = []
for i in sorted_dict.keys():
    selected_node.append(feature_list[i])
print(selected_node, len(selected_node))
'''
sorted_dict {4: 5, 5: 5, 7: 5, 8: 5, 19: 5, 23: 5, 24: 5, 1: 4, 2: 4, 3: 4, 11: 4, 12: 4, 14: 4, 15: 4, 16: 4, 17: 4, 20: 4, 22: 4, 18: 3, 6: 2}
['ReferenceType', 'VariableDeclarator', 'MemberReference', 'Literal', 'MethodDeclaration', 'BlockStatement', 'ReturnStatement', 'set', 'str', 'FieldDeclaration', 'FormalParameter', 'StatementExpression', 'Assignment', 'This', 'MethodInvocation', 'LocalVariableDeclaration', 'BasicType', 'IfStatement', 'BinaryOperation', 'ClassCreator']
'''
exit()
vote_rank = sorted(vote_count.items(), key=lambda x: -x[1])

# 计算每个特征的得分
score = {}
for i, (f, count) in enumerate(vote_rank):
    rank = i + 1
    score[f] = count * (5 - rank)

# 按得分从高到低排序，选取前 k 个特征
k = 20
top_k = sorted(score.items(), key=lambda x: -x[1])[:k]

# 输出结果
print('Top {} Features:'.format(k))
for f, s in top_k:
    print('{}: {}'.format(f, feature_list[f]))

print('S_all', len(S_all))