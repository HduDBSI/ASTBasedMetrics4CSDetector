from math import floor
import javalang
import ast
import astunparse
from javalang.ast import Node
import csv
from anytree import AnyNode
import textwrap
import hashlib
import os
import json
from tqdm import tqdm



def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(str(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    # print(node.__class__.__name__,str(node))
    # print(node.__class__.__name__, node)
    # print(node.__class__.__name__)
    return token
    
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []
 
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
           
def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    #print(type(node))
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)

def getParams_avg_max(codeTxt):
    sum_num = 0
    num=0
    max_num = 0
    #tokens = javalang.tokenizer.tokenize(source_code)
    pureAST = javalang.parse.parse(codeTxt)
    for path, node in pureAST:
        #print(type(node))
        if isinstance(node, javalang.tree.MethodDeclaration):
            num_params = len(node.parameters)
            num+=1
            sum_num += num_params
            if num_params > max_num:
                max_num = num_params
    avg_num = round(sum_num/num, 2)
    return avg_num, max_num
                
def getNodeList(tree):
    nodelist = []
    newtree=AnyNode(id=0,token=None,data=None)
    createtree(newtree, tree, nodelist)
    return newtree, nodelist

# def get_code_stm_blocks(code):
#     """
#     获取代码块列表
#     """
#     blocks = []
#     tokens = list(javalang.tokenizer.tokenize(code))
#     start = 0
#     for i in range(1, len(tokens)):
#         if tokens[i].value == ';':
#             block = ''.join([t.value for t in tokens[start:i+1]]).strip()
#             blocks.append(block)
#             start = i + 1
#     print(blocks)
#     return blocks, len(blocks)

def count_statement_blocks(node):
    """
    统计语句块数量
    """
    count = 0

    if isinstance(node, (javalang.tree.Statement, javalang.tree.BlockStatement)):
        count += 1
    for _, child_node in node.filter(javalang.tree.Statement):
        count += 1
    for _, child_node in node.filter(javalang.tree.BlockStatement):
        count += 1
    return count

def get_code_stm_blocks(code):
    parser = javalang.parse.Parser(javalang.tokenizer.tokenize(code))
    tree = parser.parse_member_declaration()
    return count_statement_blocks(tree)

# def get_code_blok_blocks(code):
#     """
#     获取代码块列表
#     """
#     blocks = []
#     tokens = list(javalang.tokenizer.tokenize(code))
#     braces = []
#     start = 0
#     for i, token in enumerate(tokens):
#         if token.value in ('{', '['):
#             braces.append(token)
#         elif token.value in ('}', ']'):
#             braces.pop()
#             if not braces:
#                 block = ''.join([t.value for t in tokens[start:i+1]]).strip()
#                 blocks.append(block)
#                 start = i + 1
#     return blocks, len(blocks)

def count_code_blocks(code):
    count = 0
    tree = javalang.parse.parse(code)
    #blocks = []
    for path, node in tree:
        if isinstance(node, (javalang.tree.BlockStatement)):
            count += 1
            #blocks.append(node)
    return count


def get_duplicate_blocks(blocks):
    """
    获取重复代码块列表
    """
    duplicates = []
    hashes = set()
    for block in blocks:
        block_hash = hashlib.sha256(block.encode()).hexdigest()
        if block_hash in hashes:
            duplicates.append(block)
        else:
            hashes.add(block_hash)
    return duplicates, len(duplicates)

# def getDuplicatesStm(codeTxt):
#     blocks, len1 = get_code_stm_blocks(codeTxt)
#     duplicates, len2 = get_duplicate_blocks(blocks)
#     print('Stm len1, len2',len1, len2)
#     return len2

# def getDuplicatesBlok(codeTxt):
#     blocks, len1 = count_code_blocks(codeTxt)
#     duplicates, len2 = get_duplicate_blocks(blocks)
#     print('Blok len1, len2',len1, len2)
#     return len2

# 读取Java文件并解析成AST树
def parse_java_file(codeTxt):
    tokens = javalang.tokenizer.tokenize(codeTxt)
    return javalang.parse.Parser(tokens).parse_member_declaration()

# 计算AST深度
def calculate_depth(node):
    if isinstance(node, AnyNode):
        #print('node.children',len(node.children))
        return 1 + max(map(calculate_depth, node.children), default=0)
    return 0

# 计算叶子节点数量
def calculate_leaf(node):
    global leaf_num
    if isinstance(node, AnyNode):
        if len(node.children) == 0:
            leaf_num +=1
        else:
            for child in node.children:
                calculate_leaf(child)
    return leaf_num
    
# 计算AST宽度
def calculate_width(node):
    max_width = 0
    queue = [(node, 1)]
    while queue:
        level_width = len(queue)
        max_width = max(max_width, level_width)
        for i in range(level_width):
            current_node, current_level = queue.pop(0)
            for child in current_node.children:
                queue.append((child, current_level+1))
    return max_width

def get_depth_width_of_ast(java_file_path):
    tree = parse_java_file(java_file_path)
    newtree, nodelist = getNodeList(tree)
    #print(newtree)
    depth = calculate_depth(newtree)
    width = calculate_width(newtree)
    leaf_num = calculate_leaf(newtree)
    # print("AST深度为：", depth)
    # print("AST宽度为：", width)
    return depth, width, len(nodelist), leaf_num

# 计算分支因子
def calculate_branch_factor(code):
    tree = javalang.parse.parse(code)
    branch_factor = 0
    for path, node in tree:
        if isinstance(node, javalang.tree.IfStatement):
            branch_factor += 1
        elif isinstance(node, javalang.tree.SwitchStatement):
            branch_factor += len(node.cases)
        elif isinstance(node, javalang.tree.WhileStatement):
            branch_factor += 1
        elif isinstance(node, javalang.tree.ForStatement):
            branch_factor += 1
        elif isinstance(node, javalang.tree.DoStatement):
            branch_factor += 1
        elif isinstance(node, javalang.tree.TryStatement):
            branch_factor += len(node.catches) + 1
    return branch_factor

# 计算节点覆盖率
def getNodeCoverage(codeTxt):
    # 解析Java源代码
    tree = javalang.parse.parse(codeTxt)
    # 统计代码节点总数和被执行的代码节点数
    total_nodes = 0
    executed_nodes = 0
    for path, node in tree:
        total_nodes += 1
        if isinstance(node, javalang.tree.Statement) and node.position is not None:
            executed_nodes += 1
    coverage = executed_nodes / total_nodes
    return round(coverage, 4)

# 计算节点重复度
def getNodeDuplicate(codeTxt):
    tree = javalang.parse.parse(codeTxt)
    nodes = set()
    total_nodes = 0
    for path, node in tree:
        if isinstance(node, javalang.tree.Node):
            total_nodes += 1
            nodes.add(str(node))
    duplicate_nodes = total_nodes - len(nodes)
    duplicate_ratio = duplicate_nodes / total_nodes
    return round(duplicate_ratio,4)


def getStructurealMetrics(codePath):
    java_file_path = codePath
    flag = ''
    if java_file_path.split('/')[-1].split('__')[0] == 'function':
        flag = 'function'
        with open(java_file_path, 'r', encoding='utf-8') as file:
            codeTxt = "class FakeClass{\n" + file.read() + "}"
    else:
        flag = 'class'
        with open(java_file_path, 'r', encoding='utf-8') as file:
            codeTxt = file.read()

    #print('codeTxt', codeTxt)

    avg_num, max_num = getParams_avg_max(codeTxt)
    #print('avg_num, max_num', avg_num, max_num)

    depth, width, node_num, leaf_num = get_depth_width_of_ast(codeTxt)
    #print('depth, width, node_num, leaf_num', depth, width, node_num, leaf_num)

    #duplicatesStms = getDuplicatesStm(codeTxt)
    #print('duplicatesStms',duplicatesStms)
    stmBlocks = get_code_stm_blocks(codeTxt)
    
    #duplicatesBlocks = getDuplicatesBlok(codeTxt)
    #print('duplicatesBlocks',duplicatesBlocks)
    codeBlocks = count_code_blocks(codeTxt)

    branch_factor = calculate_branch_factor(codeTxt)
    #print("Branch factor:", branch_factor)

    coverage = getNodeCoverage(codeTxt)
    #print('节点覆盖率：', coverage)

    duplicate_ratio = getNodeDuplicate(codeTxt)
    #print("Duplicate ratio", duplicate_ratio)

    if flag == 'function':
        depth-=1

    return [depth, width, node_num, leaf_num, max_num, stmBlocks, codeBlocks, branch_factor, coverage, duplicate_ratio]

if __name__ == '__main__':
    datasetPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/"
    sourceCodePath = datasetPath + 'sourceCode/'
    allStructurealMetricsPath = datasetPath + 'allStructuralMetrics.json'


    leaf_num = 0
    exampleCode = sourceCodePath + "function__feature_envy__major__1__9274__8068c8775ad067d75828e6360e7e0994348da9b9__DoubleValueArray__179__186.java"
    print(getStructurealMetrics(exampleCode))


    #exit()
    exceptFile = []
    allStructurealMetricsDict = {}
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
                    leaf_num = 0
                    allStructurealMetricsDict[fileName] = getStructurealMetrics(codaPath)
                    #print(getStructurealMetrics(codaPath))
                except:
                    except_num += 1
                    exceptFile.append(fileName+'\n')
                    #exit()
            #exit()
    print('except_num', except_num)
    print('allStructurealMetricsDict', len(allStructurealMetricsDict))
    # 将其保存到本地
    allStructurealMetricsFile = open(allStructurealMetricsPath, "w")
    json.dump(allStructurealMetricsDict,allStructurealMetricsFile)
    allStructurealMetricsFile.close()

    exceptItemPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/exceptItem.txt"
    with open(exceptItemPath, 'w') as f:
        f.writelines(exceptFile)