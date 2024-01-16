import json

PPmetricsDictPath = '/home/yqx/Documents/myMLCQdataset/myMLCQdataset/allCodeMetrics.json'
allStructurealMetricsPath = '/home/yqx/Documents/myMLCQdataset/myMLCQdataset/allStructurealMetrics.json'

allCommitMetricsPath = '/home/yqx/Documents/myMLCQdataset/myMLCQdataset/allCommitMetrics.json'
allCommenMetricsPath = '/home/yqx/Documents/myMLCQdataset/myMLCQdataset/allCommenMetrics.json'

with open(PPmetricsDictPath, 'r') as f:
    jsonData = f.read()
    PPmetricsDict = json.loads(jsonData)

allCommitMetricsDict = {}
allCommenMetricsDict = {}
with open(allStructurealMetricsPath, 'r') as f:
    jsonData = f.read()
    allStructurealMetricsDict = json.loads(jsonData)
    for key in allStructurealMetricsDict.keys():
        #print(key) 
        #function__feature_envy__none__1__3019__72cd0e137c4a0c3b899adfa6e19e2fd590743014__FilePart__42__45
        id = key.split('__')[4]
        #print(PPmetricsDict[id][19:])
        allCommitMetricsDict[key] = PPmetricsDict[id][:19]
        allCommenMetricsDict[key] = PPmetricsDict[id][19:]

# 分开保存
allCommitMetricsDictFile = open(allCommitMetricsPath, "w")
json.dump(allCommitMetricsDict,allCommitMetricsDictFile)
allCommitMetricsDictFile.close()

allCommenMetricsDictFile = open(allCommenMetricsPath, "w")
json.dump(allCommenMetricsDict,allCommenMetricsDictFile)
allCommenMetricsDictFile.close()