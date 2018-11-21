"""
@author: raylu
pa2-1.py
decision tree
"""
import numpy as np
import pandas as pd

class Node:
    def __init__(self):
        self.left = None                              # training examples
        self.right = None
        # self.splitdataset=splitdataset
        self.splitfeature = None
        self.splitsample=None

def splittoleftright(data,result):#left:1,right:-1
    left=[]
    right=[]
    for i in range(len(result)):
        if result[i]==1:
            left.append(i)
        else:
            right.append(i)
    return [left,right]

def splitDataSet(x,root):
    rightnode=Node([[],[]])
    leftnode=Node([[],[]])
    # rightnode.splitdataset=root.splitdataset
    threshold=x[root.splitsample][root.splitfeature]
    # leftnode.splitdataset=[[],[]]
    leftpart=root.splitdataset[0]
    rightpart=root.splitdataset[1]
    for i in leftpart:
        if x[i][root.splitfeature]<threshold:
            leftnode.splitdataset[0].append(i)
        else:
            rightnode.splitdataset[0].append(i)
    for i in rightpart:
        if x[i][root.splitfeature]<threshold:
            leftnode.splitdataset[1].append(i)
        else:
            rightnode.splitdataset[1].append(i)
    return leftnode,rightnode

def compute_gini(p,n):
    if p+n==0:
        return 0
    return 1-(p/(p+n))**2-(n/(p+n))**2

def gini_benefit(x,y,feature,sample,dataSet):
    left=[0,0]
    right=[0,0]
    threshold=x[sample][feature]
    list1 = x.T[feature]
    split = list1.index(threshold)
    iter1, iter2 = list1[:split], list1[split:]
    # np.split(A,[2],axis=0)
    # for i in dataSet[0]:
    #     if x[i][feature]<threshold:
    #         left[0]+=1
    #     else:
    #         right[0]+=1
    # for i in dataSet[1]:
    #     if x[i][feature]<threshold:
    #         left[1]+=1
    #     else:
    #         right[1]+=1
    # print(len(dataSet[0]),len(dataSet[1]))
    # print(left[0],left[1])
    # print(right[0],right[1])

    ua=compute_gini(len(dataSet[0]),len(dataSet[1]))
    ual=compute_gini(left[0],left[1])
    uar=compute_gini(right[0],right[1])
    pl=(left[0]+left[1])/(len(dataSet[0])+len(dataSet[1]))
    pr=(right[0]+right[1])/(len(dataSet[0])+len(dataSet[1]))
    return ua-pl*ual-pr*uar

# def choosefeature(x,dataSet):
#     xx=x.T
#     a=[]
#     choose=[]
#     ll=0
#     rr=0
#     for j in range(0,len(xx)):
#         for i in dataSet[0]:
#             ll+=xx[j][i]
#         for i in dataSet[1]:
#             rr+=xx[j][i]
#         el=ll/len(dataSet[0])
#         er=rr/len(dataSet[1])
#         a.append(abs(el-er))
#     b=sorted(a)
#     b=b[80:100]
#     for i in range(0,len(xx)):
#         if a[i] in b:
#             choose.append(i)
#     return choose

def chooseBestFeatureToSplit(datas,dataSet):
    features=0
    feature=[[],[]]
    # choose=choosefeature(x,dataSet)
    gini=0
    samples=dataSet[0]+dataSet[1]
    for i in range(1,len(x[0])):#feature
    # for i in choose:#feature
        print(i)
        for j in samples:#sample#
            b=gini_benefit(x,y,i,j,dataSet)
            if b>gini:
                gini=b
                features=x[j][i]
                feature=i
                sample=j
    print('threshold',sample,feature)
    return feature,sample

def finddecisiontree(datas,root,level,maxdepth): #x=[[],[]]
    feature,sample=chooseBestFeatureToSplit(datas,root)
    root.splitfeature=feature
    root.splitsample=sample
    leftnode,rightnode=splitDataSet(datas,root)
    print(len(leftnode.splitdataset[0]),len(leftnode.splitdataset[1]))
    print(len(rightnode.splitdataset[0]),len(rightnode.splitdataset[1]))
    if level<maxdepth-1:
        if leftnode.splitdataset[0]!=0 and leftnode.splitdataset[1]!=0:
            leftnode=finddecisiontree(x,y,leftnode,level+1,maxdepth)
        if rightnode.splitdataset[0]!=0 and rightnode.splitdataset[1]!=0:
            rightnode=finddecisiontree(x,y,rightnode,level+1,maxdepth)
    root.left=leftnode
    root.right=rightnode
    return root

def decisiontree(datas,maxdepth=20):
    # splitdata=splittoleftright(x,y)
    # root=Node(splitdata)
    root=Node()
    level=0
    node=finddecisiontree(datas,root,level,maxdepth)
    return node

# def compute_acc(root,acc,level,maxdepth):
#     acc[level]=acc.setdefault(level,0)+max(len(root.splitdataset[0]),len(root.splitdataset[1]))
#     if level<(maxdepth-1):
#         compute_acc(root.left,acc,level,maxdepth)
#         compute_acc(root.left,acc,level,maxdepth)
####################main#####################
maxdepth=1
df=pd.read_csv("pa3_train_reduced.csv",header=None)
xt=df.iloc[:,:].values
# yt=df.iloc[:, :1].values
df=pd.read_csv("pa3_valid_reduced.csv",header=None)
xv=df.iloc[:,:].values
# yv=df.iloc[:, :1].values
datas=xt
# result=(yt-4)*(-1)
acc={}
# print(datas)
root=decisiontree(datas,maxdepth)
# compute_acc(root,acc,0,maxdepth)
# for key, values in acc:
#     print(key,' acc: ', values/4888)
# print(root.splitfeature,root.splitsample)





