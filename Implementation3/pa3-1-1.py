"""
@author: raylu
pa2-1.py
decision tree
"""
import numpy as np
import pandas as pd
class Node:
    def __init__(self, splitdataset):
        self.left = None                              # training examples
        self.right = None
        self.splitdataset=splitdataset
        self.splitfeature = None
        self.splitsample=None
        
def splittoleftright(data,result):#left:-1,right:1
    left=[]
    right=[]
    for i in range(len(result)):
        if result[i]==-1:
            left.append(i)
        else:
            right.append(i)
    return [left,right]

def splitDataSet(x,root):
    rightnode=root
    leftnode=Node([[],[]])
    feature=root.splitfeature
    sample=root.splitsample
    value=x[sample][feature]
    leftnode.splitdataset=[[],[]]
    leftpart=rightnode.splitdataset[0]
    rightpart=rightnode.splitdataset[1]
    for i in leftpart:
        if x[i][feature]<value:
            leftnode.splitdataset[0].append(i)
            rightnode.splitdataset[0].remove(i)
    for i in rightpart:
        if x[i][feature]<value:
            leftnode.splitdataset[1].append(i)
            rightnode.splitdataset[1].remove(i)
    return leftnode,rightnode

def compute_gini(p,n):
    if p+n==0:
        return 0
    return 1-(p/(p+n))**2-(n/(p+n))**2

def gini_benefit(x,y,feature,sample,dataSet):
    left=[0,0]
    right=[0,0]
    threshold=x[sample][feature]
    for i in dataSet[0]:
        if x[i][feature]<threshold:
            left[0]+=1
        else:
            right[0]+=1
    for i in dataSet[1]:
        if x[i][feature]<threshold:
            left[1]+=1
        else:
            right[1]+=1
    ua=compute_gini(len(dataSet[0]),len(dataSet[1]))
    ual=compute_gini(left[0],left[1])
    uar=compute_gini(right[0],right[1])
    pl=(left[0]+left[1])/(len(dataSet[0])+len(dataSet[1]))
    pr=(right[0]+right[1])/(len(dataSet[0])+len(dataSet[1]))
    return 1-pl*ual-pr*uar
def choosefeature(x,dataSet):
    xx=x.T
    a=[]
    choose=[]
    ll=0
    rr=0
    for j in range(0,len(xx)):
        for i in dataSet[0]:
            ll+=xx[j][i]
        for i in dataSet[1]:
            rr+=xx[j][i]
        el=ll/len(dataSet[0])
        er=rr/len(dataSet[1])
        a.append(abs(el-er))
    b=sorted(a)
    b=b[90:100]
    for i in range(0,len(xx)):
        if a[i] in b:
            choose.append(i)
    return choose
def chooseBestFeatureToSplit(x,y,dataSet):
    # print(len(x[0]))
    features=0
    feature=[[],[]]
    choose=choosefeature(x,dataSet)
    gini=0
    for i in range(0,len(x[0])):#feature
    # for i in choose:#feature
        print(i)
        for j in range(0,len(x)):#sample#
            b=gini_benefit(x,y,i,j,dataSet)
            # print(j,i)
            if b>gini:
                gini=b
                features=x[j][i]
                feature=i
                sample=j
    return feature,sample

def finddecisiontree(x,y,root,level,maxdepth):
    feature,sample=chooseBestFeatureToSplit(x,y,root.splitdataset)
    root.splitfeature=feature
    root.splitsample=sample
    leftnode,rightnode=splitDataSet(x,root)

    if level<maxdepth-1:
        leftnode=finddecisiontree(x,y,leftnode,level+1,maxdepth)
        rightnode=finddecisiontree(x,y,rightnode,level+1,maxdepth)
    root.left=leftnode
    root.right=rightnode
    print(root.splitsample,root.splitfeature)
    print(x[root.splitsample][root.splitfeature])
    return root

def decisiontree(x,y,maxdepth=20):
    splitdata=splittoleftright(x,y)
    root=Node(splitdata)
    level=0
    b=gini_benefit(x,y,1,2,splitdata)
    print(b)
    # node=finddecisiontree(x,y,root,level,maxdepth)
    # return node
####################main#####################
maxdepth=20
df=pd.read_csv("pa3_train_reduced.csv",header=None)
xt=df.iloc[:,1:].values
yt=df.iloc[:, :1].values
# df=pd.read_csv("pa3_valid_reduced.csv",header=None)
# xv=df.iloc[:,1:].values
# yv=df.iloc[:, :1].values
datas=xt
result=(yt-4)*(-1)
root=decisiontree(datas,result,maxdepth)
# print(root.splitfeature,root.splitsample)





