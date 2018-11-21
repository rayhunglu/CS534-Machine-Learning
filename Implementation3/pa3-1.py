"""
@author: raylu
pa2-1.py
decision tree
"""
import numpy as np
import pandas as pd
import time
class Node:
    def __init__(self):
        self.left = None                              # training examples
        self.right = None
        self.splitfeature = None
        self.splitthreshold=None

# def splitDataSet(datas,feature,sample,threshold):
#     rightnode=Node()
#     leftnode=Node()
#     data=datas[datas[:, feature].argsort()]
#     leftdatas,rightdatas=np.split(datas,[sample],axis=0)  #split <T,>=T
#     left=leftdatas[leftdatas[:,0].argsort()]      #split +-1
#     right=rightdatas[rightdatas[:,0].argsort()]   #split +-1
#     llnum=np.argmax(left.T[0])          #nums of left of left node
#     rlnum=np.argmax(right.T[0])

#     return llnum,rlnum,leftdatas,rightdatas

def gini_benefit(datas,sample):
    left,right=np.split(datas,[sample],axis=0)  #split <T,>=T
    if len(left)==0:
        llnum=0
    else:
        llnum=(sum(left[:,0])-(3*len(left)))/2
    if len(right)==0:
        rlnum=0
    else:
        rlnum=(sum(right[:,0])-(3*len(right)))/2
    c=len(datas)
    l=len(left)
    r=len(right)
    # print(c)
    # print(l)
    # print(r)
    # llnum=1
    # rlnum=1
    # c=1
    # l=1
    # r=1
    if l==0 or r==0:
        return 0,left,right
    vl=(2*llnum*(l-llnum)/(c*l))
    vr=(2*rlnum*(r-rlnum)/(c*r))
    return (1-vl-vr),left,right

def chooseBestFeatureToSplit(datas,root):
    gini=0
    threshold=0
    y=datas.T[0]
    leftdatas=None
    rightdatas=None
    for i in range(1,len(datas[0])):#feature
        print(i)
        value=datas.T[i]
        data=(np.asarray([y,value])).T
        data=data[data[:,1].argsort()]
        for j in range(0,len(datas)):#  sample index
            b,left,right=gini_benefit(data,j)
            print(b)
            if b>gini:
                gini=b
                feature=i
                leftdatas=left
                rightdatas=right
                threshold=data[j][1]
    return feature,threshold,leftdatas,rightdatas,gini

def finddecisiontree(datas,root,level,maxdepth): #x=[[],[]]
    feature,threshold,leftdatas,rightdatas,gini=chooseBestFeatureToSplit(datas,root)
    root.splitfeature=feature
    root.splitthreshold=threshold
    rightnode=Node()
    leftnode=Node()
    root.left=leftnode
    root.right=rightnode
    print('threshold',threshold,feature,'benefit',gini)
    # llnum,rlnum,leftdatas,rightdatas=splitDataSet(datas,feature,sample,threshold)
    if level<maxdepth-1:
        if llnum!=0 and llnum!=len(leftdatas):
            leftnode=finddecisiontree(leftdatas,leftnode,level+1,maxdepth)
        if rlnum!=0 and rlnum!=len(rightdatas):
            rightnode=finddecisiontree(rightdatas,rightnode,level+1,maxdepth)
    return root

def decisiontree(datas,maxdepth=20):
    # splitdata=splittoleftright(x,y)
    # root=Node(splitdata)
    root=Node()
    level=0
    node=finddecisiontree(datas,root,level,maxdepth)
    return node

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
print (time.asctime( time.localtime(time.time()) ))
root=decisiontree(datas,maxdepth)
print (time.asctime( time.localtime(time.time()) ))





