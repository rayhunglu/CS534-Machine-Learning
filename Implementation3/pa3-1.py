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

def splitDataSet(datas,feature,threshold):
    # rightnode=Node()
    # leftnode=Node()
    # data=datas[datas[:, feature].argsort()]
    # leftdatas,rightdatas=np.split(datas,[sample],axis=0)  #split <T,>=T
    # left=leftdatas[leftdatas[:,0].argsort()]      #split +-1
    # right=rightdatas[rightdatas[:,0].argsort()]   #split +-1
    # llnum=np.argmax(left.T[0])          #nums of left of left node
    # rlnum=np.argmax(right.T[0])
    data=datas[datas[:,feature].argsort()]    #+-1
    for i in range(0,len(data)):
        if data[i][feature]>=threshold:
            break
    leftdatas,rightdatas=np.split(data,[i],axis=0)

    return leftdatas,rightdatas

def gini_benefit(left,right,feature,value):
    # if right[len(right)-1][feature]<value:
    #     lr=right
    #     rr=[]
    # else:
    #     for i in range(0,len(right)):
    #         if right[i][feature]>=value:
    #             break
    #     lr,rr=np.split(right,[i],axis=0)

    # if left[len(left)-1][feature]<value:
    #     ll=left
    #     lr=[]
    # else:
    #     for i in range(0,len(left)):
    #         if left[i][feature]>=value:
    #             break
    #     ll,rl=np.split(left,[i],axis=0)
    # llnum=len(ll)          #nums of left of left node
    # rlnum=len(rl)         #nums of left of right node
    # c=len(left)+len(right)
    # l=len(ll)+len(lr)
    # r=len(rl)+len(rr)
    llnum=1
    rlnum=1
    c=1
    l=1
    r=1
    if l==0 or r==0:
        return 0,llnum,rlnum
    vl=(2*llnum*(l-llnum)/(c*l))
    vr=(2*rlnum*(r-rlnum)/(c*r))
    return 1-vl-vr,llnum,rlnum

def chooseBestFeatureToSplit(datas,root):
    gini=0
    threshold=0
    feature=0
    # y=datas.T[0]
    for i in range(1,len(datas[0])):        #feature 1~101 100s
        print(i)
        value=datas.T[i]                    #the data of the test feature
        data=datas[datas[:,i].argsort()]    #+-1
        rllnum=np.argmin(data.T[0])
        right,left=np.split(data,[rllnum],axis=0)  #split -1 part, +1 part
        left=left[left[:,i].argsort()]      #sort feature in +1 part
        right=right[right[:,i].argsort()]   #sort feature in -1 part
        # print(value)
        for j in range(0,len(value)):#sample#
            b,llnum,rlnum=gini_benefit(left,right,i,value[j])
            if b>gini:
                gini=b
                ll=llnum
                rl=rlnum
                feature=i
                threshold=value[j]
                # sample=j
                # if j==len(left):
                #     threshold=left[j-1][1]
                # else:
                #     threshold=left[j][i]

    print('threshold',threshold,feature)
    return feature,threshold,ll,rl

def finddecisiontree(datas,root,level,maxdepth): #x=[[],[]]
    feature,threshold,llnum,rlnum=chooseBestFeatureToSplit(datas,root)
    root.splitfeature=feature
    root.splitthreshold=threshold
    rightnode=Node()
    leftnode=Node()
    root.left=leftnode
    root.right=rightnode
    leftdatas,rightdatas=splitDataSet(datas,feature,threshold)
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

# def compute_acc(root,acc,level,maxdepth):
#     acc[level]=acc.setdefault(level,0)+max(len(root.splitdataset[0]),len(root.splitdataset[1]))
#     if level<(maxdepth-1):
#         compute_acc(root.left,acc,level,maxdepth)
#         compute_acc(root.left,acc,level,maxdepth)
####################main#####################
maxdepth=20
df=pd.read_csv("pa3_train_reduced.csv",header=None)
xt=df.iloc[:,:].values
# yt=df.iloc[:, :1].values
df=pd.read_csv("pa3_valid_reduced.csv",header=None)
xv=df.iloc[:,:].values
# yv=df.iloc[:, :1].values
datas=xt
# result=(yt-4)*(-1)
acc={}
# print(len(datas))
print (time.asctime( time.localtime(time.time()) ))

root=decisiontree(datas,maxdepth)
print (time.asctime( time.localtime(time.time()) ))

# compute_acc(root,acc,0,maxdepth)
# for key, values in acc:
#     print(key,' acc: ', values/4888)
# print(root.splitfeature,root.splitsample)





