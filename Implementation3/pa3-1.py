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

def gini_benefit(cl,cr,ll,lr,rl,rr):
    if ll+lr==0 or rl+rr==0:
        return 0
    vl=(2*ll*lr/((cl+cr)*(ll+lr)))
    vr=(2*rl*rr/((cl+cr)*(rl+rr)))
    return (1-vl-vr)
def computechange(y,ll,lr,rl,rr):
    if y==3:
        rl-=1
        ll+=1
    else:
        rr-=1
        lr+=1
    return ll,lr,rl,rr
def chooseBestFeatureToSplit(datas):
    # print('datas',datas)
    result=datas.T[0]
    cr=(sum(result[:])-(3*len(result)))/2 #nums of 5
    cl=len(result)-cr
    # print(cl,cr)
    gini=0
    threshold=0
    feature=0
    fll,flr,frl,frr=0,0,0,0
    # final_fea=1
    # final_threshold=0
    # print(len(datas))
    for i in range(1,len(datas[0])):#feature
        # print(i)
        ll,lr,rl,rr=0,0,cl,cr
        data=datas[datas[:,i].argsort()]
        y_value=data[0][0]
        # print(y_value)
        ll,lr,rl,rr=computechange(y_value,ll,lr,rl,rr)
        # feature=i
        # fll,flr,frl,frr=ll,lr,rl,rr
        # threshold=data[0][i]
        # gini=0        
        # if len(datas)==2:
        #     feature=i
        #     fll,flr,frl,frr=ll,lr,rl,rr
        # threshold=data[j][i]
        for j in range(1,len(data)):#  sample index
            y_value=data[j][0]
            # print(ll,lr,rl,rr)
            ll,lr,rl,rr=computechange(y_value,ll,lr,rl,rr)
            # print(data[j][0])
            if data[j][0]!=data[j-1][0]:
                b=gini_benefit(cl,cr,ll,lr,rl,rr)
                # print(b)
                if b>gini:
                    gini=b
                    feature=i
                    # final_fea=feature
                    # final_threshold=threshold
                    fll,flr,frl,frr=ll,lr,rl,rr
                    threshold=data[j][i]
        # print(gini)
    # data=datas[datas[:,feature].argsort()]
    # leftdatas,rightdatas=np.split(data,[fll+flr],axis=0)
    # print('leftdatas',leftdatas,'rightdatas',rightdatas)
    return feature,threshold,fll,flr,frl,frr

def finddecisiontree(datas,root,level,maxdepth,acc): #x=[[],[]]
    feature,threshold,fll,flr,frl,frr,leftdatas,rightdatas=0,0,0,0,0,0,None,None
    feature,threshold,fll,flr,frl,frr=chooseBestFeatureToSplit(datas)
    root.splitfeature=feature
    root.splitthreshold=threshold
    rightnode=Node()
    leftnode=Node()
    root.left=leftnode
    root.right=rightnode
    # print(type(level))
    acc[level+1]=acc.setdefault(level+1,0)+min(fll,flr)+min(frl,frr)
    data=datas[datas[:,feature].argsort()]
    leftdatas,rightdatas=np.split(data,[fll+flr],axis=0)
    # print(level,':threshold',threshold,feature,'left',fll,flr,'right',frl,frr)
    print('level',level,':threshold',threshold,'feature',feature)
    if level<maxdepth-1:
        if fll!=0 and flr!=0:
            leftnode,acc=finddecisiontree(leftdatas,leftnode,level+1,maxdepth,acc)
        if frl!=0 and frr!=0:
            rightnode,acc=finddecisiontree(rightdatas,rightnode,level+1,maxdepth,acc)
    return root,acc

def decisiontree(datas,maxdepth=20):
    root=Node()
    level=0
    acc={}
    node,acc=finddecisiontree(datas,root,level,maxdepth,acc)
    for i in range(1,maxdepth+1):
        print('level ',i,'acc: ',1-(acc[i]/len(datas)))    
    return node
def testvalid(datas,root,level,vacc,maxdepth=20):
    result=datas.T[0]
    cr=(sum(result[:])-(3*len(result)))/2 #nums of 5
    cl=len(result)-cr
    # print('1',type(level))
    vacc[level]=vacc.setdefault(level,0)+min(cl,cr)
    
    if level<maxdepth:
        if cl!=0 and cr!=0:
            print(root.splitfeature)
            data=datas[datas[:,root.splitfeature].argsort()]
            for i in range(0,len(data)):
                if data[i][root.splitfeature]>=root.splitthreshold:
                    break
            leftdatas,rightdatas=np.split(data,[i],axis=0)
            vacc=testvalid(leftdatas,root.left,level+1,vacc,maxdepth)
            vacc=testvalid(rightdatas,root.right,level+1,vacc,maxdepth)
    return vacc
#####################main#####################
maxdepth=6
df=pd.read_csv("pa3_train_reduced.csv",header=None)
xt=df.iloc[:,:].values
# df=pd.read_csv("pa3_valid_reduced.csv",header=None)
# xv=df.iloc[:,:].values
datas=xt
# valid_datas=xv
# result=(yt-4)*(-1)
# wArr = np.ones(df.shape[0])
print(df[0].max())
# print (time.asctime( time.localtime(time.time()) ))
# print ('training')
# root=decisiontree(datas,maxdepth)
# print ('validation')
# vacc={}
# level=0
# vacc=testvalid(valid_datas,root,level,vacc,maxdepth)
# for i in range(1,maxdepth+1):
#     print('level ',i,'vacc: ',1-(vacc[i]/len(valid_datas)))
# print (time.asctime( time.localtime(time.time()) ))

