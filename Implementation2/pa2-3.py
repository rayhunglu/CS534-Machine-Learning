"""
@author: raylu
pa2-3.py
"""

import numpy as np
import pandas as pd
def kernel(x1, x2):
    return np.dot(x1, x2)
class Perceptron:

    # Intial global values
    def __init__(self, datas, result,vd,vr,p):
        self.x = datas                              # training examples
        self.y = result
        self.vx=vd
        self.vy=vr
        self.dataNum = len(datas)
        self.vdataNum = len(vd)  
        self.p=p                 # number of data
    def OnlinePerceptron(self, maxIter=1):
        # Initail point of weights values
        f=len(self.x[0])
        self.w = np.zeros((len(self.x[0]), 1))      #w=n*1
        #alpha=np.zeros((self.dataNum,1))
        alpha ={}
        for ite in range(maxIter):
            tt=0
            vt=0
            ttt=0
            for i in range(self.dataNum):
                k=0
                if(len(alpha)!=0):
                    for j,val in alpha.items():
                        kp=(1+np.dot(self.x[j],self.x[i].T))**p
                        k+=kp*self.y[j][0]*val
                    yj=np.sign(self.y[i][0]*k)
                else:
                    yj=0
                if(yj<=0):
                    alpha.setdefault(i, 0.0)
                    alpha[i]+=1.0
                    ttt+=1
            # for i in range(self.dataNum):
            #     k=0
            #     if(len(alpha)!=0):
            #         for j,val in alpha.items():
            #             kp=(1+np.dot(self.x[j],self.x[i].T))**p
            #             k+=kp*self.y[j][0]*val
            #         yj=np.sign(self.y[i][0]*k)
            #     else:
            #         yj=0
            #     if(yj<=0):
            #         tt=+1
            for i in range(self.vdataNum):
                k=0
                if(len(alpha)!=0):
                    for j,val in alpha.items():
                        kp=(1+np.dot(self.x[j],self.vx[i].T))**p
                        k+=kp*self.y[j][0]*val
                    vyj=np.sign(self.vy[i][0]*k)
                else:
                    vyj=0
                if(vyj<=0):
                    vt=+1
            # for i in range(self.vdataNum):
            #     k=0
            #     for j in range(self.dataNum):
            #         kp=(1+np.dot(self.x[j],self.vx[i].T))**p
            #         k+=kp*self.y[j][0]
            #     vyj=float(self.vy[i][0]*k)
            #     if(vyj<=0):
            #         vt=+1
            print(1-(ttt/self.dataNum),1-(tt/self.dataNum),1-(vt/self.vdataNum))
#======
#main
#======
maxIter=1
p=1
df=pd.read_csv("pa2_train.csv",header=None)
xt=df.iloc[:,1:].values
yt=df.iloc[:, :1].values
biast=np.ones((1,len(xt)))
df=pd.read_csv("pa2_valid.csv",header=None)
xv=df.iloc[:,1:].values
yv=df.iloc[:, :1].values
biasv=np.ones((1,len(xv)))
xxt=np.append(xt.T,biast,axis=0)
xxv=np.append(xv.T,biasv,axis=0)
lg=Perceptron(xxt.T,(yt-4)*(-1),xxv.T,(yv-4)*(-1),p)
lg.OnlinePerceptron(maxIter)
