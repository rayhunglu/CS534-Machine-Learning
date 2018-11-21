"""
@author: raylu
pa2-1.py
"""

import numpy as np
import pandas as pd

class Perceptron:

    # Intial global values
    def __init__(self, datas, result,vd,vr):
        self.x = datas                              # training examples
        self.y = result
        self.vx=vd
        self.vy=vr                             
        self.dataNum = len(datas)
        self.vdataNum = len(vd)                   # number of data
    def OnlinePerceptron(self, maxIter=15):
        # Initail point of weights values
        self.w = np.zeros((len(self.x[0]), 1))      #w=n*1
        for i in range(maxIter):
            ttt=0
            tt=0
            vt=0
            for j in range(self.dataNum):                 
                u=np.dot(self.w.T,self.x[j].T)           #(1*n) dot (n*1)
                yj=float(self.y[j][0]*u[0])
                if(yj<=0):
                    tem=np.asmatrix(self.x[j])
                    self.w=self.w+self.y[j][0]*tem.T
                    ttt+=1
            # for j in range(self.dataNum):                 
            #     u=np.dot(self.w.T,self.x[j].T)           #(1*n) dot (n*1)
            #     yj=float(self.y[j][0]*u[0])
            #     if(yj<=0):
            #         tt+=1
            for j in range(self.vdataNum):                 
                u=np.dot(self.w.T,self.vx[j].T)            #(1*n) dot (n*1)
                vyj=float(self.vy[j][0]*u[0])
                if(vyj<=0):
                    vt+=1
            print(1-(ttt/self.dataNum),1-(vt/self.vdataNum))
            #print(1-(tt/self.dataNum))
            #print(1-(vt/self.vdataNum))
#======
#main
#======
maxIter=15
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
lg=Perceptron(xxt.T,(yt-4)*(-1),xxv.T,(yv-4)*(-1))
lg.OnlinePerceptron(maxIter)
