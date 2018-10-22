"""
@author: raylu
pa2-2.py
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
        self.w = np.ones((len(self.x[0]), 1))      #w=n*1
        self.averw = np.ones((len(self.x[0]), 1))      #w=n*1
        s=0.0
        c=0.0                             
        for i in range(maxIter):
            tt=0
            vt=0
            for j in range(self.dataNum):                 
                u=np.sign(np.dot(self.w.T,self.x[j].T))          #(1*n) dot (n*1)
                yj=self.y[j]*u[0]
                if(yj<=0):
                    if((c+s)>0):
                        a=s/(c+s)
                        b=c/(c+s)
                        self.averw=a*self.averw+c*self.w
                    s=s+c
                    tem=np.asmatrix(self.x[j])
                    self.w=self.w+self.y[j][0]*tem.T   
                    c=0
                else:
                    c=c+1
            for j in range(self.dataNum):                 
                u=np.dot(self.averw.T,self.x[j].T)           #(1*n) dot (n*1)
                yj=self.y[j]*u[0]
                if(yj<=0):
                    tt+=1
            for j in range(self.vdataNum):                 
                u=np.dot(self.averw.T,self.vx[j].T)            #(1*n) dot (n*1)
                vyj=float(self.vy[j][0]*u[0])
                if(vyj<=0):
                    vt+=1
            #print(1-(tt/self.dataNum),1-(vt/self.vdataNum))
            #print(1-(tt/self.dataNum))

            print(1-(vt/self.vdataNum))

        if(c>0):
            self.averw=((s*self.averw)+(c*self.averw))/(s+c)
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
lg=Perceptron(xt,(yt-4)*(-1),xv,(yv-4)*(-1))
lg.OnlinePerceptron(maxIter)
# print(w)
