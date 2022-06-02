# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:07:12 2022

@author: laure_wkw4h
"""
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from scipy.stats import qmc

class PSO :
    def __init__(self,epsilon,xopt,xk,vk,omega,wp,wg,K,fonction,J):
        self.epsilon=epsilon
        self.xopt=xopt
        self.xk=xk
        self.vk=vk
        self.omega=omega
        self.wp=wp
        self.wg=wg
        self.K=K
        self.fonction=fonction
        self.J=J
        self.compteur=0
     
    def Iteration(self,xmin,xkmin,rk,k):
        xkminv=np.repeat([xkmin],P,axis=0)
        for i in range(P):
            if k<0.2*K or (self.compteur>=15 and k<0.7*K):
                r=3
                if self.compteur==15 and i==1:
                    print("BOOST")
            else:
                r=2.5
            if random.random() < 0.8/np.exp(np.sqrt(k)):
                self.omega = 1.1
            self.vk[:,i]=self.omega*self.vk[:,i]+rk*self.wp*(xmin[:,i] - self.xk[:,i]) + rk*self.wg*(xkminv.T[:,i] - self.xk[:,i])
            if k<0.85*K:                
                self.xk[:,i]=xkmin + (xkmin-self.xk[:,i])*((-1)**(random.randint(0, 10)))*random.random()*r + self.vk[:,i]
            else:
                self.xk[:,i]=self.xk[:,i] + self.vk[:,i]
            if np.max(self.xk[:,i])>=5.12 or np.min(self.xk[:,i])<=-5.12:
                for j in range(D):
                    if abs(self.xk[j,i])>=5.12:
                        self.xk[j,i]=((self.xk[j,i]-np.min(self.xk[:,i]))/(np.max(self.xk[:,i])-np.min(self.xk[:,i])))*10.22-5.11
            self.omega=0.8
            
    def ERX(self,xkmin):
        if not(np.linalg.norm(self.xopt)<0.0001) : 
            valk=(np.linalg.norm(xkmin - self.xopt))/np.linalg.norm(self.xopt)
        else :
            valk=np.linalg.norm(xkmin)
        return valk
    
    def calc_valJ(self,xmin):
        transpose=xmin.T
        valJ=np.array(list(map(lambda x: self.J(x),transpose)))
        return valJ
    
    def PSO_calc(self):
        xmin=deepcopy(self.xk)
        valJ=self.calc_valJ(xmin)
        indice=np.argmin(valJ)
        xkmin=xmin[:,indice]
        valk=self.ERX(xkmin)
        r=self.fonction()
        k=0
        while valk>=self.epsilon and k<self.K:
            self.Iteration(xmin,xkmin,r[k],k)
            for i in range(P):
                if self.J(self.xk[:,i])<self.J(xmin[:,i]):
                    xmin[:,i]=self.xk[:,i]
            valJ=self.calc_valJ(xmin)
            indice=np.argmin(valJ)
            potentielgb=self.J(xmin[:,indice])
            actuelgb=self.J(xkmin)
            if k%50==0:
                print(k)
            if potentielgb < actuelgb:
                print("k : ",k)
                print("valeur J(xgb) : ",self.J(xmin[:,indice]))
            if (potentielgb < actuelgb) and (actuelgb-potentielgb>0.001):
                if(self.compteur>=15):
                    print("DEBOOST")
                self.compteur=0
            else:
                self.compteur+=1
            xkmin=xmin[:,indice]
            valk=self.ERX(xkmin)
            k=k+1
        return xkmin

#//////////////     I.CHAOS MAPS     \\\\\\\\\\\\\\\

#   4.Arnold's Cat
def fct_Arnold():
    Xac = np.zeros(K)
    Yac = np.zeros(K)
    a=0.1
    Xac[0]=0.5
    Yac[0]=0.5
    for k in range(1,K):
        Xac[k]=(Xac[k-1] + Yac[k-1])%1
        Yac[k]=(Xac[k-1] + a*Yac[k-1])%1
    return Xac

#///////////////    II.FONCTIONS UTILES     \\\\\\\\\\\\\\\
   
#   fonction rk aléatoire entre 0 et 1

def rk_random(k):
    X = np.zeros(k)
    rk=X
    for i in range(k):
        rk[i] = random.random()
    return rk

def rk_norm():
    rk = np.random.randn(10)
    minr = np.min(rk)
    maxr = np.max(rk)
    for i in range(10):
        rk[i]= (rk[i]-minr)/(maxr-minr)
    return rk[0]

def rk_exp():
    l=1
    rt = np.random.exponential(l,10)
    minr = np.min(rt)
    maxr = np.max(rt)
    for i in range(10):
        rt[i]= (rt[i]-minr)/(maxr-minr)
    return rt[0]
#///////////////     III.FONCTIONS TEST      \\\\\\\\\\\\\\\

#   fonction Rastrigin

def fct_Rastrigin(x):
    A = 10
    f = A*D
    
    for i in range(D):
        f+=x[i]**2 - A*np.cos(2*pi*x[i])
    return f

def fct_booth(x):
    return (x[0]+2*x[1]-7)**2 +(2*x[0]+x[1]-5)**2

#################################################################

pi = ma.pi

#main
K=1500
D=10
P=2000

sampler = qmc.LatinHypercube(d=D)
sample = sampler.random(n=P)
x0 = (sample*10.24-5.12).T
v0=x0*0

print("debut PSO")
PSO1=PSO(0.000001,np.ones(D)*0,x0,v0,0.8,0.25,1,K,fct_Arnold,fct_Rastrigin)
val1=PSO1.PSO_calc()
val1
print("x : ",val1)
print("Valeur fonction : ",fct_Rastrigin(val1))

#Dim<=5 : 1e-11 (100 particules)
#Dim 10 : 1.9e-10 (2000 particules)
#Dim 20 : 1.9e-10 (2000 particules)

K=1500
D=2
P=100
x0=np.random.rand(D,P)*20-10
v0=x0*0
PSO2=PSO(0.000001,np.array([1,3]),x0,v0,0.8,1,1,K,fct_Arnold,fct_booth)
val2=PSO2.PSO_calc()
val2
print("x : ",val2)
print("Valeur fonction : ",fct_booth(val2))
