# -*- coding: utf-8 -*-

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import random
from copy import deepcopy

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
        
    def Iteration(self,xmin,xkmin,rk):
        xkminv=np.repeat([xkmin],P,axis=0)
        self.vk=self.omega*self.vk + rk*self.wp*(xmin - self.xk) + rk*self.wg*(xkminv.T - self.xk)
        self.xk=self.xk+self.vk
        
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
            self.Iteration(xmin,xkmin,r[k])
            for i in range(P):
                if self.J(self.xk[:,i])<self.J(xmin[:,i]):
                    xmin[:,i]=self.xk[:,i]
            valJ=self.calc_valJ(xmin)
            indice=np.argmin(valJ)
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

#///////////////     III.FONCTIONS TEST      \\\\\\\\\\\\\\\

#   fonction Rastrigin

def fct_Rastrigin(x):
    A = 10
    f = A*D
    
    for i in range(D):
        f+=x[i]**2 - A*np.cos(2*pi*x[i])
    return f

def fct_booth(x,y):
    return (x+2*y-7)**2 +(2*x+y-5)**2

#################################################################

pi = ma.pi

#main
K=1000
D=3
P=320
x0=np.random.rand(D,P)*10-5
v0=x0*0
PSO=PSO(0.01,np.ones(D)*0,x0,v0,0.8,0.1,0.1,K,fct_Arnold,fct_Rastrigin)
val2=PSO.PSO_calc()
val2
print("x : ",val2)
print("Valeur fnct : ",fct_Rastrigin(val2))
