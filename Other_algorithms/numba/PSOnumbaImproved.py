import numpy as np
from numba import jit
import random as rd
import time

def main():

    

    #We declare the initial paramters
    K=30# The max number of iterations
    D=20 # The dimension of the problem
    P=3000 # The number of particles

        #We declare the initial conditions
    x0=np.random.rand(D,P)*10-5 # The initial positions
    v0=x0*0.0

    J=fct_Rastrigin # The function to be minimized
    fonction=fct_Arnold # The pseudo-random number generator

    epsilon=0.000001 # The relative error
    xoptimal=np.ones(D)*0
    omega = 0.8 # The weight of the speed
    wp = 0.9 # The weight of the personal best
    wg = 0.9 # The weight of the global best

    (result, newxmin) = PSO(epsilon, xoptimal, x0, v0, omega, wp, wg, fonction, K, D, P, J)

    for i in range(1500):

        oldresult=result
        oldnewxmin=newxmin

        (result, newxmin) = PSO(epsilon, xoptimal, x0, v0, omega, wp, wg, fonction, K, D, P, J)
        if(oldresult<result):
            x0=oldnewxmin
        else:
            x0=newxmin-np.random.rand(D,P)*(-2/(np.exp((result-oldresult+1)/2)+2.1))*0.5

        print(result)




def PSO(epsilon, xoptimal, xk, vk, omega, wp, wg, fonction, K, D, P, J):
    """
    The function PSO takes as input the parameters epsilon, xoptimal, xk, vk, omega, wp, wg, fonction,
    K, D, P, J and returns the values xkmin, k, valk, J(xkmin)
    
    :param epsilon: the precision we want to reach
    :param xoptimal: the optimal solution
    :param xk: the current position of the particles
    :param vk: velocity
    :param omega: inertia weight
    :param wp: the weight of the particle
    :param wg: the global weight
    :param fonction: the function to be minimized
    :param K: number of iterations
    :param D: dimension of the problem
    :param P: number of particles
    :param J: the function to minimize
    :return: the best solution found, the number of iterations, the error and the value of the function
    at the best solution.
    """
    # We create the matrix with the lowest values of xk
    xmin = xk.copy()
    valJ = calc_valJ(xmin, J)
    indice = np.argmin(valJ)
    xkmin = xmin[:,indice]
    valk = ERX(xkmin, xoptimal)
    r=fonction(K)
    k=0

    while valk > epsilon and k < K:
        xkminv=np.repeat([xkmin],P,axis=0)
        (xk, vk, valJ, indice, xkmin, valk, k) = inloop(epsilon, xoptimal, xk, vk, omega, wp, wg, fonction, K, D, P, J, xkmin, valk, k, r, xkminv, xmin)
    return J(xkmin), xmin

@jit(nopython=True, parallel=True)
def inloop(epsilon, xoptimal, xk, vk, omega, wp, wg, fonction, K, D, P, J, xkmin, valk, k, r, xkminv, xmin):
    xk, vk = iterations(xmin, xkminv, xkmin, r[k], xk, vk, P, omega, wp, wg)
    for i in range(P):
        if J(xk[:,i]) < J(xmin[:,i]):
            xmin[:,i] = xk[:,i]
    valJ = calc_valJ(xmin, J)
    indice = np.argmin(valJ)
    xkmin = xmin[:,indice]
    valk = ERX(xkmin, xoptimal)
    k+=1
    return xk, vk, valJ, indice, xkmin, valk, k


@jit(nopython=True, parallel=True)
def iterations(xmin, xkminv, xkmin, rk, xk, vk, P, omega, wp, wg):
    
    vk*=omega
    vk+=rk*wp*(xmin-xk)
    vk+=rk*wg*(xkminv.T-xk)
    xk=xk+vk

    return xk, vk

@jit(nopython=True, parallel=True)
def ERX(xkmin,xopt):
    """
    > The function ERX calculates the relative error between the optimal solution and the solution
    obtained by the algorithm
    
    :param xkmin: the value of x at the minimum
    :param xopt: the optimal solution
    :return: The relative error between the optimal solution and the solution found by the algorithm.
    """
    if not(np.linalg.norm(xopt)<0.0001) : 
        valk=(np.linalg.norm(xkmin - xopt))/np.linalg.norm(xopt)
    else :
        valk=np.linalg.norm(xkmin)
    return valk

@jit(nopython=True)
def calc_valJ(xmin, J):
    """
    It takes a matrix of values for x and a function J, and returns a vector of values for J(x)
    
    :param xmin: the minimum of each points
    :param J: the function to be minimized
    :return: the value of the function J at the point xmin.
    """
    transpose=xmin.T
    valJ=np.zeros(transpose.shape[0])
    #valJ=np.array(list(map(lambda x: J(x),transpose)))
    for i in range(len(valJ)):
        valJ[i]=J(transpose[i])
    return valJ

@jit(nopython=True)
def fct_Rastrigin(x):
    A = 10

    # Get the dimension of the problem
    D = len(x)

    f = A*D
    
    for i in range(D):
        f+=x[i]**2 - A*np.cos(2*np.pi*x[i])
    return f

@jit(nopython=True)
def fct_Arnold(K):
    """
    It takes the previous two values of the sequence and adds them together, then takes the modulus of 1
    to generatare a pseudo-random number between 0 and 1. (chaos map)
    :return: the array Xac.
    """
    Xac = np.zeros(K)
    Yac = np.zeros(K)
    a=0.1
    Xac[0]=0.5
    Yac[0]=0.5
    for k in range(1,K):
        Xac[k]=(Xac[k-1] + Yac[k-1])%1
        Yac[k]=(Xac[k-1] + a*Yac[k-1])%1
    return Xac


if __name__=='__main__':

    main()
