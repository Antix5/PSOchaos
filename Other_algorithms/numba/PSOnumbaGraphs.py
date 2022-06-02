import numpy as np
from numba import jit
import time

def main():

    #We declare the initial conditions

    J=fct_Rastrigin # The function to be minimized
    fonction=fct_Arnold # The pseudo-random number generator

    epsilon=0.01 # The relative error

    omega = 0.8 # The weight of the speed
    wp = 0.1 # The weight of the personal best
    wg = 0.1 # The weight of the global best

    #K=100, P=320
    #K=200 # The max number of iterations
    #D=10 # The dimension of the problem
    #P=320 # The number of particles
    #
    #for D in [2, 5, 10, 50,100]:

    #    xoptimal=np.ones(D)*0
    #    x0=np.random.rand(D,P)*10-5 # The initial positions
    #    v0=x0*0.0
        
    #    resD = PSO(epsilon, xoptimal, x0, v0, omega, wp, wg, fonction, K, D, P, J)
    #    print("##############################################################################################")
    #    print("##############                          Dim",D,                                 "#############")
    #    print("##############################################################################################")
    #
    #    print("yd"+str(D),"=",resD)
    #    print("xd"+str(D),"=",list(range(len(resD))))

    # Evolution for K for Rastrigin
    #P=100, D=2


    ## Evolution for K for booth
    #P=10, D=2
    
    #D=2 # The dimension of the problem
    #P=10 # The number of particles
    #resK=[]
    #xoptimal=np.ones(D)*0
    #x0=np.random.rand(D,P)*10-5
    #v0=x0*0.0

    #vals=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 30, 35, 40, 45, 50, 100, 500, 1000]

    #for K in vals:

    #    resK.append(PSO(epsilon, xoptimal, x0, v0, omega, wp, wg, fonction, K, D, P, J))


    #print("yk"+str(K),"=",resK)
    #print("xk"+str(K),"=",vals)

    ## Evolution for P for booth
    #K=30, D=2
    sumpk=np.array([0,0,0,0,0,0])

    for i in range(100):
        K=1000
        D=2
        resPk=[]
        #1 3 booth
        xoptimal=np.array([0.0,0.0])


        vals=[10,20,40,80,160,320]

        for P in vals:

            x0=np.random.rand(D,P)*10-5
            v0=x0*0.0

            resPk.append(PSO(epsilon, xoptimal, x0, v0, omega, wp, wg, fonction, K, D, P, J))

        sumpk+=np.array(resPk)

    print("ypk"+str(P),"=",sumpk/100)
    print("xpk"+str(P),"=",vals)










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
        k+=1
    return k #ERJ(xkmin, J)

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

@jit(nopython=True)
def ERJ(xkmin, J):
    return J(xkmin)

@jit(nopython=True, parallel=True)
def ERX(xkmin,xopt):
    """
    > The function ERX calculates the relative error between the optimal solution and the solution
    obtained by the algorithm
    
    :param xkmin: the value of x at the minimum
    :param xopt: the optimal solution
    :return: The relative error between the optimal solution and the solution found by the algorithm.
    """
    if np.linalg.norm(xopt)>0.0001: 
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

@jit(nopython=True, parallel=True)
def fct_booth(x):
    return (x[0]+2*x[1]-7)**2 +(2*x[0]+x[1]-5)**2

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
