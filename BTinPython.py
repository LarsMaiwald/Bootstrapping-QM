#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import itertools


# In[4]:


#g = 1.
#m = 1./2
#w = 2.


# In[5]:

#gives all expectation values of powers of x up to tmax (only for the specific potential m w / 2 x^2 + g x^4)
def recursion(xQuad, E, tmax, m, w, g):
    expX = np.zeros(tmax)
    expX[0] = 1
    expX[2] = xQuad
    expX[4] = (( (4 - 3) * E* expX[4-4]  )/( (4-1)* g )) - (( (4-2) * w**2 * m * expX[4-2])/( 2 * g * (4-1) )) + (( (4-3)*(4-4)*(4-5) * expX[4-6] )/( 8 * (4-1)* m * g ))
    for t in range(4,tmax):
        if ((t & 1) == 0):
            expX[t] = (( (t - 3) * E* expX[t-4]  )/( (t-1)* g )) - (( (t-2) * w**2 * m * expX[t-2])/( 2 * g * (t-1) )) + (( (t-3)*(t-4)*(t-5) * expX[t-6] )/( 8 * (t-1)* m * g ))
    return expX

def recursion_doublewell(xQuad, E, tmax, m, w, g, V0=1.25):
    expX = np.zeros(tmax)
    expX[0] = 1
    expX[2] = xQuad
    expX[4] = (( (4 - 3) * E* expX[4-4]  )/( (4-1)* g )) + (( (4-2) * w**2 * m * expX[4-2])/( 2 * g * (4-1) )) + (( (4-3)*(4-4)*(4-5) * expX[4-6] )/( 8 * (4-1)* m * g )) - V0 * (4 - 3) * expX[4-4]
    for t in range(4,tmax):
        if ((t & 1) == 0):
            expX[t] = (( (t - 3) * E* expX[t-4]  )/( (t-1)* g )) + (( (t-2) * w**2 * m * expX[t-2])/( 2 * g * (t-1) )) + (( (t-3)*(t-4)*(t-5) * expX[t-6] )/( 8 * (t-1)* m * g )) - (V0 * (t - 3) * expX[t-4])/(g*(t-1))
    return expX

#checks for positive semidefiniteness
def is_pos_sdef(matrix, tol=1e-8):
    eigVal = np.linalg.eigvalsh(matrix)
    return np.all(eigVal > -tol)
    
#calculates the constraint matrix fom given E and <x^2>  
def calc_constrM(checkTup, K, m, w, g, double_well):
    E = checkTup[0]
    xQuad = checkTup[1]
    constrM = np.zeros((K+1, K+1))
    if double_well:
      expX = recursion_doublewell(xQuad, E, int(len(constrM[0,:]) + 1 + len(constrM[0,:] + 1)), m, w, g)
    else:
      expX = recursion(xQuad, E, int(len(constrM[0,:]) + 1 + len(constrM[0,:] + 1)), m, w, g)
    for i in range(len(constrM[0,:])):
        for j in range(len(constrM[:,0])):
            if (((i+j) & 1) == 0):
                constrM[i,j] = expX[i+j] 
    return constrM


# In[18]:

#finds the K for which in a preset grid the allowed region tops touching the edge of the grid
def checkAcc(x, E):
    switch = True
    K = 2
    icount = 0
    jcount = 0
    while switch:
        for i, j in itertools.product(range(len(x)), range(len(E))):
            if (i == 0 or i == len(x)-1 or j == 0 or j == len(E)-1 or i==1 or i==len(x)-2 or i==2 or i==len(x)-3 or j == 1 or j == len(E)-2 or j == 2 or j == len(E)-3):
                if(is_pos_sdef(calc_constrM(x[i], E[j], K))):
                    break
                icount = i
                jcount = j
        if icount == len(x) -1 and jcount == len(E) - 1:
            switch = False
        K = K+1
    return K 


# In[19]:

#After specifiyng a certain desired accuracy the function finds a fitting K using checkAcc and then finds the minimum and maximum allowed values by brute force around some specified value numRes.
def BtRegion(acc, numRes, n=3, m=1):
    x = np.linspace(numRes[0]-acc*1e-1, numRes[0]+acc*1e-1, m*10**n)
    E = np.linspace(numRes[1]-acc, numRes[1]+acc, m*10**n)
    K = checkAcc(x, E)
    Eallowed = []
    xallowed = []
    for i, j in itertools.product(range(len(x)), range(len(E))):
        if (is_pos_sdef(calc_constrM(x[i], E[j], K))):
            Eallowed.append(E[j])
            xallowed.append(x[i])
    return K, [min(Eallowed), max(Eallowed)], [min(xallowed), max(xallowed)]


# In[17]:

#For a certain K the function finds the maximum accuracy that can be achieved around a certain point numRes and then finds the minimum and maximum allowed values by brute force.
def checkK(K, numRes, n=3, acc=1e-5):
    switch = True
    icount = 0
    jconut = 0
    while switch and is_pos_sdef(calc_constrM(numRes[0], numRes[1], K)):
        x = np.linspace(numRes[0]-acc*1e-1, numRes[0]+acc*1e-1, 10**n)
        E = np.linspace(numRes[1]-acc, numRes[1]+acc, 10**n)
        for i, j in itertools.product(range(len(x)), range(len(E))):
            if (i == 0 or i == len(x)-1 or j == 0 or j == len(E)-1 or i==1 or i==len(x)-2 or i==2 or i==len(x)-3 or j == 1 or j == len(E)-2 or j == 2 or j == len(E)-3):
                if(is_pos_sdef(calc_constrM(x[i], E[j], K))):
                    break
                icount = i
                jcount = j
        if icount == len(x) -1 and jcount == len(E) - 1:
            switch = False
        acc = acc * 10
    if (is_pos_sdef(calc_constrM(numRes[0], numRes[1], K))):
        return 'e', 'e'
    else:
        Eallowed = []
        xallowed = []
        for i, j in itertools.product(range(len(x)), range(len(E))):
            if (is_pos_sdef(calc_constrM(x[i], E[j], K))):
                Eallowed.append(E[j])
                xallowed.append(x[i])
        return [min(Eallowed), max(Eallowed)], [min(xallowed), max(xallowed)]



