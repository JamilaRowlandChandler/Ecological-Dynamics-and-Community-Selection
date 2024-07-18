# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:57:36 2024

@author: jamil
"""

#%%

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

########################

#%%

def dSB_dt(t,var,
           growth,k_sat,v_max):
    
    #breakpoint()
    
    S = var[:2]
    R = var[2:]
    
    dS = S * np.sum(growth * (R/(k_sat + R)),axis=1)
    
    R0_consumption =  (v_max[0,0]/(k_sat[0,0] + R[0])) * R[0] * S[0]
    
    dR1 = -R0_consumption
    dR2 = R0_consumption - np.sum(v_max[:,1] * (R[1]/(k_sat[:,1] + R[1])) * S)
    
    dS = np.array([dS[0],dS[1],dR1,dR2])
    
    return dS

#%%

growth = np.array([[0.54,0],[0.58,0.76]])
k_sat = np.array([[0.13,1],[0.13,0.03]])
v_max = np.array([[11.2,0],[8.03,19.22]])

#%%

result = solve_ivp(dSB_dt, [0, 20], [0.1,0.1,20,0],
                   args = (growth, k_sat, v_max))

plt.plot(result.t,result.y[:2,:].T)

#%%

result2 = solve_ivp(dSB_dt, [0, 50], [0.1,0,20,0],
                   args = (growth, k_sat, v_max))

plt.plot(result2.t,result2.y[0,].T)




