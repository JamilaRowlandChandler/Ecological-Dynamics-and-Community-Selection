# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:14:46 2024

@author: jamil
"""

#cd Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/scripts/

import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from math import factorial
from scipy.integrate import solve_ivp

################################

def dCRdt_1(t,var,
            growth,carrying_capacity,consumption,production,influx):
    
    S = var[0]
    R = var[1:]
    
    dS = np.sum(growth * consumption * R) * S * (1 - S/carrying_capacity)
    dR = influx + (production * S) - (consumption * R * S)
    
    dSR = np.append(dS, dR)
    
    return dSR

def dCRdt_2(t,var,
            growth,carrying_capacity,consumption,production,influx):
    
    S = var[0]
    R = var[1:]
    
    dS = np.sum(growth * consumption * R) * S * (1 - S/carrying_capacity)
    dR = influx + (production * consumption * R * S) - (consumption * R * S)
    
    dSR = np.append(dS, dR)
    
    return dSR

def dCRdt_3(t,var,
            growth,carrying_capacity,max_consumption,half_saturation,production,
            influx):
    
    #breakpoint()
    
    S = var[0]
    R = var[1:]
    
    dS = np.sum(growth * (R/(R+half_saturation))) * S
    dR = influx + \
         (production * max_consumption * (R/(R+half_saturation)) * S) - \
         (max_consumption * (R/(R+half_saturation)) * S)
    
    dSR = np.append(dS, dR)
    
    return dSR

#########################################

t_end = 100
initial_abundance = np.array([0.05,20,0])
growth = np.array([1,1])
carrying_capacity = 1
comsumption = np.array([0,1])
production = np.array([0,1])
influx = np.array([0,0])

sol1 = solve_ivp(dCRdt_1,[0,t_end],initial_abundance,
                 args=(growth,carrying_capacity,comsumption,production,influx),
                 method='RK45')

plt.plot(sol1.t,sol1.y[0,:].T)

###############

t_end = 200
initial_abundance = np.array([0.05,20,0])
growth = np.array([0.1,1])
carrying_capacity = 1
comsumption = np.array([0.01,1])
production = np.array([0,1])
influx = np.array([0,0])

sol2 = solve_ivp(dCRdt_2,[0,t_end],initial_abundance,
                 args=(growth,carrying_capacity,comsumption,production,influx),
                 method='RK45')

plt.plot(sol2.t,sol2.y[0,:].T)

#################

t_end = 500
initial_abundance = np.array([0.05,5,0])
growth = np.array([0.1,1])
carrying_capacity = 1
half_saturation = np.array([0.5,1])
max_consumption = np.array([0.01,1])
production = np.array([0,1])
influx = np.array([0.1,0])

sol3 = solve_ivp(dCRdt_3,[0,t_end],initial_abundance,
                 args=(growth,carrying_capacity,max_consumption,half_saturation,
                       production,influx),method='RK45')

plt.plot(sol3.t,sol3.y[0,:].T)

#####

t_end = 500
initial_abundance = np.array([0.05,5,0])
growth = np.array([1,0.1])
carrying_capacity = 1
half_saturation = np.array([1,0.5])
max_consumption = np.array([1,0.01])
production = np.array([0,1])
influx = np.array([0.1,0])

sol3b = solve_ivp(dCRdt_3,[0,t_end],initial_abundance,
                 args=(growth,carrying_capacity,max_consumption,half_saturation,
                       production,influx),method='RK45')

plt.plot(sol3b.t,sol3b.y[0,:].T)

#####################################################

def dCRdt_4(t,var,
            growth,max_consumption,half_saturation,production,
            influx):
    
    S = var[0:2]
    R = var[2:]
    
    dS = np.sum(growth * (R/(R+half_saturation)),axis=1) * S
    
    # waste product can only be produced from external nutrient
    R_produced = production * (R[0]/(R[0]+half_saturation)) * max_consumption[:,0][:, np.newaxis]
    R_consumed = max_consumption * (R/(R+half_saturation))
    
    production_consumption = \
        (np.matmul(R_produced.T, S.reshape(len(S),1)) - \
         np.matmul(R_consumed.T, S.reshape(len(S),1))).reshape(R.shape)
        
    dR = influx + production_consumption
    
    dSR = np.concatenate([dS, dR])
    
    return dSR

t_end = 500
initial_abundance = np.array([0.1,0.1,5,0,0])
growth = np.array([[0.01,0,1],[0.01,1,0]])
half_saturation = np.ones((2,3))
max_consumption = np.array([[0.1,0,1],[0.1,1,0]])
production = np.array([[0,1,0],[0,0,1]])
influx = np.array([0,0,0])

sol4 = solve_ivp(dCRdt_4,[0,t_end],initial_abundance,
                 args=(growth,max_consumption,half_saturation,
                       production,influx),method='RK45')

plt.plot(sol4.t,sol4.y.T)

##########################

def dCRdt_5(t,var,
            growth,max_consumption,half_saturation,production,
            influx):
     
    S = var[0:2]
    R = var[2:]
    
    dS = np.sum(growth * (R/(R+half_saturation)),axis=1) * (S - S**2)
    
    def production_rate(i):
        
        R_remove_i = np.delete(R,i)
        max_consump_remove_i = np.delete(max_consumption,i,axis=1)
        half_sat_remove_i = np.delete(half_saturation,i,axis=1)
        
        R_i_produced = production[:,i] * \
        np.sum(max_consump_remove_i * R_remove_i/(R_remove_i+half_sat_remove_i),axis=1)
         
        return R_i_produced
    
    R_produced = np.array([production_rate(i) for i in range(len(R))])
        
    R_consumed = max_consumption * (R/(R+half_saturation))
    
    production_consumption = \
        (np.matmul(R_produced, S.reshape(len(S),1)) - \
         np.matmul(R_consumed.T, S.reshape(len(S),1))).reshape(R.shape)
        
    dR = influx + production_consumption
    
    dSR = np.concatenate([dS, dR])
    
    return dSR

t_end = 500
initial_abundance = np.array([0.1,0.1,5,0,0])
growth = np.array([[0.001,0,1],[0.001,1,0]])
half_saturation = np.ones((2,3))
max_consumption = np.array([[0.1,0,1],[0.1,1,0]])
production = np.array([[0,1,0],[0,0,1]])
influx = np.array([0,0,0])

sol5 = solve_ivp(dCRdt_5,[0,t_end],initial_abundance,
                 args=(growth,max_consumption,half_saturation,
                       production,influx),method='RK45')

plt.plot(sol5.t,sol5.y.T)
