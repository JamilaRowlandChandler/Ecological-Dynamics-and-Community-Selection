# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:46:45 2024

@author: jamil

"""

#cd Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/scripts/

import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#######################

def growth_rates_on_metabolites(max_growth,trophic_levels,f=0.6):
    
    growth_rates = max_growth * f * (1 - f)**(trophic_levels - 1)
    
    return growth_rates

def metabolite_production_rates(max_production,trophic_levels,f=0.6):
    
    production_rates = max_production * (1 - f)**(trophic_levels)
    
    def production_per_spec(production_rates,trophic_levels,i):
        
        tl_i, tl_j = np.meshgrid(trophic_levels[i,:], trophic_levels[i,:])
        
        def does_x_make_y(r1,r2):
            
            return r1 < r2
        
        probability_matrix = does_x_make_y(tl_i,tl_j).astype(int)
        
        production_rates_spec = \
            np.sum(probability_matrix, axis = 1) * production_rates[i,:]
        
        return production_rates_spec
     
    production_rates = [production_per_spec(production_rates, trophic_levels, i) \
                              for i in range(trophic_levels.shape[0])]
            
    return np.array(production_rates)

##############

def dSRdt(t,var,
          no_species,
          growth,inhibition,
          influx,consumption,production):
    
    S = var[:no_species]
    R = var[no_species:]
    
    dS = S * (np.sum(growth * consumption * R, axis = 1) - \
              np.sum(inhibition * R, axis = 1))
    
    def production_rate(i):
        
        R_remove_i = np.delete(R,i)
        consump_remove_i = np.delete(consumption,i,axis=1)
         
        R_i_produced = production[:,i] * np.sum(consump_remove_i * R_remove_i,axis=1)
         
        return R_i_produced
    
    R_produced = np.array([production_rate(i) for i in range(len(R))])
    R_consumed = consumption * R
    
    production_consumption = \
        (np.matmul(R_produced, S.reshape(len(S),1)) - \
         np.matmul(R_consumed.T, S.reshape(len(S),1))).reshape(R.shape)
     
    dR = influx + production_consumption
    
    dSR = np.concatenate([dS,dR])
    
    return dSR



####################################################################




trophic_levels = np.array([[1,2,2],
                           [1,2,2]])
max_production = np.array([[0,1,0],
                        [0,0,1]])
max_growth = np.array([[0.01,0,0.1],
                       [0.01,0.1,0]])

true_production = metabolite_production_rates(production, trophic_levels)
true_growth = growth_rates_on_metabolites(max_growth, trophic_levels)







def growth_rates(max_growth,trophic_levels,f=0.6):
    
    def growth_per_trophic_level(rates,f,t_l):
        
        breakpoint()
        
        t_l_rates = rates * f * (1 - f)**(t_l-1)
        
        return t_l_rates
    
    growth_rates_per_resource = \
        np.array([growth_per_trophic_level(rates, f, t_l) \
                  for rates, t_l in zip(max_growth,trophic_levels)])
            
    return growth_rates_per_resource

def dSRdt(t,var,
          growth,death,
          influx,consumption,production):
    
    S = var[:2]
    R = var[2:]
    
    dS = np.sum(growth * consumption * R, axis=1)*S
    
    def production_rate(i):
        
        R_remove_i = np.delete(R,i)
        consump_remove_i = np.delete(consumption,i,axis=1)
         
        R_i_produced = production[:,i] * np.sum(consump_remove_i * R_remove_i,axis=1)
         
        return R_i_produced
    
    R_produced = np.array([production_rate(i) for i in range(len(R))])
    R_consumed = consumption * R
    
    production_consumption = \
        (np.matmul(R_produced, S.reshape(len(S),1)) - \
         np.matmul(R_consumed.T, S.reshape(len(S),1))).reshape(R.shape)
     
    dR = influx + production_consumption
    
    dSR = np.concatenate([dS,dR])
    
    return dSR

###########################################################################

growth = np.array([[0.01,0,1],
                   [0.01,1,0]])
production = np.array([[0,1,0],
                        [0,0,1]])

true_growth = 



influx = np.zeros(3)
production = np.array([[0,1,0],
                        [0,0,1]])
consumption = np.array([[0.1,0,1],
                        [0.1,1,0]])

t_end = 70
initial_abundance = np.array([0.1,0.1,5,0,0])
