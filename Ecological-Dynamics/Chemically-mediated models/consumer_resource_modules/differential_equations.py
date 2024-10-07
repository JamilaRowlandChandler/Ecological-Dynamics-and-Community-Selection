# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:29:00 2024

@author: jamil
"""

# %%

import numpy as np
from scipy.integrate import solve_ivp

from initial_abundances import InitialConditionsInterface

#%%

class DifferentialEquationsInterface(InitialConditionsInterface):
    
    def simulate_community(self, lineages, t_end, init_cond_func='Mallmin',
                           model_version = 'growth_consumption_uncoupled',
                           assign = True, **kwargs):
     
        initial_abundances = \
            self.generate_initial_conditions(lineages, init_cond_func, **kwargs)
            
        ODE_sols = {'lineage ' + str(lineage) : self.CR_simulation(t_end, initial_abundances[:,i], model_version) \
                         for i, lineage in enumerate(lineages)}
            
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
            
    def CR_simulation(self, t_end, initial_abundance, model_version):
        
        match model_version:
            
            case 'growth_consumption_uncoupled':
                
                model = dCR_dt
                
            case 'growth_function_of_consumption':
                
                model = dCR_dt_2
                
            case 'consumption_funtion_of_growth':
                
                model = dCR_dt_3
                
        unbounded_growth.terminal = True
                   
        return solve_ivp(model, [0, t_end], initial_abundance,
                         args=(self.no_species, self.growth, self.death,
                               self.consumption, self.influx, self.dispersal),
                         method = 'RK45', rtol = 1e-14, atol = 1e-14,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
# %%

def dCR_dt(t, var,
           no_species,
           growth, death, consumption, influx,
           dispersal):

    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal
    
    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + dispersal

    return np.concatenate((dSdt, dRdt))


def dCR_dt_2(t, var,
             no_species,
             growth, death, consumption, influx,
             dispersal):
    
    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * consumption.T * resources, axis=1) - death) + dispersal

    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + dispersal

    return np.concatenate((dSdt, dRdt))

def dCR_dt_3(t, var,
             no_species,
             growth, death, consumption, influx,
             dispersal):
    
    species = var[:no_species]
    resources = var[no_species:]

    dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal

    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(growth.T * consumption * species, axis=1)) + dispersal

    return np.concatenate((dSdt, dRdt))

def unbounded_growth(t, var, *args):
    
    if np.any(np.log(np.abs(var)) > 4) or np.isnan(np.log(np.abs(var))).any():
        
        return 0
    
    else: 
        
        return 1
    