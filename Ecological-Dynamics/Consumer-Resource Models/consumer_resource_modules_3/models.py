# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:01:05 2025

@author: jamil
"""

import numpy as np
from scipy.integrate import solve_ivp

from parameters import ParametersInterface
from differential_equations import DifferentialEquationsInterface
from differential_equations import unbounded_growth
from community_level_properties import CommunityPropertiesInterface
#from initial_abundances import InitialConditionsInterface

# %%

def Consumer_Resource_Model(model, no_species, no_resources):
    
    match model:
        
        case "Self-limiting resource supply":
            
            instance = SL_CRM(no_species, no_resources)
            
        case "Externally-supplied resources":
            
            #instance = ES_CRM(no_species, no_resources)
            pass
        
        case _:
            
            raise Exception('You have not selected an exisiting model.\n' + \
                  'Please chose from either "Self-limiting resource supply"' + \
                      ' or "Externally-supplied resources"')
    return instance

# %%

class SL_CRM(ParametersInterface, DifferentialEquationsInterface, CommunityPropertiesInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
        
    def model_specific_rates(self, death_method, death_args,
                             resource_growth_method, resource_growth_args):
        
        p_labels = ['d', 'b']
        dims_list = [(self.no_species, ), (self.no_resources, )]
        
        for p_method, p_args, p_label, dims in \
            zip([death_method, resource_growth_method],
                [death_args, resource_growth_args],
                p_labels, dims_list):
                
                self.other_parameter_methods(p_method, p_args, p_label, dims)
    
    #####################################################################
    
    def simulation(self, t_end, initial_abundance):
        
        def model(t, y,
                  S, G, C, D, K):
            
            species, resources = y[:S], y[S:]

            dSdt = species * (np.sum(G * resources, axis = 1) - D)
            
            dRdt = (resources * (K - resources)) - \
                (resources * np.sum(C * species, axis=1))

            return np.concatenate((dSdt, dRdt)) + 1e-8
        
        unbounded_growth.terminal = True
        
        # call the ODE solver with the unbounded growth event function
        # the ode solver stops when the event function is true (returns 0)           
        return solve_ivp(model, [0, t_end], initial_abundance, 
                         args = (self.no_species, self.growth, self.consumption, 
                                 self.d, self.b),
                         method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                         t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)

# %%

'''

class ES_CRM(ParametersInterface, CommunityPropertiesInterface, InitialConditionsInterface):
    
    def __init__(self, no_species, no_resources):
        
        self.no_species = no_species
        self.no_resources = no_resources
    
    def model_specific_rates(death, influx, outflux):
        
        pass
    
    def simulate_community(self, t_end, no_init_cond, init_cond_func='Mallmin',
                           assign = True, **kwargs):
        
        def simulation(t_end, initial_abundance):
            
            def model(t, y,
                      no_species, growth, death, consumption, influx, outflux,
                      dispersal):
             
             
             #ODEs for the consumer-resource model with externally supplied resources.
             
             
             species = y[:no_species]
             resources = y[no_species:]
             
             #breakpoint()

             dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal

             dRdt = (influx - outflux * resources) - \
                     (resources * np.sum(consumption * species, axis=1)) + dispersal

             return np.concatenate((dSdt, dRdt))
            
            unbounded_growth.terminal = True
            
            # call the ODE solver with the unbounded growth event function
            # the ode solver stops when the event function is true (returns 0)           
            return solve_ivp(model, [0, t_end], initial_abundance, 
                             args = (self.no_species, self.growth, self.death,
                                     self.consumption, self.influx, self.dispersal),
                             method = 'LSODA', rtol = 1e-7, atol = 1e-9,
                             t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
        
        # generate initial species and resource abundances
        initial_abundances = \
            self.generate_initial_conditions(np.arange(no_init_cond), init_cond_func, **kwargs)
        
        # simulate community dynamics for each set of initial conditions
        ODE_sols = [simulation(t_end, initial_abundance) 
                    for initial_abundance in initial_abundances]
        
        # should simulations be assigned to a class attribute
        if assign is True:
            
            self.ODE_sols = ODE_sols
        
        else:
            
            return ODE_sols
        
'''