# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:20:16 2024

@author: jamil
"""

# %%

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

########################

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe
import pandas as pd
import seaborn as sns
import sys
from copy import deepcopy
import pickle
import colorsys
from scipy.integrate import solve_ivp

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules')

from models import Consumer_Resource_Model
from community_level_properties import max_le

#%%

def partial_differential_dRdt(t, var,
                              no_species, 
                              growth, death, consumption, influx, dispersal):
    
    #breakpoint()
    
    species = var[:no_species]
    resources = var[no_species:]
    
    dRdt = (resources * (influx - resources)) - \
        (resources * np.sum(consumption * species, axis=1)) + dispersal
        
    dSdt = np.zeros(len(species))
    
    return np.concatenate((dSdt, dRdt))

def partial_differential_dSdt(t, var,
                              no_species, 
                              growth, death, consumption, influx, dispersal):
    
    species = var[:no_species]
    resources = var[no_species:]
    
    dSdt = species * (np.sum(growth * resources, axis=1) - death) + dispersal
    
    dRdt = np.zeros(len(resources))
    
    return np.concatenate((dSdt, dRdt))

def partial_differential_dSdt_c(t, var,
                                no_species, 
                                growth, death, consumption, influx, dispersal):
    
    species = var[:no_species]
    resources = var[no_species:]
    
    dSdt = species * (np.sum(growth * consumption.T * resources, axis=1) - death) + dispersal
    
    dRdt = np.zeros(len(resources))
    
    return np.concatenate((dSdt, dRdt))

#%%

def unbounded_growth(t, var, *args):
    
    if np.any(np.log(np.abs(var)) > 4) or np.isnan(np.log(np.abs(var))).any():
        
        return 0
    
    else: 
        
        return 1
    
#%%

def simulate_relaxation(partial_differential_equation,
                        t_end, initial_abundance, 
                        no_species, growth, death, consumption, influx, dispersal):
    
    match partial_differential_equation:
        
        case 'partial_R':
            
            model = partial_differential_dRdt
            
        case 'partial_S':
            
            model = partial_differential_dSdt
            
        case 'partial_S_coupled':
            
            model = partial_differential_dSdt_c
            
    unbounded_growth.terminal = True
    
    solution = solve_ivp(model, [0, t_end], initial_abundance,
                     args=(no_species, growth, death, consumption, influx, dispersal),
                     method = 'RK45', rtol = 1e-14, atol = 1e-14,
                     t_eval = np.linspace(0, t_end, 200), events = unbounded_growth)
    
    breakpoint()
               
    return solution

#%%

def fluctuations(dynamics, extinction_threshold):
         
    extant_species = dynamics[np.any(dynamics > extinction_threshold, axis=1), :]

    return np.std(extant_species, axis=1)/np.mean(extant_species, axis=1)

#%%

def relaxation_time(simulation, partial_differential_equation, no_species, stability_threshold = 5e-2):
    
    breakpoint()
    
    if partial_differential_equation == 'partial_R':
        
        dynamics = simulation.y[no_species:, :]
        extinction_threshold = 1e-2
            
    else:
            
        dynamics = simulation.y[:no_species, :]
        extinction_threshold = 1e-3
            
    variation_in_dynamics = fluctuations(dynamics, extinction_threshold)
    
    final_relaxation_time = simulation.t[np.all(variation_in_dynamics < stability_threshold, axis = 0)][0]
    average_relaxation_time = \
        np.mean(simulation.t[np.where(variation_in_dynamics < stability_threshold, axis = 1)])
    
    return {'Final relaxation' : final_relaxation_time,
            'Average relaxation' : average_relaxation_time}

#%%

def simulate_and_calculate_relaxations(community, simulation, partial_differential_equation, no_species):
     
    partial_differential_simulation = simulate_relaxation(partial_differential_equation,
                                                            50, simulation.y[:, -1],
                                                            no_species, community.growth,
                                                            community.death, community.consumption,
                                                            community.influx, community.dispersal) 
    
    #relaxation_times = relaxation_time(partial_differential_simulation,
    #                                   partial_differential_equation, no_species)
    
    relaxation_rates = gradient_estimate(simulation, partial_differential_equation)
    
    return np.mean(relaxation_rates)

#%%

def gradient_estimate(simulation, partial_differential_equation):
    
    if partial_differential_equation == 'partial_R':
        
        dynamics = simulation.y[no_species:, :]
            
    else:
            
        dynamics = simulation.y[:no_species, :]
        
    breakpoint()
    
    return np.diff(dynamics, axis = 1)/np.diff(simulation.t)

# %%

def consumer_resource_dynamics_relaxation(model_version, mu_c, sigma_c,
                                          mu_g, sigma_g, no_species, no_resources):
    
    no_communities = 10
 
    def community_dynamics(model_version,
                           mu_c, sigma_c, mu_g, sigma_g,
                           no_species, no_resources):
        
        #breakpoint()
        
        #community = Consumer_Resource_Model(no_species, no_resources, {'mu_g' : mu_g, 'sigma_g' : sigma_g},
        #                                    {'mu_c' : mu_c, 'sigma_c' : sigma_c})
        community = Consumer_Resource_Model(no_species, no_resources, {'mu_g' : mu_g/no_species,
                                                                       'sigma_g' : sigma_g/np.sqrt(no_species)},
                                            {'mu_c' : mu_c/no_species, 'sigma_c' : sigma_c/np.sqrt(no_species)})
        community.generate_parameters()
        #consumption_rates = deepcopy(community.consumption)
        #community.generate_parameters(method = 'user supplied', growth = consumption_rates.T,
        #                              consumption = consumption_rates)
        community.simulate_community(np.arange(5), 3000, model_version = model_version,
                                                    assign = True)
        # relaxation times
        
        match model_version:
            
            case 'growth_consumption_uncoupled':
                
                species_relaxations = [simulate_and_calculate_relaxations(community, simulation, 'partial_S', no_species)
                                       for simulation in community.ODE_sols.values()]
            
            case 'growth_function_of_consumption':
                
                species_relaxations = [simulate_and_calculate_relaxations(community, simulation, 'partial_S_coupled', no_species)
                                       for simulation in community.ODE_sols.values()]
                
        resource_relaxations = [simulate_and_calculate_relaxations(community, simulation, 'partial_R', no_species)
                                for simulation in community.ODE_sols.values()]
                
        return community, resource_relaxations, species_relaxations
            
    communities_list = [deepcopy(community_dynamics(model_version, mu_c, sigma_c,
                                                    mu_g, sigma_g, no_species,
                                                    no_resources))
                              for _ in range(no_communities)]
    
    return communities_list

#%%

model_version = 'growth_consumption_uncoupled'
mu_c = 200 #0.5
sigma_c = 3.5 #0.1
mu_g = 200 #1
sigma_g = 3.5 #0.1
no_species = 250
no_resources = 250

relaxation_uncoupled_model = consumer_resource_dynamics_relaxation(model_version, mu_c, sigma_c, mu_g, sigma_g, no_species, no_resources)

#%%

model_version = 'growth_function_of_consumption'
mu_c = 0.5
sigma_c = 0.1
mu_g = 1
sigma_g = 0.1
no_species = 250
no_resources = 250

relaxation_coupled_model = consumer_resource_dynamics_relaxation(model_version, mu_c, sigma_c, mu_g, sigma_g, no_species, no_resources)
