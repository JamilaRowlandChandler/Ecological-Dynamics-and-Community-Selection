# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:19:33 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
from copy import deepcopy
import pickle

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')
from models import Consumer_Resource_Model
from community_level_properties import max_le

# %%

def pickle_dump(filename,data):
    
    '''
    
    Pickle data.

    Parameters
    ----------
    filename : string
        Pickle file name. Should end with .pkl
    data : any
        Data to pickle.

    Returns
    -------
    None.

    '''
    
    with open(filename, 'wb') as fp:
        
        pickle.dump(data, fp)

# %%

def consumer_resource_model_dynamics(no_species, no_resources, parameters,
                                     growth_consumption_function,
                                     no_communities = 5, no_lineages = 2,
                                     t_end = 3500):
    
    #parameter_methods = {'death' : 'normal', 'influx' : 'normal'}
    parameter_methods = {'death' : 'constant', 'influx' : 'constant'}
    
    def community_dynamics(i, lineages, no_species, no_resources, parameters,
                           growth_consumption_function, no_lineages, t_end):
        
        community = Consumer_Resource_Model(no_species, no_resources, parameters)
        
        community.generate_parameters(growth_consumption_method = growth_consumption_function,
                                      other_parameter_methods = parameter_methods)
        
        community.simulate_community(lineages, t_end, model_version = 'self-limiting resource supply',
                                     assign = True)
        
        community.calculate_community_properties(lineages, t_end - 500)
        community.lyapunov_exponent = \
            max_le(community, 500, community.ODE_sols['lineage 0'].y[:, -1],
                   1e-3, 'self-limiting resource supply', dt = 20, separation = 1e-3)
    
        return community 

    communities_list = [deepcopy(community_dynamics(i, np.arange(no_lineages),
                                                    no_species, no_resources,
                                                    parameters,
                                                    growth_consumption_function,
                                                    no_lineages,
                                                    t_end))
                                  for i in range(no_communities)]
    
    return communities_list

# %%
    
def CR_dynamics_df(communities_list, parameters, parameter_cols):
    
    simulation_data = {'Species Volatility' : [volatility for community in communities_list for volatility in community.species_volatility.values()],
                       'Resource Volatility' : [volatility for community in communities_list for volatility in community.resource_volatility.values()],
                       'Species Fluctuation CV' : [fluctuations for community in communities_list for fluctuations in community.species_fluctuations.values()],
                       'Resource Fluctuation CV' : [fluctuations for community in communities_list for fluctuations in community.resource_fluctuations.values()],
                       'Species diversity' : [diversity for community in communities_list for diversity in community.species_survival_fraction.values()],
                       'Resource diversity' : [diversity for community in communities_list for diversity in community.resource_survival_fraction.values()],
                       'Max. lyapunov exponent' : np.concatenate([np.repeat(community.lyapunov_exponent, len(community.ODE_sols)) 
                                                                  for community in communities_list]),
                       'Divergence measure' : [simulation.t[-1] for community in communities_list for simulation in community.ODE_sols.values()]}
                       
    simulation_data['Species packing'] = \
        np.array(simulation_data['Species diversity'])/np.array(simulation_data['Resource diversity'])
        
    properties_df = pd.DataFrame.from_dict(simulation_data)
    
    more_properties_df = CR_abundance_distribution(communities_list)
    
    parameter_array = np.array([np.concatenate([np.repeat(getattr(community, parameter),
                                                          len(community.ODE_sols))
                                 for community in communities_list]) for parameter in parameters])
    
    if parameter_cols is None: 
        
        columns = parameters
        
    else:
        
        columns = parameter_cols
        
    parameter_df = pd.DataFrame(parameter_array.T, columns = columns)
     
    df = pd.concat([parameter_df, properties_df, more_properties_df], axis = 1)
    
    return df

# %%

def CR_abundance_distribution(communities_list):
    
    def distribution_properties(simulation, no_species):
        
        final_species_abundances = simulation.y[:no_species, -1]
        final_resource_abundances = simulation.y[no_species:, -1]
        
        spec_survive_frac = np.count_nonzero(final_species_abundances > 1e-4)/len(final_species_abundances)
        spec_mean = np.mean(final_species_abundances)
        spec_sq_mean = np.mean(final_species_abundances**2)
        
        res_survive_frac = np.count_nonzero(final_resource_abundances > 1e-4)/len(final_resource_abundances)
        res_mean = np.mean(final_resource_abundances)
        res_sq_mean = np.mean(final_resource_abundances**2)
        
        return spec_survive_frac, spec_mean, spec_sq_mean, res_survive_frac, \
                res_mean, res_sq_mean
     
    dist_properties_array = np.vstack([distribution_properties(simulation, community.no_species)
                                       for community in communities_list 
                                       for simulation in community.ODE_sols.values()])
    
    df = pd.DataFrame(dist_properties_array, columns = ['phi_N', 'N_mean', 'q_N', 
                                                        'phi_R', 'R_mean', 'q_R'])
    
    return df

#%%

def create_and_delete_CR(filename, no_species, no_resources, parameters, **kwargs):
    
    CR_communities = consumer_resource_model_dynamics(no_species, no_resources,
                                                      parameters, **kwargs)
    
    pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl",
                CR_communities)
    del CR_communities
    
# %%

def create_df_and_delete_simulations(filename, parameters, parameter_cols = None):
    
    CR_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
    
    df = CR_dynamics_df(CR_communities, parameters, parameter_cols)
    
    return df

# %%

def prop_chaotic(x,
                instability_threshold = 0.0025):
        
    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

# %%

def species_packing(df):
    
    M, S, phi_N, phi_R = df['no_resources'], df['no_species'], df['phi_N'], df['phi_R']
    
    gamma = M/S
    
    return (phi_N/phi_R)/gamma

# %%

def distance_from_instability(df):
    
    M, S, rho, phi_N, phi_R = df['no_resources'], df['no_species'], df['rho'], df['phi_N'], df['phi_R']
    gamma = M/S
    
    return rho**2 - phi_N/(phi_R * gamma)

# %%

def distance_from_infeasibility(df):
    
    M, S, phi_N, phi_R = df['no_resources'], df['no_species'], df['phi_N'], df['phi_R']
    gamma = M/S
    
    return phi_R - phi_N/gamma