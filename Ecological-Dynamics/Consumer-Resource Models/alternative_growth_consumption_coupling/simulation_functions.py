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
import seaborn as sns
from matplotlib import pyplot as plt

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

def create_df_and_delete_simulations_2(path, file, parameters, parameter_cols = None):
    
    CR_communities = pd.read_pickle(path + file)
    
    df = CR_dynamics_df(CR_communities, parameters, parameter_cols)
    
    return df

# %%

def prop_chaotic(x,
                instability_threshold = 0.00):
        
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

# %%

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None):
    
    '''

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    variables : TYPE
        DESCRIPTION.
    variable_label : TYPE
        DESCRIPTION.
    cmaps : TYPE
        DESCRIPTION.
    titles : TYPE
        DESCRIPTION.
    fig_dims : TYPE
        DESCRIPTION.
    is_logs : TYPE, optional
        DESCRIPTION. The default is None.
    specify_min_max : TYPE, optional
        DESCRIPTION. The default is None.
    mosaic : TYPE, optional
        DESCRIPTION. The default is None.
    gridspec_kw : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axs : TYPE
        DESCRIPTION.

    '''
    
    if pivot_functions is None:
    
        pivot_tables = {variable : df.pivot(index = y, columns = x, values = variable)
                        for variable in variables}
        
    else:
        
        pivot_tables = {variable : (df.pivot(index = y, columns = x, values = variable)
                                    if pivot_functions[variable] is None 
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x,
                                                              values = variable)[0]) 
                        for variable in variables}
    if is_logged is None:
        
        pivot_tables_plot = pivot_tables
        
    else:
    
        pivot_tables_plot = pivot_tables | \
                            {variable : np.log10(np.abs(pivot_tables[variable]))
                             for variable in is_logged}
    
    start_v_min_max = {variable : [np.min(pivot_table), np.max(pivot_table)]
                       for variable, pivot_table in pivot_tables_plot.items()}
    
    if specify_min_max:
        
        v_min_max = start_v_min_max | specify_min_max
        
    else:
        
        v_min_max = start_v_min_max
        
    sns.set_style('white')
    
    if mosaic:
        
        fig, axs = plt.subplot_mosaic(mosaic, figsize = figsize,
                                      gridspec_kw = gridspec_kw, layout = 'constrained')
    
    else:
        
        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize)

    fig.supxlabel(xlabel, fontsize = 16, weight = 'bold')
    fig.supylabel(ylabel, fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')
    
    if fig_dims == (1,1):
        
        subfig = sns.heatmap(pivot_tables_plot[variables[0]], ax = axs,
                             vmin = v_min_max[variables[0]][0],
                             vmax = v_min_max[variables[0]][1],
                             cbar = True, cmap = cmaps)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot_tables_plot[variables[0]].shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot_tables_plot[variables[0]].shape[1], 0, 1,
                       color = 'black', linewidth = 2)

        axs.set_yticks([0.5, len(np.unique(df[y])) - 0.5],
                      labels = [np.round(np.min(df[y]), 3),
                                np.round(np.max(df[y]), 3)], fontsize = 14)
        axs.set_xticks([0.5, len(np.unique(df[x])) - 0.5], 
                      labels = [np.round(np.min(df[x]), 3),
                                np.round(np.max(df[x]), 3)],
                      fontsize = 14, rotation = 0)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.invert_yaxis()
        axs.set_title(titles, fontsize = 16, weight = 'bold', y = 1.05)
        
    else:
        
        for ax, variable, cmap, title in zip(axs.flatten()[:len(variables)], 
                                             variables, cmaps, titles):
            
            #breakpoint()
            
            subfig = sns.heatmap(pivot_tables_plot[variable], ax = ax,
                                 vmin = v_min_max[variable][0],
                                 vmax = v_min_max[variable][1],
                                 cbar = True, cmap = cmap)
            
            subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axhline(pivot_tables_plot[variables[0]].shape[0], 0, 1,
                           color = 'black', linewidth = 2)
            subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axvline(pivot_tables_plot[variables[0]].shape[1], 0, 1,
                           color = 'black', linewidth = 2)
    
            ax.set_yticks([0.5, len(np.unique(df[y])) - 0.5],
                          labels = [np.round(np.min(df[y]), 3),
                                    np.round(np.max(df[y]), 3)], fontsize = 14)
            ax.set_xticks([0.5, len(np.unique(df[x])) - 0.5], 
                          labels = [np.round(np.min(df[x]), 3),
                                    np.round(np.max(df[x]), 3)],
                          fontsize = 14, rotation = 0)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.invert_yaxis()
            ax.set_title(title, fontsize = 16, weight = 'bold', y = 1.05)
                    
    return fig, axs