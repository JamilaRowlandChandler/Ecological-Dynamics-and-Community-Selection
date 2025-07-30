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
import os
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/consumer_resource_modules_3')
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


def CRM_across_parameter_space(parameter_sets, subdirectory, parms_for_filenames):
    
    '''
    
    Create and simulate communities across parameter space, using the 
    Consumer_Resource_Model class

    Parameters
    ----------
    parameter_sets : list of dicts
        List of parameters for the Consumer_Resource_Model class.
    subdirectory : string
        Folder to save the data in.
    parms_for_filenames : str
        Parameters used to name each file.

    Returns
    -------
    None.

    '''
    
    # extract the parameters used to name files
    key0, key1 = parms_for_filenames
    
    # create the directory where the communities should be saved (if the directory
    #   doesn't already exist)
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + subdirectory
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
    
    # Iterate through the parameter space, creating and simulating community dynamics
    for parm_set in tqdm(parameter_sets, position = 0, leave = True):
        
        CRMs_create_and_save(subdirectory + "/simulations_" + \
                                 str(np.round(parm_set[key0], 4)) + "_" + \
                                 str(np.round(parm_set[key1], 4)),
                             parm_set['S'], parm_set['M'],
                             dict(method = 'growth function of consumption',
                                  mu_c = parm_set['mu_c'],
                                  sigma_c = parm_set['sigma_c'],
                                  mu_g = parm_set['mu_y'],
                                  sigma_g = parm_set['sigma_y']),
                             dict(death_method = 'constant',
                                  death_args =  {'d' : parm_set['d']},
                                  resource_growth_method = 'constant',
                                  resource_growth_args = {'b' : parm_set['b']}),
                             no_communities = 20, t_end = 7000)
            
#%%

def CRMs_create_and_save(filepath,
                         no_species, no_resources,
                         growth_consumption_rates_args,
                         model_specific_rates_args,
                         **kwargs):
    
    '''
    
    Simulate communites where model parameters are sampled from the same distribution

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    no_species : TYPE
        DESCRIPTION.
    no_resources : TYPE
        DESCRIPTION.
    growth_consumption_rates_args : TYPE
        DESCRIPTION.
    model_specific_rates_args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    communities = consumer_resource_model_dynamics(no_species, no_resources,
                                                   growth_consumption_rates_args,
                                                   model_specific_rates_args,
                                                   **kwargs)
    
    pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + \
                filepath + ".pkl",
                communities)
    del communities

# %%

def consumer_resource_model_dynamics(no_species, no_resources,
                                     growth_consumption_rates_args,
                                     model_specific_rates_args,
                                     no_communities = 5, no_init_conds = 2,
                                     t_end = 3500):
    
    def community_dynamics(no_init_conds,
                           no_species, no_resources,
                           growth_consumption_rates_args,
                           model_specific_rates_args,
                           t_end):
    
        community = Consumer_Resource_Model("Self-limiting resource supply",
                                            no_species, no_resources)

        community.growth_consumption_rates(**growth_consumption_rates_args)
        community.model_specific_rates(**model_specific_rates_args)

        community.simulate_community(t_end, no_init_conds)
        community.calculate_community_properties()
        community.lyapunov_exponent = max_le(community, 500, community.ODE_sols[0].y[:, -1],
                                             1e-3, dt = 20, separation = 1e-3)
        
        return community 

    communities = [deepcopy(community_dynamics(no_init_conds,
                                                    no_species, no_resources,
                                                    growth_consumption_rates_args,
                                                    model_specific_rates_args,
                                                    t_end))
                                  for _ in range(no_communities)]
    
    return communities

# %%

def generate_simulation_df(directory):
    
    parameters = ['no_species', 'no_resources', 'mu_c', 'sigma_c', 'mu_y',
                  'sigma_y', 'd_val', 'b_val'] 
    
    df = CRM_df(directory, parameters)
    
    for var in ['rho', 'mu_c', 'mu_y', 'sigma_c', 'sigma_y', 'mu_c/M',
                'sigma_c/root_M']:
        
        df[var] = np.round(df[var], 6)
    
    df['no_resources'] = np.int32(df['no_resources'])
    
    df.rename(columns = {'no_resources' : 'M', 'no_species' : 'S'},
              inplace = True)
    
    return df

# %%

def CRM_df(directory, parameters):
    
    def load_data_create_df(filepath):
    
        communities = pd.read_pickle(filepath)
        
        df = community_dynamics_df(communities, parameters)
    
        return df
    
    df = pd.concat([load_data_create_df(directory + "/" + file) 
                    for file in os.listdir(directory)],
                   axis = 0, ignore_index = True)
        
    return df

# %%
    
def community_dynamics_df(communities, parameters):
    
    parameters = np.array(parameters)
    
    parameters[np.where(parameters == 'mu_y')[0]] = 'mu_g'
    parameters[np.where(parameters == 'sigma_y')[0]] = 'sigma_g'

    properties_df = pd.DataFrame.from_dict({'phi_N' : [phi_N for community in communities for phi_N in community.species_survival_fraction],
                                            'N_mean' : [N_mean for community in communities for N_mean in community.species_avg_abundance],
                                            'q_N' : [q_N for community in communities for q_N in community.species_abundance_fluctuations],
                                            'phi_R' : [phi_R for community in communities for phi_R in community.resource_survival_fraction],
                                            'R_mean' : [R_mean for community in communities for R_mean in community.resource_avg_abundance],
                                            'q_R' : [q_R for community in communities for q_R in community.resource_abundance_fluctuations],
                                            'Max. lyapunov exponent' : np.concatenate([np.repeat(community.lyapunov_exponent, len(community.ODE_sols)) 
                                                                                       for community in communities]),
                                            'Divergence measure' : [simulation.t[-1] for community in communities for simulation in community.ODE_sols],
                                            'rho_est' : np.concatenate([np.repeat(pearsonr(community.consumption.T.flatten(),
                                                                            community.growth.flatten())[0], len(community.ODE_sols)) 
                                                                        for community in communities])})
                            
    parameter_df = pd.DataFrame.from_dict({parameter : \
                                           np.concatenate([np.repeat(getattr(community, parameter),
                                                                     len(community.ODE_sols))
                                                           for community in communities]) 
                                           for parameter in parameters})
            
    parameter_df.rename(columns = {'mu_c' : 'mu_c/M', 'sigma_c' : 'sigma_c/root_M',
                                   'mu_g' : 'mu_y', 'sigma_g' : 'sigma_y'},
                        inplace = True)
    
    df = pd.concat([parameter_df, properties_df], axis = 1)
    
    df['mu_c'] = df['mu_c/M'] * df['no_resources']
    df['sigma_c'] = df['sigma_c/root_M'] * np.sqrt(df['no_resources'])
    
    df['rho'] = np.sqrt(1 / (1 + \
                             ((df['sigma_y']/df['mu_y'])**2 * (1 + \
                                                               ((df['mu_c']**2)/(df['no_resources'] * df['sigma_c']**2))))))
    
    df['Species packing'] = (df['phi_N']*df['no_species'])/(df['phi_R']*df['no_resources'])
    df['Instability distance'] = df['rho']**2 - df['Species packing']
    
    df['Infeasibility distance'] = df['phi_R'] - df['phi_N']/(df['no_resources']/df['no_species'])
    
    return df

# %%

def prop_chaotic(x,
                instability_threshold = 0.00):
        
    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None, **kwargs):
    
    if pivot_functions is None:
    
        pivot_tables = {variable : df.pivot(index = y, columns = x, values = variable)
                        for variable in variables}
        
    else:
        
        pivot_tables = {variable : (df.pivot(index = x, columns = y, values = variable)
                                    if pivot_functions[variable] is None 
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x,
                                                              values = variable)[0]) 
                        for variable in variables}
        
        #breakpoint()
    
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
        
        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                                sharex = True, sharey = True, layout = 'constrained')

    fig.supxlabel(xlabel, fontsize = 16, weight = 'bold')
    fig.supylabel(ylabel, fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')
    
    if fig_dims == (1,1):
        
        axs.set_facecolor('grey')
        
        subfig = sns.heatmap(pivot_tables_plot[variables[0]], ax = axs,
                    vmin = v_min_max[variables[0]][0], vmax = v_min_max[variables[0]][1],
                    cbar = True, cmap = cmaps, **kwargs)
        
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
        axs.set_title(titles, fontsize = 16, weight = 'bold')
        
    else:

        for ax, variable, cmap, title in zip(axs.values(), variables, cmaps, titles):
            
            ax.set_facecolor('grey')
            
            subfig = sns.heatmap(pivot_tables_plot[variable], ax = ax,
                        vmin = v_min_max[variable][0], vmax = v_min_max[variable][1],
                        cbar = True, cmap = cmap, **kwargs)
            
            subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axhline(pivot_tables_plot[variable].shape[0], 0, 1,
                           color = 'black', linewidth = 2)
            subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
            subfig.axvline(pivot_tables_plot[variables].shape[1], 0, 1,
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
            ax.set_title(title, fontsize = 16, weight = 'bold')
        
    return fig, axs
