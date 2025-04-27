# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 20:20:02 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
from copy import copy
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
from time import time

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/self_limiting_alternative_growth_consumption_coupling')

from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations, prop_chaotic, distance_from_instability, \
    distance_from_infeasibility, species_packing, pickle_dump

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')
from models import Consumer_Resource_Model
from community_level_properties import max_le

# %%

def solve_for_mu_c(x):
    
    correlation, sigma_c, sigma_g, mu_g, M = x
    
    solved_mu_c = ((sigma_c * np.sqrt(M))/sigma_g) * np.sqrt((mu_g**2)*(1/(correlation**2) - 1) - sigma_g**2)
    
    return solved_mu_c

# %%

def solve_for_mu_g(x):
    
    rho, sigma_c, sigma_g, mu_c, M = x
        
    solved_mu_g = (sigma_g/sigma_c)*np.sqrt(((rho**2)/(1 - rho**2)) * ((mu_c**2)/M + sigma_c**2))
    
    return solved_mu_g

# %% 

def generate_parameters(no_resources, scaling, unscaled_mu_c_range, unscaled_sigma_range,
                        fixed_parameters):
    
    fixed_parameters_copy = copy(fixed_parameters)
    
    match scaling:
        
        case 'asymmetric':
    
            mu_c_range = unscaled_mu_c_range/no_resources
            sigma_range = unscaled_sigma_range/np.sqrt(no_resources)
            
        case 'symmetric':
            
            mu_c_range = unscaled_mu_c_range/np.sqrt(no_resources)
            sigma_range = unscaled_sigma_range/(no_resources**0.25)
            
            fixed_parameters_copy = copy(fixed_parameters)
            fixed_parameters_copy['mu_g'] /= np.sqrt(no_resources)
        
    # generate n values of rho and sigma within range
    mu_c_sigma_combinations = np.unique(sce.parameter_combinations([mu_c_range,
                                                                    sigma_range],
                                                                   n), axis = 1)
   
    # array of variable parameter combinations
    variable_parameters = np.vstack([mu_c_sigma_combinations,
                                     mu_c_sigma_combinations[1, :]])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               ['mu_c', 'sigma_c', 'sigma_g'],
                                               fixed_parameters_copy)
    
    return parameters

# %%

def feasibility_related_dynamics(system_size, parameters, growth_consumption_function,
                                 subdirectory, n, no_communities,
                                 filename_vars, **kwargs): 
    
    '''
    =======================
    Create folder
    =======================
    '''
    
    complete_subdirectory = subdirectory + "/infeasibility"
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + complete_subdirectory
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
        
    '''
    =====================
    Community dynamics
    =====================
    
    '''
    
    for parameter_set in tqdm(parameters, position = 0, leave = True):

        filename_CR = complete_subdirectory + "/CR_self_limiting_" + \
                        str(np.round(parameter_set[filename_vars[0]], 3)) + "_" + \
                            str(np.round(parameter_set[filename_vars[1]], 3))
        
        create_and_delete_CR(filename_CR, system_size, system_size, parameter_set,
                             no_communities = no_communities,
                             growth_consumption_function = growth_consumption_function,
                             **kwargs)
        
# %%

def find_infeasibility_brute(system_size, scaling, unscaled_mu_c_range, unscaled_sigma_range,
                             fixed_parameters, growth_consumption_function,
                             subdirectory, n, no_communities,
                             filename_vars = ['mu_c', 'sigma_c'],
                             **kwargs):
    
    parameters = generate_parameters(system_size, scaling, unscaled_mu_c_range,
                                     unscaled_sigma_range, fixed_parameters)
    
    feasibility_related_dynamics(system_size, parameters, growth_consumption_function,
                                 subdirectory, n, no_communities, filename_vars,
                                 **kwargs)
    
# %%

def find_infeasibility_supply_parameters(system_size, variable_parameters, v_names,
                                         fixed_parameters,
                                         growth_consumption_function,
                                         subdirectory, n, no_communities,
                                         filename_vars = ['mu_c', 'sigma_c'],
                                         **kwargs):
            
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters, v_names,
                                               fixed_parameters)
    
    feasibility_related_dynamics(system_size, parameters,
                                 growth_consumption_function, subdirectory, n,
                                 no_communities, filename_vars, **kwargs)
        
# %%

def infeasibility_df(system_size, subdirectory, supplied_parameters = None,
                     unscaled_mu_c_range = None, unscaled_sigma_range = None,
                     n = None, scaling = 'asymmetric',
                     parm_attributes = ['no_species', 'no_resources', 'mu_c',
                                        'sigma_c', 'mu_g', 'sigma_g', 'm', 'K'],
                     do_round = True):
    
    '''
    =======================
    Create folder
    =======================
    '''
    
    if supplied_parameters is not None:
        
        parameters = supplied_parameters
        
    else:
        
        match scaling:
            
            case 'asymmetric':
        
                mu_c_range = unscaled_mu_c_range/system_size
                sigma_range = unscaled_sigma_range/np.sqrt(system_size)
                
            case 'symmetric':
            
                mu_c_range = unscaled_mu_c_range/np.sqrt(system_size)
                sigma_range = unscaled_sigma_range/(system_size**0.25)
    
        # generate n values of mu and sigma within range
        mu_c_sigma_combinations = sce.parameter_combinations([mu_c_range, sigma_range], n)
        
        # array of all parameter combinations
        parameters = sce.variable_fixed_parameters(mu_c_sigma_combinations, ['mu_c', 'sigma_c'],
                                                   {})
    
    directory_prefix = subdirectory + "/infeasibility/CR_self_limiting_"
    
    key1, key2 = parameters[0].keys()
    
    if do_round == True:

        dfs = [create_df_and_delete_simulations(directory_prefix + \
                                                str(np.round(parameter_set[key1], 3)) + \
                                                "_" + str(np.round(parameter_set[key2], 3)),
                                                parm_attributes)
               for parameter_set in parameters]
            
    elif do_round == False:
        
        dfs = [create_df_and_delete_simulations(directory_prefix + \
                                                str(parameter_set[key1]) + \
                                                "_" + str(parameter_set[key2]),
                                                parm_attributes)
               for parameter_set in parameters]
        
    return dfs

# %%

def covariance_correlation(df, scaling):
    
    M = df['no_resources']
    
    match scaling:
        
        case 'asymmetric':
            
            mu_c, mu_g = df['mu_c']*M, df['mu_g']
            
            sigma_c, sigma_g = df['sigma_c']*np.sqrt(M), df['sigma_g']
            
            denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2)/M + (sigma_c * sigma_g)**2
            
        case 'symmetric':
            
            mu_c, mu_g = df['mu_c']*np.sqrt(M), df['mu_g']*np.sqrt(M)
            
            sigma_c, sigma_g = df['sigma_c']*(M**0.25), df['sigma_g']*(M**0.25)
            
            denominator = (mu_g * sigma_c)**2 + (mu_c * sigma_g)**2  + \
                                    np.sqrt(M)*(sigma_c * sigma_g)**2
         
    covariance = (mu_g * sigma_c**2)
    correlation = (mu_g * sigma_c)/np.sqrt(denominator)
    
    return covariance, correlation

# %%

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

def N_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'N_mean', aggfunc = 'mean')]

def phi_N_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'phi_N', aggfunc = 'mean')]

def rho_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns, 
                          values = 'rho', aggfunc = 'mean')]

def di_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns, 
                          values = 'instability distance', aggfunc = 'mean')]

def divergence_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns, 
                          values = 'Divergence measure', aggfunc = 'mean')]

def infeasible_pivot(df, index = 'sigma_c', columns = 'mu_c'):
    
    return [pd.pivot_table(df, index = index, columns = columns, 
                          values = 'infeasibility distance', aggfunc = 'mean')]

# %%

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None, **kwargs):
    
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
    
        pivot_tables = {variable : df.pivot(index = x, columns = y, values = variable)
                        for variable in variables}
        
    else:
        
        pivot_tables = {variable : (df.pivot(index = x, columns = y, values = variable)
                                    if pivot_functions[variable] is None 
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x)[0]) 
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

# %%

def generic_heatmaps_multiple_dfs_same_v(dfs, x, y, xlabel, ylabel, variable,
                                         cbar_label, cmap, titles,
                                         fig_dims, figsize,
                                         pivot_function = None, is_logged = None, 
                                         specify_min_max = None,
                                         mosaic = None, gridspec_kw = None,
                                         **kwargs):
    
    if pivot_function is None:
    
        pivot_tables = [df.pivot(index = x, columns = y, values = variable)
                        for df in dfs]
        
    else:
                
        pivot_tables = [pivot_function(df, index = y, columns = x)[0] 
                        for df in dfs]
    
    if is_logged:
    
        pivot_tables = [np.log10(np.abs(pivot_table)) 
                        for pivot_table in pivot_tables]
        
    if specify_min_max:
        
        v_min_max = specify_min_max
        
    else:
        
        v_min_max = [np.min([np.min(pivot_table) for pivot_table in pivot_tables]),
                     np.max([np.max(pivot_table) for pivot_table in pivot_tables])]
        
    xtick_position = np.max([len(np.unique(df[x])) for df in dfs])
    xtick_vals = [np.round(np.min([np.min(df[x]) for df in dfs]), 3),
                  np.round(np.max([np.max(df[x]) for df in dfs]), 3)]
    
    ytick_position = np.max([len(np.unique(df[y])) for df in dfs])
    ytick_vals = [np.round(np.min([np.min(df[y]) for df in dfs]), 3),
                  np.round(np.max([np.max(df[y]) for df in dfs]), 3)]
    
    top_border = np.max([pivot_table.shape[0] for pivot_table in pivot_tables])
    right_border = np.max([pivot_table.shape[1] for pivot_table in pivot_tables])
    
    sns.set_style('white')
    
    if mosaic:
        
        fig, axs = plt.subplot_mosaic(mosaic, figsize = figsize,
                                      gridspec_kw = gridspec_kw, layout = 'constrained',
                                      sharex = True, sharey = True)
    
    else:
        
        fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                                layout = 'constrained', sharex = True,
                                sharey = True)

    fig.supxlabel(xlabel, fontsize = 16, weight = 'bold')
    fig.supylabel(ylabel, fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')

    for i, (ax, df, pivot_table, title) in enumerate(zip(axs.flatten(), dfs,
                                                        pivot_tables, titles)):
    
        subfig = sns.heatmap(pivot_table, ax = ax,
                             vmin = v_min_max[0], vmax = v_min_max[1],
                             cbar = False, cmap = cmap, **kwargs)
        
        ax.set_facecolor('grey')
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(top_border, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(right_border, 0, 1, color = 'black', linewidth = 2)

        ax.set_yticks([0.5, ytick_position], labels = ytick_vals,
                      fontsize = 14)
        ax.set_xticks([0.5, xtick_position], labels = xtick_vals,
                      fontsize = 14, rotation = 0)
        ax.invert_yaxis()
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.set_title(title, fontsize = 16, weight = 'bold')
        
        if i == 0:
            
            mappable = subfig.get_children()[0]
    
    if fig_dims[0] == 1 or fig_dims[1] == 1:
        
        cbar = plt.colorbar(mappable, ax = axs[-1], orientation = 'vertical')
    else:
        
        #breakpoint()
        
        cbar = plt.colorbar(mappable,
                            ax = [axs[i, fig_dims[1] - 1] for i in range(fig_dims[0])],
                            orientation = 'vertical')
        
    cbar.set_label(label = cbar_label, size = '16')
    cbar.ax.tick_params(labelsize=12)
        
    return fig, axs

# %%

def plot_dynamics(simulations, no_species = 150, no_resources = 150):
    
   
    species = np.arange(no_species)
    resources = np.arange(no_species, no_species + no_resources)
    
    s_colour_index = np.arange(no_species)
    np.random.shuffle(s_colour_index)
    
    r_colour_index = np.arange(no_resources)
    np.random.shuffle(r_colour_index)

    cmap_s = LinearSegmentedColormap.from_list('custom YlGBl',
                                               ['#e9a100ff','#1fb200ff',
                                                '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                 N = no_species)
    
    cmap_r = LinearSegmentedColormap.from_list('custom YlGBl',
                                               ['#e9a100ff','#1fb200ff',
                                                '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                 N = no_resources)
    
    sns.set_style('white')

    fig, axs = plt.subplots(2, len(simulations), figsize = (5*len(simulations), 7),
                            layout = 'constrained',
                           sharex = True, sharey = True)
    
    for ax, data in zip(axs[0, :], simulations):
        
        for i, spec in zip(s_colour_index, species):
        
            ax.plot(data.t, data.y[spec,:].T, color = 'black', linewidth = 3.75)
            ax.plot(data.t, data.y[spec,:].T, color = cmap_s(i), linewidth = 3)
        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
    for ax, data in zip(axs[1, :], simulations):
        
        for i, res in zip(r_colour_index, resources):
        
            ax.plot(data.t, data.y[res,:].T, color = 'black', linewidth = 3.75)
            ax.plot(data.t, data.y[res,:].T, color = cmap_r(i), linewidth = 3)
        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    sns.despine()
    
    return fig, axs
    
##################################################################################################################################

# %%

def main():
    
    system_size = 250
    unscaled_sigma_range = np.array([1, 1.6])
    rho_range = [0.1, 0.99]
    mu_c = 150

    n = 12

    rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                        n)

    mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                 rho_sigma_combinations[1, :]/10,
                                 np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

    solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)
    
    #breakpoint()

    variable_parameters = np.vstack([solved_mu_gs,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     rho_sigma_combinations[1, :]/10])
    
    fixed_parameters = {'mu_c' : mu_c/system_size, 'm' : 1, 'K' : 1, 'gamma' : 1}

    ################################

    growth_consumption_function = 'growth function of consumption'

    subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect(larger system)"

    no_communities = 5

    find_infeasibility_supply_parameters(system_size, variable_parameters,
                                         ['mu_g', 'sigma_c', 'sigma_g'],
                                         fixed_parameters, growth_consumption_function,
                                         subdirectory, n, no_communities,
                                         filename_vars=['mu_g', 'sigma_c'],
                                         t_end = 7000)
    
if __name__ == '__main__':
    
    main()
    
# %%


'''
    ================================================================================
    Investigating the interplay between mu_c, correlation, and community feasibility
    ================================================================================
    
    ===========
    Simulations
    ===========
'''

system_size = 150
unscaled_sigma_range = np.array([1, 1.6])
rho_range = [0.1, 0.99]
mu_g = 1

n = 12


rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/np.sqrt(system_size),
                             np.repeat(mu_g, n**2), np.repeat(system_size, n**2)])

solved_mu_cs = np.apply_along_axis(solve_for_mu_c, axis = 0, arr = mu_solve_inputs)

variable_parameters = np.vstack([solved_mu_cs/system_size,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size)])

fixed_parameters = {'mu_g' : mu_g, 'm' : 1, 'K' : 1, 'gamma' : 1}

################################

growth_consumption_function = 'growth function of consumption'

subdirectory = "alternative_coupling_gc_infeasibility_interplay"

no_communities = 10

find_infeasibility_supply_parameters(system_size, variable_parameters,
                                     ['mu_c', 'sigma_c', 'sigma_g'],
                                     fixed_parameters, growth_consumption_function,
                                     subdirectory, n, no_communities)

# %%

#######################

'''
    With actual growth ineffiency (won't get that if mu_g = 1).
    I should also test when mu_g > 1, as growth efficiency will improve
'''

def growth_efficiency_infeasibility(mu_g, subsubdirectory):

    system_size = 150
    unscaled_sigma_range = np.array([1, 1.6])
    rho_range = [0.1, 0.99]
    
    n = 12
    
    rho_sigma_combinations_1 = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                          n)

    unscaled_sigma_range = np.array([1.6, 2.2])   
    rho_sigma_combinations_2 = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                          n)
    
    new_n = np.int32((n**2)/2)
    rho_sigma_combinations_2 = rho_sigma_combinations_2[:, n : new_n + n]
    
    rho_sigma_combinations = np.hstack((rho_sigma_combinations_1, rho_sigma_combinations_2))
    
    mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 np.repeat(mu_g, rho_sigma_combinations.shape[1]),
                                 np.repeat(system_size, rho_sigma_combinations.shape[1])])
    
    solved_mu_cs = np.apply_along_axis(solve_for_mu_c, axis = 0, arr = mu_solve_inputs)
    
    variable_parameters = np.vstack([solved_mu_cs/system_size,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size)])
    
    variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]
    
    fixed_parameters = {'mu_g' : mu_g, 'm' : 1, 'K' : 1, 'gamma' : 1}
    
    ################################
    
    growth_consumption_function = 'growth function of consumption'
    
    subdirectory = "alternative_coupling_gc_infeasibility_interplay" + subsubdirectory
    
    no_communities = 10
    
    find_infeasibility_supply_parameters(system_size, variable_parameters,
                                         ['mu_c', 'sigma_c', 'sigma_g'],
                                         fixed_parameters, growth_consumption_function,
                                         subdirectory, variable_parameters.shape[1],
                                         no_communities, t_end = 7000)

growth_efficiency_infeasibility(0.75, "/inefficient_growth")
growth_efficiency_infeasibility(1.25, "/efficient_growth")
growth_efficiency_infeasibility(0.5, "/extra_inefficient_growth")
growth_efficiency_infeasibility(1, "/equal_growth")

# %%

'''
    ==========
    Dataframes
    ==========
'''

system_size = 150

unscaled_sigma_range = np.array([1, 1.6])
additional_sigma_range = np.array([1.6, 2.2])
rho_range = [0.1, 0.99]

n = 12

def growth_efficiency_infeasibility_df(system_size, unscaled_sigma_range,
                                       rho_range, mu_g, n, subdirectory, do_round,
                                       additional_sigmas = None):
    
    if additional_sigmas is None:
        
        rho_sigma_combinations = sce.parameter_combinations([rho_range,
                                                             unscaled_sigma_range],
                                                            n)
        mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     np.repeat(mu_g, n**2),
                                     np.repeat(system_size, n**2)])
        
    else:

        rho_sigma_combinations_1 = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                              n)
    
        rho_sigma_combinations_2 = sce.parameter_combinations([rho_range, additional_sigmas],
                                                              n)
        
        new_n = np.int32((n**2)/2)
        rho_sigma_combinations_2 = rho_sigma_combinations_2[:, n : new_n + n]
        
        rho_sigma_combinations = np.hstack((rho_sigma_combinations_1, rho_sigma_combinations_2))
        
        mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     np.repeat(mu_g, rho_sigma_combinations.shape[1]),
                                     np.repeat(system_size, rho_sigma_combinations.shape[1])])
         
    solved_mu_cs = np.apply_along_axis(solve_for_mu_c, axis = 0, arr = mu_solve_inputs)
    
    variable_parameters = np.vstack([solved_mu_cs/system_size,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size)])
    
    variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]

    file_parameters = sce.variable_fixed_parameters(variable_parameters,
                                                    ['mu_c', 'sigma_c'],
                                                    {})

    df = pd.concat(infeasibility_df(system_size, subdirectory,
                                                     supplied_parameters = file_parameters,
                                                     do_round = do_round),
                                    axis = 0, ignore_index = True)

    df[['covariance', 'rho']] = \
        pd.DataFrame(df.apply(covariance_correlation, axis = 1,
                              scaling = 'asymmetric').to_list())
        
    df['instability distance'] = df.apply(distance_from_instability, axis = 1)
    df['infeasibility distance'] = df.apply(distance_from_infeasibility, axis = 1) 
    df['species packing 2'] = df.apply(species_packing, axis = 1) 
        
    df['sigma_c'] = np.round(df['sigma_c'], 6)
    df['rho'] = np.round(df['rho'], 6)
    
    return df 


subdirectories = ["alternative_coupling_gc_infeasibility_interplay/extra_inefficient_growth",
                  "alternative_coupling_gc_infeasibility_interplay/inefficient_growth",
                  "alternative_coupling_gc_infeasibility_interplay/equal_growth",
                  "alternative_coupling_gc_infeasibility_interplay/efficient_growth"]

mu_gs = [0.5, 0.75, 1, 1.25]

infsbl_grwth_effcncy = {str(mu_g) : growth_efficiency_infeasibility_df(system_size,
                                                                       unscaled_sigma_range,
                                                                       rho_range,
                                                                       mu_g, n, subdirectory,
                                                                       do_round = True,
                                                                       additional_sigmas = additional_sigma_range)
                        for mu_g, subdirectory in zip(mu_gs, subdirectories)}

# %%

'''
    ========
    Plotting
    ========
    
    Relationship betwwen mu_c, rho, and N_mean
    
'''
dfs = [df for df in infsbl_grwth_effcncy.values()]

fig, axs = generic_heatmaps_multiple_dfs_same_v(dfs, 'sigma_c', 'rho',
                                                'std. in the consumption rate, ' + \
                                                    r'$\frac{\sigma_c}{\sqrt{M}}$',
                                                'correlation between growth and\nconsumption, ' + r'$\rho$',
                                                'Max. lyapunov exponent', 
                                                'Proportion of simulations with\nmax. LE ' + r'$> 0.0025$',
                                                'Purples',
                                                [r'$\mu_g = 0.5$',
                                                 r'$\mu_g = 0.75$',
                                                 r'$\mu_g = 1$',
                                                 r'$\mu_g = 1.25$'], (2, 2),
                                                (9, 6),
                                                pivot_function = le_pivot,
                                                specify_min_max = [0,1])

fig.suptitle('Average increase in yield per unit resource consumed (growth efficiency)',
             fontsize = 16, weight = 'bold')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_maxle.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_maxle.svg",
            bbox_inches='tight')

del dfs

# %%

dfs = [df for df in infsbl_grwth_effcncy.values()]

fig, axs = generic_heatmaps_multiple_dfs_same_v([dfs[1], dfs[3]], 'sigma_c', 'rho',
                                                'std. in the consumption rate, ' + \
                                                    r'$\frac{\sigma_c}{\sqrt{M}}$',
                                                'correlation between growth and\nconsumption, ' + r'$\rho$',
                                                'Max. lyapunov exponent', 
                                                'Proportion of simulations with\nmax. LE ' + r'$> 0.0025$',
                                                'Purples',
                                                ['low resource efficiency, ' + \
                                                 r'$\mu_g = 0.75$',
                                                 'high resource efficiency, ' + \
                                                 r'$\mu_g = 1.25$'], (2, 1),
                                                (6, 7),
                                                pivot_function = le_pivot,
                                                specify_min_max = [0,1])

fig.suptitle("I'm not sure, something about the trade off between correlation and species packing",
             fontsize = 16, weight = 'bold')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_maxle_2.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_maxle_2.svg",
            bbox_inches='tight')

del dfs

# %%

dfs = [df for df in infsbl_grwth_effcncy.values()]

vmin = np.min(np.concatenate([df.groupby(['rho', 'sigma_c'])['instability distance'].mean()
               for df in dfs]))
vmax = np.max(np.concatenate([df.groupby(['rho', 'sigma_c'])['instability distance'].mean()
               for df in dfs]))
norm = TwoSlopeNorm(vcenter = 0, vmin = vmin, vmax = vmax)


fig, axs = generic_heatmaps_multiple_dfs_same_v(dfs, 'sigma_c', 'rho',
                                                'std. in the consumption rate, ' + \
                                                    r'$\frac{\sigma_c}{M}$',
                                                'correlation between growth and\nconsumption, ' + r'$\rho$',
                                                'instability distance', 
                                                r'$\rho^2 - \frac{\phi_N \gamma^{-1}}{\phi_R}$',
                                                'RdBu',
                                                [r'$\mu_g = 0.5$',
                                                 r'$\mu_g = 0.75$',
                                                 r'$\mu_g = 1$',
                                                 r'$\mu_g = 1.25$'], (2, 2),
                                                (9, 6),
                                                pivot_function = di_pivot,
                                                norm = norm)


del dfs

# %%

dfs = [df for df in infsbl_grwth_effcncy.values()]

fig, axs = generic_heatmaps_multiple_dfs_same_v(dfs, 'sigma_c', 'rho',
                                                'std. in the consumption rate,' + r'$\sigma_c$',
                                                'correlation between growth and\nconsumption, ' + r'$\rho$',
                                                'N_mean', 
                                                'Average species abundance, ' + \
                                                r'$\langle N \rangle$',
                                                'viridis_r',
                                                [r'$\mu_g = 0.5$',
                                                 r'$\mu_g = 0.75$',
                                                 r'$\mu_g = 1$',
                                                 r'$\mu_g = 1.25$'], (2, 2),
                                                (9, 6),
                                                pivot_function = N_pivot,
                                                norm = PowerNorm(gamma = 0.5))

fig.suptitle('Average increase in yield per unit resource consumed (growth efficiency)',
             fontsize = 16, weight = 'bold')
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_powernorm.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_powernorm.svg",
            bbox_inches='tight')

v_min = np.min([np.min(df['N_mean']) for df in infsbl_grwth_effcncy.values()])
v_max = np.max([np.max(df['N_mean']) for df in infsbl_grwth_effcncy.values()])

fig, axs = generic_heatmaps_multiple_dfs_same_v(dfs, 'sigma_c', 'rho',
                                                'std. in the consumption rate,' + r'$\sigma_c$',
                                                'correlation between growth and\nconsumption, ' + r'$\rho$',
                                                'N_mean', 
                                                'Average species abundance, ' + \
                                                r'$\langle N \rangle$',
                                                'viridis_r',
                                                [r'$\mu_g = 0.5$',
                                                 r'$\mu_g = 0.75$',
                                                 r'$\mu_g = 1$',
                                                 r'$\mu_g = 1.25$'], (2, 2),
                                                (9, 6),
                                                pivot_function = N_pivot,
                                                norm = LogNorm(vmin = v_min,
                                                               vmax = v_max))

fig.suptitle('Average increase in yield per unit resource consumed (growth efficiency)',
             fontsize = 16, weight = 'bold')
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_lognorm.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_powernorm.svg",
            bbox_inches='tight')

del dfs

# %%

'''
    ========
    Dynamics
    ========
'''

def growth_efficiency_dynamics(subdirectory, df, approx_rho, approx_sigma):
    
    rho_distance = np.abs(np.round(df['rho'], 6) - approx_rho)
    sigma_distance = np.abs(np.round(df['sigma_c'], 6) - approx_sigma)
    
    index = np.where((rho_distance == rho_distance.min()) & 
                     (sigma_distance == sigma_distance.min()))[0][0]
    
    mu_c_str = str(np.round(df.iloc[index]['mu_c'], 3))
    sigma_c_str = str(np.round(df.iloc[index]['sigma_c'], 3))
    
    communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/alternative_coupling_gc_infeasibility_interplay/" + \
                                 subdirectory + "/infeasibility/CR_self_limiting_"+ \
                                 mu_c_str + "_" + sigma_c_str + ".pkl")
        
    return communities

# %%

subdirectories = ['extra_inefficient_growth', 'inefficient_growth',
                  'equal_growth', 'efficient_growth']

plot_sigma = 0.125
plot_rhos = [0.1, 0.35, 0.9]
plot_combos = [[rho, plot_sigma] for rho in plot_rhos]

selected_parameter_spaces = {mu_g : plot_combos 
                             for mu_g in ['0.5', '0.75', '1', '1.25']}

communities_dynamics = {key : [growth_efficiency_dynamics(subdirect,
                                                          infsbl_grwth_effcncy[key],
                                                          rho_sigma[0],
                                                          rho_sigma[1])
                               for rho_sigma in rho_sigmas]
                        for subdirect, (key, rho_sigmas) in zip(subdirectories,
                                                                selected_parameter_spaces.items())}

for key, communities in communities_dynamics.items():

    plot_dynamics([communities[0][0].ODE_sols['lineage 0'],
                   communities[1][0].ODE_sols['lineage 0'],
                   communities[2][0].ODE_sols['lineage 0']])
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_dynamics_" + \
                key + ".png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_dynamics_" + \
                key + ".svg",
                bbox_inches='tight')
        
# %%

subdirectories = ['inefficient_growth', 'efficient_growth']

plot_sigma = 0.13
plot_rhos = [0.15, 0.4, 0.9]
plot_combos = [[rho, plot_sigma] for rho in plot_rhos]

selected_parameter_spaces = {mu_g : plot_combos 
                             for mu_g in ['0.75', '1.25']}

communities_dynamics = {key : [growth_efficiency_dynamics(subdirect,
                                                          infsbl_grwth_effcncy[key],
                                                          rho_sigma[0],
                                                          rho_sigma[1])
                               for rho_sigma in rho_sigmas]
                        for subdirect, (key, rho_sigmas) in zip(subdirectories,
                                                                selected_parameter_spaces.items())}

for key, communities in communities_dynamics.items():

    plot_dynamics([communities[0][0].ODE_sols['lineage 0'],
                   communities[1][0].ODE_sols['lineage 0'],
                   communities[2][0].ODE_sols['lineage 0']])
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_dynamics_" + \
                key + "_2.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_infeasibility_dynamics_" + \
                key + "_2.svg",
                bbox_inches='tight')
        
# %%

df = infsbl_grwth_effcncy['0.75'].iloc[np.where(np.round(infsbl_grwth_effcncy['0.75']['sigma_c'], 3) == 0.157)]

sns.lineplot(x = df['mu_c'], y = df['rho']**2)
sns.lineplot(df, x = 'mu_c', y = 'species packing 2')
sns.lineplot(df, x = 'mu_c', y = 'instability distance')

plt.xlim([0, 5])

del df

# %%

df = infsbl_grwth_effcncy['1.25'].iloc[np.where(np.round(infsbl_grwth_effcncy['1.25']['sigma_c'], 3) == 0.157)]

sns.lineplot(x = df['mu_c'], y = df['rho']**2)
sns.lineplot(df, x = 'mu_c', y = 'species packing 2')
sns.lineplot(df, x = 'mu_c', y = 'instability distance')

plt.xlim([0, 5])

del df

# %%

'''
    ===========================================================================================================
    Comparing the emergence of infeaisble dynamics with rho in the C-R model with linear parameter combinations
    ===========================================================================================================

    ===========
    Simulations
    ===========
'''

system_size = 150
n = 12

linear_rho_sigma_combinations = sce.parameter_combinations([[0.1, 0.99], [1, 3.5]],
                                                          n)
variable_parameters = np.vstack([linear_rho_sigma_combinations[0, :],
                                 linear_rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 linear_rho_sigma_combinations[1, :]/np.sqrt(system_size)])

fixed_parameters = {'mu_c': 200/system_size, 'mu_g' : 200/system_size, 'm' : 1,
                    'K' : 1, 'gamma' : 1}

growth_consumption_function = 'coupled by rho'

subdirectory = "alternative_coupling_gc_infeasibility_interplay/linear"

no_communities = 10

find_infeasibility_supply_parameters(variable_parameters, ['rho', 'sigma_c', 'sigma_g'],
                                     fixed_parameters, growth_consumption_function,
                                     subdirectory, n, no_communities,
                                     filename_vars = ['rho', 'sigma_c'])

# %%

'''
    ==========
    Dataframes
    ==========
'''

system_size = 150
subdirectory = "alternative_coupling_gc_infeasibility_interplay/linear"

n = 12

###################

linear_rho_sigma_combinations = sce.parameter_combinations([[0.1, 0.99], [1, 3.5]],
                                                          n)
variable_parameters = np.vstack([linear_rho_sigma_combinations[0, :],
                                 linear_rho_sigma_combinations[1, :]/np.sqrt(system_size)])

file_parameters = sce.variable_fixed_parameters(variable_parameters,
                                                ['rho', 'sigma_c'],
                                           {})

linear_infeasibility_df = pd.concat(infeasibility_df(system_size, subdirectory,
                                                 supplied_parameters = file_parameters,
                                                 parm_attributes = ['no_species',
                                                                    'no_resources',
                                                                    'rho',
                                                                    'mu_c',
                                                                    'sigma_c',
                                                                    'mu_g',
                                                                    'sigma_g',
                                                                    'm',
                                                                    'K'],
                                                 do_round = False),
                                axis = 0, ignore_index = True)

linear_infeasibility_df['sigma_c'] = np.round(linear_infeasibility_df['sigma_c'],
                                              6)
linear_infeasibility_df['rho'] = np.round(linear_infeasibility_df['rho'], 6)
linear_infeasibility_df['instability distance'] = \
    linear_infeasibility_df.apply(distance_from_instability, axis = 1)
linear_infeasibility_df['infeasibility distance'] = \
    linear_infeasibility_df.apply(distance_from_infeasibility, axis = 1)

# %%

other_mu_sigma = file_parameters[-1]
other_community_check = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/alternative_coupling_gc_infeasibility_interplay/linear/infeasibility/CR_self_limiting_" + \
                                        str(other_mu_sigma['rho']) + "_" + \
                                        str(other_mu_sigma['sigma_c']) + ".pkl")
    
plt.plot(other_community_check[3].ODE_sols['lineage 0'].t,
         other_community_check[3].ODE_sols['lineage 0'].y[:system_size, :].T)


# %%

'''
    ========
    Plotting
    ========
    
    Relationship betwwen mu_c, rho, and N_mean
    
'''


df_plot = copy(linear_infeasibility_df)
df_plot['sigma_c'] *= np.sqrt(df_plot['no_resources'])

fig, axs = generic_heatmaps(df_plot,
                            'sigma_c', 'rho', r'$\sigma_c$',  r'$\rho$',
                            ['Max. lyapunov exponent'], 'Purples',
                            '',
                            (1, 1), (5, 4),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            cbar_kws = {'label': r'$ \langle N \rangle $'})

axs.figure.axes[-1].yaxis.label.set_size(14)

del df_plot

# %%

df_plot = copy(linear_infeasibility_df)
df_plot['sigma_c'] *= np.sqrt(df_plot['no_resources'])

plot_max = np.max(df_plot['infeasibility distance'])

fig, axs = generic_heatmaps(df_plot,
                            'sigma_c', 'rho', r'$\sigma_c$',  r'$\rho$',
                            ['infeasibility distance'], 'Reds',
                            '',
                            (1, 1), (5, 4),
                            pivot_functions = {'infeasibility distance' : infeasible_pivot},
                            specify_min_max = {'infeasibility distance' :[0, plot_max]})

del df_plot

# %%


'''
    ================================================================================
    Investigating the interplay between mu_g, correlation, and community feasibility
    ================================================================================
    
    ===========
    Simulations
    ===========
'''

system_size = 150
unscaled_sigma_range = np.array([1, 1.6])
rho_range = [0.1, 0.99]
mu_c = 150

n = 12


rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/np.sqrt(system_size),
                             np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)

variable_parameters = np.vstack([solved_mu_gs,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size)])

fixed_parameters = {'mu_c' : mu_c/system_size, 'm' : 1, 'K' : 1, 'gamma' : 1}

################################

growth_consumption_function = 'growth function of consumption'

subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect"

no_communities = 10

find_infeasibility_supply_parameters(system_size, variable_parameters,
                                     ['mu_g', 'sigma_c', 'sigma_g'],
                                     fixed_parameters, growth_consumption_function,
                                     subdirectory, n, no_communities,
                                     filename_vars=['mu_g', 'sigma_c'],
                                     t_end = 7000)

# %%

system_size = 150
unscaled_sigma_range = np.array([1, 1.6])
rho_range = [0.1, 0.99]
mu_c = 150
subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect"
n = 12

rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/np.sqrt(system_size),
                             np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)

variable_parameters = np.vstack([solved_mu_gs,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size)])

variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]

file_parameters = sce.variable_fixed_parameters(variable_parameters,
                                                ['mu_g', 'sigma_c'],
                                                {})

infeasibility_mu_g_df = pd.concat(infeasibility_df(system_size, subdirectory,
                                                   supplied_parameters = file_parameters,
                                                   do_round = True),
                                  axis = 0, ignore_index = True)

infeasibility_mu_g_df[['covariance', 'rho']] = \
    pd.DataFrame(infeasibility_mu_g_df.apply(covariance_correlation, axis = 1,
                                             scaling = 'asymmetric').to_list())
    
infeasibility_mu_g_df['sigma_c'] = np.round(infeasibility_mu_g_df['sigma_c'], 6)
infeasibility_mu_g_df['rho'] = np.round(infeasibility_mu_g_df['rho'], 6)

infeasibility_mu_g_df['instability distance'] = \
    infeasibility_mu_g_df.apply(distance_from_instability, axis = 1)
    
infeasibility_mu_g_df['infeasibility distance'] = \
    infeasibility_mu_g_df.apply(distance_from_infeasibility, axis = 1)
    
infeasibility_mu_g_df['species packing 2'] = \
    infeasibility_mu_g_df.apply(species_packing, axis = 1)
   
# %%

'''
    mu_g vs rho vs stability
'''

df_plot = copy(infeasibility_mu_g_df)
#df_plot['sigma_c'] *= np.sqrt(df_plot['no_resources'])

fig, axs = generic_heatmaps(df_plot,
                            'sigma_c', 'rho',
                            'std. in the consumption rate, ' + \
                                r'$\frac{\sigma_c}{\sqrt{M}}$',
                            'correlation between growth and\nconsumption, ' + r'$\rho$',
                            ['Max. lyapunov exponent'], 'Purples',
                            'Altering the correlation between growth and\nconsumption through avg. resource efficiency ' + \
                                r'$(\mu_g)$' + '\ndrives weak phase transitions.',
                            (1, 1), (6, 5),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max = {'Max. lyapunov exponent' : [0, 1]})
    
cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with\nmax. LE ' + r'$> 0.0025$',
               size = '16')
cbar.ax.tick_params(labelsize = 12)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_rho_le.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_rho_le.svg",
            bbox_inches='tight')

del df_plot

# %%

'''
    ========
    Dynamics
    ========
'''

def growth_efficiency_dynamics_2(df, approx_rho, approx_sigma):
    
    rho_distance = np.abs(np.round(df['rho'], 6) - approx_rho)
    sigma_distance = np.abs(np.round(df['sigma_c'], 6) - approx_sigma)
    
    index = np.where((rho_distance == rho_distance.min()) & 
                     (sigma_distance == sigma_distance.min()))[0][0]
    
    mu_g_str = str(np.round(df.iloc[index]['mu_g'], 3))
    sigma_c_str = str(np.round(df.iloc[index]['sigma_c'], 3))
    
    communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/alternative_coupling_gc_infeasibility_interplay/mu_g_effect/infeasibility/CR_self_limiting_" + \
                                 mu_g_str + "_" + sigma_c_str + ".pkl")
        
    return communities

plot_sigma = 0.13
plot_rhos = [0.35, 0.9]
rho_sigmas = [[rho, plot_sigma] for rho in plot_rhos]

communities_dynamics = [growth_efficiency_dynamics_2(infeasibility_mu_g_df,
                                                     rho_sigma[0],
                                                     rho_sigma[1])
                               for rho_sigma in rho_sigmas]

# %%

focal_community = communities_dynamics[0][9]
no_simulations, batch_size = 1000, 10
no_iterations = np.int32(no_simulations/batch_size)

directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/alternative_coupling_gc_infeasibility_interplay/mu_g_effect/infeasibility/ms_check"
if not os.path.exists(directory): os.makedirs(directory)
  
for i in tqdm(range(no_iterations)):
    
    temp_community = copy(focal_community)
    temp_community.simulate_community(np.arange(batch_size), 7000,
                                      model_version = 'self-limiting resource supply',
                                      assign = True)
    
    pickle_dump(directory + "/batch_" + str(i) + ".pkl", temp_community)
        
# %%

def final_composition(i, directory):
    
    batch = pd.read_pickle(directory + "/batch_" + str(i) + ".pkl")
    batch.calculate_community_properties(np.arange(len(batch.ODE_sols)), 6500)
    
    breakpoint()
    
    stable_survivors = np.array([np.argwhere(sol.y[:batch.no_species, -1] > 1e-6) 
                                 for lineage, sol in batch.ODE_sols.items() 
                                 if batch.species_volatility[lineage] < 0.1])
    
    return stable_survivors

all_stable_survivors = np.concatenate([final_composition(i, directory)
                                       for i in range(no_iterations)])

# %%

plot_dynamics([communities_dynamics[0][0].ODE_sols['lineage 0'],
               communities_dynamics[0][1].ODE_sols['lineage 0'],
               communities_dynamics[1][0].ODE_sols['lineage 0']])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_rho.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_rho.svg",
            bbox_inches='tight')
                            
# %%

df = infeasibility_mu_g_df.iloc[np.where(np.round(infeasibility_mu_g_df['sigma_c'], 3) == 0.131)]

sns.lineplot(x = df['mu_g'], y = df['rho']**2)
sns.lineplot(df, x = 'mu_g', y = 'species packing 2')
sns.lineplot(df, x = 'mu_g', y = 'instability distance')

#plt.xlim([0, 5])

del df      
                        
# %%


'''
    ================================================================================
    Investigating the interplay between mu_g, correlation, and community feasibility
    ================================================================================
    
    ===========
    Simulations
    ===========
'''

system_size = 250
unscaled_sigma_range = np.array([1, 1.6])
rho_range = [0.1, 0.99]
mu_c = 150

n = 12

rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/10,
                             np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)

#breakpoint()

variable_parameters = np.vstack([solved_mu_gs,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 rho_sigma_combinations[1, :]/10])

fixed_parameters = {'mu_c' : mu_c/system_size, 'm' : 1, 'K' : 1, 'gamma' : 1}

################################

growth_consumption_function = 'growth function of consumption'

subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect(larger system)"

no_communities = 5

# %%

system_size = 250
unscaled_sigma_range = np.array([1, 1.6])
rho_range = [0.1, 0.99]
mu_c = 150
subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect(larger system)"

n = 12

rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/10,
                             np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)

variable_parameters = np.vstack([solved_mu_gs,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size)])

variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]

file_parameters = sce.variable_fixed_parameters(variable_parameters,
                                                ['mu_g', 'sigma_c'],
                                                {})

infeasibility_mu_g_df = pd.concat(infeasibility_df(system_size, subdirectory,
                                                   supplied_parameters = file_parameters,
                                                   do_round = True),
                                  axis = 0, ignore_index = True)

infeasibility_mu_g_df[['covariance', 'rho']] = \
    pd.DataFrame(infeasibility_mu_g_df.apply(covariance_correlation, axis = 1,
                                             scaling = 'asymmetric').to_list())
    
infeasibility_mu_g_df['sigma_c'] = np.round(infeasibility_mu_g_df['sigma_c'], 6)
infeasibility_mu_g_df['rho'] = np.round(infeasibility_mu_g_df['rho'], 6)

infeasibility_mu_g_df['instability distance'] = \
    infeasibility_mu_g_df.apply(distance_from_instability, axis = 1)
    
infeasibility_mu_g_df['infeasibility distance'] = \
    infeasibility_mu_g_df.apply(distance_from_infeasibility, axis = 1)
    
# %%

df_plot = copy(infeasibility_mu_g_df)
df_plot['sigma_c'] *= np.sqrt(df_plot['no_resources'])

fig, axs = generic_heatmaps(df_plot,
                            'sigma_c', 'rho', r'$\sigma_c$',  r'$\rho$',
                            ['Max. lyapunov exponent'], 'Purples',
                            '',
                            (1, 1), (5, 4),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max = {'Max. lyapunov exponent' : [0, 1]})

del df_plot

# %%

'''
    mu_g vs rho vs stability
'''

df_plot = copy(infeasibility_mu_g_df)
df_plot['sigma_c'] *= np.sqrt(df_plot['no_resources'])

vmin = np.min(infeasibility_mu_g_df.groupby(['rho', 'sigma_c'])['instability distance'].mean())
vmax = np.max(infeasibility_mu_g_df.groupby(['rho', 'sigma_c'])['instability distance'].mean())
norm = TwoSlopeNorm(vcenter = 0, vmin = vmin, vmax = vmax)

fig, axs = generic_heatmaps(df_plot,
                            'sigma_c', 'rho', r'$\sigma_c$',  r'$\rho$',
                            ['instability distance'], 'RdBu',
                            '',
                            (1, 1), (5, 4),
                            pivot_functions = {'instability distance' : di_pivot},
                            norm = norm)

del df_plot