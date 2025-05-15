# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:37:40 2025

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:25:57 2025

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

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling/" + \
             "trade_offs_community_properties_stability")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling")
from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations, prop_chaotic, distance_from_instability, \
    distance_from_infeasibility, species_packing, pickle_dump

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')

# %%

'''
==============================================================================
                                Functions
==============================================================================
'''

def solve_for_mu_g(x):
    
    rho, sigma_c, sigma_g, mu_c, M = x
        
    solved_mu_g = (sigma_g/sigma_c)*np.sqrt(((rho**2)/(1 - rho**2)) * ((mu_c**2)/M + sigma_c**2))
    
    return solved_mu_g

# %%

def solve_for_mu_c(x):
    
    rho, sigma_c, sigma_g, mu_g, M = x
    
    solved_mu_c = ((sigma_c * np.sqrt(M))/sigma_g) * np.sqrt((mu_g**2)*(1/(rho**2) - 1) - sigma_g**2)
    
    return solved_mu_c

# %% 

def generate_parameters(system_size, rho_range, unscaled_sigma_range, n,
                        vary_rho_by, mu):
    
    # create all combinations of rho and sigma with range
    rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                        n)
    
    mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 np.repeat(mu, n**2), np.repeat(system_size, n**2)])
    
    match vary_rho_by:
        
        case 'mu_g':
            
            v_parm_names = ['mu_g', 'sigma_c']
            
            # solve for mu_g using mu_c, sigma_c, sigma_g and rho 
            solved_mus = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)
            
        case 'mu_c':
            
            v_parm_names = ['mu_c', 'sigma_c']
            
            # solve for mu_c using mu_g, sigma_c, sigma_g and rho 
            solved_mus = np.apply_along_axis(solve_for_mu_c, axis = 0, arr = mu_solve_inputs)
            
    # get variable parameters (mu_g, sigma_c (which sigma_g scales with))
    variable_parameters = np.vstack([solved_mus,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size)])
    # remove any nan values (which could occur in the solve as some values blow up to inf)
    variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]
    
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               v_parm_names, {})
    
    return parameters
        
# %%

def generate_df(directory, system_size, rho_range, sigma_range, n, vary_rho_by, mu):
    
    file_parameters = generate_parameters(system_size, rho_range, sigma_range,
                                          n, vary_rho_by, mu)

    df = pd.concat(community_properties_df(directory, file_parameters),
                   axis = 0, ignore_index = True)
    
    df[['covariance', 'rho']] = pd.DataFrame(df.apply(covariance_correlation,
                                                      axis = 1).to_list())

    # calculate the stability metric (rho^2 - phi_N/(gamma * phi_R)) from the 
    #   cavity solution
    df['instability distance'] = df.apply(distance_from_instability, axis = 1)

    # calcualte the infeasibily metric (phi_R - phi_N/gamma) from the cavity solution
    df['infeasibility distance'] = df.apply(distance_from_infeasibility, axis = 1)

    # calculate the species packing ratio, phi_N/(gamma * phi_R)
    df['species packing 2'] = df.apply(species_packing, axis = 1)
    
    for var in ['rho', 'mu_g', 'sigma_c', 'sigma_g']:
        
        df[var] = np.round(df[var], 6)
    
    return df

# %%

def community_properties_df(directory, parameters,
                            parm_attributes = ['no_species', 'no_resources',
                                               'mu_c', 'sigma_c', 'mu_g',
                                               'sigma_g', 'm', 'K']):
    

    directory_prefix = directory + "/CR_self_limiting_"
    
    key1, key2 = parameters[0].keys()

    dfs = [create_df_and_delete_simulations(directory_prefix + \
                                            str(np.round(parameter_set[key1], 3)) + \
                                            "_" + str(np.round(parameter_set[key2], 3)),
                                            parm_attributes)
           for parameter_set in parameters]
            
    return dfs

# %%

def covariance_correlation(df):
    
    M = df['no_resources']
        
    mu_c, mu_g = df['mu_c']*M, df['mu_g']
    
    sigma_c, sigma_g = df['sigma_c']*np.sqrt(M), df['sigma_g']
    
    denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2)/M + (sigma_c * sigma_g)**2
         
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

def generic_heatmaps_multiple_dfs_same_v(dfs, x, y, xlabel, ylabel, variable,
                                         cbar_label, cmap, titles,
                                         fig_dims, figsize,
                                         pivot_function = None, is_logged = None, 
                                         specify_min_max = None,
                                         **kwargs):
    
    if pivot_function is None:
    
        pivot_tables = [df.pivot(index = x, columns = y, values = variable)
                        for df in dfs]
        
    else:
                
        pivot_tables = [pivot_function(df, index = y, columns = x,
                                       values = variable)[0] 
                        for df in dfs]
    
    if is_logged:
    
        pivot_tables = [np.log10(np.abs(pivot_table)) 
                        for pivot_table in pivot_tables]
        
    if specify_min_max:
        
        v_min_max = specify_min_max
        
    else:
        
        v_min_max = [np.min([np.min(pivot_table) for pivot_table in pivot_tables]),
                     np.max([np.max(pivot_table) for pivot_table in pivot_tables])]
    
    sns.set_style('white')
    
    fig, axs = plt.subplots(fig_dims[0], fig_dims[1], figsize = figsize,
                            layout = 'constrained')

    fig.supxlabel(xlabel, fontsize = 16, weight = 'bold')
    fig.supylabel(ylabel, fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')

    for i, (ax, df, pivot_table, title) in enumerate(zip(axs.flatten(), dfs,
                                                        pivot_tables, titles)):
        
        xtick_position = len(np.unique(df[x]))
        xtick_vals = [np.round(np.min(df[x]), 3), np.round(np.max(df[x]), 3)]
        
        ytick_position = len(np.unique(df[y]))
        ytick_vals = [np.round(np.min(df[y]), 3), np.round(np.max(df[y]), 3)]
        
        top_border, right_border = pivot_table.shape[0], pivot_table.shape[1]
    
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

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

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

# %%

'''
==============================================================================
                                    Main
==============================================================================
'''

def rho_controlled_by_mu_g_df():
    
    df_gc_c = generate_df("/growth_coupled_to_consumption",
                          'growth function of consumption', system_size,
                          mu_c_range_gc_c, unscaled_sigma_range, n)

    df_g_cg = generate_df("/consumption_coupled_to_growth",
                          'consumption function of growth', system_size,
                          mu_c_range_g_cg, unscaled_sigma_range, n)


    '''
        Vary rho through varying mu_g, create parameter combinations of rho and sigma
    '''
    
    # where the simulation data will be saved
    subdirectory = "alternative_coupling_gc_infeasibility_interplay/mu_g_effect"
    
    # no_resources (= no_species)
    system_size = 150
    
    # range of sigma_c (also controls sigma_g: sigma_g = sigma_c/sqrt(system_size))
    unscaled_sigma_range = np.array([1, 1.6])
    
    # range of rhos/correlations between growth and consumption
    rho_range = [0.1, 0.99]
    
    # mu_c is fixed
    mu_c = 150
    
    # no. parameters combinations = n^2
    n = 12
    
    #######################
    
    # create all combinations of rho and sigma with range
    rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                        n)
    
    # solve for mu_g using mu_c, sigma_c, sigma_g and rho 
    mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])
    solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)
    
    # get variable parameters (mu_g, sigma_c (which sigma_g scales with))
    variable_parameters = np.vstack([solved_mu_gs,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size)])
    # remove any nan values (which could occur in the solve as some values blow up to inf)
    variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]
    
    '''
        Create dataframe
    '''
    
    # get parameters used to name the files containing simulations
    variable_parameters = variable_parameters[:2, :]
    file_parameters = sce.variable_fixed_parameters(variable_parameters,
                                                    ['mu_g', 'sigma_c'],
                                                    {})
    
    # Generate the data frame
    infeasibility_mu_g_df = pd.concat(infeasibility_df(system_size, subdirectory,
                                                       supplied_parameters = file_parameters,
                                                       do_round = True),
                                      axis = 0, ignore_index = True)
    
    # calculate the covariance and correlation between growth and consumption
    infeasibility_mu_g_df[['covariance', 'rho']] = \
        pd.DataFrame(infeasibility_mu_g_df.apply(covariance_correlation,
                                                 axis = 1).to_list())
    
    # calculate the stability metric (rho^2 - phi_N/(gamma * phi_R)) from the 
    #   cavity solution
    infeasibility_mu_g_df['instability distance'] = \
        infeasibility_mu_g_df.apply(distance_from_instability, axis = 1)
    
    # calcualte the infeasibily metric (phi_R - phi_N/gamma) from the cavity solution
    infeasibility_mu_g_df['infeasibility distance'] = \
        infeasibility_mu_g_df.apply(distance_from_infeasibility, axis = 1)
    
    # calculate the species packing ratio, phi_N/(gamma * phi_R)
    infeasibility_mu_g_df['species packing 2'] = \
        infeasibility_mu_g_df.apply(species_packing, axis = 1)
    
    # round values for plotting    
    for var in ['rho', 'mu_g', 'sigma_c', 'sigma_g']:
        
        infeasibility_mu_g_df[var] = np.round(infeasibility_mu_g_df[var], 6)
        
    return ...
    