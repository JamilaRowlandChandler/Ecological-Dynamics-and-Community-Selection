# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 23:52:57 2025

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
             "comparing_phase_transitions_across_models")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling")
from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations, prop_chaotic, distance_from_instability, \
    distance_from_infeasibility, species_packing, pickle_dump

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %% 

def generate_parameters(growth_consumption_function,
                        no_resources, unscaled_mu_c_range, unscaled_sigma_range,
                        fixed_parameters, n,
                        v_parm_names = ['mu_c', 'sigma_c', 'sigma_g']):
    
    fixed_parameters_copy = copy(fixed_parameters)
    
    match growth_consumption_function:
        
        case 'growth function of consumption': 
            
            mu_c_range = unscaled_mu_c_range/no_resources
            
        case 'consumption function of growth':
            
            mu_c_range = unscaled_mu_c_range
            
            if 'mu_g' in fixed_parameters_copy.keys():
                
                fixed_parameters_copy['mu_g'] /= no_resources
            
    sigma_range = unscaled_sigma_range/np.sqrt(no_resources)
                            
    # generate n values of rho and sigma within range
    mu_c_sigma_combinations = np.unique(sce.parameter_combinations([mu_c_range,
                                                                    sigma_range],
                                                                    n), axis = 1)
   
    # array of variable parameter combinations
    variable_parameters = np.vstack([mu_c_sigma_combinations,
                                     mu_c_sigma_combinations[1, :]])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               v_parm_names,
                                               fixed_parameters_copy)
    
    return parameters

# %%

def dynamics(system_size, parameters, growth_consumption_function, subdirectory,
             no_communities, filename_vars, **kwargs): 
    
    '''
    =======================
    Create folder
    =======================
    '''
    
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + subdirectory
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
        
    '''
    =====================
    Community dynamics
    =====================
    
    '''
    
    for parameter_set in tqdm(parameters, position = 0, leave = True):

        filename_CR = subdirectory + "/CR_self_limiting_" + \
                        str(np.round(parameter_set[filename_vars[0]], 3)) + "_" + \
                            str(np.round(parameter_set[filename_vars[1]], 3))
        
        create_and_delete_CR(filename_CR, system_size, system_size, parameter_set,
                             no_communities = no_communities,
                             growth_consumption_function = growth_consumption_function,
                             **kwargs)
        
# %%

def generate_parameters_simulate_dynamics(system_size, unscaled_mu_c_range,
                                          unscaled_sigma_range,
                                          fixed_parameters, growth_consumption_function,
                                          subdirectory, n, no_communities,
                                          filename_vars = ['mu_c', 'sigma_c'],
                                          **kwargs):
    
    parameters = generate_parameters(growth_consumption_function, system_size,
                                     unscaled_mu_c_range,
                                     unscaled_sigma_range, fixed_parameters, n)
    
    dynamics(system_size, parameters, growth_consumption_function,
             subdirectory, no_communities, filename_vars,
             **kwargs)
    
# %%

def generate_df(directory, growth_consumption_function, system_size,
                mu_c_range, sigma_range, n):
    
    file_parameters = generate_parameters(growth_consumption_function,
                                          system_size, mu_c_range, sigma_range,
                                          {}, n, 
                                          v_parm_names = ['mu_c', 'sigma_c'])

    df = pd.concat(community_properties_df("phase_transtions_between_models" + \
                                           directory,
                                           file_parameters),
                   axis = 0, ignore_index = True)
    
    df[['covariance', 'rho']] = pd.DataFrame(df.apply(covariance_correlation, axis = 1,
                                                      growth_consumption_function = \
                                                          growth_consumption_function).to_list())

    # calculate the stability metric (rho^2 - phi_N/(gamma * phi_R)) from the 
    #   cavity solution
    df['instability distance'] = df.apply(distance_from_instability, axis = 1)

    # calcualte the infeasibily metric (phi_R - phi_N/gamma) from the cavity solution
    df['infeasibility distance'] = df.apply(distance_from_infeasibility, axis = 1)

    # calculate the species packing ratio, phi_N/(gamma * phi_R)
    df['species packing 2'] = df.apply(species_packing, axis = 1)
    
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

def covariance_correlation(df, growth_consumption_function):
    
    M = df['no_resources']
    
    match growth_consumption_function:
        
        case 'growth function of consumption': 
            
            mu_c, mu_g = df['mu_c']*M, df['mu_g']

            sigma_c, sigma_g = df['sigma_c']*np.sqrt(M), df['sigma_g']
            
            denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2)/M + (sigma_c * sigma_g)**2
                 
            covariance = (mu_g * sigma_c**2)
            correlation = (mu_g * sigma_c)/np.sqrt(denominator)
            
        
        case 'consumption function of growth':
            
            mu_c, mu_g = df['mu_c'], df['mu_g']*M
            
            sigma_c, sigma_g = df['sigma_c'], df['sigma_g']*np.sqrt(M)
            
            denominator = ((mu_g * sigma_c)**2)/M + (mu_c * sigma_g)**2 + (sigma_c * sigma_g)**2
                 
            covariance = (mu_c * sigma_g**2)
            correlation = (mu_c * sigma_g)/np.sqrt(denominator)
            
    return covariance, correlation

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

def main():
    
    ######################################################################
    
    def add_sigmas(): 
    
        system_size = 150
        unscaled_sigma_range = np.array([1, 1.6])
        higher_directory = "phase_transtions_between_models"
        n = 15
        no_communities = 10
        
        mu_start = np.array([-100, 100])
        mu_c_base = np.linspace(mu_start[0], mu_start[1], n)[-4 :]/system_size
        
        fixed_parameters_gc_c  = {'mu_g' : 1, 'm' : 1, 'K' : 1, 'gamma' : 1}
        
        parameters_gc_c = generate_parameters('growth function of consumption',
                                              system_size, mu_start,
                                              unscaled_sigma_range,
                                              fixed_parameters_gc_c, n)
        parameters_gc_c = [parm_set for parm_set in parameters_gc_c
                           if parm_set['mu_c'] > mu_c_base[0] - 1e-5
                           and parm_set['mu_c'] < mu_c_base[-1]]
        
        dynamics(system_size, parameters_gc_c, 'growth function of consumption',
                 higher_directory + "/growth_coupled_to_consumption",
                 no_communities, ['mu_c', 'sigma_c'], t_end = 7000)
        
        fixed_parameters_g_cg  = {'mu_g' : 150, 'm' : 1, 'K' : 1, 'gamma' : 1}
        
        parameters_g_cg = generate_parameters('consumption function of growth',
                                              system_size, mu_start/system_size,
                                              unscaled_sigma_range,
                                              fixed_parameters_g_cg, n)
        parameters_g_cg = [parm_set for parm_set in parameters_g_cg
                           if parm_set['mu_c'] > mu_c_base[0] - 1e-5
                           and parm_set['mu_c'] < mu_c_base[-1]]
        
        dynamics(system_size, parameters_g_cg, 'consumption function of growth',
                 higher_directory + "/consumption_coupled_to_growth",
                 no_communities, ['mu_c', 'sigma_c'], t_end = 7000) 
    
    add_sigmas()
    
if __name__ == '__main__':
    
    main()
        
# %%

system_size = 150
unscaled_sigma_range = np.array([1, 1.6])
higher_directory = "phase_transtions_between_models"

n = 15
no_communities = 10

mu_c_range_gc_c = np.array([100, 300])
fixed_parameters_gc_c  = {'mu_g' : 1, 'm' : 1, 'K' : 1, 'gamma' : 1}

mu_c_range_g_cg = np.array([100, 300])/system_size
fixed_parameters_g_cg  = {'mu_g' : 150, 'm' : 1, 'K' : 1, 'gamma' : 1}

# %%

generate_parameters_simulate_dynamics(system_size, mu_c_range_gc_c,
                                      unscaled_sigma_range, fixed_parameters_gc_c,
                                      'growth function of consumption',
                                      higher_directory + "/growth_coupled_to_consumption",
                                      n, no_communities, t_end = 7000) 

generate_parameters_simulate_dynamics(system_size, mu_c_range_g_cg,
                                      unscaled_sigma_range, fixed_parameters_g_cg,
                                      'consumption function of growth',
                                      higher_directory + "/consumption_coupled_to_growth",
                                      n, no_communities, t_end = 7000) 
    
# %%

df_gc_c = generate_df("/growth_coupled_to_consumption",
                      'growth function of consumption', system_size,
                      mu_c_range_gc_c, unscaled_sigma_range, n)

df_g_cg = generate_df("/consumption_coupled_to_growth",
                      'consumption function of growth', system_size,
                      mu_c_range_g_cg, unscaled_sigma_range, n)

# %%

dfs_plot = [copy(df_gc_c), copy(df_g_cg)]
dfs_plot[0]['mu_c'] *= system_size
dfs_plot[0]['sigma_c'] *= np.sqrt(system_size)

fig, axs = generic_heatmaps_multiple_dfs_same_v(dfs_plot, 'mu_c', 'sigma_c',
                                                '(parameter underlying)' + \
                                                ' average consumption rate, '  + \
                                                r'$\mu_c$',
                                                'std. in the parameter underlying' + \
                                                ' consumption rate, ' + r'$\sigma_c$',
                                                'Max. lyapunov exponent', 
                                                'Proportion of simulations with\nmax. LE ' + \
                                                r'$> 0.0025$',
                                                'Purples',
                                                ['', ''], (2, 1), (6, 7),
                                                pivot_function = le_pivot,
                                                specify_min_max = [0, 1])
    
del dfs_plot
