# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:27:31 2025

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
             "finite_size_effects")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling")
from simulation_functions import \
    prop_chaotic, distance_from_instability, \
    distance_from_infeasibility, species_packing, pickle_dump, create_and_delete_CR, \
    create_df_and_delete_simulations_2

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_fixed_C(Ms, C_range, sigma_C, n, fixed_parameters):
    
    subdirectory = 'finite_effects_fixed_C_2'
    
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + subdirectory
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
        
    for M in tqdm(Ms, position = 0, leave = True):
        
        mu_c_range = C_range/M 
        sigma = sigma_C/np.sqrt(M)
        
        parameters = generate_parameters(mu_c_range, [sigma, sigma], n,
                                         fixed_parameters)
        for parm_set in parameters: parm_set['sigma_g'] = sigma_C/np.sqrt(150)
        
        for parm_set in tqdm(parameters, position = 0, leave = True):
    
            create_and_delete_CR(subdirectory + "/CR_self_limiting_" + str(M) + \
                                 str(parm_set['mu_c']),
                                 M, M, parm_set,
                                 growth_consumption_function = 'growth function of consumption',
                                 no_communities = 10, t_end = 7000)
                
# %%

def generate_parameters(mu_range, sigma_range, n, fixed_parameters,
                        v_parm_names = ['mu_c', 'sigma_c', 'sigma_g']):
    
    mu_sigma_combinations = np.unique(sce.parameter_combinations([mu_range,
                                                                  sigma_range],
                                                                 n), axis = 1)
   
    # array of variable parameter combinations
    variable_parameters = np.round(np.vstack([mu_sigma_combinations,
                                              mu_sigma_combinations[1, :]]),
                                   6)
    
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               v_parm_names,
                                               fixed_parameters)
    
    return parameters

# %%

def generate_df():
    
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + 'finite_effects_fixed_C_2'
    
    df = pd.concat(community_properties_df(full_directory), 
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
    
    for var in ['rho', 'mu_c', 'mu_g', 'sigma_c', 'sigma_g']:
        
        df[var] = np.round(df[var], 6)
    
    return df
    
# %%

def community_properties_df(directory,
                            parm_attributes = ['no_species', 'no_resources',
                                               'mu_c', 'sigma_c', 'mu_g',
                                               'sigma_g', 'm', 'K']):
    
    dfs = [create_df_and_delete_simulations_2(directory + "/", file, parm_attributes)
           for file in os.listdir(directory)]
            
    return dfs

# %%

def covariance_correlation(df):
            
    mu_c, mu_g, sigma_c, sigma_g = df['mu_c'], df['mu_g'], df['sigma_c'], df['sigma_g']

    denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2) + (sigma_c * sigma_g)**2
         
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

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None, **kwargs):
    
    if pivot_functions is None:
    
        pivot_tables = {variable : df.pivot(index = x, columns = y, values = variable)
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


# %%

resource_pool_sizes = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# %%

M_effect_fixed_C(resource_pool_sizes, np.array([100, 300]), 1.6,
                 15, {'mu_g': 1, 'K' : 1, 'm' : 1, 'gamma' : 1})

# %%

df_C_M = generate_df()
df_C_M['no_resources'] = np.int32(df_C_M['no_resources'])
df_C_M['<C>'] = np.round(df_C_M['mu_c'] * df_C_M['no_resources'], 2)

# %%

fig, axs = generic_heatmaps(df_C_M,
                            'no_resources', '<C>', 
                           'resource pool size, ' + r'$M$',
                           'average total consumption rate of a resource, ' + r'$<C>$',
                            ['Max. lyapunov exponent'], 'Purples',
                            '(For a given total consumption rate), ' + \
                            'communities\nstabilise with increasing resource pool size',
                            (1, 1), (6.5, 6.5),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                          labels = resource_pool_sizes, fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M.svg",
            bbox_inches='tight')

plt.show()

