# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:43:25 2025

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
#from matplotlib.colors import TwoSlopeNorm
#from matplotlib.colors import LogNorm
#from matplotlib.colors import PowerNorm
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
                            layout = 'constrained')
#                            ,
#                           sharex = True, sharey = True)
    
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
    ===========================================================================================================
    Comparing the emergence of infeaisble dynamics with rho in the C-R model with linear parameter combinations
    ===========================================================================================================

'''

'''
    Parameters
'''

subdirectory = "alternative_coupling_gc_infeasibility_interplay/linear"

system_size = 150
n = 12

linear_rho_sigma_combinations = sce.parameter_combinations([[0.1, 0.99], [1, 3.5]],
                                                          n)
variable_parameters = np.vstack([linear_rho_sigma_combinations[0, :],
                                 linear_rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 linear_rho_sigma_combinations[1, :]/np.sqrt(system_size)])

# %%

'''
    ===========
    Simulations
    ===========
'''

fixed_parameters = {'mu_c': 200/system_size, 'mu_g' : 200/system_size, 'm' : 1,
                    'K' : 1, 'gamma' : 1}

growth_consumption_function = 'coupled by rho'

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

#linear_infeasibility_df['sigma_c'] = np.round(linear_infeasibility_df['sigma_c'],
#                                              6)
#linear_infeasibility_df['rho'] = np.round(linear_infeasibility_df['rho'], 6)
linear_infeasibility_df['instability distance'] = \
    linear_infeasibility_df.apply(distance_from_instability, axis = 1)
linear_infeasibility_df['infeasibility distance'] = \
    linear_infeasibility_df.apply(distance_from_infeasibility, axis = 1)

# %%

'''
    ========
    Plotting
    ========
    
    Relationship betwwen mu_c, rho, and N_mean
    
'''

fig, axs = generic_heatmaps(linear_infeasibility_df,
                            'sigma_c', 'rho',
                           'std. in the consumption rate, ' + \
                               r'$\frac{\sigma_c}{\sqrt{M}}$',
                           'correlation between growth and\nconsumption, ' + r'$\rho$',
                            ['Max. lyapunov exponent'], 'Purples',
                            'Increasing the correlation  between ' + \
                            'growth and consumption\ninduces a stronger phase transition ' + \
                            ' when it is\nuncoupled to the mean growth rate.',
                            (1, 1), (6, 5),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with\nmax. LE ' + r'$> 0.0025$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_rho_le_linear.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_rho_le_linear.svg",
            bbox_inches='tight')


# %%


'''
=========================================================================
    Simulation dynamics and checking for multistability
=========================================================================
'''

# extract simulation data
def growth_efficiency_dynamics(df, approx_rho, approx_sigma):
    
    rho_distance = np.abs(np.round(df['rho'], 6) - approx_rho)
    sigma_distance = np.abs(np.round(df['sigma_c'], 6) - approx_sigma)
    
    index = np.where((rho_distance == rho_distance.min()) & 
                     (sigma_distance == sigma_distance.min()))[0][0]
    
    rho_str = str(df.iloc[index]['rho'])
    sigma_c_str = str(df.iloc[index]['sigma_c'])
    
    communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/alternative_coupling_gc_infeasibility_interplay/linear/infeasibility/CR_self_limiting_" + \
                                 rho_str + "_" + sigma_c_str + ".pkl")
        
    return communities

# rhos and sigmas of interest
plot_sigma = 0.13
plot_rhos = [0.35, 0.9]
rho_sigmas = [[rho, plot_sigma] for rho in plot_rhos]

communities_dynamics = [growth_efficiency_dynamics(linear_infeasibility_df,
                                                   rho_sigma[0],
                                                   rho_sigma[1])
                               for rho_sigma in rho_sigmas]

# plotting some example dynamics
plot_dynamics([communities_dynamics[0][0].ODE_sols['lineage 0'],
               communities_dynamics[0][1].ODE_sols['lineage 0'],
               communities_dynamics[1][0].ODE_sols['lineage 0']])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_linear_rho.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_linear_rho.svg",
            bbox_inches='tight')
