# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:14:24 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
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

# %%

def solve_for_mu_c(x):
    
    correlation, sigma_c, sigma_g, mu_g, M = x
    
    solved_mu_c = ((sigma_c * np.sqrt(M))/sigma_g) * np.sqrt((mu_g**2)*(1/(correlation**2) - 1) - sigma_g**2)
    
    return solved_mu_c

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

def growth_efficiency_infeasibility(mu_g, subsubdirectory):
    
    ############ generate parameters ##########
    
    # no_resources (= no_species)
    system_size = 150

    # range of sigma_c (also controls sigma_g: sigma_g = sigma_c/sqrt(system_size))
    unscaled_sigma_range = np.array([1, 1.6])

    # range of rhos/correlations between growth and consumption
    rho_range = [0.1, 0.99]

    # no. parameters combinations = n^2
    n = 12

    # create all combinations of rho and sigma with range
    rho_sigma_combinations_1 = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                          n)
    
    # add more sigmas, from about 1.6 to 1.9
    unscaled_sigma_range = np.array([1.6, 2.2])   
    rho_sigma_combinations_2 = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                          n)
    new_n = np.int32((n**2)/2)
    rho_sigma_combinations_2 = rho_sigma_combinations_2[:, n : new_n + n]
    
    # get all combinations of rho and sigma
    rho_sigma_combinations = np.hstack((rho_sigma_combinations_1, rho_sigma_combinations_2))
    
    # solve for mu_c using mu_g, sigma_c, sigma_g and rho 
    mu_solve_inputs = np.vstack([rho_sigma_combinations,
                                 rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                 np.repeat(mu_g, rho_sigma_combinations.shape[1]),
                                 np.repeat(system_size, rho_sigma_combinations.shape[1])])
    solved_mu_cs = np.apply_along_axis(solve_for_mu_c, axis = 0, arr = mu_solve_inputs)

    # get variable parameters (mu_g, sigma_c (which sigma_g scales with))
    variable_parameters = np.vstack([solved_mu_cs,
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size),
                                     rho_sigma_combinations[1, :]/np.sqrt(system_size)])
    # remove any nan values (which could occur in the solve as some values blow up to inf)
    variable_parameters = variable_parameters[:, ~np.isnan(variable_parameters).any(axis=0)]
    
    # Set fixed parameters (mu_g, instrinsic resource growth rates, species death rates,
    #                       ratio of resources to species)
    fixed_parameters = {'mu_g' : mu_g, 'm' : 1, 'K' : 1, 'gamma' : 1}
    
    ############ simulate dynamics ##########
    
    # function used to generate proper growth and consumption matrices by the 
    #   Consumer_Resource_Model class
    growth_consumption_function = 'growth function of consumption'
    
    subdirectory = "alternative_coupling_gc_infeasibility_interplay" + subsubdirectory
    
    # No. communities per distribution of growth and consumption rates
    no_communities = 10
    
    # Simulate dynamics and save
    find_infeasibility_supply_parameters(system_size, variable_parameters,
                                         ['mu_c', 'sigma_c', 'sigma_g'],
                                         fixed_parameters, growth_consumption_function,
                                         subdirectory, variable_parameters.shape[1],
                                         no_communities, t_end = 7000)
        
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
    
############################################################################################

# %%

'''
==============================================================================
                                    Main
==============================================================================
'''

'''
    Investigating the interplay between mu_c, correlation, and community feasibility
    Vary rho through varying mu_c, create parameter combinations of rho and sigma

'''

# where the simulation data will be saved
subdirectory = "alternative_coupling_gc_infeasibility_interplay"



# %%

#######################

'''
    Simulate dynamics with different growth efficiencies (mu_gs)
'''

# Run function for different growth efficiencies
growth_efficiency_infeasibility(0.75, "/inefficient_growth")
growth_efficiency_infeasibility(1.25, "/efficient_growth")
growth_efficiency_infeasibility(0.5, "/extra_inefficient_growth")
growth_efficiency_infeasibility(1, "/equal_growth")

# %%

'''
    Generate dataframes 
'''

system_size = 150

unscaled_sigma_range = np.array([1, 1.6])
additional_sigma_range = np.array([1.6, 2.2])
rho_range = [0.1, 0.99]

n = 12

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
    Plotting rho vs sigma vs stability 
'''
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
'''

'''
    Subset of the data
'''

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

'''
    Plotting rho vs sigma vs avg. species abundance 
'''

# %%

dfs = [df for df in infsbl_grwth_effcncy.values()]

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
=========================================================================
    Simulation dynamics
=========================================================================
'''

# extract simulation data
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

'''
# data locations
subdirectories = ['extra_inefficient_growth', 'inefficient_growth',
                  'equal_growth', 'efficient_growth']

# rhos and sigmas of interest - from the infeasible, dynamic and stable regions
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

# plot dynamics from each region for each growth efficiency, then save
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
'''        

'''
    Subset of the data
'''
# data locations
subdirectories = ['inefficient_growth', 'efficient_growth']

# rhos and sigmas of interest - from the infeasible, dynamic and stable regions
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

# plot dynamics from each region for each growth efficiency, then save
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
    