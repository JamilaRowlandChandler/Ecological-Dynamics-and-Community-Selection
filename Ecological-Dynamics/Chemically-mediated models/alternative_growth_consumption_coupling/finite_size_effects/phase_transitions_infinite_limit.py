# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 23:46:22 2025

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
from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations_2, prop_chaotic, distance_from_instability, \
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
                                     (mu_c_sigma_combinations[1, :])*np.sqrt(no_resources)/np.sqrt(150)])

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
    
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/phase_transtions_between_models/" \
                        + directory
    
    df = pd.concat(community_properties_df(full_directory), 
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
    
    for var in ['rho', 'mu_c', 'mu_g', 'sigma_c', 'sigma_g']:
        
        df[var] = np.round(df[var], 6)
    
    return df
    
# %%

def community_properties_df(directory,
                            parm_attributes = ['no_species', 'no_resources',
                                               'mu_c', 'sigma_c', 'mu_g',
                                               'sigma_g', 'm', 'K']):
    
    dfs = [create_df_and_delete_simulations_2(directory, file, parm_attributes)
           for file in os.listdir(directory)]
            
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
                            layout = 'constrained')
                            #,
                           #sharex = True, sharey = True)
    
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

def main():
    
        system_size = 250
        unscaled_sigma_range = np.array([1, 1.6])
        higher_directory = "phase_transtions_between_models/infinite_limit"

        n = 15
        no_communities = 5

        mu_c_range_gc_c = np.array([100, 300])
        fixed_parameters_gc_c  = {'mu_g' : 1, 'm' : 1, 'K' : 1, 'gamma' : 1}


        generate_parameters_simulate_dynamics(system_size, mu_c_range_gc_c,
                                              unscaled_sigma_range, fixed_parameters_gc_c,
                                              'growth function of consumption',
                                              higher_directory + "/growth_coupled_to_consumption/try_again",
                                              n, no_communities, t_end = 7000) 

        #generate_parameters_simulate_dynamics(system_size, mu_c_range_g_cg,
        #                                      unscaled_sigma_range, fixed_parameters_g_cg,
        #                                      'consumption function of growth',
        #                                      higher_directory + "/consumption_coupled_to_growth",
        #                                      n, no_communities, t_end = 7000) 
    
if __name__ == '__main__':
    
    main()
        
# %%

system_size = 250
unscaled_sigma_range = np.array([1, 1.6])
higher_directory = "phase_transtions_between_models/infinite_limit"

n = 15
no_communities = 5

mu_c_range_gc_c = np.array([100, 300])
fixed_parameters_gc_c  = {'mu_g' : 1, 'm' : 1, 'K' : 1, 'gamma' : 1}

# %%

generate_parameters_simulate_dynamics(system_size, mu_c_range_gc_c,
                                      unscaled_sigma_range, fixed_parameters_gc_c,
                                      'growth function of consumption',
                                      higher_directory + "/growth_coupled_to_consumption",
                                      n, no_communities, t_end = 7000)

# %% 

df_small = generate_df("growth_coupled_to_consumption/",
                      'growth function of consumption', 150,
                      mu_c_range_gc_c, unscaled_sigma_range, 15)

df_large = generate_df("infinite_limit/growth_coupled_to_consumption/try_again/",
                      'growth function of consumption', 250,
                      mu_c_range_gc_c, unscaled_sigma_range, 15)

# %%

df_plots = [copy(df_small), copy(df_large)]

df_plots[0]['mu_c'] *= 150
df_plots[0]['sigma_c'] *= np.sqrt(150)
df_plots[0] = df_plots[0].iloc[np.where(df_plots[0]['mu_c'] > 99)]
df_plots[0].dropna(axis = 0, inplace = True)

df_plots[1]['mu_c'] *= 250
df_plots[1]['sigma_c'] *= np.sqrt(250)
df_plots[1] = df_plots[1].iloc[np.where(df_plots[1]['mu_c'] > 99)]


fig, axs = generic_heatmaps_multiple_dfs_same_v(df_plots, 'mu_c', 'sigma_c',
                                                'average consumption rate, ' + \
                                                    r'$<c>$',
                                                'std. in consumption rate, ' + \
                                                    r'$\sigma_c \sqrt{M}$',
                                                'Max. lyapunov exponent', 
                                                'Proportion of simulations with\nmax. LE ' + \
                                                r'$> 0.0025$',
                                                'Purples',
                                                ['Resource pool size, M = 150', 'M = 250'],
                                                (1, 2), (8.5, 3.5),
                                                pivot_function = le_pivot,
                                                specify_min_max = [0, 1])
    
#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_sigma_c_le.png",
 #           bbox_inches='tight')
#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_sigma_c_le.svg",
 #           bbox_inches='tight')

##################################################

fig, ax = plt.subplots(1, 1, figsize = (5, 5), sharex = True, sharey = True,
                        layout = 'constrained')

sns.lineplot(pd.concat([df.iloc[np.where(df['sigma_c'] > 1.59)]
                        for df in df_plots]),
             x = 'mu_c', y = 'rho', hue = 'no_resources', ax = ax,
             palette = 'viridis', linewidth = 5)

ax.set_xlabel('average consumption rate, ' + r'$<c>$', fontsize = 16,
              weight = 'bold')
ax.set_ylabel('correlation between growth and\nconsumption, ' + r'$\rho$',
              fontsize = 16, weight = 'bold',)
ax.tick_params(axis='both', which='major', labelsize = 12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['150', '250'], title = 'Resource pool size (M)',
          fontsize = 14, title_fontsize = 14)

sns.despine()

#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_line.png",
#            bbox_inches='tight')
#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_rho_line.svg",
#            bbox_inches='tight')

del df_plots

# %%

stable_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/phase_transtions_between_models/growth_coupled_to_consumption/CR_self_limiting_0.381_0.131.pkl")
chaotic_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/phase_transtions_between_models/growth_coupled_to_consumption/CR_self_limiting_2.0_0.127.pkl")

plot_dynamics([stable_communities[0].ODE_sols['lineage 0'],
               chaotic_communities[1].ODE_sols['lineage 0']])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_dynamics.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_c_dynamics.svg",
            bbox_inches='tight')