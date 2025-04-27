# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:15:08 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
from copy import deepcopy
import pickle
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import colormaps as cmaps
from matplotlib import colors

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models')
import self_consistency_equation_functions as sce

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
                                     no_communities = 5, no_lineages = 2):
    
    parameter_methods = {'death' : 'normal', 'influx' : 'normal', 'outflux' : 'normal'}
    
    def community_dynamics(i, lineages, no_species, no_resources, parameters):
       
        community = Consumer_Resource_Model(no_species, no_resources, parameters)
        
        community.generate_parameters(growth_consumption_method = 'coupled by rho',
                                      other_parameter_methods = parameter_methods)
        
        community.simulate_community(lineages, 3500, model_version = 'external resource supply',
                                     assign = True)
        
        community.calculate_community_properties(lineages, 3000)
        community.lyapunov_exponent = \
            max_le(community, 1000, community.ODE_sols['lineage 0'].y[:, -1],
                   1e-3, 'external resource supply', dt = 20, separation = 1e-3)
    
        return community 

    communities_list = [deepcopy(community_dynamics(i, np.arange(no_lineages),
                                                    no_species, no_resources,
                                                    parameters))
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
        
        spec_survive_frac = np.count_nonzero(final_species_abundances > 1e-4)/no_species
        spec_mean = np.mean(final_species_abundances)
        spec_sq_mean = np.mean(final_species_abundances**2)
        
        res_mean = np.mean(final_resource_abundances)
        res_sq_mean = np.mean(final_resource_abundances**2)
        
        return spec_survive_frac, spec_mean, spec_sq_mean, res_mean, res_sq_mean
     
    dist_properties_array = np.vstack([distribution_properties(simulation, community.no_species)
                                       for community in communities_list 
                                       for simulation in community.ODE_sols.values()])
    
    df = pd.DataFrame(dist_properties_array, columns = ['phi_N', 'N_mean', 'q_N', 
                                                        'R_mean', 'q_R'])
    
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

'''
    =======================================================
    
                        Vary sigma and rho
    
    =======================================================

'''

no_species = 100
no_resources = 100

sigma_range = np.array([0, 2.5])/np.sqrt(no_resources)
rho_range = np.array([0.001, 1])
n = 20

# generate n values of rho and sigma within range
rho_sigma_combinations = sce.parameter_combinations([rho_range, sigma_range], n)

# array of variable parameter combinations
variable_parameters = np.vstack([rho_sigma_combinations, rho_sigma_combinations[1, :]])

# fixed parameters
fixed_parameters = {'mu_c' : 1/no_resources, 'mu_g' : 1/no_resources, 'mu_m' : 1,
                    'sigma_m' : 0.1, 'mu_K' : 1, 'sigma_K' : 0.1,
                    'gamma' : no_resources/no_species, 'mu_D': 1, 'sigma_D' : 0}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters, ['rho', 'sigma_c', 'sigma_g'],
                                           fixed_parameters)

# %%

for parameter_set in tqdm(parameters):

    filename_CR = "CR_external_resource_" + str(parameter_set['rho']) + "_" + \
        str(parameter_set['sigma_c'])
    
    create_and_delete_CR(filename_CR, no_species, no_resources, parameter_set,
                         no_communities = 15)
    
# %%

parm_attributes = ['no_species', 'no_resources', 'rho', 'mu_c', 'sigma_c', 'mu_g',
                   'sigma_g', 'mu_m', 'sigma_m', 'mu_K', 'sigma_K', 'mu_D', 'sigma_D']

external_dfs = [create_df_and_delete_simulations("CR_external_resource_" + str(parameter_set['rho']) + "_" + \
                                                 str(parameter_set['sigma_c']),
                                                 parm_attributes) 
                for parameter_set in parameters]

externally_supplied_df = pd.concat(external_dfs, axis = 0, ignore_index = True)

externally_supplied_df['Species Fluctuation CV'] = \
    externally_supplied_df['Species Fluctuation CV']/externally_supplied_df['no_species']
externally_supplied_df['Resurces Fluctuation CV'] = \
    externally_supplied_df['Resource Fluctuation CV']/externally_supplied_df['no_resources']

pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/externally_supplied_df.pkl",
            externally_supplied_df)

# %%

species_pivot = pd.pivot_table(externally_supplied_df, columns = 'sigma_c', index = 'rho',
                               values = 'Species diversity', aggfunc = 'mean')

le_pivot = pd.pivot_table(externally_supplied_df, columns = 'sigma_c',
                          index = 'rho', values = 'Max. lyapunov exponent',
                          aggfunc = prop_chaotic)

fluct_pivot = pd.pivot_table(externally_supplied_df, columns = 'sigma_c', index = 'rho',
                               values = 'Species Fluctuation CV', aggfunc = 'mean')

divergence_pivot = pd.pivot_table(externally_supplied_df, columns = 'sigma_c',
                                  index = 'rho', values = 'Divergence measure',
                                  aggfunc = 'mean')

cmap_list = ['Purples', 'Blues', 'Greens',  'Oranges']

v_min_max = [[0, 1], [0, 1], [0, 0.5], [0, np.max(divergence_pivot)]]

titles = ['Proportion of simulations with LEs ' + r'$ > 0.0025$',
          'Species flucutation coefficient',
          'Species survival fraction, ' + r'$\phi_N$',
          'Divergence measure (simulation end time)']

sns.set_style('white')

fig, axs = plt.subplots(2, 2, figsize = (12, 10), sharex = True, sharey = True, 
                        layout = 'constrained')

fig.suptitle('There is no feasible dynamic phase in the Consumer-Resource model with external nutrient supply',
             fontsize = 20, weight = 'bold')
fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
              fontsize = 18, weight = 'bold')
fig.supylabel('Correlation between growth and consumption rates ' + r'$(\rho)$',
              fontsize = 18, weight = 'bold', horizontalalignment = 'center',
              verticalalignment = 'center')

for ax, data, cmap, v, title in zip(axs.flatten(),
                                    [le_pivot, fluct_pivot, species_pivot, divergence_pivot],
                                    cmap_list, v_min_max, titles):
    
    subfig = sns.heatmap(data, ax=ax, vmin = v[0], vmax = v[1], cbar = True, cmap = cmap)

    ax.set_yticks([0.5, n - 0.5], labels = np.round(rho_range, 2), fontsize = 14)
    ax.set_xticks([0.5, n - 0.5], labels = sigma_range, fontsize = 14, rotation = 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.set_title(title, fontsize = 16, weight = 'bold')
        
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/external_resource_phases.svg",
            bbox_inches='tight')

# %%

stable_feasible_filename = "CR_external_resource_" + \
                                str(parameters[-1]['rho']) + "_" + \
                                    str(parameters[-1]['sigma_c'])
                            
unstable_infeasible_1_filename = "CR_external_resource_" + \
                                    str(parameters[41]['rho']) + "_" + \
                                        str(parameters[41]['sigma_c'])
                                
#unstable_infeasible_2_filename = "CR_external_resource_" + \
#                                    str(parameters[8]['rho']) + "_" + \
#                                        str(parameters[1]['sigma_c'])
                                        
community_list = [pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")[0]
                  for filename in [stable_feasible_filename, unstable_infeasible_1_filename]]

colour_index = np.arange(100)
np.random.shuffle(colour_index)

cmap = colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                ['#e9a100ff','#1fb200ff',
                                                 '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                N = no_species)

fig, axs = plt.subplots(1, 2,figsize=(11,4),layout='constrained')

for d_i, (ax, community) in enumerate(zip(axs.flatten(), community_list)):
    
    data = community.ODE_sols['lineage 0']
    
    for spec, i in enumerate(colour_index):
    
        ax.plot(data.t, data.y[spec,:].T, color = 'black', linewidth = 3.75)
        ax.plot(data.t, data.y[spec,:].T, color = cmap(i), linewidth = 3)
    
    ax.set_xticks([0, np.round(data.t[-1], 0)])
    ax.set_xticklabels([0, np.round(data.t[-1], 0)], fontsize = 16)
    ax.set_yticks([0, np.round(np.max(data.y[:100, :]), 0)])
    ax.set_yticklabels([0, np.round(np.max(data.y[:100, :]), 0)], fontsize = 16)

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/external_resource_dynamics.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/external_resource_dynamics.png",
            bbox_inches='tight')

# %%

sol_external = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_1_ls_2.pkl")

species_pivot_sim = pd.pivot_table(externally_supplied_df, columns = 'sigma_c', index = 'rho',
                                   values = 'phi_N', aggfunc = 'mean')

species_pivot_solve = sol_external.pivot_table(index = 'rho',
                                               columns = 'sigma_c',
                                               values = 'phi_N')

ms_pivot_sim = pd.pivot_table(externally_supplied_df, columns = 'sigma_c',
                              index = 'rho', values = 'Max. lyapunov exponent',
                              aggfunc = prop_chaotic)

ms_pivot_solve = sol_external.pivot_table(index = 'rho',
                                          columns = 'sigma_c',
                                          values = 'dNde')
                                          #values = 'dRde')
ms_pivot_solve = np.log10(np.abs(ms_pivot_solve))

rho_range = [np.min(sol_external['rho']), np.max(sol_external['rho'])]
sigma_range = [np.min(sol_external['sigma_c']), np.max(sol_external['sigma_c'])]

cmap_list = ['Greens', 'Purples',
             'Greens', 'Purples']

v_min_max = [[0, 0.5], [np.min(ms_pivot_solve), np.max(ms_pivot_solve)],
             [0, 0.5], [0, 1]]

titles_0 = ['Species survival fraction, ' + r'$\phi_N$',
            'Multistability metric',
            '.', '.']
title_col = ['black', 'black', 'white', 'white']

titles_1 = ['Cavity calculation', 'Simulation']

cbar_labels = ['Species survival fraction, ' + r'$\phi_N$',
               r'$\mathbf{\log_{10}\langle (dN^+/d\epsilon)^2 \rangle}$',
               'Avg. proportion of species with\nabundances, ' + r'$ > e^{-4}$',
               'Proportion of simulations with\nLEs ' + r'$ > 0.0025$']


sns.set_style('white')

fig, axs = plt.subplots(2, 2, figsize = (12, 10), sharex = True, sharey = True, 
                        layout = 'constrained')

fig.suptitle('The dynamics of the multistable region are biologically infeasible (total species extinction)',
             fontsize = 22, weight = 'bold', y = 1.05)
fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
              fontsize = 18, weight = 'bold')
fig.supylabel('Correlation between growth and consumption rates ' + r'$(\rho)$',
              fontsize = 18, weight = 'bold', horizontalalignment = 'center',
              verticalalignment = 'center')

for ax, data, cmap, v, title, t_col, c_label in zip(axs.flatten(),
                                                    [species_pivot_solve,
                                                     ms_pivot_solve,
                                                     species_pivot_sim,
                                                     ms_pivot_sim],
                                                    cmap_list, v_min_max,
                                                    titles_0, title_col, cbar_labels):

    subfig = sns.heatmap(data, ax=ax, vmin = v[0], vmax = v[1], cbar = True,
                         cmap = cmap)

    ax.set_yticks([0.5, n - 0.5], labels = np.round(rho_range, 2), fontsize = 14)
    ax.set_xticks([0.5, n - 0.5], labels = sigma_range, fontsize = 14, rotation = 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.set_title(title, fontsize = 18, weight = 'bold', y = 1.15, color = t_col)
    
    ax.figure.axes[-1].set_ylabel(c_label, size = 14)
    
fig.text(0.5, 0.94, 'Cavity calculation', fontsize = 18, weight = 'bold',
         horizontalalignment = 'center', verticalalignment = 'center')
fig.text(0.5, 0.475, 'Simulations', fontsize = 18, weight = 'bold',
         horizontalalignment = 'center', verticalalignment = 'center')

#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_sim_exre_1.svg",
#            bbox_inches='tight')
#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_sim_exre_1.png",
#            bbox_inches='tight')

# %%

'''

==================================

    With bad-fitting equations 
    
==================================

'''

sol_external = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_1_ls_2.pkl")
bad_solve = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_ls_wengping_eq.pkl")

species_pivot_sim = pd.pivot_table(externally_supplied_df, columns = 'sigma_c', index = 'rho',
                                   values = 'Species diversity', aggfunc = 'mean')

species_pivot_solve = sol_external.pivot_table(index = 'rho',
                                               columns = 'sigma_c',
                                               values = 'phi_N')
species_pivot_bad_solve = bad_solve.pivot_table(index = 'rho',
                                                columns = 'sigma_c',
                                                values = 'phi_N')

titles = ['Cavity calculation (Cui et at.)', 'Cavity calculation (mine)',
          'Simulations']

rho_range = [np.min(sol_external['rho']), np.max(sol_external['rho'])]
sigma_range = [np.min(sol_external['sigma_c']), np.max(sol_external['sigma_c'])]

sns.set_style('white')

fig, axs = plt.subplots(1, 3, figsize = (15, 4), sharex = True, sharey = True, 
                        layout = 'constrained')

fig.suptitle('My equations solve better for non-reciprocal interactions ' + \
             r'$(\rho < 1)$' + ' than Cui et al.',
             fontsize = 22, weight = 'bold', y = 1.1)
fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
              fontsize = 18, weight = 'bold')
fig.supylabel('Correlation between growth\nand consumption rates ' + r'$(\rho)$',
              fontsize = 18, weight = 'bold', horizontalalignment = 'center',
              verticalalignment = 'center')

for ax, data, title in zip(axs.flatten(), [species_pivot_bad_solve,
                                           species_pivot_solve,
                                           species_pivot_sim],
                           titles):
        
    if ax == axs[-1]:
        
        subfig = sns.heatmap(data, ax=ax, vmin = 0, vmax = 0.5, cbar = True,
                             cmap = 'Greens')
        
        ax.figure.axes[-1].set_ylabel('Species survival fraction, ' + r'$\phi_N$',
                                      size = 14)
    
    else:

        subfig = sns.heatmap(data, ax=ax, vmin = 0, vmax = 0.5, cbar = False,
                             cmap = 'Greens')

    ax.set_yticks([0.5, n - 0.5], labels = np.round(rho_range, 2), fontsize = 14)
    ax.set_xticks([0.5, n - 0.5], labels = sigma_range, fontsize = 14, rotation = 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.set_title(title, fontsize = 18, weight = 'bold', y = 1.05)
    
    ax.axhline(y = 0, linewidth = 2, color = 'black')
    ax.axhline(y = data.shape[1], linewidth = 2, color = 'black')
    ax.axvline(x = 0, linewidth = 2, color = 'black')
    ax.axvline(x = data.shape[0], linewidth = 2, color = 'black')
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_cui_mine_sim.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_cui_mine_sim.png",
            bbox_inches='tight')

# %%

sim_rho_1 = externally_supplied_df.iloc[np.where((externally_supplied_df['rho'] == 1) &
                                            (externally_supplied_df['sigma_c'] > 0.5/np.sqrt(no_resources)))]
sim_rho_1['sigma_c'] = sim_rho_1['sigma_c']*np.sqrt(sim_rho_1['no_resources'])


solve_rho_1 = sol_external.iloc[np.where((sol_external['rho'] == 1) &
                                         (sol_external['sigma_c'] > 0.5))]
bad_solve_rho_1 = bad_solve.iloc[np.where((bad_solve['rho'] == 1) &
                                          (bad_solve['sigma_c'] > 0.5))]

sns.set_style('white')

fig, axs = plt.subplots(1, 3, figsize = (14, 4), sharex = True,
                        layout = 'constrained')

fig.suptitle('The cavity equations (currently) fit poorly to the simulations',
             fontsize = 22, weight = 'bold', y = 1.1)
fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
              fontsize = 18, weight = 'bold')

for ax, y_solve, y_sim, title in zip(axs.flatten(), ['phi_N', 'N_mean', 'q_N'],
                                     ['Species diversity', 'N_mean', 'q_N'],
                                     [r'$\phi_N$', r'$\langle N \rangle$', r'$q_N$']):

    sns.lineplot(solve_rho_1, x = 'sigma_c', y = y_solve, color = 'black',
                 linewidth = 3, ax = ax)
    sns.lineplot(bad_solve_rho_1, x = 'sigma_c', y = y_solve, color = 'grey',
                 linewidth = 3, ax = ax)
    sns.lineplot(sim_rho_1, x = 'sigma_c', y = y_sim, err_style = "bars",
                 errorbar=("se", 2), ax = ax, linewidth = 3)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize = 18, weight = 'bold')
    
textstr = '\n'.join(('KEY',
                     '',
                     'Cavity equations (mine) = black',
                     'Cavity equations (Cui et al.) = grey',
                     'Simulations = blue'))
props = dict(boxstyle='round', facecolor='white')

fig.text(1.01, 0.75, textstr,  fontsize=14, verticalalignment='top', bbox=props)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_cui_mine_sim_line.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cavity_cui_mine_sim_line.png",
            bbox_inches='tight')

# %%

'''
    =======================================================
    
                        Vary mu and rho
    
    =======================================================

'''

no_species = 100
no_resources = 100

mu_range = np.array([0.25, 15.25])/no_resources
rho_range = np.array([0.001, 1])
n = 20

# generate n values of rho and sigma within range
rho_sigma_combinations = sce.parameter_combinations([rho_range, mu_range], n)

# array of variable parameter combinations
variable_parameters = np.vstack([rho_sigma_combinations, rho_sigma_combinations[1, :]])

# fixed parameters
fixed_parameters = {'sigma_c' : 1/np.sqrt(no_resources), 'sigma_g' : 1/np.sqrt(no_resources),
                    'mu_m' : 1, 'sigma_m' : 0.1, 'mu_K' : 1, 'sigma_K' : 0.1,
                    'gamma' : no_resources/no_species, 'mu_D': 1, 'sigma_D' : 0}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters, ['rho', 'mu_c', 'mu_g'],
                                           fixed_parameters)

# %%

for parameter_set in tqdm(parameters):

    filename_CR = "CR_external_resource_mu_rho_" + str(np.round(parameter_set['rho'], 3)) + "_" + \
        str(np.round(parameter_set['mu_c'], 3))
    
    create_and_delete_CR(filename_CR, no_species, no_resources, parameter_set)
    
# %%

parm_attributes = ['no_species', 'no_resources', 'rho', 'mu_c', 'sigma_c', 'mu_g',
                   'sigma_g', 'mu_m', 'sigma_m', 'mu_K', 'sigma_K', 'mu_D', 'sigma_D']

external_dfs_mu = [create_df_and_delete_simulations("CR_external_resource_mu_rho_" + \
                                                    str(np.round(parameter_set['rho'], 3)) + \
                                                        "_" + \
                                                    str(np.round(parameter_set['mu_c'], 3)),
                                                 parm_attributes) 
                for parameter_set in parameters]

externally_supplied_df_mu = pd.concat(external_dfs_mu, axis = 0, ignore_index = True)

externally_supplied_df_mu['Species Fluctuation CV'] = \
    externally_supplied_df_mu['Species Fluctuation CV']/externally_supplied_df_mu['no_species']

# %%

species_pivot = pd.pivot_table(externally_supplied_df_mu, columns = 'mu_c', index = 'rho',
                               values = 'Species diversity', aggfunc = 'mean')

le_pivot = pd.pivot_table(externally_supplied_df_mu, columns = 'mu_c',
                          index = 'rho', values = 'Max. lyapunov exponent',
                          aggfunc = prop_chaotic)

fluct_pivot = pd.pivot_table(externally_supplied_df_mu, columns = 'mu_c', index = 'rho',
                               values = 'Species Fluctuation CV', aggfunc = 'mean')

divergence_pivot = pd.pivot_table(externally_supplied_df_mu, columns = 'mu_c',
                                  index = 'rho', values = 'Divergence measure',
                                  aggfunc = 'mean')

cmap_list = ['Purples', 'Blues', 'Greens',  'Oranges']

v_min_max = [[0, 1], [0, 1], [0, 0.5], [0, np.max(divergence_pivot)]]

titles = ['Proportion of simulations with LEs ' + r'$ > 0.004$',
          'Species flucutation coefficient',
          'Species survival fraction, ' + r'$\phi_N$',
          'Divergence measure (simulation end time)']

sns.set_style('white')

fig, axs = plt.subplots(2, 2, figsize = (12, 10), sharex = True, sharey = True, 
                        layout = 'constrained')

fig.suptitle('There is no feasible dynamic phase in the Consumer-Resource model with external nutrient supply',
             fontsize = 20, weight = 'bold')
fig.supxlabel('avg. growth/consumption rate ' + r'$(\mu_c$' + ' or ' + r'$\mu_g)$',
              fontsize = 18, weight = 'bold')
fig.supylabel('Correlation between growth and consumption rates ' + r'$(\rho)$',
              fontsize = 18, weight = 'bold', horizontalalignment = 'center',
              verticalalignment = 'center')

for ax, data, cmap, v, title in zip(axs.flatten(),
                                    [le_pivot, fluct_pivot, species_pivot, divergence_pivot],
                                    cmap_list, v_min_max, titles):
    
    subfig = sns.heatmap(data, ax=ax, vmin = v[0], vmax = v[1], cbar = True, cmap = cmap)

    ax.set_yticks([0.5, n - 0.5], labels = np.round(rho_range, 2), fontsize = 14)
    ax.set_xticks([0.5, n - 0.5], labels = sigma_range, fontsize = 14, rotation = 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.set_title(title, fontsize = 16, weight = 'bold')

