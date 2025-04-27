# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:46:01 2025

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
from matplotlib import colormaps as cmaps
from matplotlib.colors import TwoSlopeNorm

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/self_limiting_alternative_growth_consumption_coupling')

from simulation_functions import create_and_delete_CR, \
    create_df_and_delete_simulations, prop_chaotic, distance_from_instability

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')
from models import Consumer_Resource_Model

# %%

def mu_c_size_effects(system_size, scaling, unscaled_mu_c_range, unscaled_sigma_range,
                      fixed_parameters, growth_consumption_function,
                      subdirectory,
                      n, no_communities):
    
    '''
    =======================
    Create folder
    =======================
    '''
    
    complete_subdirectory = subdirectory + "/system_size_" + str(system_size)
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + complete_subdirectory
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
        
    '''
    =====================
    Create parameters
    =====================
    
    '''
    
    no_species, no_resources = system_size, system_size 
    
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
    mu_c_sigma_combinations = sce.parameter_combinations([mu_c_range, sigma_range], n)

    # array of variable parameter combinations
    variable_parameters = np.vstack([mu_c_sigma_combinations,
                                     mu_c_sigma_combinations[1, :]])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters, ['mu_c', 'sigma_c', 'sigma_g'],
                                               fixed_parameters_copy)
    
    breakpoint()
    
    '''
    =====================
    Community dynamics
    =====================
    
    '''
    
    for parameter_set in tqdm(parameters):

        filename_CR = complete_subdirectory + "/CR_self_limiting_" + \
                        str(parameter_set['mu_c']) + "_" + str(parameter_set['sigma_c'])
        
        create_and_delete_CR(filename_CR, no_species, no_resources, parameter_set,
                             no_communities = no_communities,
                             growth_consumption_function = growth_consumption_function)
        
# %%

def size_effect_df(system_size, unscaled_mu_c_range, unscaled_sigma_range,
                   subdirectory, scaling = 'asymmetric'):
    
    '''
    =======================
    Create folder
    =======================
    '''
    
    match scaling:
        
        case 'asymmetric':
    
            mu_c_range = unscaled_mu_c_range/system_size
            sigma_range = unscaled_sigma_range/np.sqrt(system_size)
            
            directory_end = "/CR_external_resource_"
            
        case 'symmetric':
        
            mu_c_range = unscaled_mu_c_range/np.sqrt(system_size)
            sigma_range = unscaled_sigma_range/(system_size**0.25)
            
            directory_end = "/CR_self_limiting_"
            
    # generate n values of mu and sigma within range
    mu_c_sigma_combinations = sce.parameter_combinations([mu_c_range, sigma_range], n)
    
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(mu_c_sigma_combinations, ['mu_c', 'sigma_c'],
                                               {})
    
    complete_subdirectory = subdirectory + "/system_size_" + str(system_size)
    
    parm_attributes = ['no_species', 'no_resources', 'mu_c', 'sigma_c', 'mu_g',
                       'sigma_g', 'm', 'K']

    dfs = [create_df_and_delete_simulations(complete_subdirectory + \
                                                directory_end + \
                                                    str(parameter_set['mu_c']) + "_" + \
                                                        str(parameter_set['sigma_c']),
                                            parm_attributes)
           for parameter_set in parameters] 
        
    return dfs

# %%

def covariance_correlation(df, scaling):
    
    #breakpoint()
    
    M = df['no_resources']
    
    match scaling:
        
        case 'asymmetric':
            
            mu_c, mu_g = df['mu_c']*M, df['mu_g']
            
            sigma_c, sigma_g = df['sigma_c']*np.sqrt(M), df['sigma_g']
            
            denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2)/M + (sigma_c * sigma_g)**2
            
        case 'symmetric':
            
            #breakpoint()
            
            mu_c, mu_g = df['mu_c']*np.sqrt(M), df['mu_g']*np.sqrt(M)
            
            sigma_c, sigma_g = df['sigma_c']*(M**0.25), df['sigma_g']*(M**0.25)
            
            denominator = (mu_g * sigma_c)**2 + (mu_c * sigma_g)**2  + \
                                    np.sqrt(M)*(sigma_c * sigma_g)**2
            
        
    covariance = (mu_g * sigma_c**2)
    correlation = (mu_g * sigma_c)/np.sqrt(denominator)
    
    return covariance, correlation

# %%

def le_pivot(df):
    
    return [pd.pivot_table(df, index = 'sigma_c', columns = 'mu_c',
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def N_pivot(df):
    
    return [pd.pivot_table(df, index = 'sigma_c', columns = 'mu_c',
                          values = 'N_mean', aggfunc = 'mean')]

def rho_pivot(df):
    
    return [pd.pivot_table(df, index = 'sigma_c', columns = 'mu_c', 
                          values = 'rho', aggfunc = 'mean')]

def di_pivot(df):
    
    return [pd.pivot_table(df, index = 'sigma_c', columns = 'mu_c', 
                          values = 'instability distance', aggfunc = 'mean')]

# %%

def size_effects_plots(pivot_tables, scaling, title, cbar_label,
                       cmap, v_min_max = [0, 1], **kwargs):
    
    
    sigma_tick_labels = [np.round(np.array([np.min(data.index),
                                           np.max(data.index)]), 3)
                         for data in pivot_tables.values()]
    
    
    mu_tick_labels = [np.round(np.array([np.min(data.columns),
                                           np.max(data.columns)]), 3)
                         for data in pivot_tables.values()]
    
    sns.set_style('white')

    fig, axs = plt.subplots(2, 2, figsize = (9, 7.5), layout = 'constrained')

    fig.suptitle(title, fontsize = 24)
    fig.supxlabel('Average consumption rate, ' + r'$\frac{\mu_c}{M}$', fontsize = 22)
    fig.supylabel('Std. in consumption rate, ' + r'$\frac{\sigma_c}{\sqrt{M}}$',
                  fontsize = 22)

    for (i, ax), sigma_label, mu_label, (n_spec, data) in zip(enumerate(axs.flatten()),
                                                                sigma_tick_labels,
                                                                mu_tick_labels,
                                                                pivot_tables.items()):
                
        subfig = sns.heatmap(data, ax = ax, vmin = v_min_max[0],
                             vmax = v_min_max[1], cbar = False, cmap = cmap,
                             **kwargs)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(5, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(5, 0, 1, color = 'black', linewidth = 2)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()
        
        ax.set_xticks([0.5, 4.5], labels = mu_label, fontsize = 14,
                      rotation = 0)
        
        ax.set_yticks([0.5, 4.5], labels = sigma_label, fontsize = 14,
                      rotation = 0)
        
        ax.set_title(r'$M = $' + str(np.int32(np.rint(n_spec))), fontsize = 22)
                    
        if i == 0:
            
            mappable = subfig.get_children()[0]

            
    cbar = plt.colorbar(mappable, ax = [axs[0, 1], axs[1, 1]],
                        orientation = 'vertical')
    cbar.set_label(label = cbar_label, size = '18')
    cbar.ax.tick_params(labelsize=14)

    return fig, axs

# %%

###############################################################################################

'''

    Asymmetric scaling

'''

system_sizes = np.array([100, 150, 200, 250])

unscaled_mu_c_range = np.array([100, 200])
unscaled_sigma_range = np.array([1, 2])

fixed_parameters = {'mu_g' : 1, 'm' : 1, 'K' : 2, 'gamma' : 1}

growth_consumption_function = 'growth function of consumption'

subdirectory = "alternative_coupling_gc_finite_size_effects"

n = 5
no_communities = 5

for system_size in tqdm(system_sizes):

    mu_c_size_effects(system_size, 'asymmetric', unscaled_mu_c_range, unscaled_sigma_range,
                      fixed_parameters, growth_consumption_function, subdirectory,
                      n, no_communities)
    
# %%

system_sizes = np.array([100, 150, 200, 250])

unscaled_mu_c_range = np.array([100, 200])
unscaled_sigma_range = np.array([1, 2])

subdirectory = "alternative_coupling_gc_finite_size_effects"

n = 5

self_limiting_gc_size_effects_df_list = [size_effect_df(system_size,
                                                        unscaled_mu_c_range,
                                                        unscaled_sigma_range,
                                                        subdirectory)
                                         for system_size in system_sizes]
    
self_limiting_gc_size_effects = pd.concat([df 
                                           for df_list in self_limiting_gc_size_effects_df_list
                                           for df in df_list],
                                          axis = 0, ignore_index = True)

self_limiting_gc_size_effects[['covariance', 'rho']] = \
    pd.DataFrame(self_limiting_gc_size_effects.apply(covariance_correlation,
                                                     axis = 1, scaling = 'asymmetric').to_list())
    
del self_limiting_gc_size_effects_df_list

# %%

chaos_size_effects = self_limiting_gc_size_effects.groupby('no_species').apply(le_pivot).to_dict()
for n_spec, data in chaos_size_effects.items(): chaos_size_effects[n_spec] = data[0]

fig, axs = size_effects_plots(chaos_size_effects, 'asymmetric',
                              'The effect of ' + r'$\mu_c$' + \
                              ' on stability vanishes in the infinite size limit',
                              'Proportion of simulations with max. LE ' + \
                                  r'$> 0.0025$',
                              'Purples')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_phase.png",
            bbox_inches='tight')

# %%

N_mean_size_effects = self_limiting_gc_size_effects.groupby('no_species').apply(N_pivot).to_dict()
for n_spec, data in N_mean_size_effects.items(): N_mean_size_effects[n_spec] = data[0]

min_max_N_mean = [np.min([np.min(N_mean_size) 
                          for N_mean_size in N_mean_size_effects.values()]),
                  np.max([np.max(N_mean_size) 
                          for N_mean_size in N_mean_size_effects.values()])]

fig, axs = size_effects_plots(N_mean_size_effects, 'asymmetric',
                              'Increasing ' + r'$\mu_c$' + ' decreases avg. species abundance independent?' + \
                              '\nof system size (effect on ' + r'$\kappa$' + ' probably dominates).',
                              'Average ' r'$\langle N \rangle$' + ' across simulations',
                              'Oranges', min_max_N_mean)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_N.png",
            bbox_inches='tight')

# %%

rho_size_effects = self_limiting_gc_size_effects.groupby('no_species').apply(rho_pivot).to_dict()
for n_spec, data in rho_size_effects.items(): rho_size_effects[n_spec] = data[0]

fig, axs = size_effects_plots(rho_size_effects, 'asymmetric',
                              'Increasing ' + r'$\mu_c$' + \
                              ' decreases the correlation between growth\nand consumption,' + \
                              ' but this effect diminishes with system size.',
                              r'$ \rho $' + '(growth, consumption)',
                              'Greens')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_rho.png",
            bbox_inches='tight')

# %%

self_limiting_gc_size_effects['instability distance'] = \
    self_limiting_gc_size_effects.apply(distance_from_instability, axis = 1)

di_size_effects = self_limiting_gc_size_effects.groupby('no_species').apply(di_pivot).to_dict()
for n_spec, data in di_size_effects.items(): di_size_effects[n_spec] = data[0]

v_min = np.min([np.min(pivot) for pivot in di_size_effects.values()])
v_max = np.max([np.max(pivot) for pivot in di_size_effects.values()])

norm = TwoSlopeNorm(vmin = v_min, vcenter=0, vmax = v_max)

fig, axs = size_effects_plots(di_size_effects, 'asymmetric',
                              '(Validation of analytical results) ' + \
                              'Communities approach the\nanalytical instability boundary as they approach phase transition.',
                              r'$ \rho^2 - \frac{\phi_N \gamma^{-1}}{\phi_R}$', 'RdBu_r',
                              norm = norm)
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_unstable_cond.png",
            bbox_inches='tight')
    
# %%

'''
 Symmetric scaling with non-fixed mu_g/sqrt(M) over M, larger variance
'''

system_sizes = np.array([100, 150, 200, 250])

unscaled_mu_c_range = np.array([10, 20])
unscaled_sigma_range = np.array([1/np.sqrt(10), 2/np.sqrt(10)])

fixed_parameters = {'mu_g' : 10, 'm' : 1, 'K' : 1, 'gamma' : 1}

growth_consumption_function = 'growth function of consumption'

subdirectory = "alternative_coupling_gc_finite_size_effects_symm_larger_v"

n = 5
no_communities = 5

for system_size in tqdm(system_sizes):

    mu_c_size_effects(system_size, 'symmetric', unscaled_mu_c_range, unscaled_sigma_range,
                      fixed_parameters, growth_consumption_function, subdirectory,
                      n, no_communities)
    
# %%

system_sizes = np.array([100, 150, 200, 250])

unscaled_mu_c_range = np.array([10, 20])
unscaled_sigma_range = np.array([1/np.sqrt(10), 2/np.sqrt(10)])

n = 5
    
subdirectory = "alternative_coupling_gc_finite_size_effects_symm_larger_v"

self_limiting_gc_size_effects_symm_df_list = [size_effect_df(system_size,
                                                             unscaled_mu_c_range,
                                                             unscaled_sigma_range,
                                                             subdirectory,
                                                             scaling = 'symmetric')
                                              for system_size in system_sizes]
    
sl_gc_symm_size_effects = pd.concat([df for df_list in self_limiting_gc_size_effects_symm_df_list
                                     for df in df_list],
                                    axis = 0, ignore_index = True)

sl_gc_symm_size_effects[['covariance', 'rho']] = \
    pd.DataFrame(sl_gc_symm_size_effects.apply(covariance_correlation, axis = 1,
                                               scaling = 'symmetric').to_list())
    
del self_limiting_gc_size_effects_symm_df_list

# %%

chaos_size_effects = sl_gc_symm_size_effects.groupby('no_species').apply(le_pivot).to_dict()
for n_spec, data in chaos_size_effects.items(): chaos_size_effects[n_spec] = data[0]

fig, axs = size_effects_plots(chaos_size_effects, 'symmetric',
                              'The effect of ' + r'$\mu_c$' + \
                              ' is independent of system size in\nmodel 1 with symmetric scaling',
                              'P(chaos)' r'$ = P( \lambda > 0.0025)$', 'Purples')
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_phase_symm.png",
            bbox_inches='tight')

# %%

rho_size_effects = sl_gc_symm_size_effects.groupby('no_species').apply(rho_pivot).to_dict()
for n_spec, data in rho_size_effects.items(): rho_size_effects[n_spec] = data[0]

fig, axs = size_effects_plots(rho_size_effects, 'symmetric',
                              'Increasing ' + r'$\mu_c$' + \
                              ' decreases the correlation between growth\nand consumption,' + \
                              ' independent of system size.',
                              r'$ \rho $' + '(growth, consumption)',
                              'Greens')
    
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_size_effect_mu_c_rho_symm.png",
            bbox_inches='tight')

# %%

sl_gc_symm_size_effects['instability distance'] = \
    sl_gc_symm_size_effects.apply(distance_from_instability, axis = 1)

di_size_effects = sl_gc_symm_size_effects.groupby('no_species').apply(di_pivot).to_dict()
for n_spec, data in di_size_effects.items(): di_size_effects[n_spec] = data[0]

v_min = np.min([np.min(pivot) for pivot in di_size_effects.values()])
v_max = np.max([np.max(pivot) for pivot in di_size_effects.values()])

norm = TwoSlopeNorm(vmin = v_min, vcenter=0, vmax = v_max)

fig, axs = size_effects_plots(di_size_effects, 'symmetric', '', '', "RdBu_r",
                              v_min_max = [0,1], norm = norm)

