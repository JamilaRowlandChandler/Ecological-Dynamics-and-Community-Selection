# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:05:35 2025

@author: jamil
"""

# %%

########################

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe
import pandas as pd
import seaborn as sns
import colorsys
import os
import sys
from tqdm.auto import tqdm

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

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

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     is_logged = None, specify_min_max = None,
                     mosaic = None, gridspec_kw = None):
    
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
    
    pivot_tables = {variable : df.pivot(index = x, columns = y, values = variable)
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
        
        sns.heatmap(pivot_tables_plot[variables[0]], ax = axs,
                    vmin = v_min_max[variables[0]][0], vmax = v_min_max[variables[0]][1],
                    cbar = True, cmap = cmaps)

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
            
            sns.heatmap(pivot_tables_plot[variable], ax = ax,
                        vmin = v_min_max[variable][0], vmax = v_min_max[variable][1],
                        cbar = True, cmap = cmap)
    
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

def solve_self_limit(parameters):
    
    sol_self_limit = sce.boundary(parameters, equation_func = sce.self_consistency_equations_sl_gc_c,
                                   solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                        'phi_R', 'R_mean', 'q_R', 'chi_R'],
                                   bounds = bounds, x_init = x_init,
                                   solver = sce.solve_equations_least_squares)
    
    sol_self_limit[['dNde', 'dRde', 'ms_loss']] = \
        pd.DataFrame(sol_self_limit.apply(sce.solve_for_multistability, axis = 1,
                                          multistability_equation_func = 'self-limiting gc c').to_list())
        
    return sol_self_limit

# %%

'''
 
    Solve the self consistency equations for a range of mu and sigma
    
    ####### Parameters #######

'''

system_sizes = np.array([150, 250, 500])

mu_c_range = np.array([100, 200])
sigma_range = np.array([1, 2])
n = 10

mu_c_sigma_combinations = sce.parameter_combinations([mu_c_range, sigma_range], n)

# array of variable parameter combinations
variable_parameters_sizes = [np.vstack([mu_c_sigma_combinations,
                                        mu_c_sigma_combinations[1, :]/np.sqrt(system_size)])
                             for system_size in system_sizes]
# fixed parameters
fixed_parameters_no_size = {'mu_g' : 1, 'mu_m' : 1, 'sigma_m' : 0, 'mu_K' : 1,
                            'sigma_K' : 1, 'gamma' : 1}
fixed_parameters_sizes = [fixed_parameters_no_size | {'M' : system_size} 
                          for system_size in system_sizes]

# array of all parameter combinations
parameters_sizes = [sce.variable_fixed_parameters(variable_parameters, ['mu_c', 'sigma_c', 'sigma_g'],
                                                  fixed_parameters)
                    for variable_parameters, fixed_parameters in
                    zip(variable_parameters_sizes, fixed_parameters_sizes)]

'''
    ######## Variables ########
    
    We are solving for phi_N, phi_R, N_mean, R_mean,q_N, q_R, v_N, and chi_R.

'''

# variable bounds 
bounds = ([0, 0, 0, -1000, 0, 0, 0, 0],
          [1, 1000, 1000, 0, 1, 1000, 1000, 1000])

# initial values
x_init = [0.5, 0.01, 0.1, -0.1, 0.5, 0.01, 0.1, 0.1]

# %%

solved_self_consist_sizes = [solve_self_limit(parameters) 
                             for parameters in tqdm(parameters_sizes, position = 0,
                                                    leave = True)]

# %%

test_plot = solved_self_consist_sizes[2]
#test_plot.iloc[np.where(test_plot['loss'] > - 15)]['N_mean'] = np.nan

generic_heatmaps(test_plot,
                 'mu_c', 'sigma_c', r'$\mu_c / M$', r'$\sigma_c / \sqrt{M}$',
                 ['loss'], 'Greens', '', (1, 1), (5, 4))
                 #,
                 #specify_min_max = {'N_mean' : [0, 0.025]})

# %%

mu_c_range = np.array([0.3, 1.3])
sigma_range = np.array([1, 1.6])
n = 10

system_size = 250

mu_c_sigma_combinations = sce.parameter_combinations([mu_c_range, sigma_range], n)

# array of variable parameter combinations
variable_parameters = np.vstack([mu_c_sigma_combinations[0, :],
                                 mu_c_sigma_combinations[1, :],
                                 mu_c_sigma_combinations[1, :]/np.sqrt(system_size)])
# fixed parameters
fixed_parameters = {'mu_g' : 1, 'mu_m' : 1, 'sigma_m' : 0, 'mu_K' : 1,
                    'sigma_K' : 1, 'gamma' : 1}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters,
                                           ['mu_c', 'sigma_c', 'sigma_g'],
                                           fixed_parameters)

'''
    ######## Variables ########
    
    We are solving for phi_N, phi_R, N_mean, R_mean,q_N, q_R, v_N, and chi_R.

'''

# variable bounds 
bounds = ([0, 0, 0, -1000, 0, 0, 0, 0],
          [1, 1000, 1000, 0, 1, 1000, 1000, 1000])

# initial values
x_init = [0.5, 0.01, 0.1, -0.1, 0.5, 0.01, 0.1, 0.1]

sol_self_limit = sce.boundary(parameters, equation_func = sce.self_consistency_equations_sl_g_cg,
                               solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                    'phi_R', 'R_mean', 'q_R', 'chi_R'],
                               bounds = bounds, x_init = x_init,
                               solver = sce.solve_equations_least_squares)

sol_self_limit[['dNde', 'dRde', 'ms_loss']] = \
    pd.DataFrame(sol_self_limit.apply(sce.solve_for_multistability, axis = 1,
                                      multistability_equation_func = 'self-limiting g cg inf').to_list())
#%%
generic_heatmaps(sol_self_limit, 'mu_c', 'sigma_c', r'$\mu_c$', r'$\sigma_c$',
                 ['loss'], 'Greens', '', (1, 1), (5, 4))

# %%

generic_heatmaps(sol_self_limit, 'mu_c', 'sigma_c', r'$\mu_c$', r'$\sigma_c$',
                 ['dNde'], 'Purples', '', (1, 1), (5, 4))
                 #, is_logged = ['dNde'])

# %%

system_size = 10000
unscaled_sigma_range = np.array([0.1, 1])
rho_range = [0.1, 0.99]
mu_c = 150

n = 12

rho_sigma_combinations = sce.parameter_combinations([rho_range, unscaled_sigma_range],
                                                    n)

mu_solve_inputs = np.vstack([rho_sigma_combinations,
                             rho_sigma_combinations[1, :]/10,
                             np.repeat(mu_c, n**2), np.repeat(system_size, n**2)])

solved_mu_gs = np.apply_along_axis(solve_for_mu_g, axis = 0, arr = mu_solve_inputs)

variable_parameters = np.vstack([solved_mu_gs,
                                 rho_sigma_combinations[1, :],
                                 rho_sigma_combinations[1, :]/10])

fixed_parameters = {'mu_c' : mu_c, 'mu_m' : 1, 'sigma_m' : 0, 'mu_K' : 1,
                    'sigma_K' : 1, 'gamma' : 1}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters,
                                           ['mu_g', 'sigma_c', 'sigma_g'],
                                           fixed_parameters)

'''
    ######## Variables ########
    
    We are solving for phi_N, phi_R, N_mean, R_mean,q_N, q_R, v_N, and chi_R.

'''

# variable bounds 
bounds = ([0, 0, 0, -1000, 0, 0, 0, 0],
          [1, 1000, 1000, 0, 1, 1000, 1000, 1000])

# initial values
x_init = [0.5, 0.01, 0.1, -0.1, 0.5, 0.01, 0.1, 0.1]

sol_self_limit = sce.boundary(parameters,
                              equation_func = sce.self_consistency_equations_sl_gc_c_inf,
                               solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                    'phi_R', 'R_mean', 'q_R', 'chi_R'],
                               bounds = bounds, x_init = x_init,
                               solver = sce.solve_equations_least_squares)

def correlation_gc_c_inf(df, M):
    

    mu_c, mu_g, sigma_c, sigma_g = df['mu_c'], df['mu_g'], df['sigma_c'], df['sigma_g']
            
    correlation = (mu_g * sigma_c)/np.sqrt((mu_g * sigma_c)**2 + \
                                           (sigma_c * sigma_g)**2 + \
                                           ((mu_c * sigma_g)**2)/M)
    
    return correlation

sol_self_limit['rho'] = sol_self_limit.apply(correlation_gc_c_inf, axis = 1,
                                             M = system_size)

# %%

sol_self_limit['rho'] = np.round(sol_self_limit['rho'], 6)
sol_self_limit['sigma_c'] = np.round(sol_self_limit['sigma_c'], 6)
sol_self_limit['mu_c'] = np.round(sol_self_limit['mu_c'], 6)

generic_heatmaps(sol_self_limit, 'rho', 'sigma_c', r'$\rho$', r'$\sigma_c$',
                 ['loss'], 'Greens', '', (1, 1), (5, 4))

# %%

sol_self_limit[['dNde', 'dRde', 'ms_loss']] = \
    pd.DataFrame(sol_self_limit.apply(sce.solve_for_multistability, axis = 1,
                                      multistability_equation_func = 'self-limiting gc c inf').to_list())

# %%

generic_heatmaps(sol_self_limit, 'rho', 'sigma_c', r'$\rho$', r'$\sigma_c$',
                 ['dNde'], 'Purples', '', (1, 1), (5, 4), is_logged = ['dNde'])
