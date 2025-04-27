# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:38:02 2025

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

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models')
import self_consistency_equation_functions as sce

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_2')
#from models import Consumer_Resource_Model

# %%

def useful_plots(df, variable = 'phi_N',
                 variable_label = 'Species surivial fraction, ' + r'$\mathbf{\phi_N}$'):
    
    loss_pivot = df.pivot(index = 'rho', columns = 'sigma_c', values = 'loss')

    val_check_pivot = df.pivot(index = 'rho', columns = 'sigma_c', values = variable)

    multistability_pivot = df.pivot(index = 'rho', columns = 'sigma_c', values = 'dNde')
    multistability_pivot = np.log10(np.abs(multistability_pivot))

    cmaps = ['Greens', 'Purples', 'Blues']
    v_min_max = [[np.min(loss_pivot), np.max(loss_pivot)], [0, np.max(val_check_pivot)],
                 [0, np.max(multistability_pivot)]]
    titles = [r'$\mathbf{\log_{10}}$' + '(Loss Function)',
              variable_label,
              'Multi-stability metric, ' + \
                  r'$\mathbf{\log_{10}\langle (dN^+/d\epsilon)^2 \rangle}$']

    sns.set_style('white')

    fig, axs = plt.subplots(1, 3, figsize = (15, 4.5), sharex = True, sharey = True, 
                            layout = 'constrained')

    fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
                  fontsize = 16, weight = 'bold')
    fig.supylabel('Correlation between growth and\nconsumption rates ' + r'$(\rho)$',
                  fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')

    for ax, data, cmap, v, title in zip(axs, [loss_pivot, val_check_pivot, multistability_pivot],
                                        cmaps, v_min_max, titles):
        
        subfig = sns.heatmap(data, ax=ax, vmin = v[0], vmax = v[1], cbar = True, cmap = cmap)

        ax.set_yticks([0.5, n - 0.5],
                      labels = [np.round(np.min(df['rho']), 3),
                                np.round(np.max(df['rho']), 3)], fontsize = 14)
        ax.set_xticks([0.5, n - 0.5], labels = [np.round(np.min(df['sigma_c']), 3),
                                                np.round(np.max(df['sigma_c']), 3)],
                      fontsize = 14, rotation = 0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()
        ax.set_title(title, fontsize = 16, weight = 'bold')
        
    return fig, axs
    
# %%

def generic_heatmaps(df, variables, cmaps, titles, fig_dims, figsize,
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
    
    pivot_tables = {variable : df.pivot(index = 'rho', columns = 'sigma_c',
                                        values = variable)
                    for variable in variables}
    
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

    fig.supxlabel('std. in growth/consumption rates ' + r'$(\sigma_c$' + ' or ' + r'$\sigma_g)$',
                  fontsize = 16, weight = 'bold')
    fig.supylabel('Correlation between growth and\nconsumption rates ' + r'$(\rho)$',
                  fontsize = 16, weight = 'bold', horizontalalignment = 'center',
                  verticalalignment = 'center')

    for ax, variable, cmap, title in zip(axs.values(), variables, cmaps, titles):
        
        sns.heatmap(pivot_tables_plot[variable], ax = ax,
                    vmin = v_min_max[variable][0], vmax = v_min_max[variable][1],
                    cbar = True, cmap = cmap)

        ax.set_yticks([0.5, len(np.unique(df['rho'])) - 0.5],
                      labels = [np.round(np.min(df['rho']), 3),
                                np.round(np.max(df['rho']), 3)], fontsize = 14)
        ax.set_xticks([0.5, len(np.unique(df['sigma_c'])) - 0.5], 
                      labels = [np.round(np.min(df['sigma_c']), 3),
                                np.round(np.max(df['sigma_c']), 3)],
                      fontsize = 14, rotation = 0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()
        ax.set_title(title, fontsize = 16, weight = 'bold')
        
    return fig, axs

# %%

'''
 ================================================================================
     Calculations from Blumenthal et al. (2024), for proof of concept
 ================================================================================
 
    In these models, the resource supply is self-limiting (has logistic growth).
    
 
    Solve the self consistency equations for a range of mu and sigma
    
    ####### Parameters #######

'''

sigma_range = [2, 5]
rho_range = [0.001, 1]
n = 20

# generate n values of rho and sigma within range
rho_sigma_combinations = sce.parameter_combinations([rho_range, sigma_range], n)

# array of variable parameter combinations
variable_parameters = np.vstack([rho_sigma_combinations, rho_sigma_combinations[1, :]])

# fixed parameters
fixed_parameters = {'mu_c' : 200, 'mu_g' : 200, 'mu_m' : 1, 'sigma_m' : 0.1,
                    'mu_K' : 1, 'sigma_K' : 0.1, 'gamma' : 256/256}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters, ['rho', 'sigma_c', 'sigma_g'],
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

# solve the self-consistency equations
sol_self_limit = sce.boundary(parameters, equation_func = sce.self_consistency_equations_e,
                              solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                   'phi_R', 'R_mean', 'q_R', 'chi_R'],
                              bounds = bounds, x_init = x_init,
                              solver = sce.solve_equations_least_squares)

'''

    Multistability calculation

'''

sol_self_limit[['dNde', 'dRde']] = \
    pd.DataFrame(sol_self_limit.apply(sce.solve_for_multistability, axis = 1,
                                      multistability_equation_func = 'self-limiting').to_list())
    
'''

    Plotting

'''

fig, axs = useful_plots(sol_self_limit)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_consistency_blumenthal.png",
            bbox_inches='tight')

plt.show()

sol_self_limit['CE'] = sol_self_limit['phi_N']/sol_self_limit['phi_R']
sol_self_limit['CE'].iloc[np.where(sol_self_limit['rho'] < 0.1)] = np.nan

fig, axs = useful_plots(sol_self_limit, 'CE', r'$\phi_N / \phi_R$')
plt.show()

# %%

'''
 ================================================================================
                             Calculations for my models
 ================================================================================
 
    In my models, resources are externally-supplied (chemostat-style growth).
    
 
    Solve the self consistency equations for a range of mu and sigma
    
    ####### Parameters #######

'''

no_species = 100
no_resources = 100

sigma_range = np.array([0.001, 2.5])
rho_range = np.array([0.001, 1])
n = 20

# generate n values of rho and sigma within range
rho_sigma_combinations = sce.parameter_combinations([rho_range, sigma_range], n)

# array of variable parameter combinations
variable_parameters = np.vstack([rho_sigma_combinations, rho_sigma_combinations[1, :]])

# fixed parameters
fixed_parameters = {'mu_c' : 1, 'mu_g' : 1, 'mu_m' : 1, 'sigma_m' : 0.1,
                    'mu_K' : 1, 'sigma_K' : 0.1, 'gamma' : no_species/no_resources,
                    'mu_D': 1, 'sigma_D' : 0}

# array of all parameter combinations
parameters = sce.variable_fixed_parameters(variable_parameters, ['rho', 'sigma_c', 'sigma_g'],
                                           fixed_parameters)

'''
    ######## Variables ########
    
    We are solving for phi_N, N_mean, R_mean,q_N, q_R, v_N, and chi_R.

'''

# variable bounds 
bounds = ((0, 1), (0, 100), (0, 100), (-100, 0), (0, 100), (0, 100), (0, 100))

# initial values, aided by simulations
example_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_external_resource_1.0_0.11842105263157894.pkl")
example_community = example_communities[0]

phi_N = example_community.species_survival_fraction['lineage 0']
N_abundances = example_community.ODE_sols['lineage 0'].y[:example_community.no_species, -1]
N_mean = np.mean(N_abundances)
q_N = np.mean(N_abundances**2)

R_abundances = example_community.ODE_sols['lineage 0'].y[example_community.no_species:, -1]
R_mean = np.mean(R_abundances)
q_R = np.mean(R_abundances**2)

known_quantities = {key : value for key, value in zip(['phi_N', 'N_mean', 'q_N',
                                                       'R_mean', 'q_R'],
                                                      np.round(np.array([phi_N,
                                                                          N_mean,
                                                                          q_N,
                                                                          R_mean, 
                                                                          q_R]), 1))}

x_init = [known_quantities['phi_N'], known_quantities['N_mean'],
          known_quantities['q_N'], -1, known_quantities['R_mean'],
          known_quantities['q_R'], 0.5]

# %%

# solve the self-consistency equations with differential evolution solver
sol_external = sce.boundary(parameters, equation_func = sce.self_consistency_equations,
                            solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                 'R_mean', 'q_R', 'chi_R'],
                             bounds = bounds, x_init = x_init,
                             solver = sce.solve_equations_different_evolve)

#sol_external = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_1.pkl")

'''

    Multistability calculation

'''

sol_external[['dNde', 'dRde']] = \
    pd.DataFrame(sol_external.apply(sce.solve_for_multistability, axis = 1,
                                    multistability_equation_func = 'externally supplied').to_list())
    
sce.pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_1.pkl",
                sol_external)
    
fig, axs = useful_plots(sol_external[sol_external['sigma_c'] > 0.001], variable = 'phi_N')
plt.show()

# %%

# Repeat with least squares

bounds2 = ((0, 0, 0, -1e6, 0, 0, 0),
           (1, 1e6, 1e6, 0, 1e6, 1e6, 1e6))

sol_external_2 = sce.boundary(parameters, equation_func = sce.self_consistency_equations,
                              solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                   'R_mean', 'q_R', 'chi_R'],
                              bounds = bounds2, x_init = x_init,
                              solver = sce.solve_equations_least_squares)

'''

    Multistability calculation

'''

sol_external_2[['dNde', 'dRde']] = \
    pd.DataFrame(sol_external_2.apply(sce.solve_for_multistability, axis = 1,
                                      multistability_equation_func = 'externally supplied').to_list())
    
sol_external_2['AB - 1'] = sol_external_2.apply(sce.distance_from_multistability_threshold,
                                                axis = 1)
    
'''

    Plotting

'''

fig, axs = useful_plots(sol_external_2[sol_external_2['sigma_c'] > 0.001], variable = 'phi_N')
plt.show()

#sce.pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_ls_wengping_eq.pkl",
#                sol_external_2)

# %%

sol_external_2 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/sce_solved_1_ls_2.pkl")

# %%

def recalculate_dRde(y):
    
    eq_kwargs_names = ['mu_c', 'sigma_c', 'sigma_g', 'rho', 'gamma', 'mu_D',
                       'sigma_D', 'mu_K',
                       'v_N', 'phi_N', 'N_mean', 'q_N', 'dNde']
    
    eq_kwargs = {key : y[key] for key in eq_kwargs_names}
    
    def dRde_equation(rho, gamma, mu_c, sigma_c, sigma_g, mu_K, mu_D, sigma_D,
                      phi_N, N_mean, q_N, v_N, dNde):
    
        omega = mu_D + (mu_c * N_mean)/gamma
        B = (sigma_g * sigma_c * rho * v_N)/gamma
        x = (phi_N * sigma_c**2)/gamma

        dRde = (dNde + 1) * (1/(2*B))**2 * (x*(1 - \
                                               (4*omega)/(np.sqrt(omega**2 - 4*B*mu_K)) + \
                                              4*(omega**2 + (sigma_c*sigma_D)**2)/(omega**2 - 4*B*mu_K)) + \
                                            (x**2)*(q_N/(omega**2 - 4*B*mu_K)))
    
        return dRde
    
    return dRde_equation(**eq_kwargs)

sol_external_2['dRde recalc'] = sol_external_2.apply(recalculate_dRde, axis = 1)

# %%

dVde_min_max = [np.min(np.log10(np.abs(np.concatenate((sol_external_2['dNde'],
                                                       sol_external_2['dRde']))))),
                np.max(np.log10(np.abs(np.concatenate((sol_external_2['dNde'],
                                                       sol_external_2['dRde'])))))]

fig, axs = generic_heatmaps(sol_external_2, 
                            variables = ['phi_N', 'dRde', 'chi_R', 'dNde', 'AB - 1'],
                            titles = ['Species survival fraction, ' + r'$\phi_N$',
                                      r'$\mathbf{\log_{10}\langle (dR/d\epsilon)^2 \rangle}$',
                                      r'$\log_{10}(\chi_{(R)})$',
                                      r'$\mathbf{\log_{10}\langle (dN^+/d\epsilon)^2 \rangle}$',
                                      'Multistability threshold, ' +
                                      r'$\log_{10}|AB - 1|$'],
                            fig_dims = (2, 3), figsize = (11, 12),
                            cmaps = ['Greens', 'Purples', 'Blues', 'Purples', 
                                     'viridis_r'],
                            is_logged = ['chi_R', 'dNde',
                                         'dRde',
                                         'AB - 1'],
                            specify_min_max = {'phi_N' : [0, 0.5],
                                               'dNde' : dVde_min_max,
                                               'dRde' : dVde_min_max
                                               },
                            mosaic = [['top1', 'top2'], 
                                      ['middle1', 'middle2'],
                                      ['bottom1', '.']],
                            gridspec_kw = {'width_ratios' : [1, 1],
                                           'wspace' : 0.15, 'hspace' : 0.15,
                                           'height_ratios' : [1, 1, 1]})

fig.text(0.5, 1.02, 'In the infeasible region (where no species survive), '
         + r'$\langle (dR/d\epsilon)^2 \rangle = 0$.', size = 20, weight = 'bold',
         horizontalalignment = 'center', verticalalignment = 'center')

fig.text(0.5, 0.68, 'Because ' + r'$\chi_{(R)} = 0$' + ' in the infeasible region, '
         + r'$\langle (dN/d\epsilon)^2 \rangle \rightarrow \infty$.', size = 20,
         weight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')

fig.text(0.5, 0.34, 'This means the multistability condition, ' + r'$AB - 1 = 0$' + 
         ', is not met', size = 20, weight = 'bold', horizontalalignment = 'center',
         verticalalignment = 'center')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/multistability_infeasibility_discussion.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/multistability_infeasibility_discussion.png",
            bbox_inches='tight')
