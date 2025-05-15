# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:22:32 2025

@author: jamil
"""

########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
from copy import copy
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import KroghInterpolator
from scipy.interpolate import make_smoothing_spline
import scipy.interpolate as interpolate

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling")
from simulation_functions import distance_from_instability, \
    distance_from_infeasibility, species_packing, \
        create_df_and_delete_simulations_2, prop_chaotic
    
# %%

def finite_sized_parm_sets(Ms, C_range, sigma_C, n, fixed_parameters):
    
    def generate_M_dependent_parms(M):
        
        mu_c_range = C_range/M 
        sigma = sigma_C/np.sqrt(M)
        
        fixed_parameters_copy = copy(fixed_parameters)
        fixed_parameters_copy['M'] = M
        
        parameters = generate_parameters(mu_c_range, [sigma, sigma], n,
                                         fixed_parameters_copy)
        
        return parameters
    
    parameters = list(np.concatenate([generate_M_dependent_parms(M) for M in Ms]))
    for parm_set in parameters: parm_set['sigma_g'] = sigma_C/np.sqrt(150)
    
    return parameters
                
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

def solve_sces_finite(parameters, solver_kwargs, x_init = None):
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
     
    # variable bounds 
    
    #bounds = [(0, 1), (1e-10, 1e15), (1e-10, 1e15), (-1e15, -1e-10),
    #          (0, 1), (1e-10, 1e15), (1e-10, 1e15), (1e-10, 1e15)]

    # initial values
    
    if x_init:
        
        #solver_kwargs['x_scale'] = np.power(10, np.floor(np.log10(np.abs(x_init[0]))))
        xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
        solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                         for xscale in xscales]
    
    else:

        x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
        solver_kwargs['x_scale'] = np.power(10, np.floor(np.log10(np.abs(x_init))))
    
    sol = sce.solve_sces(parameters, 'self-limiting gc c M',
                         solved_quantities = solved_quantities,
                         bounds = bounds, x_init = x_init,
                         solver = sce.solve_equations_least_squares,
                         solver_kwargs = solver_kwargs)
 #                     solver = sce.solve_equations_basinhopping,
       
    sol[['dNde', 'dRde', 'ms_loss']] = \
        pd.DataFrame(sol.apply(sce.solve_for_multistability, axis = 1,
                               multistability_equation_func = 'self-limiting gc c M').to_list())
        
    return sol

# %%

def solve_phase_boundary(parameters, solved_quantities, bounds, x_init,
                         solver_kwargs):
    
    xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
    solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                     for xscale in xscales]
    
    sol = sce.solve_sces_2(parameters, 'self-limiting gc c M',
                           solved_quantities = solved_quantities,
                           bounds = bounds, x_init = x_init,
                           solver = sce.solve_equations_least_squares,
                           solver_kwargs = solver_kwargs, 
                           include_multistability = True)
    
    sol['rho'] = (sol['mu_g'] * sol['sigma_c'])/np.sqrt((sol['mu_g'] * sol['sigma_c'])**2 + \
                                                        (sol['mu_c'] * sol['sigma_g'])**2 + \
                                                        (sol['sigma_g'] * sol['sigma_c'])**2) 
    
    sol['no_species'] = sol['M']/sol['gamma'] 
    sol.rename(columns = {'M' : 'no_resources'}, inplace = True)
    sol['instability_condition'] = sol.apply(distance_from_instability, axis = 1)
    sol.rename(columns = {'no_resources' : 'M'}, inplace = True)
    sol.drop('no_species', axis = 1, inplace = True)
       
    sol[['dNde', 'dRde', 'ms_loss']] = \
        pd.DataFrame(sol.apply(sce.solve_for_multistability, axis = 1,
                               multistability_equation_func = 'self-limiting gc c M').to_list())
        
    return sol

# %%

def generic_heatmaps(df, x, y, xlabel, ylabel, variables, cmaps, titles,
                     fig_dims, figsize,
                     pivot_functions = None, is_logged = None, specify_min_max = None,
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
    
    if pivot_functions is None:
    
        pivot_tables = {variable : df.pivot(index = y, columns = x, values = variable)
                        for variable in variables}
        
    else:
        
        pivot_tables = {variable : (df.pivot(index = y, columns = x, values = variable)
                                    if pivot_functions[variable] is None 
                                    else
                                    pivot_functions[variable](df, index = y,
                                                              columns = x,
                                                              values = variable)[0]) 
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

def generate_simulation_df():
    
    def community_properties_df(directory,
                                parm_attributes = ['no_species', 'no_resources',
                                                   'mu_c', 'sigma_c', 'mu_g',
                                                   'sigma_g', 'm', 'K']):
        
        dfs = [create_df_and_delete_simulations_2(directory + "/", file, parm_attributes)
               for file in os.listdir(directory)]
                
        return dfs

    def covariance_correlation(df):
                
        mu_c, mu_g, sigma_c, sigma_g = df['mu_c'], df['mu_g'], df['sigma_c'], df['sigma_g']

        denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2) + (sigma_c * sigma_g)**2
             
        covariance = (mu_g * sigma_c**2)
        correlation = (mu_g * sigma_c)/np.sqrt(denominator)
            
        return covariance, correlation
    
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

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

df_simulation = generate_simulation_df()
df_simulation['no_resources'] = np.int32(df_simulation['no_resources'])
df_simulation['<C>'] = np.round(df_simulation['mu_c'] * df_simulation['no_resources'], 2)

# %%

parms_x_inits = df_simulation.groupby(['no_resources', 'mu_c', 'sigma_c'])[['phi_N',
                                                                      'N_mean',
                                                                      'q_N',
                                                                      'phi_R', 
                                                                      'R_mean',
                                                                      'q_R']].mean().reset_index()

parms_x_inits = np.round(parms_x_inits, 6)
parms_x_inits.rename(columns = {'no_resources' : 'M'}, inplace = True)

variable_parameters = parms_x_inits[['M', 'mu_c', 'sigma_c']].to_dict('records')
parameters = sce.variable_fixed_parameters(variable_parameters, None,
                                           {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                            'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                            'sigma_m' : 0, 'gamma' : 1})

x_inits = parms_x_inits[['phi_N', 'N_mean', 'q_N', 'phi_R', 'R_mean', 'q_R']].to_dict('records')

complete_x_inits = [[x0['phi_N'], x0['N_mean'], x0['q_N'], -0.1,
                    x0['phi_R'], x0['R_mean'], x0['q_R'], 0.05]
                    for x0 in x_inits]

solved_sces_close_xinit = solve_sces_finite(parameters,
                                            solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                                            x_init = complete_x_inits)

solved_sces_close_xinit['M'] = np.int32(solved_sces_close_xinit['M'])
solved_sces_close_xinit['<C>'] = np.round(solved_sces_close_xinit['mu_c'] * solved_sces_close_xinit['M'], 2)

# %%

generic_heatmaps(solved_sces_close_xinit,
                 'M', '<C>', r'$M$', r'$<C>$',
                 ['dNde'], 'Blues',
                 r'$\log_{10} (\mid \langle dN / d \epsilon \rangle^2 \mid)$',
                 (1, 1), (6.5, 6.5),
                 is_logged = ['dNde'])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_analyticalphase.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M__analyticalphase.svg",
            bbox_inches='tight')
                                          
# %%

################# Solving for the phase boundary ##############################

max_dnde_by_M = solved_sces_close_xinit.groupby('M')['dNde'].max().to_numpy()
sorter = np.argsort(solved_sces_close_xinit['dNde'].to_numpy())
max_dNde_df = solved_sces_close_xinit.iloc[sorter[np.searchsorted(solved_sces_close_xinit['dNde'].to_numpy(),
                                                       max_dnde_by_M,
                                                       sorter=sorter)]]

max_dNde_df = max_dNde_df.iloc[np.where(max_dNde_df['M'] > 25)]

variable_parameters_p = max_dNde_df[['M', 'sigma_c']].to_dict('records')
parameters_p = sce.variable_fixed_parameters(variable_parameters_p, None,
                                           {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                            'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                            'sigma_m' : 0, 'gamma' : 1})

x_inits_p = max_dNde_df[['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R',
                         'mu_c']].to_dict('records')
x_inits_p = [list(x_init_p.values()) for x_init_p in x_inits_p]

solved_phase = solve_phase_boundary(parameters_p,
                                      solved_quantities = ['phi_N', 'N_mean',
                                                           'q_N', 'v_N',
                                                           'phi_R', 'R_mean',
                                                           'q_R', 'chi_R',
                                                           'mu_c'],
                                      bounds = ([1e-10, 1e-10, 1e-10, -1e15,
                                                 1e-10, 1e-10, 1e-10, 1e-10,
                                                 0.4],
                                                [1, 1e15, 1e15, 1e-10,
                                                 1, 1e15, 1e15, 1e15,
                                                 6]),
                                      x_init = x_inits_p,
                                      solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13})

solved_phase['<C>'] = np.round(solved_phase['mu_c'] * solved_phase['M'], 2)

# %%

solved_phase_plot = solved_phase.iloc[np.where(solved_phase['loss'] < -10)]

resource_pool_sizes = np.unique(df_simulation['no_resources'])

fig, axs = generic_heatmaps(df_simulation,
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

y_phase = solved_phase_plot['<C>'].to_numpy() - np.min(solved_sces_close_xinit['<C>'])
diffs = np.unique(np.round(np.abs(np.diff(solved_sces_close_xinit.iloc[np.where(solved_sces_close_xinit['M'] == 100)]['<C>'])), 1))
divider = diffs[diffs > 0]

y_vals = (y_phase/divider) + 0.5

x_vals = (np.min(solved_phase_plot['M'])/25 - 1) + \
            np.arange(0.5, len(solved_phase_plot['M']) + 0.5, 1)
            
xy_interpolator = KroghInterpolator(x_vals, y_vals)
x_interpolated = np.arange(x_vals[0], x_vals[-1] + 0.2, 0.2)
y_interpolated = xy_interpolator(x_interpolated)

smoothed_xy = interpolate.m

make_smoothing_spline(x_interpolated, y_interpolated)
    
sns.lineplot(x = x_interpolated, y = smoothed_xy(x_interpolated),
             ax = axs, color = 'black', linewidth = 5)
sns.scatterplot(x = x_vals, y = y_vals, ax = axs, color = 'black', s = 150)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase.svg",
            bbox_inches='tight')

# %%

################# Solving for the phase boundary ##############################

# with interpolation, but this can be done another time.

max_dnde_by_M = solved_sces_close_xinit.groupby('M')['dNde'].max().to_numpy()
sorter = np.argsort(solved_sces_close_xinit['dNde'].to_numpy())
max_dNde_df = solved_sces_close_xinit.iloc[sorter[np.searchsorted(solved_sces_close_xinit['dNde'].to_numpy(),
                                                       max_dnde_by_M,
                                                       sorter=sorter)]]

max_dNde_df = max_dNde_df.iloc[np.where(max_dNde_df['M'] > 75)]

interpolator = BarycentricInterpolator(max_dNde_df['M'],
                                       max_dNde_df[['phi_N', 'N_mean', 'q_N', 'v_N',
                                                    'phi_R', 'R_mean', 'q_R', 'chi_R',
                                                    'mu_c', 'sigma_c']])

interpolated_data = pd.DataFrame(interpolator(np.arange(np.min(max_dNde_df['M']),
                                                        np.max(max_dNde_df['M']) + 5, 5)),
                                 columns = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                            'phi_R', 'R_mean', 'q_R', 'chi_R',
                                            'mu_c', 'sigma_c'])

interpolated_data['M'] = np.arange(np.min(max_dNde_df['M']),
                                   np.max(max_dNde_df['M']) + 5, 5)

variable_parameters_p2 = interpolated_data[['M', 'sigma_c']].to_dict('records')
parameters_p2 = sce.variable_fixed_parameters(variable_parameters_p2, None,
                                              {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                               'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                               'sigma_m' : 0, 'gamma' : 1})

x_inits_p2 = interpolated_data[['phi_N', 'N_mean', 'q_N', 'v_N',
                                'phi_R', 'R_mean', 'q_R', 'chi_R',
                                'mu_c']].to_dict('records')
x_inits_p2 = [list(x_init_p.values()) for x_init_p in x_inits_p2]

solved_phase2 = solve_phase_boundary(parameters_p2,
                                     solved_quantities = ['phi_N', 'N_mean',
                                                          'q_N', 'v_N',
                                                          'phi_R', 'R_mean',
                                                          'q_R', 'chi_R',
                                                          'mu_c'],
                                     bounds = ([1e-10, 1e-10, 1e-10, -1e15,
                                                1e-10, 1e-10, 1e-10, 1e-10,
                                                0.4],
                                               [1, 1e15, 1e15, 1e-10,
                                                1, 1e15, 1e15, 1e15,
                                                6]),
                                     x_init = x_inits_p2,
                                     solver_kwargs = {'xtol' : 1e-15, 'ftol' : 1e-15})

solved_phase2['<C>'] = np.round(solved_phase2['mu_c'] * solved_phase2['M'], 2)

sns.scatterplot(solved_phase2.iloc[np.where(solved_phase2['loss'] < -10)],
                x = 'M', y = '<C>')

solved_phase2.loc[-1] = solved_phase.loc[1]
solved_phase2.index = solved_phase2.index + 1  
solved_phase2.sort_index(inplace=True)

# %%

resource_pool_sizes = np.unique(df_simulation['no_resources'])

fig, axs = generic_heatmaps(df_simulation,
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

y_phase = solved_phase2['<C>'].to_numpy() - np.min(solved_sces_close_xinit['<C>'])
diffs = np.unique(np.round(np.abs(np.diff(solved_sces_close_xinit.iloc[np.where(solved_sces_close_xinit['M'] == 100)]['<C>'])), 1))
divider = diffs[diffs > 0]
y_vals = (y_phase/divider) + 0.5

x_vals = np.concatenate([[2.5], np.arange(3.5, 9.7, 0.2)])

smoothed_vals = make_smoothing_spline(x_vals[solved_phase2['loss'] < -26],
                                      y_vals[solved_phase2['loss'] < -26])

extrapolated_xvals = np.concatenate([[2.5], np.arange(3.5, 10.3, 0.2)])
sns.lineplot(x = extrapolated_xvals, y = smoothed_vals(extrapolated_xvals),
             ax = axs, color = 'black', linewidth = 6)
    
#sns.lineplot(x = x_vals, y = y_vals, ax = axs, color = 'black', linewidth = 7.5)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase_intrplt.png",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase_intrplt.svg",
            bbox_inches='tight')
