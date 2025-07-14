# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:22:32 2025

@author: jamil
"""

########################

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from copy import copy
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import make_splrep

from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.colors import LinearSegmentedColormap

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Chemically-mediated models/alternative_growth_consumption_coupling")
from simulation_functions import distance_from_instability, \
    distance_from_infeasibility, species_packing, \
        create_df_and_delete_simulations_2, prop_chaotic, pickle_dump, \
        generic_heatmaps
        
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

def solve_sces_finite(parameters, solver_kwargs,
                      x_init = None,
                      solver = sce.solve_equations_least_squares,
                      other_kwargs = {}):
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    # variable bounds 
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
     
    # initial values
    if x_init:
        
        xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
        solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                         for xscale in xscales]
    
    else:

        x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
        solver_kwargs['x_scale'] = np.power(10, np.floor(np.log10(np.abs(x_init))))
    
    sol = sce.solve_sces(parameters, 'self-limiting gc c M',
                         solved_quantities = solved_quantities,
                         bounds = bounds, x_init = x_init,
                         solver = solver,
                         solver_kwargs = solver_kwargs,
                         other_kwargs = other_kwargs)
       
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
    
    sol = sce.solve_sces(parameters, 'self-limiting gc c M',
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

def generate_simulation_df():
    
    def community_properties_df(directory,
                                parm_attributes = ['no_species', 'no_resources',
                                                   'mu_c', 'sigma_c', 'mu_g',
                                                   'sigma_g', 'm', 'K']):
        
        dfs = [create_df_and_delete_simulations_2(directory + "/", file, parm_attributes)
               for file in os.listdir(directory)]
                
        return dfs

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
    
    df['no_resources'] = np.int32(df['no_resources'])
    df.rename(columns = {'no_resources' : 'M', 'no_species' : 'S'}, inplace = True)
    
    df['<C>'] = np.round(df['mu_c'] * df['M'], 2)

    return df

def covariance_correlation(df):
            
    mu_c, mu_g, sigma_c, sigma_g = df['mu_c'], df['mu_g'], df['sigma_c'], df['sigma_g']

    denominator = (mu_g * sigma_c)**2 + ((mu_c * sigma_g)**2) + (sigma_c * sigma_g)**2
         
    covariance = (mu_g * sigma_c**2)
    correlation = (mu_g * sigma_c)/np.sqrt(denominator)
        
    return covariance, correlation
    

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

# Locally solve self-consistency equations with initial conditions from simulations #

def Local_Optimum_InitCondSpecific(df_simulation):

    parms_x_inits = df_simulation.groupby(['M', 'mu_c', 'sigma_c'])[['phi_N',
                                                                     'N_mean',
                                                                     'q_N',
                                                                     'phi_R', 
                                                                     'R_mean',
                                                                     'q_R']].mean().reset_index()
    
    parms_x_inits = np.round(parms_x_inits, 6)
     
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
    
    return solved_sces_close_xinit

# %%

################# Solving for the phase boundary ##############################

def Phase_Boundary_Interpolated_M(solved_sces):

    max_dnde_by_M = solved_sces.groupby('M')['dNde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dNde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dNde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    max_dNde_df_interp = max_dNde_df.iloc[np.where(max_dNde_df['M'] > 75)]
    
    interpolator = BarycentricInterpolator(max_dNde_df_interp['M'],
                                           max_dNde_df_interp[['phi_N', 'N_mean', 'q_N', 'v_N',
                                                               'phi_R', 'R_mean', 'q_R', 'chi_R',
                                                               'mu_c', 'sigma_c']])
    
    interpolated_M = np.arange(np.min(max_dNde_df_interp['M']),
                               np.max(max_dNde_df_interp['M']) + 5, 5)
    
    interpolated_data = pd.DataFrame(interpolator(interpolated_M),
                                     columns = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                                'phi_R', 'R_mean', 'q_R', 'chi_R',
                                                'mu_c', 'sigma_c'])
    interpolated_data['M'] = interpolated_M
    
    interpolated_data = pd.concat([max_dNde_df.iloc[np.where(max_dNde_df['M'] == 75)][list(interpolated_data.columns)],
                                   interpolated_data])
    interpolated_data.reset_index(drop = True, inplace = True)
    
    variable_parameters = interpolated_data[['M', 'sigma_c']].to_dict('records')
    parameters = sce.variable_fixed_parameters(variable_parameters, None,
                                               {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                                'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                                'sigma_m' : 0, 'gamma' : 1})
    
    x_inits = interpolated_data[['phi_N', 'N_mean', 'q_N', 'v_N',
                                 'phi_R', 'R_mean', 'q_R', 'chi_R',
                                 'mu_c']].to_dict('records')
    x_inits = [list(x_init.values()) for x_init in x_inits]
    
    solved_phase = solve_phase_boundary(parameters,
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
                                        x_init = x_inits,
                                        solver_kwargs = {'xtol' : 1e-15, 'ftol' : 1e-15})
    
    solved_phase['<C>'] = np.round(solved_phase['mu_c'] * solved_phase['M'], 2)
    
    return solved_phase

# %%

############################# global optimization, not reliant on initial conditions ######################

def Global_Optimisation(df_simulation):

    parms_x_inits = df_simulation.groupby(['M', 'mu_c', 'sigma_c'])[['phi_N',
                                                                     'N_mean',
                                                                     'q_N',
                                                                     'phi_R', 
                                                                     'R_mean',
                                                                     'q_R']].mean().reset_index()
    
    parms_x_inits = np.round(parms_x_inits, 6)
     
    variable_parameters = parms_x_inits[['M', 'mu_c', 'sigma_c']].to_dict('records')
    parameters = sce.variable_fixed_parameters(variable_parameters, None,
                                               {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                                'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                                'sigma_m' : 0, 'gamma' : 1})
    
    solved_sces = solve_sces_finite(parameters,
                                    solver_kwargs = {'xtol' : 1e-13,
                                                     'ftol' : 1e-13},
                                    solver = sce.solve_equations_basinhopping,
                                    other_kwargs = {'niter' : 200})
    
    solved_sces['M'] = np.int32(solved_sces['M'])
    solved_sces['<C>'] = np.round(solved_sces['mu_c'] * solved_sces['M'], 2)
    
    ######### Correcting bad solves with more iterations of basinhopping ########################
    
    bad_solves = solved_sces.iloc[np.where(solved_sces['loss'] > - 25)]
    
    variable_parameters_bs = bad_solves[['M', 'mu_c', 'sigma_c']].to_dict('records')
    parameters_bs = sce.variable_fixed_parameters(variable_parameters_bs, None,
                                                  {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                                   'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                                   'sigma_m' : 0, 'gamma' : 1})
    
    resolved_bad_sces = solve_sces_finite(parameters_bs,
                                          solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                                          solver = sce.solve_equations_basinhopping,
                                          other_kwargs = {'niter' : 500})
    
    # correcting leftover bad solves
    
    variable_parameters_bs2 = resolved_bad_sces.iloc[-1][['M', 'mu_c', 'sigma_c']].to_dict()
    parameters_bs2 = sce.variable_fixed_parameters([variable_parameters_bs2], None,
                                                  {'mu_g': 1, 'sigma_g' : 1.6/np.sqrt(150),
                                                   'mu_K' : 1, 'sigma_K' : 0, 'mu_m' : 1,
                                                   'sigma_m' : 0, 'gamma' : 1})
    
    resolved_bad_sces2 = solve_sces_finite(parameters_bs2,
                                          solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                                          solver = sce.solve_equations_basinhopping,
                                          other_kwargs = {'niter' : 1000})
    
    # update bad solves in original dataframe with new good solves 
    
    solved_sces.drop('<C>', axis = 1, inplace = True)
    
    resolved_bad_sces.set_index(np.where(solved_sces['loss'] > - 25)[0],
                                inplace = True)
    resolved_bad_sces2.set_index(resolved_bad_sces.iloc[np.where(resolved_bad_sces['loss'] > - 25)].index,
                                 inplace = True)
    
    solved_sces.loc[resolved_bad_sces.index] = resolved_bad_sces
    solved_sces.loc[resolved_bad_sces2.index] = resolved_bad_sces2
    
    solved_sces['<C>'] = np.round(solved_sces['mu_c'] * solved_sces['M'], 2)
    
    # calculate correlation, instability condition, infeasibility condition etc
    
    solved_sces.rename(columns = {'M' : 'no_resources'}, inplace = True)
    solved_sces['no_species'] = solved_sces['no_resources']/solved_sces['gamma']
    
    solved_sces[['covariance', 'rho']] = pd.DataFrame(solved_sces.apply(covariance_correlation,
                                                      axis = 1).to_list())
    
    # calculate the stability metric (rho^2 - phi_N/(gamma * phi_R)) from the 
    #   cavity solution
    solved_sces['instability distance'] = solved_sces.apply(distance_from_instability, axis = 1)
    
    # calcualte the infeasibily metric (phi_R - phi_N/gamma) from the cavity solution
    solved_sces['infeasibility distance'] = solved_sces.apply(distance_from_infeasibility, axis = 1)
    
    # calculate the species packing ratio, phi_N/(gamma * phi_R)
    solved_sces['species packing 2'] = solved_sces.apply(species_packing, axis = 1)
    
    solved_sces.rename(columns = {'no_resources' : 'M'}, inplace = True)
    solved_sces.drop('no_species', axis = 1, inplace = True)
    
    # save data
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                          + "cavity_solutions/self_limiting_finite"
    if not os.path.exists(directory): os.makedirs(directory) 
    
    pickle_dump(directory + "/M_vs_C.pkl", solved_sces)
    
    return solved_sces
    
##################################################################################################################################

# %%

def All_Solves():
    
    df_simulation = generate_simulation_df()

    #### Local solve #####

    locally_solved_sces = Local_Optimum_InitCondSpecific(df_simulation)
    solved_phase = Phase_Boundary_Interpolated_M(locally_solved_sces)

    #### Global solve ####

    globally_solved_sces = Global_Optimisation(df_simulation) # automatically pickled
    
# %% 

df_simulation = generate_simulation_df()
locally_solved_sces = Local_Optimum_InitCondSpecific(df_simulation) 

# %%

sns.heatmap(locally_solved_sces.pivot_table(index = '<C>', columns = 'M', values = 'dNde'),
            cmap = 'Greens')
plt.show()

# %%

# Simulation data

df_simulation = generate_simulation_df()

#### Global solve ####

globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_finite/M_vs_C.pkl")

# (locally-solved) phase boundary
solved_phase = Phase_Boundary_Interpolated_M(globally_solved_sces)

# %%

resource_pool_sizes = np.unique(df_simulation.iloc[np.where(df_simulation['M'] > 25)]['M'])

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot():
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot(df_simulation.iloc[np.where((df_simulation['<C>'] <= 260) &
                                                               (df_simulation['M'] > 25))],
                                   columns = 'M', index = '<C>')[0]
    
    sns.set_style('white')
    
    mosaic = [["P", ".", ".", ".", ".", "I_C"],
              ["P", ".", "D1", "D2", ".", "I_C"],
              ["P", ".", "D3", "D4", ".", "I_C"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (18, 5),
                                  width_ratios = [6, 0, 2.5, 2.5, 1.2, 6],
                                  height_ratios = [2, 2.5, 2.5],
                                  gridspec_kw = {'hspace' : 0.3})
    
    subfig = sns.heatmap(stability_sim_pivot, ax = axs["P"],
                         vmin = 0, vmax = 1, cbar = True, cmap = 'Purples')
        
    subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
    
    axs["P"].set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                        labels = resource_pool_sizes[::2], fontsize = 14)

    axs["P"].set_yticks([0.5, len(stability_sim_pivot.index) - 0.5],
                        labels = [np.round(stability_sim_pivot.index[0], 3),
                                  np.round(stability_sim_pivot.index[-1], 3)],
                                fontsize = 14)
    
    axs["P"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 14,
                        weight = 'bold')
    axs["P"].set_ylabel('average total consumption rate of a\nresource, ' + \
                        r'$<C>$', fontsize = 14, weight = 'bold')
    axs["P"].invert_yaxis()
    
    axs['P'].text(1.2, 1.2,
                  'Increasing the resource pool size stabilises community dynamics ' + \
                  '(for a fixed total\nconsumption rate)',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["P"].transAxes)
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
                   size = '14')
    cbar.ax.tick_params(labelsize = 12)
    
    # Analytically-derived phase boundary

    y_phase = solved_phase['<C>'].to_numpy() - np.min(globally_solved_sces['<C>'])
    diffs = np.unique(np.round(np.abs(np.diff(globally_solved_sces.iloc[np.where(globally_solved_sces['M'] == 100)]['<C>'])), 1))
    divider = diffs[diffs > 0]
    y_vals = (y_phase/divider) + 0.5

    x_vals = np.concatenate([[1.5], np.arange(2.5, 8.7, 0.2)])

    smoothed_vals = make_splrep(x_vals, y_vals, k = 2, s = 0.7)
    extrapolated_xvals = np.concatenate([[1.5], np.arange(2.5, 9.3, 0.2)])

    sns.lineplot(x = extrapolated_xvals, y = smoothed_vals(extrapolated_xvals),
                 ax = axs["P"], color = 'black', linewidth = 6)
    
    #################### Instability condition vs M #####################
    
    # Example relathips with <C> = 128.57

    df_plot = globally_solved_sces.iloc[np.where((globally_solved_sces['<C>'] == 128.57) &
                                                 (globally_solved_sces['M']> 25))]
    dfl = pd.melt(df_plot[['M', 'rho', 'species packing 2']], ['M'])
    dfl.loc[dfl['variable'] == 'rho', 'value'] = dfl.loc[dfl['variable'] == 'rho', 'value']**2
    
    smoother_2 = make_splrep(solved_phase['M'], solved_phase['<C>'],
                             k = 2, s = 0.7)
    M_vals = np.arange(50, 250, 0.1)
    smoothed_vals_2 = smoother_2(M_vals)
    M_stability_threshold = M_vals[np.abs(smoothed_vals_2 - 128.57).argmin()]
    
    axs["I_C"].vlines(M_stability_threshold, np.min(dfl['value']), np.max(dfl['value']),
                      color = 'gray', linestyle = '--', linewidth = 3, zorder = 0)

    subfig1 = sns.lineplot(dfl, x = 'M', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 5, marker = 'o', markersize = 13,
                           palette = sns.color_palette(['#39568cff', '#1f968bff'], 2),
                           zorder = 100)

    axs["I_C"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 14,
                          weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=14)
    axs["I_C"].set_xticks(resource_pool_sizes[::2],
                          labels = resource_pool_sizes[::2], fontsize = 14)

    # y-axis label
    ybox1 = TextArea('(correlation between\ngrowth and consumption)' + r'$^2$',
                     textprops=dict(color='#39568cff', size=14, rotation='vertical',
                                    multialignment='center', weight = 'bold'))
    ybox2 = TextArea('and ',
                     textprops=dict(color = "black", size=14,rotation='vertical'))
    ybox3 = TextArea('species packing ratio',
                     textprops=dict(color='#1f968bff', size=14,rotation='vertical',
                                    weight = 'bold'))

    ybox_t = VPacker(children=[ybox1], align="center", pad=0, sep=0)
    anchored_ybox1 = AnchoredOffsetbox(loc='center', child=ybox_t, pad=0., frameon=False,
                                       bbox_to_anchor=(-0.25, 0.5),
                                       bbox_transform=axs["I_C"].transAxes, borderpad=0)

    ybox_b = VPacker(children=[ybox3, ybox2], align="center", pad=0, sep=0)
    anchored_ybox2 = AnchoredOffsetbox(loc='center', child=ybox_b, pad=0., frameon=False,
                                       bbox_to_anchor=(-0.15, 0.5),
                                       bbox_transform=axs["I_C"].transAxes, borderpad=0)

    axs["I_C"].add_artist(anchored_ybox1)
    axs["I_C"].add_artist(anchored_ybox2)

    axs["I_C"].legend_.remove()
    
    axs['I_C'].text(0.5, 1.2,
                  'Increasing the resource pool size stabilises communities ' + \
                  'by\nincreasing the correlation between growth and\nconsumption rates',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["I_C"].transAxes)
        
    axs["I_C"].text(0.4, 0.98, "Unstable", color='#7300e3ff', fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')],
                horizontalalignment='right', verticalalignment='top',
                transform=axs["I_C"].transAxes)

    axs["I_C"].text(0.48, 0.98, "Stable", color='white', fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')],
                horizontalalignment='left', verticalalignment='top',
                transform=axs["I_C"].transAxes)

    axs["I_C"].annotate("Stability\nthreshold", xytext=(55, 0.6), xy=(134, 0.6),
                    color = 'grey', fontsize = 14, weight = 'bold',
                    va = 'center', multialignment = 'center',
                    arrowprops={'arrowstyle': '-|>', 'color' : 'gray', 'lw' : 2})
    
    ####################### Example population dynamics ######################
    
    # M = 75 and 225, <C> = 128.57
    example_M = [75, 225]
    
    stable_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_C_2/" + \
                                         "CR_self_limiting_2250.571429.pkl")
    
    chaotic_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_C_2/" + \
                                         "CR_self_limiting_751.714286.pkl")
    
    def indices_and_cmaps(M):
        
        species, resources = np.arange(M), np.arange(M, M*2)
        
        s_colour_index, r_colour_index = np.arange(M), np.arange(M)
        np.random.shuffle(s_colour_index)
        np.random.shuffle(r_colour_index)
        
        cmap_s = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        cmap_r = LinearSegmentedColormap.from_list('custom YlGBl',
                                                   ['#e9a100ff','#1fb200ff',
                                                    '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                   N = M)
        
        return [species, s_colour_index, cmap_s], [resources, r_colour_index,
                                                   cmap_r]
    
    def plot_dynamics(ax, simulation, i_c_rp_M, title):
        
        #breakpoint()
            
        var_pos, colour_index, cmap = i_c_rp_M
        data = simulation.ODE_sols['lineage 0']
        
        for i, v in zip(colour_index, var_pos):
        
            ax.plot(data.t, data.y[v,:].T, color = 'black', linewidth = 1)
            ax.plot(data.t, data.y[v,:].T, color = cmap(i), linewidth = 0.75)
        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            ax.set_title(title, fontsize = 12)
        
        return ax
    
    i_c_rp = [indices_and_cmaps(M) for M in example_M]
    i_c_rp = [i_c for i_c_rp_M in i_c_rp for i_c in i_c_rp_M]

    for ax, simulation, i_c_rp_M, title in \
        zip([axs['D1'], axs['D2'], axs['D3'], axs['D4']],
            [chaotic_populations[0], chaotic_populations[0],
             stable_populations[0], stable_populations[0]],
            i_c_rp,
            ['species', 'resources', '', '']):
        
        plot_dynamics(ax, simulation, i_c_rp_M, title)
        
    axs['D1'].text(1.1, 1.2, "M = 75", fontsize = 14, weight = 'bold',
                   verticalalignment = 'center', horizontalalignment = 'center',
                   transform=axs["D1"].transAxes)
    
    axs['D3'].text(1.1, 1.1, "M = 225", fontsize = 14, weight = 'bold',
                   verticalalignment = 'center', horizontalalignment = 'center',
                   transform=axs["D3"].transAxes)
    
    axs['D3'].text(1.1, -0.15, "time", fontsize = 14, weight = 'bold',
                   verticalalignment = 'center', horizontalalignment = 'center',
                   transform=axs["D3"].transAxes)
    
    axs['D3'].text(-0.1, 1.15, "abundances", fontsize = 14, weight = 'bold',
                   verticalalignment = 'center', horizontalalignment = 'center',
                   transform=axs["D3"].transAxes, rotation = 90)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase_intrplt.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_fixed_C_M_sim_and_analyticalphase_intrplt.svg",
                bbox_inches='tight')
        
    plt.show()

Stability_Plot()
