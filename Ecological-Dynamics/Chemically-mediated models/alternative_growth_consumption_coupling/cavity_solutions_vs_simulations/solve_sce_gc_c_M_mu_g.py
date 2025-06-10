# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:21:31 2025

@author: jamil
"""

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
from matplotlib.colors import TwoSlopeNorm
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
                         solver_kwargs, 
                         solver = sce.solve_equations_least_squares,
                         other_kwargs = {}):
    
    xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
    solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                     for xscale in xscales]
    
    sol = sce.solve_sces(parameters, 'self-limiting gc c M',
                         solved_quantities = solved_quantities,
                         bounds = bounds, x_init = x_init,
                         solver = solver,
                         solver_kwargs = solver_kwargs, 
                         include_multistability = True,
                         other_kwargs = other_kwargs)
    
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
                        + 'finite_effects_fixed_mu_g_3'
    
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
    
# %%

def le_pivot(df, index = 'sigma_c', columns = 'mu_c', values = 'Max. lyapunov exponent'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                          values = 'Max. lyapunov exponent', aggfunc = prop_chaotic)]

def agg_pivot(df, values, index = 'sigma_c', columns = 'mu_c', aggfunc = 'mean'):
    
    return [pd.pivot_table(df, index = index, columns = columns,
                           values = values, aggfunc = aggfunc)]

# %%

############################# global optimization, not reliant on initial conditions ######################

def Global_Optimisation(df_simulation):
    
    extractable_parameters = df_simulation.groupby(['M',
                                                    'mu_g'])[['mu_c',
                                                              'sigma_c',
                                                              'sigma_g']].mean().reset_index().to_dict('records')
    
    parameters = sce.variable_fixed_parameters(extractable_parameters, None,
                                               {'mu_K' : 1, 'sigma_K' : 0,
                                                'mu_m' : 1, 'sigma_m' : 0,
                                                'gamma' : 1})
       
    solved_sces = solve_sces_finite(parameters,
                                    solver_kwargs = {'xtol' : 1e-13,
                                                     'ftol' : 1e-13},
                                    solver = sce.solve_equations_basinhopping,
                                    other_kwargs = {'niter' : 200})
    
    solved_sces['M'] = np.int32(solved_sces['M'])
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
    
    pickle_dump(directory + "/M_vs_C_vs_mu_g.pkl", solved_sces)
    
    return solved_sces

# %%

################# Solving for the phase boundary ##############################

'''

def Phase_Boundary_Interpolated_M(solved_sces):
    
    solved_sces['dNde'] = np.log10(np.abs(solved_sces['dNde']))

    max_dnde_by_M = solved_sces.groupby('M')['dNde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dNde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dNde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    max_dNde_df['sigma_C'] =  np.round(max_dNde_df['sigma_c'] * np.sqrt(max_dNde_df['M']), 4)
    
    interpolator = BarycentricInterpolator(max_dNde_df['M'],
                                           max_dNde_df[['phi_N', 'N_mean', 'q_N', 'v_N',
                                                        'phi_R', 'R_mean', 'q_R', 'chi_R',
                                                        '<C>', 'sigma_C', 'mu_g', 'sigma_g',
                                                        'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                                        'gamma']])
    
    interpolated_M = np.arange(np.min(max_dNde_df['M']),
                               np.max(max_dNde_df['M']) + 5, 5)
    
    interpolated_matrix = interpolator(interpolated_M)
    smoothed_interps = [make_splrep(interpolated_M, interpolated_y,
                                    k = 2, s = 0.7)
                        for interpolated_y in interpolated_matrix.T]
    
    interpolated_data = pd.DataFrame([smooth_y(interpolated_M)
                                      for smooth_y in smoothed_interps],
                                     index = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                              'phi_R', 'R_mean', 'q_R', 'chi_R',
                                              '<C>', 'sigma_C', 'mu_g', 'sigma_g',
                                              'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                              'gamma']).T
    
    interpolated_data['M'] = interpolated_M
    interpolated_data['mu_c'] = interpolated_data['<C>']/interpolated_data['M']
    interpolated_data['sigma_c'] = interpolated_data['sigma_C']/np.sqrt(interpolated_data['M'])
    
    parameters = interpolated_data[['M', 'mu_c', 'sigma_c', 'sigma_g',
                                    'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                    'gamma']].to_dict('records')
    
    x_inits = interpolated_data[['phi_N', 'N_mean', 'q_N', 'v_N',
                                 'phi_R', 'R_mean', 'q_R', 'chi_R',
                                 'mu_g']].to_dict('records')
    x_inits = [list(x_init.values()) for x_init in x_inits]
    
    solved_phase = solve_phase_boundary(parameters,
                                        solved_quantities = ['phi_N', 'N_mean',
                                                            'q_N', 'v_N',
                                                            'phi_R', 'R_mean',
                                                            'q_R', 'chi_R',
                                                            'mu_g'],
                                        bounds = ([1e-10, 1e-10, 1e-10, -1e15,
                                                   1e-10, 1e-10, 1e-10, 1e-10,
                                                   0.05],
                                                  [1, 1e15, 1e15, 1e-10,
                                                   1, 1e15, 1e15, 1e15,
                                                   3]),
                                        x_init = x_inits,
                                        solver_kwargs = {'xtol' : 1e-15, 'ftol' : 1e-15})
    
    solved_phase['<C>'] = np.round(solved_phase['mu_c'] * solved_phase['M'], 2)
    
    return solved_phase

'''

################################################################

def Phase_Boundary_Interpolated_M(solved_sces):
    
    solved_sces['dNde'] = np.log10(np.abs(solved_sces['dNde']))

    max_dnde_by_M = solved_sces.groupby('M')['dNde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dNde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dNde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    max_dNde_df['sigma_C'] =  np.round(max_dNde_df['sigma_c'] * np.sqrt(max_dNde_df['M']), 4)
    
    interpolator = BarycentricInterpolator(max_dNde_df['M'],
                                           max_dNde_df[['phi_N', 'N_mean', 'q_N', 'v_N',
                                                        'phi_R', 'R_mean', 'q_R', 'chi_R',
                                                        '<C>', 'sigma_C', 'mu_g', 'sigma_g',
                                                        'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                                        'gamma']])
    
    interpolated_M = np.arange(np.min(max_dNde_df['M']),
                               np.max(max_dNde_df['M']) + 5, 5)
    
    interpolated_matrix = interpolator(interpolated_M)
    smoothed_interps = [make_splrep(interpolated_M, interpolated_y,
                                    k = 2, s = 0.7)
                        for interpolated_y in interpolated_matrix.T]
    
    interpolated_data = pd.DataFrame([smooth_y(interpolated_M)
                                      for smooth_y in smoothed_interps],
                                     index = ['phi_N', 'N_mean', 'q_N', 'v_N',
                                              'phi_R', 'R_mean', 'q_R', 'chi_R',
                                              '<C>', 'sigma_C', 'mu_g', 'sigma_g',
                                              'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                              'gamma']).T
    
    interpolated_data['M'] = interpolated_M
    interpolated_data['mu_c'] = interpolated_data['<C>']/interpolated_data['M']
    interpolated_data['sigma_c'] = interpolated_data['sigma_C']/np.sqrt(interpolated_data['M'])
    
    parameters = interpolated_data[['M', 'mu_c', 'sigma_c', 'sigma_g',
                                    'mu_K', 'sigma_K', 'mu_m', 'sigma_m',
                                    'gamma']].to_dict('records')
    
    x_inits = interpolated_data[['phi_N', 'N_mean', 'q_N', 'v_N',
                                 'phi_R', 'R_mean', 'q_R', 'chi_R',
                                 'mu_g']].to_dict('records')
    x_inits = [list(x_init.values()) for x_init in x_inits]
    
    solved_phase = solve_phase_boundary(parameters,
                                        solved_quantities = ['phi_N', 'N_mean',
                                                            'q_N', 'v_N',
                                                            'phi_R', 'R_mean',
                                                            'q_R', 'chi_R',
                                                            'mu_g'],
                                        bounds = ([1e-10, 1e-10, 1e-10, -1e15,
                                                   1e-10, 1e-10, 1e-10, 1e-10,
                                                   0.05],
                                                  [1, 1e15, 1e15, 1e-10,
                                                   1, 1e15, 1e15, 1e15,
                                                   3]),
                                        x_init = x_inits,
                                        solver_kwargs = {'xtol' : 1e-15, 'ftol' : 1e-15})
                                        #,
#                                        solver = sce.solve_equations_basinhopping,
#                                        other_kwargs = {'niter' : 200})
    
    solved_phase['<C>'] = np.round(solved_phase['mu_c'] * solved_phase['M'], 2)
    
    return solved_phase

# %%

df_simulation = generate_simulation_df()

globally_solved_sces = Global_Optimisation(df_simulation)

# (locally-solved) phase boundary
solved_phase = Phase_Boundary_Interpolated_M(globally_solved_sces[globally_solved_sces['mu_g'] <= 2])

# %%

df_simulation = generate_simulation_df()
globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_finite/M_vs_C_vs_mu_g.pkl")

# %%

resource_pool_sizes = np.unique(df_simulation['M'])

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot():
    
    #breakpoint()
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot(df_simulation[df_simulation['mu_g'] <= 2],
                                   columns = 'M', index = 'mu_g')[0]
    
    sns.set_style('white')
    
    mosaic = [["P", "D1", "D2", ".", "I_C"],
              ["P", ".", ".", ".", "I_C"],
              ["P", "D3", "D4", ".", "I_C"],
              ["P",  "D5", "D6", ".", "I_C"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (16.5, 4.5),
                                  width_ratios = [8, 2.5, 2.5, 1, 6],
                                  height_ratios = [1, 0.5, 1, 1])
    
    subfig = sns.heatmap(stability_sim_pivot, ax = axs["P"],
                         vmin = 0, vmax = 1, cbar = True, cmap = 'Purples')
        
    subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
    
    axs["P"].set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                        labels = resource_pool_sizes, fontsize = 14)

    axs["P"].set_yticks([0.5, len(stability_sim_pivot.index) - 0.5],
                        labels = [np.round(stability_sim_pivot.index[0], 3),
                                  np.round(stability_sim_pivot.index[-1], 3)],
                                fontsize = 14)
    
    axs["P"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 14,
                        weight = 'bold')
    axs["P"].set_ylabel('average resource use efficiency, ' + r'$\mu_g$',
                        fontsize = 14, weight = 'bold')
    axs["P"].invert_yaxis()
    axs["P"].set_title('Altering the average resource use efficiency\n' + \
                       'has a weak effect on community stability',
                       fontsize = 16, weight = 'bold', y = 1.05)
        
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
                   size = '14')
    cbar.ax.tick_params(labelsize = 12)
    
    # Analytically-derived phase boundary
    
    good_solves = solved_phase[solved_phase['loss'] <= -28]
    good_solves = good_solves.iloc[np.where((good_solves['M'] >= 60) & 
                                            (good_solves['M'] <= 245))]

    smoother = make_splrep(good_solves['M'], good_solves['mu_g'], k = 2, s = 0.85)
    smoothed_x = np.arange(60, 285, 5)

    y_phase = smoother(smoothed_x) - np.min(globally_solved_sces['mu_g'])
    diffs = np.unique(np.round(np.abs(np.diff(globally_solved_sces.iloc[np.where(globally_solved_sces['M'] == 100)]['mu_g'])), 1))
    divider = diffs[diffs > 0]
    y_vals = (y_phase/divider) + 0.5

    x_vals = 0.9 + np.arange(0, len(smoothed_x)*0.2, 0.2)

    sns.lineplot(x = x_vals, y = y_vals,
                 ax = axs["P"], color = 'black', linewidth = 6,
                 linestyle = '--')
    
    #################### Instability condition vs M #####################
    
    # Example relathips with <C> = 128.57

    df_plot = globally_solved_sces.iloc[np.where((globally_solved_sces['M'] == 100) &
                                                 (globally_solved_sces['mu_g'] <= 2))]
    dfl = pd.melt(df_plot[['mu_g', 'rho', 'species packing 2']], ['mu_g'])
    dfl.loc[dfl['variable'] == 'rho', 'value'] = dfl.loc[dfl['variable'] == 'rho', 'value']**2

    #g_stability_threshold = df_plot.iloc[np.abs(df_plot['instability distance'] - 0).argmin()]['mu_g']
    #axs["I_C"].vlines(g_stability_threshold, np.min(dfl['value']), np.max(dfl['value']),
    #                  color = 'gray', linestyle = '--', linewidth = 3, zorder = 0)

    subfig1 = sns.lineplot(dfl, x = 'mu_g', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 5, marker = 'o', markersize = 13,
                           palette = sns.color_palette(['#39568cff', '#1f968bff'], 2),
                           zorder = 100)

    axs["I_C"].set_xlabel('average resource use efficiency, ' + r'$\mu_g$',
                          fontsize = 14, weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=14)
    axs["I_C"].set_xticks(np.unique(df_plot['mu_g']),
                          labels = np.unique(df_plot['mu_g']))
    
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
                                       bbox_to_anchor=(-0.26, 0.5),
                                       bbox_transform=axs["I_C"].transAxes, borderpad=0)

    ybox_b = VPacker(children=[ybox3, ybox2], align="center", pad=0, sep=0)
    anchored_ybox2 = AnchoredOffsetbox(loc='center', child=ybox_b, pad=0., frameon=False,
                                       bbox_to_anchor=(-0.14, 0.5),
                                       bbox_transform=axs["I_C"].transAxes, borderpad=0)

    axs["I_C"].add_artist(anchored_ybox1)
    axs["I_C"].add_artist(anchored_ybox2)

    axs["I_C"].legend_.remove()

    axs["I_C"].set_title('Increasing resource use efficiency simultaneously' + \
                     '\nincreases the correlation between growth and\n' + \
                     'consumption rates and the species packing ratio',
                     fontsize = 16, weight = 'bold', y = 1.05)
    
    ####################### Example population dynamics ######################
    
    stable_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_mu_g_3/" + \
                                         "CR_self_limiting_1002.0.pkl")
    
    chaotic_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_mu_g_3/" + \
                                         "CR_self_limiting_1000.5.pkl")
    
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
    
    i_c_rp = indices_and_cmaps(100)
    i_c_rp = [i_c for _ in range(3) for i_c in i_c_rp]
    
    for ax, simulation, i_c, title in \
        zip([axs['D1'], axs['D2'], axs['D3'], axs['D4'], axs['D5'], axs['D6']],
            [stable_populations[2], stable_populations[2],
             chaotic_populations[0], chaotic_populations[0],
             chaotic_populations[1], chaotic_populations[1]],
            i_c_rp, ['species', 'resources', '', '', '', '']):
        
        plot_dynamics(ax, simulation, i_c, title)
       
    fig.text(0.53, 0.05, "time", fontsize = 14, weight = 'bold',
             verticalalignment = 'center', horizontalalignment = 'right')
    
    fig.text(0.41, 0.5, 'abundances', fontsize = 14, weight = 'bold',
             verticalalignment = 'center', horizontalalignment = 'center',
             rotation = 90)
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_M_sim_and_analyticalphase_intrplt.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_mu_g_M_sim_and_analyticalphase_intrplt.svg",
                bbox_inches='tight')
    
    plt.show()
    
Stability_Plot()

# %%

resource_pool_sizes = np.unique(df_simulation['M'])

fig, axs = generic_heatmaps(df_simulation[df_simulation['mu_g'] <= 2],
                            'M', 'mu_g', 
                           'resource pool size, ' + r'$M$',
                           'average resource use efficiency, ' + r'$\mu_g$',
                            ['infeasibility distance'], 'Blues',
                            '',
                            (1, 1), (6.5, 4),
                            pivot_functions = {'infeasibility distance' : agg_pivot},
                            specify_min_max = {'infeasibility distance' : 
                                               [0, np.max(df_simulation['infeasibility distance'])]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                          labels = resource_pool_sizes, fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = r'$\phi_N$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.show()