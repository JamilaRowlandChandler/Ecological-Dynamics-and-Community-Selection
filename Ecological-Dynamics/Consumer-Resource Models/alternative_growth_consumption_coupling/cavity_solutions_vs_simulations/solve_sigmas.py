# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:37:22 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from copy import copy
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generate_simulation_df, le_pivot, pickle_dump
    
# %%

def solve_sces_yc_c(parameters, solved_quantities, bounds, x_init, solver_name,
                    solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                    other_kwargs = {}, include_multistability = False):
    
    if isinstance(x_init[0], list):
        
        xscales = list(np.power(10, np.floor(np.log10(np.abs(x_init)))))
        solver_kwargs = [dict(list(solver_kwargs.items()) + [('x_scale', xscale)]) 
                         for xscale in xscales]
        
    else:
        
        solver_kwargs['x_scale'] = np.power(10, np.floor(np.log10(np.abs(x_init))))
    
    sol = sce.solve_self_consistency_equations(model = 'self-limiting, yc c',
                                               parameters = parameters,
                                               solved_quantities = solved_quantities,
                                               bounds = bounds,
                                               x_init = x_init,
                                               solver_name = solver_name,
                                               solver_kwargs = solver_kwargs,
                                               other_kwargs = other_kwargs,
                                               include_multistability = include_multistability)
    
    sol['rho'] = np.sqrt(1 / (1 + \
                             ((sol['sigma_y']/sol['mu_y'])**2 * (1 + \
                                                               ((sol['mu_c']**2)/(sol['M'] * sol['sigma_c']**2))))))
    
    sol['Species packing'] = sol['phi_N']/(sol['phi_R'] * sol['gamma'])
    sol['Instability distance'] = sol['rho']**2 - sol['Species packing']
    
    sol['Infeasibility distance'] = sol['phi_R'] - sol['phi_N']/sol['gamma']
        
    return sol

# %%

def Local_Solve_Phase_Boundary(solved_sces, solved_quantity = 'sigma_c',
                               quantity_bounds = [0.1, 10]):
    
    parm_names = ['mu_c', 'sigma_c', 'mu_y', 'sigma_y',
                  'mu_b', 'sigma_b', 'mu_d', 'sigma_d',
                  'gamma']
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    solved_sces = solved_sces[solved_sces['loss'] < -30]
    solved_sces['dNde'] = np.log10(np.abs(solved_sces['dNde']))

    max_dnde_by_M = solved_sces.groupby('M')['dNde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dNde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dNde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    interpolated_M = np.arange(np.min(max_dNde_df['M']),
                               np.max(max_dNde_df['M']), 5)
    
    interpolators = [np.poly1d(np.polyfit(max_dNde_df.loc[max_dNde_df['M'] > 125,
                                                          'M'],
                                          max_dNde_df.loc[max_dNde_df['M'] > 125,
                                                          col],
                                          1))
                     for col in parm_names + solved_quantities]
    
    interpolated_data = pd.DataFrame([np.round(interpolator(interpolated_M), 7)
                                      for interpolator in interpolators],
                                     index = parm_names + solved_quantities).T
    
    for col in parm_names + ['phi_N', 'N_mean', 'q_N', 'phi_R', 'R_mean', 'q_R']:
        
        interpolated_data.loc[interpolated_data[col] < 0, col] = 1e-10
     
    interpolated_data['M'] = interpolated_M
    
    parm_names.append('M')
    parm_names.remove(solved_quantity)
    parameters = interpolated_data[parm_names].to_dict('records')
    
    solved_quantities.append(solved_quantity)
    x_init_dicts = interpolated_data[solved_quantities].to_dict('records')
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10,
                quantity_bounds[0]], 
               [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15,
                quantity_bounds[1]])
    
    x_inits = [list(x_init.values()) for x_init in x_init_dicts]
    
    solved_phase = solve_sces_yc_c(parameters, solved_quantities, bounds, x_inits,
                                   'least-squares', 
                                   solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                                   include_multistability = True)
     
    return solved_phase
      
# %%

########################## sigma_cs ################################

# load in simulation data
df_simulation_c = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                         + 'finite_effects_sigma_c_final')

# load in simulation data and solved sces
globally_solved_sces_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                        + "cavity_solutions/self_limiting_yc_c/M_vs_sigma_c_for_sim.pkl")

# (locally-solved) phase boundary - very quick
solved_phase_c = Local_Solve_Phase_Boundary(globally_solved_sces_c)

# %%

########################## sigma_ys ################################

df_simulation_y = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                         + 'finite_effect_sigma_y_final')

globally_solved_sces_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                        + "cavity_solutions/self_limiting_yc_c/" + \
                                            "M_sigma_y_2.pkl")    

solved_phase_y = Local_Solve_Phase_Boundary(globally_solved_sces_y)

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot():
    
    resource_pool_sizes = np.unique(df_simulation_c['M'])
    sigma_cs = np.unique(df_simulation_c['sigma_c'])
    sigma_ys = np.unique(df_simulation_y['sigma_y'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot_c = le_pivot(df_simulation_c, columns = 'M',
                                   index = 'sigma_c')[0]
    
    stability_sim_pivot_y = le_pivot(df_simulation_y, columns = 'M',
                                   index = 'sigma_y')[0]
    
    sns.set_style('ticks')
    
    #mosaic = [["P", ".", "D1", "D2", ".", "I_C"],
    #          ["P", ".", ".", ".", ".", "I_C"],
    #          ["P", ".", "D3", "D4", ".", "I_C"],
    #          ["P", ".",  "D5", "D6", ".", "I_C"]]
    
    mosaic = [["P_sigma_y", ".", "P_sigma_c", ".", "I_C_sigma_y"],
              ["P_sigma_y", ".", "P_sigma_c", ".", "I_C_sigma_c"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (8.3, 2.5),
                                  width_ratios = [6.6, 1, 6.6, 0.8, 6.7],
                                  height_ratios = [1, 1],
                                  gridspec_kw = {'hspace' : 0.2, 'wspace' : 0.1})
    
    #########################
    
    def phase_diagram(ax, stability_sim_pivot, sigmas, ylabel,
                      cbar = False):
    
        subfig = sns.heatmap(stability_sim_pivot, ax = ax,
                             vmin = 0, vmax = 1, cbar = cbar, cmap = 'Purples')
            
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                       color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                       color = 'black', linewidth = 2)
        
        ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                            labels = resource_pool_sizes[::2], fontsize = 6,
                            rotation = 0)
    
        ax.set_yticks(np.arange(0.5, len(sigmas) + 0.5, 2), 
                      labels = [r'$' + str(sigma) + '^2$' 
                                for sigma in sigmas[::2]], fontsize = 6,
                      rotation = 0)
        
        ax.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                      weight = 'bold')
        ax.set_ylabel(ylabel, fontsize = 10, weight = 'bold')
        ax.invert_yaxis()
        
    phase_diagram(axs["P_sigma_y"], stability_sim_pivot_y, sigma_ys,
                  'variance in yield conversion, ' +  r'$\sigma_y^2$')
    
    phase_diagram(axs["P_sigma_c"], stability_sim_pivot_c, sigma_cs,
                  'variance in total consumption, ' +  r'$\sigma_c^2$',
                  cbar = True)
    
    #axs['P'].text(1, 1.35,
    #              'Increasing the variance in consumption rate ' + \
    #                r'$(\sigma_c^2)$' + ' stabilises\ncommunity dynamics',
    #              fontsize = 11, weight = 'bold',
    #              verticalalignment = 'top', horizontalalignment = 'center',
    #              transform=axs["P"].transAxes)
         
    cbar = axs["P_sigma_c"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with stable dynamics',
                   size = '8', horizontalalignment = 'center', 
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 6)
    
    ########## Analytically-derived phase boundary ###################
    
    #### sigma_ys ####
    
    good_solves_y = solved_phase_y.loc[solved_phase_y['loss'] <= -28, :]
     
    smoother = np.poly1d(np.polyfit(good_solves_y['M'], good_solves_y['sigma_y'], 2))
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    y_phase = smoother(smoothed_x) - np.min(sigma_ys)
    
    divider = np.unique(np.round(np.abs(np.diff(sigma_ys)), 8))
    y_vals = (1/divider)*y_phase + 0.5 
    
    x_vals = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
        np.arange(0, len(smoothed_x), 1)/25

    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P_sigma_y"], color = 'black',
                 linewidth = 3)
    
    #### sigma_cs ####
    
    good_solves_c = solved_phase_c.loc[solved_phase_c['loss'] <= -28, :]
    
    def hyperbolic(x, a, b):
        
        return a + b/x
    
    fit_p, _ = curve_fit(hyperbolic, good_solves_c['M'], good_solves_c['sigma_c'],
                         bounds = [0, [1e6, 1e6]])
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    y_phase = hyperbolic(smoothed_x, *fit_p) - np.min(sigma_cs)
    divider = np.unique(np.round(np.abs(np.diff(sigma_cs)), 8))
    y_vals = (1/divider)*y_phase + 0.5
    
    x_vals = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
        np.arange(0, len(smoothed_x), 1)/25

    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P_sigma_c"], color = 'black',
                 linewidth = 3)
    
    #################### Instability condition vs M #####################
    
    '''
    
    example_M = 225

    df_plot = globally_solved_sces[globally_solved_sces['M'] == example_M]
    dfl = pd.melt(df_plot[['sigma_c', 'rho', 'Species packing']], ['sigma_c'])
    #dfl.loc[dfl['variable'] == 'rho', 'value'] = dfl.loc[dfl['variable'] == 'rho', 'value']**2
    dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
        np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
        
    axs['I_C'].add_patch(Rectangle((np.min(sigma_cs), np.min(dfl['value'])),
                                   hyperbolic(example_M, *fit_p) - np.min(sigma_cs),
                                   np.max(dfl['value']) - np.min(dfl['value']),
                                   fill = True, color = '#b6b6d8ff', zorder = 0))
    
    axs["I_C"].vlines(hyperbolic(example_M, *fit_p), np.min(dfl['value']),
                      np.max(dfl['value']),
                      color = 'black', linewidth = 2.5, zorder = 0)
    
    subfig1 = sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 3,
                           palette = sns.color_palette(['black', 'black'], 2),
                           zorder = 10)

    subfig1 = sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 2, marker = 'o', markersize = 8,
                           palette = sns.color_palette(['#4a6fb5ff', '#3dc27aff'], 2),
                           zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    axs["I_C"].set_ylim([np.min(dfl['value']) - 0.02, np.max(dfl['value']) + 0.02])
    
    axs["I_C"].set_xlabel('variance in total consumption rate, ' + r'$\sigma_c^2$',
                          fontsize = 10, weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=6)
    axs["I_C"].set_xticks(sigma_cs[::2], labels = [r'$' + str(sigma) + '^2$' 
                                                   for sigma in sigma_cs[::2]])
    
    axs["I_C"].legend_.remove()
    
    #axs['I_C'].text(0.5, 1.35,
    #                'Increasing ' + r'$M$' ' increases interaction\n' + \
    #                    'reciprocity faster than the\nspecies packing ratio',
    #                  fontsize = 11, weight = 'bold',
    #                  verticalalignment = 'top', horizontalalignment = 'center',
    #                  transform=axs["I_C"].transAxes)
        
    #axs['I_C'].text(0.58, 0.1,
    #                'Stable', fontsize = 10, weight = 'bold', color = 'white',
    #                verticalalignment = 'top', horizontalalignment = 'center',
    #                path_effects = [patheffects.withStroke(linewidth=0.8,
    #                                                       foreground='black')],
    #                transform=axs["I_C"].transAxes)
    
    #axs['I_C'].text(0.29, 0.1,
    #                'Unstable', fontsize = 10, weight = 'bold', color = '#3f007dff',
    #                verticalalignment = 'top', horizontalalignment = 'center',
    #                path_effects = [patheffects.withStroke(linewidth=0.1,
    #                                                       foreground='black')],
    #                transform=axs["I_C"].transAxes)
        
    axs["I_C"].annotate("Stability threshold",
                        xytext=(hyperbolic(example_M, *fit_p),
                                np.max(dfl['value']) + 0.08),
                        xy=(hyperbolic(example_M, *fit_p), np.max(dfl['value'])),
                        color = 'black', fontsize = 10, weight = 'bold',
                        va = 'center', ha = 'center', multialignment = 'center',
                        arrowprops={'arrowstyle': '-|>', 'color' : 'black',
                                    'lw' : 1},
                        transform=axs["I_C"].transAxes)
    
    sns.despine(ax = axs["I_C"])
    
    '''
    
    ################################################

    
    ##################################################
    
    #plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_sigma_c_M_sim_and_analyticalphase_intrplt.png",
    #            bbox_inches='tight')
    #plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_sigma_c_M_sim_and_analyticalphase_intrplt.svg",
    #            bbox_inches='tight')
        
    plt.show()

Stability_Plot()
