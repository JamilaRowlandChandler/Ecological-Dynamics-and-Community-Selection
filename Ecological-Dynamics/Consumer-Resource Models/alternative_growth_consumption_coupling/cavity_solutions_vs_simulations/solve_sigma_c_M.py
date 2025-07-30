# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:43:55 2025

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
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

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

def Global_Solve_SCEs(df_simulation, filename):
    
    extractable_parameters = df_simulation.groupby(['M',
                                                    'sigma_c'])[['mu_c', 'mu_y',
                                                                 'sigma_y']].mean().reset_index().to_dict('records')
    
    # Solver arguments                                                         
    parameters = sce.variable_fixed_parameters(extractable_parameters,
                                               {'mu_b' : 1, 'sigma_b' : 0,
                                                'mu_d' : 1, 'sigma_d' : 0,
                                                'gamma' : 1})
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10,  1e-10, 1e-10],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
    
    x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
    
    solved_sces = solve_sces_yc_c(parameters, solved_quantities, bounds, x_init,
                                  'basin-hopping', other_kwargs = {'niter' : 200})
    
    solved_sces['S'] = solved_sces['M']/solved_sces['gamma']
    
    final_sces = clean_bad_solves(solved_sces)
    
    solved_sces['M'] = np.int32(solved_sces['M'])

    # save data
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                          + "cavity_solutions/self_limiting_yc_c"
    if not os.path.exists(directory): os.makedirs(directory) 
    
    pickle_dump(directory + filename, final_sces)
    
    return solved_sces

# %%

def clean_bad_solves(sces, other_kwargs = {'niter' : 500}):
    
    bad_solves = sces.loc[sces['loss'] > -30, :]
    
    if bad_solves.empty is True:
        
        return sces
    
    else:
    
        parameters = bad_solves[['M', 'mu_c', 'mu_y','sigma_c', 
                                 'sigma_y', 'mu_b', 'sigma_b',
                                 'mu_d', 'sigma_d', 'gamma']].to_dict('records')
        
        solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                             'phi_R', 'R_mean', 'q_R', 'chi_R']
        
        bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10],
                  [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
        
        x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
        
        cleaned_sces = solve_sces_yc_c(parameters, solved_quantities, bounds, x_init,
                                      'basin-hopping', other_kwargs = other_kwargs)
        cleaned_sces['S'] = cleaned_sces['M']/cleaned_sces['gamma']
        
        final_sces = copy(sces)
        bad_solve_idx = final_sces.loc[final_sces['loss'] > -30, :].index.tolist()
        cleaned_sces.rename(index={old_idx : new_idx for old_idx, new_idx in
                                   zip(cleaned_sces.index.tolist(), bad_solve_idx)},
                            inplace = True)
        final_sces.update(cleaned_sces)
        
        return final_sces

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

# load in simulation data
df_simulation = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                       + 'finite_effects_sigma_c_final')

# %%

# globally solved self consistency equations - slow
globally_solved_sces = Global_Solve_SCEs(df_simulation,
                                         "/M_vs_sigma_c_for_sim.pkl")

# %%

# load in simulation data and solved sces
globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/M_vs_sigma_c_for_sim.pkl")

# %%

# (locally-solved) phase boundary - very quick
solved_phase = Local_Solve_Phase_Boundary(globally_solved_sces)

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot():
    
    resource_pool_sizes = np.unique(df_simulation['M'])
    sigma_cs = np.unique(df_simulation['sigma_c'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot(df_simulation, columns = 'M',
                                   index = 'sigma_c')[0]
    
    sns.set_style('white')
    
    mosaic = [["P",".", "I_C"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (14.8, 5),
                                  width_ratios = [6, 1.2, 6],
                                  height_ratios = [1],
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
    axs["P"].set_ylabel('standard deviation in total consumption rate, ' + \
                        r'$\sigma_c$', fontsize = 14, weight = 'bold')
    axs["P"].invert_yaxis()
    
    axs['P'].text(0.5, 1.2,
                  'Increasing the standard deviation in consumption rate\n' + \
                    r'$(\sigma_c)$' + ' weakly stabilises communities',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["P"].transAxes)
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
                   size = '14')
    cbar.ax.tick_params(labelsize = 12)
    
    # Analytically-derived phase boundary
    
    good_solves = solved_phase.loc[solved_phase['loss'] <= -28, :]
    
    def hyperbolic(x, a, b):
        
        return a + b/x
    
    fit_p, _ = curve_fit(hyperbolic, good_solves['M'], good_solves['sigma_c'],
                         bounds = [0, [1e6, 1e6]])
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 15,
                           np.max(resource_pool_sizes) + 15,
                           1)
    
    y_phase = hyperbolic(smoothed_x, *fit_p) - np.min(sigma_cs)
    divider = np.unique(np.round(np.abs(np.diff(sigma_cs)), 8))
    y_vals = (1/divider)*y_phase + 0.5
    
    x_vals = 0.5 + np.arange(0, len(smoothed_x), 1)/25

    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P"], color = 'black',
                 linewidth = 6, linestyle = '--')
    
    #################### Instability condition vs M #####################
    
    example_M = 225

    df_plot = globally_solved_sces[globally_solved_sces['M'] == example_M]
    dfl = pd.melt(df_plot[['sigma_c', 'rho', 'Species packing']], ['sigma_c'])
    dfl.loc[dfl['variable'] == 'rho', 'value'] = dfl.loc[dfl['variable'] == 'rho', 'value']**2
     
    axs["I_C"].vlines(hyperbolic(example_M, *fit_p), np.min(dfl['value']),
                      np.max(dfl['value']),
                      color = 'black', linestyle = '--', linewidth = 3, zorder = 0)

    subfig1 = sns.lineplot(dfl, x = 'sigma_c', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 5, marker = 'o', markersize = 13,
                           palette = sns.color_palette(['#39568cff', '#1f968bff'], 2),
                           zorder = 100)

    axs["I_C"].set_xlabel('standard deviation in total consumption rate, ' + r'$\sigma_c$',
                          fontsize = 14, weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=14)
    axs["I_C"].set_xticks(sigma_cs, labels = sigma_cs)
    
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
                  'Increasing ' + r'$\sigma_c$' ' increases the correlation and species\n' + \
                      'packing ratio, weakening destabilising effects',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["I_C"].transAxes)
        
    axs["I_C"].annotate("Stability\nthreshold", xytext=(1.7, 0.3),
                        xy=(hyperbolic(example_M, *fit_p) + 0.01, 0.3),
                        color = 'black', fontsize = 14, weight = 'bold',
                        va = 'center', multialignment = 'center',
                        arrowprops={'arrowstyle': '-|>', 'color' : 'black', 'lw' : 2})
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_sigma_c_M_sim_and_analyticalphase_intrplt.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_sigma_c_M_sim_and_analyticalphase_intrplt.svg",
                bbox_inches='tight')
        
    plt.show()

Stability_Plot()
