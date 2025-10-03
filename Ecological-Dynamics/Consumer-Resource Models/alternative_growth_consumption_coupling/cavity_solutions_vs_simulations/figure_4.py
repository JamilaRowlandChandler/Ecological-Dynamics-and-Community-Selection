# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:50:49 2025

@author: jamil
"""


import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import le_pivot_r

# %%

################################ sigma_c ################################

df_simulation_c = pd.read_pickle("simulations/M_vs_sigma_c.pkl")
globally_solved_sces_c = pd.read_pickle("self_consistency_equations/M_vs_sigma_c.pkl")
solved_boundary_c = pd.read_pickle("self_consistency_equations/M_vs_sigma_c_stable_bound.pkl")

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot_sigma_c():
    
    resource_pool_sizes = np.unique(df_simulation_c['M'])
    sigma_cs = np.unique(df_simulation_c['sigma_c'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot_r(df_simulation_c, columns = 'M',
                                     index = 'sigma_c')[0]
    
    sns.set_style('ticks')
  
    mosaic = [["P", ".", "I_C"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (8.65, 2.5),
                                  width_ratios = [6.6, 6.55, 6.6],
                                  gridspec_kw = {'hspace' : 0.2, 'wspace' : 0.1})
    
    subfig = sns.heatmap(stability_sim_pivot, ax = axs["P"],
                         vmin = 0, vmax = 1, cbar = True, cmap = 'Purples_r')
        
    subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
    
    axs["P"].set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                        labels = resource_pool_sizes[::2], fontsize = 6,
                        rotation = 0)

    axs["P"].set_yticks(np.arange(0.5, len(sigma_cs) + 0.5, 2), 
                        labels = [r'$' + str(sigma) + '^2$' 
                                  for sigma in sigma_cs[::2]], fontsize = 6,
                        rotation = 0)
    
    axs["P"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                        weight = 'bold')
    axs["P"].set_ylabel('total consumption rate variance, ' + \
                        r'$\sigma_c^2$', fontsize = 10, weight = 'bold')
    axs["P"].invert_yaxis()
    
    axs['P'].text(1, 1.35,
                  'Increasing the variance in consumption rate ' + \
                    r'$(\sigma_c^2)$' + ' stabilises\ncommunity dynamics',
                  fontsize = 11, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["P"].transAxes)
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with stable dynamics',
                   size = '8', horizontalalignment = 'center', 
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 6)
    
    # Analytically-derived phase boundary
    
    good_solves = solved_boundary_c.loc[solved_boundary_c['loss'] <= -28, :]
    
    def hyperbolic(x, a, b):
        
        return a + b/x
    
    fit_p, _ = curve_fit(hyperbolic, good_solves['M'], good_solves['sigma_c'],
                         bounds = [0, [1e6, 1e6]])
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    y_phase = hyperbolic(smoothed_x, *fit_p) - np.min(sigma_cs)
    divider = np.unique(np.round(np.abs(np.diff(sigma_cs)), 8))
    y_vals = (1/divider)*y_phase + 0.5
    
    x_vals = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
        np.arange(0, len(smoothed_x), 1)/25

    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P"], color = 'black',
                 linewidth = 3)
    
    #################### Instability condition vs M #####################
    
    example_M = 225

    df_plot = globally_solved_sces_c[globally_solved_sces_c['M'] == example_M]
    dfl = pd.melt(df_plot[['sigma_c', 'rho', 'Species packing']], ['sigma_c'])
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
                           palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                           zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    axs["I_C"].set_ylim([np.min(dfl['value']) - 0.02, np.max(dfl['value']) + 0.02])
    
    axs["I_C"].set_xlabel('variance in total consumption rate, ' + r'$\sigma_c^2$',
                          fontsize = 10, weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=6)
    axs["I_C"].set_xticks(sigma_cs[::2], labels = [r'$' + str(sigma) + '^2$' 
                                                   for sigma in sigma_cs[::2]])
    
    axs["I_C"].legend_.remove()
    
    axs['I_C'].text((0.5*(np.max(sigma_cs) + hyperbolic(example_M, *fit_p)) - np.min(sigma_cs))/(np.max(sigma_cs) - np.min(sigma_cs)),
                    1.03,
                    'Stable', fontsize = 10, weight = 'bold', color = 'black',
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)
    
    axs['I_C'].text((0.5*(hyperbolic(example_M, *fit_p) - np.min(sigma_cs)))/(np.max(sigma_cs) - np.min(sigma_cs)),
                    1.03,
                    'Unstable', fontsize = 10, weight = 'bold', color = 'black',
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)
    
    sns.despine(ax = axs["I_C"])
    
    ###############################################
    
    plt.show()

Stability_Plot_sigma_c()

# %%

################################ sigma_y ################################

df_simulation_y = pd.read_pickle("simulations/M_vs_sigma_y.pkl")
globally_solved_sces_y = pd.read_pickle("self_consistency_equations/M_vs_sigma_y.pkl")
solved_boundary_y = pd.read_pickle("self_consistency_equations/M_vs_sigma_y_stable_bound.pkl")

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot_sigma_y():
    
    resource_pool_sizes = np.unique(df_simulation_y['M'])
    sigma_ys = np.unique(df_simulation_y['sigma_y'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot_r(df_simulation_y, columns = 'M',
                                     index = 'sigma_y')[0]
    
    sns.set_style('ticks')
    
    mosaic = [["P", ".", "I_C"]]
    
    fig, axs = plt.subplot_mosaic(mosaic, figsize = (8.65, 2.5),
                                  width_ratios = [6.6, 6.55, 6.6],
                                  gridspec_kw = {'hspace' : 0.2, 'wspace' : 0.1})
    
    subfig = sns.heatmap(stability_sim_pivot, ax = axs["P"],
                         vmin = 0, vmax = 1, cbar = True, cmap = 'Purples_r')
        
    subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axhline(stability_sim_pivot.shape[0], 0, 1,
                   color = 'black', linewidth = 2)
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(stability_sim_pivot.shape[1], 0, 1,
                   color = 'black', linewidth = 2)
    
    axs["P"].set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                        labels = resource_pool_sizes[::2], fontsize = 6,
                        rotation = 0)

    axs["P"].set_yticks(np.arange(0.5, len(sigma_ys) + 0.5, 2), 
                        labels = [r'$' + str(sigma) + '^2$' 
                                  for sigma in sigma_ys[::2]], fontsize = 6,
                        rotation = 0)
    
    axs["P"].set_xlabel('resource pool size, ' + r'$M$', fontsize = 10,
                        weight = 'bold')
    axs["P"].set_ylabel('variance in yield conversion, ' + \
                        r'$\sigma_y^2$', fontsize = 10, weight = 'bold')
    axs["P"].invert_yaxis()
    
    axs['P'].text(1, 1.35,
                  'Increasing the variance in yield conversion ' + \
                    r'$(\sigma_y^2)$' + ' destabilises\ncommunity dynamics',
                  fontsize = 11, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["P"].transAxes)
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with stable dynamics',
                   size = '8', horizontalalignment = 'center', 
                   verticalalignment = 'top')
    cbar.ax.tick_params(labelsize = 6)
    
    # Analytically-derived phase boundary
    
    good_solves = solved_boundary_y.loc[solved_boundary_y['loss'] <= -28, :]
     
    smoother = np.poly1d(np.polyfit(good_solves['M'], good_solves['sigma_y'], 2))
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 25,
                           np.max(resource_pool_sizes) + 25,
                           1)
    
    y_phase = smoother(smoothed_x) - np.min(sigma_ys)
    
    divider = np.unique(np.round(np.abs(np.diff(sigma_ys)), 8))
    y_vals = (1/divider)*y_phase + 0.5 
    
    x_vals = (0.5 - (np.min(resource_pool_sizes) - np.min(smoothed_x))/25) + \
        np.arange(0, len(smoothed_x), 1)/25

    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P"], color = 'black',
                 linewidth = 3)
    
    #################### Instability condition vs M #####################
    
    example_M = 225

    df_plot = globally_solved_sces_y[globally_solved_sces_y['M'] == example_M]
    dfl = pd.melt(df_plot[['sigma_y', 'rho', 'Species packing']], ['sigma_y'])
    dfl.loc[dfl['variable'] == 'Species packing', 'value'] = \
        np.sqrt(dfl.loc[dfl['variable'] == 'Species packing', 'value'])
        
    axs['I_C'].add_patch(Rectangle((smoother(example_M), np.min(dfl['value'])),
                                   np.max(sigma_ys) - smoother(example_M),
                                   np.max(dfl['value']) - np.min(dfl['value']),
                                   fill = True, color = '#6950a3ff', zorder = 0))
     
    axs["I_C"].vlines(smoother(example_M), np.min(dfl['value']), np.max(dfl['value']),
                      color = 'black', linewidth = 2.5, zorder = 0)
    
    subfig1 = sns.lineplot(dfl, x = 'sigma_y', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 3,
                           palette = sns.color_palette(['black', 'black'], 2),
                           zorder = 10)

    subfig1 = sns.lineplot(dfl, x = 'sigma_y', y = 'value', hue = 'variable',
                           ax = axs["I_C"], linewidth = 2, marker = 'o', markersize = 8,
                           palette = sns.color_palette(['#00557aff', '#3dc27aff'], 2),
                           zorder = 10, markeredgewidth = 0.4, markeredgecolor = 'black')
    
    axs["I_C"].set_xlabel('variance in yield conversion, ' + r'$\sigma_y^2$',
                          fontsize = 10, weight = 'bold')
    axs["I_C"].set_ylabel('')
    axs["I_C"].tick_params(axis='both', which='major', labelsize=6)
    axs["I_C"].set_xticks(sigma_ys[::2], labels = [r'$' + str(sigma) + '^2$' 
                                                   for sigma in sigma_ys[::2]])
    
    axs["I_C"].legend_.remove()
    
    axs['I_C'].text((0.5*(np.max(sigma_ys) + smoother(example_M)) - np.min(sigma_ys))/(np.max(sigma_ys) - np.min(sigma_ys)),
                    1.02,
                    'Unstable', fontsize = 10, weight = 'bold', color = 'black',
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)
    
    axs['I_C'].text((0.5*(smoother(example_M) - np.min(sigma_ys)))/(np.max(sigma_ys) - np.min(sigma_ys)),
                    1.02,
                    'Stable', fontsize = 10, weight = 'bold', color = 'black',
                    verticalalignment = 'top', horizontalalignment = 'center',
                    transform=axs["I_C"].transAxes)

    sns.despine(ax = axs["I_C"])
    
    plt.show()
    
Stability_Plot_sigma_y()
