# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:15:12 2025

@author: jamil
"""

import numpy as np
import pandas as pd
from copy import copy
import seaborn as sns
from matplotlib import pyplot as plt

# %%

def plot_same_variable_multiple_dfs(dfs, variable, x, ys, x_label, y_labels,
                                    variable_label, figdims, figsize,
                                    cmap = 'Blues', log = False,
                                    min_max = [None, 'symmetrical']):
    
    pivot_tables = [df.pivot(index = y, columns = x, values = variable)
                    for df, y in zip(dfs, ys)]
    
    if log is True:
        
        pivot_tables = [np.log10(np.abs(pivot_table))
                        for pivot_table in pivot_tables]
        
    if min_max[0] is not None:
        
        id_min, id_max = min_max
    
    else:
        
        id_min = np.min(np.concatenate([df[variable].to_numpy() for df in dfs]))
        id_max = np.max(np.concatenate([df[variable].to_numpy() for df in dfs]))
        
        if min_max[1] == 'symmetrical':
        
            if id_max > np.abs(id_min): id_min = -id_max 
            else: id_max = -id_min
            
    sns.set_style('white')
    
    fig, axs = plt.subplots(figdims[0], figdims[1], figsize = figsize,
                            sharex = True, layout = 'constrained')
    #fig.subplots_adjust(hspace = 0.125, wspace = 0.75)
    
    cbar_ax = fig.add_axes([1.02, 0.3, 0.02, 0.4])
    
    fig.supxlabel(x_label, fontsize = 16, weight = 'bold')
             
    def plot_ax(ax, pivot_table, y, i):
         
        subfig = sns.heatmap(pivot_table, ax = ax, vmin = id_min,
                             vmax = id_max, cbar = i == len(dfs) - 1,
                             cmap = cmap,
                             cbar_ax = cbar_ax if i == len(dfs) - 1 else None)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot_table.shape[0], 0, 1, color = 'black',
                       linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot_table.shape[1], 0, 1, color = 'black',
                       linewidth = 2)
        
        ax.set_xticks(np.arange(0.5, len(pivot_table.columns) + 0.5, 2),
                      labels = pivot_table.columns[::2], fontsize = 14,
                      rotation = 0)
        
        ax.set_yticks(np.arange(0.5, len(pivot_table.index) + 0.5, 2),
                      labels = pivot_table.index[::2], fontsize = 14,
                      rotation = 0)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.invert_yaxis()
    
        return subfig, ax

    if figdims == (1, 1):
        
        subfig, axs = plot_ax(axs, pivot_tables[0], ys[0], 0)
        
        fig.supylabel(y_labels, fontsize = 16, weight = 'bold')
        cbar = axs.collections[0].colorbar
            
    else: 
        
        for i, (ax, pivot_table, y, y_label) in enumerate(zip(axs.flatten()[:len(dfs)],
                                                              pivot_tables,
                                                              ys, y_labels)):
            
            subfig, ax = plot_ax(ax, pivot_table, y, i)
            ax.set_ylabel(y_label, fontsize = 16, weight = 'bold',
                          loc = 'center')
            
        cbar = axs.flatten()[len(dfs) - 1].collections[0].colorbar
        
    cbar.set_label(label = variable_label, size = '14')
    cbar.ax.tick_params(labelsize = 12)
    cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
    
    return fig, axs

# %%

def plot_multiple_variables_same_df(df, variables, x, y, x_label, y_label,
                                    variable_labels, figdims, figsize,
                                    cmaps = None, logs = None, min_maxs = None):
    
    if not cmaps:
        
        all_cmaps = plt.colormaps()
        cmaps = np.tile(all_cmaps,
                        np.int32(np.ceil(len(variables)/len(all_cmaps))))[:len(variables)]
    
    pivot_tables = [df.pivot(index = y, columns = x, values = variable)
                    for variable in variables]
    
    if logs:
        
        pivot_tables = [np.log10(np.abs(pivot_table)) if log is True else pivot_table
                        for log, pivot_table in zip(logs, pivot_tables)]
        
    if not min_maxs[0]:
        
        min_maxs = [[np.min(df[variable].to_numpy()), np.max(df[variable].to_numpy())]
                    for variable in variables]
        
        if min_maxs[1] == 'symmetrical':
            
            min_maxs = [[-mm[1], mm[1]] if mm[1] > np.abs(mm[0]) 
                        else [mm[0], -mm[0]] for mm in min_maxs]
   
    sns.set_style('white')
    
    fig, axs = plt.subplots(figdims[0], figdims[1], figsize = figsize,
                            sharex = True, layout = 'constrained', wspace = 0.25)
    
    fig.supxlabel(x_label, fontsize = 16, weight = 'bold')
    fig.supxlabel(y_label, fontsize = 16, weight = 'bold')
    
    def plot_ax(ax, pivot_table, variable_label, cmap, id_min, id_max):
    
        ax.set_facecolor('grey')
        
        subfig = sns.heatmap(pivot_table, ax = ax, vmin = id_min,
                             vmax = id_max, cbar = False, cmap = cmap)
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(pivot_table.shape[0], 0, 1, color = 'black',
                       linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(pivot_table.shape[1], 0, 1, color = 'black',
                       linewidth = 2)
        
        axs.set_xticks(np.arange(0.5, len(pivot_table.columns) + 0.5, 2),
                       labels = pivot_table.columns[::2], fontsize = 14,
                       rotation = 0)
        
        axs.set_xticks(np.arange(0.5, len(pivot_table.index) + 0.5, 2),
                       labels = pivot_table.index[::2], fontsize = 14,
                       rotation = 0)
        
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.invert_yaxis()
        
        cbar = ax.collections.colorbar
        cbar.set_label(label = variable_label, size = '14')
        cbar.ax.tick_params(labelsize = 12)
    
        return subfig, ax
    
    for ax, pivot_table, variable_label, cmap, min_max in zip(axs.flatten(),
                                                              pivot_tables,
                                                              variable_labels,
                                                              cmaps,
                                                              min_maxs):
        
        subfig, ax = plot_ax(ax, pivot_table, variable_label, cmap, min_max[0],
                             min_max[1])
        
    plt.show()
    
    return fig, axs

# %%

def stability_infeasibility_transitions_other_parms():
    
    solved_sces_mu_d = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_mu_d_2.pkl")
        
    solved_sces_mu_b = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_mu_b_2.pkl")

    solved_sces_sigma_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_sigma_c_2.pkl")

    solved_sces_sigma_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_sigma_y_2.pkl")

    solved_sces_sigma_d = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_sigma_d_2.pkl")
        
    solved_sces_sigma_b = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c/" + \
                                          "M_sigma_b_2.pkl")
            
    sces_list = [np.round(sces, 7)
                 for sces in [solved_sces_sigma_y,
                              solved_sces_mu_d, solved_sces_sigma_d,
                              solved_sces_mu_b, solved_sces_sigma_b]]
    
    quantities = ['sigma_y', 'mu_d', 'sigma_d', 'mu_b', 'sigma_b']
    
    y_labels = ['std deviation in yield\nconversion efficiency, $\sigma_y$',
                'average death rate, $\mu_d$', 
                'std deviation in death\nrate, $\sigma_d$',
                'average intrinsic\nresource growth rate, $\mu_b$',
                'std deviation in intrinsic\nresource growth rate, $\sigma_b$']
    
    fig, axs = plot_same_variable_multiple_dfs(sces_list, 'Instability distance',
                                               'M', quantities, 
                                               'resource pool size, ' + r'$M$',
                                               y_labels,
                                               'Stability condition, ' + \
                                                   r'$\rho^2 - S^*/M^*$',
                                               (2, 3), (12, 6),
                                               cmap = 'RdBu')
        
    fig.delaxes(axs.flatten()[-1])
        
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                "self_limit_instability_M_others.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                "self_limit_instability_M_others.svg",
                bbox_inches='tight')
        
    fig, axs = plot_same_variable_multiple_dfs(sces_list, 'Infeasibility distance',
                                               'M', quantities, 
                                               'resource pool size, ' + r'$M$',
                                               y_labels,
                                               'Infeasibility condition, ' + \
                                                   r'$M^* - S^*$',
                                               (2, 3), (12, 6),
                                               cmap = 'viridis_r', min_max = [0, 1])
        
    fig.delaxes(axs.flatten()[-1])
        
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                "self_limit_infeasibility_M_others.png",
                bbox_inches='tight')
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                "self_limit_infeasibility_M_others.svg",
                bbox_inches='tight')
        
stability_infeasibility_transitions_other_parms()

# %%














                                                                                














