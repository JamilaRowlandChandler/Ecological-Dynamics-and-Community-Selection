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
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import make_splrep

from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.colors import LinearSegmentedColormap

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import CRM_df, \
    le_pivot, generic_heatmaps, pickle_dump
    
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

def Global_Solve_SCEs(df_simulation):
    
    extractable_parameters = df_simulation.groupby(['M',
                                                    'mu_c'])[['mu_y',
                                                              'sigma_c',
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
    
    solved_sces['M'] = np.int32(solved_sces['M'])
    solved_sces['S'] = solved_sces['M']/solved_sces['gamma']
    
    def clean_bad_solves(sces, other_kwargs = {'niter' : 500}):
        
        bad_solves = sces.loc[sces['loss'] > -30, :]
        
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
        
        cleaned_sces['M'] = np.int32(cleaned_sces['M'])
        cleaned_sces['S'] = cleaned_sces['M']/cleaned_sces['gamma']
        
        final_sces = sces
        final_sces.loc[final_sces['loss'] > -30, :] = cleaned_sces.to_numpy()
        
        return final_sces
    
    solved_sces = clean_bad_solves(solved_sces)

    # save data
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                          + "cavity_solutions/self_limiting_yc_c"
    if not os.path.exists(directory): os.makedirs(directory) 
    
    pickle_dump(directory + "/M_vs_mu_c.pkl", solved_sces)
    
    return solved_sces

# %%

def Local_Solve_Phase_Boundary(solved_sces, solved_quantity = 'mu_c',
                               quantity_bounds = [80, 260]):
    
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
    
    interpolator = BarycentricInterpolator(max_dNde_df['M'],
                                           max_dNde_df[parm_names + solved_quantities])
    
    interpolated_M = np.arange(np.min(max_dNde_df['M']) + 10, # 100
                               np.max(max_dNde_df['M']) - 10, 5)
    
    interpolated_matrix = interpolator(interpolated_M)
     
    smoothed_interps = [np.poly1d(np.polyfit(interpolated_M, interpolated_y, 2))
                        for interpolated_y in interpolated_matrix.T]
    
    interpolated_data = pd.DataFrame([np.round(smooth_y(interpolated_M), 7)
                                      for smooth_y in smoothed_interps],
                                     index = parm_names + solved_quantities).T
    
    for col in parm_names + ['phi_N', 'N_mean', 'q_N', 'phi_R', 'R_mean', 'q_R']:
        
        interpolated_data.loc[interpolated_data[col] < 0, col] = 1e-10
     
    interpolated_data['M'] = interpolated_M
    
    parm_names.append('M')
    parm_names.remove(solved_quantity)
    parameters = interpolated_data[parm_names].to_dict('records')
    
    solved_quantities.append(solved_quantity)
    x_init_dicts = interpolated_data[solved_quantities].to_dict('records')
    
    bounds = [([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10,
                0.85 * x_is[solved_quantity]], 
               [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15,
                1.15 * x_is[solved_quantity]])
              for x_is in x_init_dicts]
    
    x_inits = [list(x_init.values()) for x_init in x_init_dicts]
    
    solved_phase = solve_sces_yc_c(parameters, solved_quantities, bounds, x_inits,
                                   'least-squares', 
                                   solver_kwargs = {'xtol' : 1e-13, 'ftol' : 1e-13},
                                   include_multistability = True)
     
    return solved_phase
      

# %%

def generate_simulation_df():
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + 'finite_effects_fixed_C_final'
                        
    parameters = ['no_species', 'no_resources', 'mu_c', 'sigma_c', 'mu_y',
                  'sigma_y', 'd_val', 'b_val']
     
    df = CRM_df(directory, parameters)
    
    for var in ['rho', 'mu_c', 'mu_y', 'sigma_c', 'sigma_y', 'mu_c/M',
                'sigma_c/root_M']:
        
        df[var] = np.round(df[var], 6)
    
    df['no_resources'] = np.int32(df['no_resources'])
    df.rename(columns = {'no_resources' : 'M', 'no_species' : 'S'}, inplace = True)
    
    return df

# %%

# load in simulation data
df_simulation = generate_simulation_df()

# %%

# globally solved self consistency equations - slow
globally_solved_sces = Global_Solve_SCEs(df_simulation)

# %%

# load in simulation data and solved sces
globally_solved_sces = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c//M_vs_mu_c.pkl")

# %%

# (locally-solved) phase boundary - very quick
solved_phase = Local_Solve_Phase_Boundary(globally_solved_sces)

# %%

plt.plot(solved_phase['M'], solved_phase['mu_c'])
plt.show()

# %%

############# Phase diagram + analytically-derived boundary #####################

def Stability_Plot():
    
    resource_pool_sizes = np.unique(df_simulation['M'])
    mu_cs = np.unique(df_simulation['mu_c'])
    
    ######################## Phase diagram ######################################
    
    # Simulation data
    
    stability_sim_pivot = le_pivot(df_simulation, columns = 'M',
                                   index = 'mu_c')[0]
    
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
                        r'$\mu_c$', fontsize = 14, weight = 'bold')
    axs["P"].invert_yaxis()
    
    axs['P'].text(1.2, 1.2,
                  'Increasing the resource pool size stabilises community dynamics',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["P"].transAxes)
         
    cbar = axs["P"].collections[0].colorbar
    cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
                   size = '14')
    cbar.ax.tick_params(labelsize = 12)
    
    # Analytically-derived phase boundary
    
    good_solves = solved_phase.loc[solved_phase['loss'] <= -28, :]
    
    smoother = np.poly1d(np.polyfit(good_solves['M'], good_solves['mu_c'], 2))
    #smoother = make_splrep(good_solves['M'], good_solves['mu_c'], k = 3, s = 2)
    
    smoothed_x = np.arange(np.min(resource_pool_sizes) - 15,
                           np.max(resource_pool_sizes) + 15,
                           1)
    
    y_phase = smoother(smoothed_x) - np.min(mu_cs)
    divider = np.unique(np.abs(np.diff(mu_cs)))
    y_vals = (1/divider)*y_phase + 0.5
    
    x_vals = 0.5 + np.arange(0, len(smoothed_x), 1)/25
    
    #breakpoint()
    
    sns.lineplot(x = x_vals, y = y_vals, ax = axs["P"], color = 'black',
                 linewidth = 6)
    
    #################### Instability condition vs M #####################
    
    # Example relathips with mu_c = 145
    example_mu_c = 145

    df_plot = globally_solved_sces.loc[globally_solved_sces['mu_c'] == example_mu_c, :]
    dfl = pd.melt(df_plot[['M', 'rho', 'Species packing']], ['M'])
    dfl.loc[dfl['variable'] == 'rho', 'value'] = dfl.loc[dfl['variable'] == 'rho', 'value']**2
    
    M_stability_threshold = smoothed_x[np.abs(smoother(smoothed_x) - example_mu_c).argmin()]
    
    axs["I_C"].vlines(M_stability_threshold, np.min(dfl['value']), np.max(dfl['value']),
                      color = 'black', linestyle = '--', linewidth = 3, zorder = 0)

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
                  'Increasing the resource pool size stabilises\ncommunities ' + \
                  'by increasing the correlation\nfaster than the packing ratio',
                  fontsize = 16, weight = 'bold',
                  verticalalignment = 'top', horizontalalignment = 'center',
                  transform=axs["I_C"].transAxes)
        
    #axs["I_C"].text(M_stability_threshold/(np.max(resource_pool_sizes) - np.min(resource_pool_sizes)) - 0.05,
    #                0.98,
    #                "Unstable", color='#7300e3ff', fontsize=16,
    #                path_effects=[patheffects.withStroke(linewidth=1.5,
    #                                                     foreground='black')],
    #                horizontalalignment='right', verticalalignment='top',
    #                transform=axs["I_C"].transAxes)

    #axs["I_C"].text(0.48, 0.98, "Stable", color='white', fontsize=16,
    #            path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')],
    #            horizontalalignment='left', verticalalignment='top',
    #            transform=axs["I_C"].transAxes)

    axs["I_C"].annotate("Stability\nthreshold",
                        xytext=(M_stability_threshold - 75, 0.6),
                        xy=(M_stability_threshold - 0.01, 0.6),
                        color = 'black', fontsize = 14, weight = 'bold',
                        va = 'center', multialignment = 'center',
                        arrowprops={'arrowstyle': '-|>', 'color' : 'black', 'lw' : 2})
    
    ####################### Example population dynamics ######################
    
    # M = 75 and 225, mu_c = 145
    
    example_M = [75, 225]
    
    chaotic_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_C_final/" + \
                                         "simulations_75_1.9333.pkl")
        
    stable_populations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/finite_effects_fixed_C_final/" + \
                                         "simulations_225_0.6444.pkl")
                 
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
        data = simulation.ODE_sols[0]
        
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
             stable_populations[2], stable_populations[2]],
            i_c_rp,
            ['species', 'resources', '', '']):
        
        plot_dynamics(ax, simulation, i_c_rp_M, title)
        
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