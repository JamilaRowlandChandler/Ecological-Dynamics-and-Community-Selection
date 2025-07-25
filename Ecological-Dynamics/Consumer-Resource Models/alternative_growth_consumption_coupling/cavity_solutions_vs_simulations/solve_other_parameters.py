# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:30:06 2025

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

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generic_heatmaps, pickle_dump
    
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

def Global_Solve_SCEs(solved_quantity, quantity_range,
                      M_vals, fixed_parameters,
                      n = 10):

    variable_parameters = np.unique(sce.parameter_combinations([M_vals,
                                                                quantity_range],
                                                               n),
                                    axis = 1)
    
    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['M', solved_quantity]).tolist()
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10,  1e-10, 1e-10],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15])
    
    x_init = np.array([0.5, 0.01, 0.01, -0.1, 0.7, 0.001, 0.01, 0.05])
    
    solved_sces = solve_sces_yc_c(parameters, solved_quantities, bounds, x_init,
                                  'basin-hopping', other_kwargs = {'niter' : 200})
    
    solved_sces['M'] = np.int32(solved_sces['M'])
    solved_sces['S'] = solved_sces['M']/solved_sces['gamma']
    
    final_sces = clean_bad_solves(solved_sces)
    
    # save data
    
    directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                          + "cavity_solutions/self_limiting_yc_c"
    if not os.path.exists(directory): os.makedirs(directory) 
    
    pickle_dump(directory + "/M_" + solved_quantity + "_2.pkl", final_sces)
    
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
        
        final_sces['M'] = np.int32(final_sces['M'])
        
        return final_sces

# %%

def dict_without_key(dictionary, key):
    
    dict_copy = dict(dictionary)
    
    dict_copy.pop(key)
    
    return dict_copy

# %%

resource_pool_sizes = np.arange(50, 275, 25)
any_fixed_parameters = dict(mu_c = 130, sigma_c = 1.6, mu_y = 1, sigma_y = 0.130639,
                            mu_d = 1, sigma_d =  0, mu_b = 1, sigma_b = 0, 
                            gamma = 1)

# %%

# Testing the effect of mu_d

solved_sces_mu_d = Global_Solve_SCEs('mu_d', [0.5, 2.5],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'mu_d'),
                                        n = 9)

# Testing the effect of mu_b

solved_sces_mu_b = Global_Solve_SCEs('mu_b', [0.5, 2.5],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'mu_b'),
                                        n = 9)

# Testing the effect of sigma_c

solved_sces_sigma_c = Global_Solve_SCEs('sigma_c', [0.5, 2.5],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'sigma_c'),
                                        n = 9)

# Testing the effect of sigma_y

solved_sces_sigma_y = Global_Solve_SCEs('sigma_y', [0.05, 0.25],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'sigma_y'),
                                        n = 9)

solved_sces_sigma_d = Global_Solve_SCEs('sigma_d', [0, 0.25],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'sigma_d'),
                                        n = 9)

# Testing the effect of sigma_b

solved_sces_sigma_b = Global_Solve_SCEs('sigma_b', [0, 0.25],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'sigma_b'),
                                        n = 9)

# %%

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
        
# %%

def plot_instability_distances():

    sces_list = [sces.loc[sces['loss'] < -30, :] 
                 for sces in [solved_sces_sigma_c, solved_sces_sigma_y,
                              solved_sces_mu_d, solved_sces_sigma_d,
                              solved_sces_mu_b, solved_sces_sigma_b]]
    
    quantities = ['sigma_c', 'sigma_y', 'mu_d', 'sigma_d', 'mu_b', 'sigma_b']
    
    y_labels = ['std deviation in total consumption rate, $\sigma_c$',
                'std deviation in yield conversion efficiency, $\sigma_y$',
                'average death rate, $\mu_d$', 
                'std deviation in death rate, $\sigma_d$',
                'average intrinsic resource growth rate, $\mu_b$',
                'std deviation in intrinsic resource growth rate, $\sigma_b$']
    
    id_max = np.max(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    id_min = np.min(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    
    if id_max > np.abs(id_min): id_min = -id_max 
    else: id_max = -id_min
    
    for (sces, quantity, y_label) in zip(sces_list, quantities, y_labels):
    
        generic_heatmaps(sces, 'M', quantity,
                         "resource pool size, $M$", y_label,
                         ["Instability distance"], "RdBu", "", (1, 1), (5.5, 5),
                         specify_min_max = {'Instability distance' : [id_min, id_max]})
        
        plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                    "self_limit_instability_M_" + quantity + ".png",
                    bbox_inches='tight')
        plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/" + \
                    "self_limit_instability_M_" + quantity + ".svg",
                    bbox_inches='tight')
        
        plt.show()
    
plot_instability_distances()

# %%

'''

def Local_Solve_Phase_Boundary(solved_sces, solved_quantity, quantity_bounds):
    
    parm_names = ['mu_c', 'sigma_c', 'mu_y', 'sigma_y',
                  'mu_b', 'sigma_b', 'mu_d', 'sigma_d',
                  'gamma']
    
    solved_quantities = ['phi_N', 'N_mean', 'q_N', 'v_N',
                         'phi_R', 'R_mean', 'q_R', 'chi_R']
    
    solved_sces = solved_sces[solved_sces['loss'] < -30]
    solved_sces.loc[:, 'dRde'] = np.abs(solved_sces['dRde'])
    
    max_dnde_by_M = solved_sces.groupby('M')['dRde'].max().to_numpy()
    sorter = np.argsort(solved_sces['dRde'].to_numpy())
    max_dNde_df = solved_sces.iloc[sorter[np.searchsorted(solved_sces['dRde'].to_numpy(),
                                                           max_dnde_by_M,
                                                           sorter=sorter)]]
    
    interpolator = BarycentricInterpolator(max_dNde_df['M'],
                                           max_dNde_df[parm_names + solved_quantities])
    
    interpolated_M = np.arange(np.min(max_dNde_df['M']),
                               np.max(max_dNde_df['M']) + 5, 5)
    
    interpolated_matrix = interpolator(interpolated_M)
    smoothed_interps = [make_splrep(interpolated_M, interpolated_y,
                                    k = 2, s = 0.7)
                        for interpolated_y in interpolated_matrix.T]
    
    interpolated_data = pd.DataFrame([np.round(smooth_y(interpolated_M), 7)
                                      for smooth_y in smoothed_interps],
                                     index = parm_names + solved_quantities).T
    
    for col in parm_names + ['phi_N', 'N_mean', 'q_N', 'phi_R', 'R_mean', 'q_R']:
        
        interpolated_data.loc[interpolated_data[col] < 0, col] = 1e-5
     
    interpolated_data['M'] = interpolated_M
    parm_names.append('M')
    
    parm_names.remove(solved_quantity)
    parameters = interpolated_data[parm_names].to_dict('records')
    
    solved_quantities.append(solved_quantity)
    
    bounds = ([1e-10, 1e-10, 1e-10, -1e15, 1e-10, 1e-10, 1e-10, 1e-10,
               quantity_bounds[0]],
              [1, 1e15, 1e15, 1e-10, 1, 1e15, 1e15, 1e15,
               quantity_bounds[1]])
    
    x_inits = interpolated_data[solved_quantities].to_dict('records')
    x_inits = [list(x_init.values()) for x_init in x_inits]
    
    solved_phase = solve_sces_yc_c(parameters, solved_quantities, bounds, x_inits,
                                   'least-squares', 
                                   solver_kwargs = {'xtol' : 1e-15, 'ftol' : 1e-15},
                                   include_multistability = True)
     
    return solved_phase

# phase diagrams

phase_mu_d = Local_Solve_Phase_Boundary(solved_sces_mu_d, 'mu_d', [0, 20])
phase_mu_b = Local_Solve_Phase_Boundary(solved_sces_mu_b, 'mu_b', [0, 20])
phase_sigma_c = Local_Solve_Phase_Boundary(solved_sces_sigma_c, 'sigma_c', [0, 20])
phase_sigma_y = Local_Solve_Phase_Boundary(solved_sces_sigma_y, 'sigma_y', [0, 20])
phase_sigma_d = Local_Solve_Phase_Boundary(solved_sces_sigma_d, 'sigma_d', [0, 20])
phase_sigma_b = Local_Solve_Phase_Boundary(solved_sces_sigma_b, 'sigma_b', [0, 20])


def plot_phases(quantities, y_labels, phases, sces_sets, steps_from_boundary,
                example_Ms, fig_dims):
    
    fig, axs = plt.subplots(fig_dims[0], fig_dims[1],
                            figsize = (6*fig_dims[0], 9*fig_dims[1]),
                            layout = 'constrained')    

    def plot_phase(ax, quantity, y_label, phase, sces, step_from_boundary,
                   example_M):
        
        good_phase = phase.loc[phase['loss'] < -25, :]
          
        smoother = np.poly1d(np.polyfit(good_phase.loc[:, 'M'],
                                        good_phase.loc[:, quantity], 2))
        phase.loc[:, quantity] = smoother(phase.loc[:, 'M'].to_numpy())
        
        example_q = phase.loc[phase['M'] == example_M, quantity].to_numpy()[0]
        
        sces_M = sces.loc[sces['M'] == example_M,
                          [quantity, 'Instability distance']].reset_index(drop=True)
        
        smaller_than = sces_M.loc[np.abs(sces_M[quantity] - \
                                         (example_q - step_from_boundary)).argmin(),
                                  'Instability distance']
        greater_than = sces_M.loc[np.abs(sces_M[quantity] - \
                                         (example_q + step_from_boundary)).argmin(),
                                  'Instability distance']
            
        ######################################
        
        sns.lineplot(phase, x = 'M', y = quantity, ax = ax, color = 'black',
                     linewidth = 5)
        
        if greater_than > smaller_than:
            
            color_thresh = 0
            
            ax.fill_between(phase['M'], phase[quantity], color_thresh,
                            color = "#6600cbff")
        
        elif smaller_than < greater_than:
            
            color_thresh = np.max(phase.loc[:, quantity].to_numpy())
        
            ax.fill_between(phase['M'], phase[quantity], color_thresh,
                            color = "#6600cbff")
        
        ax.set_xlabel('')
        ax.set_ylabel(y_label, fontsize = 16, weight = 'bold',
                      horizontalalignment = 'center', verticalalignment = 'center')
        
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=10)
        
        return ax
    
    for (ax, quantity, y_label, phase, sces, step_from_boundary, example_M) \
        in zip(axs.flatten(), quantities, y_labels, phases, sces_sets,
            steps_from_boundary, example_Ms):
    
        plot_phase(ax, quantity, y_label, phase, sces, step_from_boundary,
                   example_M)
    
    fig.supxlabel('resource pool size, $M$', fontsize = 16, weight = 'bold')
    
    sns.despine()
    plt.show()
    
plot_phases(['sigma_c', 'sigma_y', 'mu_d', 'sigma_d', 'mu_b', 'sigma_b'],
            ['std deviation in total consumption rate, $\sigma_c$',
             'std deviation in yield conversion efficiency, $\sigma_y$',
             'average death rate, $\mu_d$', 
             'average intrinsic resource growth rate, $\mu_b$',
             'std deviation in death rate, $\sigma_d$',
             'std deviation in intrinsic resource growth rate, $\sigma_b$'],
            [copy(phase) for phase in [phase_sigma_c, phase_sigma_y,
                                       phase_mu_d, phase_sigma_d,
                                       phase_mu_b, phase_sigma_b]],
            [copy(sces) for sces in [solved_sces_sigma_c, solved_sces_sigma_y,
                                     solved_sces_mu_d, solved_sces_sigma_d,
                                     solved_sces_mu_b, solved_sces_sigma_b]],
            [0.5, 0.1, 0.5, 0.1, 0.5, 0.1],
            [125, 125, 125, 125, 125, 125],
            (3, 2))
    
#plot_phase('mu_d', "average death rate, $\mu_d$", phase_mu_d, solved_sces_mu_d)

'''
