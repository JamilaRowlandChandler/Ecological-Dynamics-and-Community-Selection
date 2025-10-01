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
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import generic_heatmaps_multi, pickle_dump
    
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
                      n = 10, suffix = "_2"):

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
    
    pickle_dump(directory + "/M_" + solved_quantity + suffix + ".pkl", final_sces)
    
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

# Testing the effect of sigma_y

any_fixed_parameters = dict(mu_c = 160, sigma_c = 1.6, mu_y = 1, sigma_y = 0.130639,
                            mu_d = 1, sigma_d =  0, mu_b = 1, sigma_b = 0, 
                            gamma = 1)


solved_sces_sigma_y = Global_Solve_SCEs('sigma_y', [0.05, 0.25],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'sigma_y'),
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

solved_sces_mu_c = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c//M_vs_mu_c.pkl")
    
solved_sces_mu_y = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + "cavity_solutions/self_limiting_yc_c//M_vs_mu_y_2.pkl")

# %%

def plot_instability_distances():

    #sces_list = [sces.loc[sces['loss'] < -30, :] 
    #             for sces in [solved_sces_mu_c, solved_sces_sigma_c,
    #                          solved_sces_mu_y, solved_sces_sigma_y,
    #                          solved_sces_mu_d, solved_sces_sigma_d,
    #                          solved_sces_mu_b, solved_sces_sigma_b]]
    
    sces_list = [sces.loc[sces['loss'] < -30, :] 
                 for sces in [solved_sces_mu_y,
                              solved_sces_mu_d, solved_sces_sigma_d,
                              solved_sces_mu_b, solved_sces_sigma_b]]
    
    quantities = ['mu_y', 'mu_d', 'sigma_d', 'mu_b', 'sigma_b']
    
    y_labels = ['average yield conversio\nefficiency, $\mu_y$',
                'average death rate, $\mu_d$', 
                'std deviation in death rate,\n$\sigma_d$',
                'average intrinsic resource\ngrowth rate, $\mu_b$',
                'std deviation in intrinsic\nresource growth rate, $\sigma_b$']
    
    
    for sces, quantity in zip(sces_list, quantities): 
        
        sces['Instability distance'] = sces['rho'] - np.sqrt(sces['Species packing'])
        sces[quantity] = np.round(sces[quantity], 7)
    
    #quantities = ['mu_c', 'sigma_c', 'mu_y', 'sigma_y', 'mu_d', 'sigma_d',
    #              'mu_b', 'sigma_b']
    
    
    
    #y_labels = ['average total consumtion rate, $\mu_c$',
    #            'std deviation in total consumption rate, $\sigma_c$',
    #            'average yield conversion efficiency, $\mu_y$',
    #            'std deviation in yield conversion efficiency, $\sigma_y$',
    #            'average death rate, $\mu_d$', 
    #            'std deviation in death rate, $\sigma_d$',
    #            'average intrinsic resource growth rate, $\mu_b$',
    #            'std deviation in intrinsic resource growth rate, $\sigma_b$']
    
    id_max = np.max(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    id_min = np.min(np.concatenate([sces['Instability distance'].to_numpy() 
                                    for sces in sces_list]))
    
    if id_max > np.abs(id_min): id_min = -id_max 
    else: id_max = -id_min
    
    #breakpoint()
    
    fig, axs = generic_heatmaps_multi(sces_list, 'M',
                                      quantities,
                                      "resource pool size, $M$",
                                      y_labels, "Instability distance", "RdBu",
                                      (2, 3), (8.5, 4.4),
                                      specify_min_max = np.tile([id_min, id_max],
                                                                len(sces)).reshape((len(sces), 2)),
                                      cbar_pos = 2)
    
    cbar = axs.flatten()[2].collections[0].colorbar
    cbar.set_label(label = r'$\text{reciprocity} - \sqrt{\text{packing ratio}}$',
                   size = '10')
    cbar.ax.tick_params(labelsize = 8)
    
    fig.delaxes(axs.flatten()[-1])
    
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_other_parms.png",
                bbox_inches='tight', dpi = 400)
    plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_other_parms.svg",
                bbox_inches='tight')
    
    plt.show()
    
plot_instability_distances_()
