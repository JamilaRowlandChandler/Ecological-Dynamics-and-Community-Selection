# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:49:08 2025

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
from simulation_functions import generic_heatmaps, pickle_dump

from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

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

solved_sces = Global_Solve_SCEs('mu_c', [265, 340],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'mu_c'),
                                        n = 6, suffix = "_extra")

del solved_sces

# %%

rho = 0.15

mu_c_lim = np.sqrt(50 * 1.6**2 * ((1**2/0.130639**2) * (1/(rho**2) - 1) - 1))

# %%

solved_sces = Global_Solve_SCEs('mu_c', [420, 570],
                                        resource_pool_sizes,
                                        dict_without_key(any_fixed_parameters,
                                                         'mu_c'),
                                        n = 6, suffix = "_extra_2")

del solved_sces

# %%

solved_sces_1 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                  + "cavity_solutions/self_limiting_yc_c/" + \
                                      "M_mu_c_extra.pkl")
    
solved_sces_2 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                  + "cavity_solutions/self_limiting_yc_c/" + \
                                      "M_mu_c_extra_2.pkl")
    
solved_sces = pd.concat([solved_sces_1, solved_sces_2])

del solved_sces_1, solved_sces_2

# %%

fig, axs = generic_heatmaps(solved_sces[solved_sces['loss'] < -25],
                            'M', 'mu_c', 
                           'resource pool size, ' + r'$M$',
                           'average total consumption  rate, ' + r'$\mu_c$',
                            ['rho', 'Infeasibility distance'], ['Blues', 'magma'],
                            ['', ''],
                            (1, 2), (8.5, 4)) #,
                            #specify_min_max={'rho' : [0,1], 'Infeasibility distance' : [0,1]})

for ax in axs.flatten():
    
    ax.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 2),
                  labels = resource_pool_sizes[::2], fontsize = 14)

#cbar = axs.flatten()[0].collections[0].colorbar
#cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
#               size = '14')
#cbar.ax.tick_params(labelsize = 12)

plt.show()

# %%

def niche_encroach(n,
                   mu_c = 145, sigma_c = 1.6, mu_y = 1, sigma_y = 0.13):
   
    C = (mu_c/n + (sigma_c/np.sqrt(n))*np.random.randn(n**2)).reshape(n, n)
    G = (mu_y + sigma_y*np.random.randn(n**2)).reshape(n, n) * C
    
    quick_niche = 1 - cdist(G, C, 'cosine')
    
    #breakpoint()
    
    mask = np.zeros(quick_niche.shape, dtype=bool)
    np.fill_diagonal(mask, 1)
    between_niche = np.ma.masked_array(quick_niche, mask).max(axis=0).mean()
    
    #between_niche = np.round(np.mean(np.max(quick_niche[~np.eye(quick_niche.shape[0],
    #                                                     dtype = 'bool')], axis = 0)), 5)
    
    within_niche = np.round(np.mean(np.diag(quick_niche)), 5)
    
    return {'M' : n, 'between niche' : between_niche,
            'within niche' : within_niche}

# %%
    
resource_pool_sizes = np.arange(50, 275, 25)
n_rep = 10

data = pd.DataFrame([niche_encroach(M) for M in resource_pool_sizes for _ in range(100)])
data = data.groupby('M').apply('mean')
data.rename(columns = {'between niche' : 'Distance between niches',
                       'within niche' : 'Own niche similarity (baseline)'},
            inplace = True)

data.reset_index(inplace = True)

# %%

dfl = pd.melt(data[['M',
                    'Distance between niches',
                    'Own niche similarity (baseline)']], ['M'])

sns.set_style('ticks')

fig, ax = plt.subplots(1, 1, figsize = (2.5, 2.5), layout = 'constrained')

sns.lineplot(dfl, x = 'M', y = 'value', hue = 'variable', ax = ax,
             palette = ['black', 'gray'], linewidth = 2.5)

ax.set_xlabel('resource pool size, ' + r'$M$', fontsize = 10, weight = 'bold')
ax.set_ylabel('measure of niche encroachment \n(cosine similarity between\nnearest consumer)',
              fontsize = 10, weight = 'bold')

ax.get_legend().remove()

ax.set_title("Increasing the resource pool size decreases\nniche encroachment",
             fontsize = 11, weight = 'bold')

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_cosine.png",
            bbox_inches='tight', dpi = 400)
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/self_limit_M_cosine.svg",
            bbox_inches='tight')


plt.show()


