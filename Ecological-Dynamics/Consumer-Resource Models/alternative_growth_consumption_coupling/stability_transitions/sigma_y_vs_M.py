# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 23:32:42 2025

@author: jamil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:32:30 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/" + \
             "stability_transitions")
    
sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import CRM_across_parameter_space, generate_simulation_df, \
    le_pivot, generic_heatmaps

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/cavity_method_functions')
import self_consistency_equation_functions as sce

# %%

def M_effect_sigma_y(M_range, sigma_y_range, mu_C, sigma_C, n, fixed_parameters):
    
    parameters = generate_parameters_M_sigma_y(M_range, sigma_y_range, mu_C, sigma_C,
                                               n, fixed_parameters)
    
    CRM_across_parameter_space(parameters, 'finite_effect_sigma_y_final',
                               ['M', 'sigma_y'])
                
# %%

def generate_parameters_M_sigma_y(M_range, sigma_y_range, mu_C, sigma_C, n,
                               fixed_parameters):
    
    M_sigma_y_combinations = np.unique(sce.parameter_combinations([M_range,
                                                                   sigma_y_range],
                                                                  n),
                                    axis = 1)
    
    variable_parameters = np.vstack([M_sigma_y_combinations[0, :]/fixed_parameters['gamma'],
                                     M_sigma_y_combinations,
                                     np.repeat(mu_C, M_sigma_y_combinations.shape[1])/M_sigma_y_combinations[0, :],
                                     np.repeat(sigma_C, M_sigma_y_combinations.shape[1])/np.sqrt(M_sigma_y_combinations[0, :])])

    # array of all parameter combinations
    parameters = sce.variable_fixed_parameters(variable_parameters,
                                               fixed_parameters,
                                               ['S', 'M', 'sigma_y', 'mu_c', 'sigma_c'])
    
    for parms in parameters:
        
        parms['S'] = np.int32(parms['S'])
        parms['M'] = np.int32(parms['M'])
    
    return parameters
    
# %%

resource_pool_sizes = np.arange(50, 275, 25)

# %%

M_effect_sigma_y(resource_pool_sizes, (0.05, 0.25), 160, 1.6, 9,
                 {'mu_y' : 1, 'b' : 1, 'd' : 1, 'gamma' : 1})

# %%

df_sigma_y_M = generate_simulation_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                                      + 'finite_effect_sigma_y_final')

# %%

fig, axs = generic_heatmaps(df_sigma_y_M,
                            'M', 'sigma_y', 
                           'resource pool size, ' + r'$M$',
                           'std. dev. yield conversion, ' + r'$\sigma_y$',
                            ['Max. lyapunov exponent'], 'Purples',
                            '',
                            (1, 1), (5.5, 4),
                            pivot_functions = {'Max. lyapunov exponent' : le_pivot},
                            specify_min_max={'Max. lyapunov exponent' : [0,1]})

axs.set_xticks(np.arange(0.5, len(resource_pool_sizes) + 0.5, 1),
                          labels = resource_pool_sizes, fontsize = 14)

cbar = axs.collections[0].colorbar
cbar.set_label(label = 'Proportion of simulations with max. LE ' + r'$> 0.00$',
               size = '14')
cbar.ax.tick_params(labelsize = 12)

plt.show()