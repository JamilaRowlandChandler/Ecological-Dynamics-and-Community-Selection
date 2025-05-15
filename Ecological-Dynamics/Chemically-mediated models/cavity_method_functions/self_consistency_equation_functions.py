# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:56:36 2025

@author: jamil
"""

# %%

########################

import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

from scipy.special import erfc
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping

import os
os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/cavity_method_functions')

import self_limiting_rho_equations as slr
import externally_supplied_equations as es
import self_limiting_gc_c_equations as slgc
import self_limiting_gc_c_finite_equations as slgcM
import self_limiting_g_cg_equations as slcg

# %% 

def parameter_combinations(parameter_ranges, n):
    
    variable_parameter_vals = np.meshgrid(*[np.linspace(*val_range, n)
                                            for val_range in parameter_ranges])
     
    v_p_v_flattened = np.array([v_p_v.flatten() for v_p_v in variable_parameter_vals])
    
    return v_p_v_flattened

# %%

def variable_fixed_parameters(variable_parameters,  v_names,
                              fixed_parameters):
    
    if isinstance(variable_parameters[0], (list, np.ndarray)) == True:
    
        def variable_dict(v_p, v_p_names, fixed_parameters):
            
            return dict(zip(v_p_names, v_p)) | fixed_parameters 
        
        variable_list = np.apply_along_axis(variable_dict, 0, variable_parameters,
                                              v_p_names = v_names,
                                              fixed_parameters = fixed_parameters)
        
    elif isinstance(variable_parameters[0], dict) == True:
        
        variable_list = [v_p | fixed_parameters for v_p in variable_parameters]
                         
    return variable_list
    

# %%

def solve_sces(parameters, equation_func, solved_quantities, bounds, x_init, solver,
               solver_kwargs):
    
    match equation_func:
        
        case 'self-limiting':
            
            ##module = slr
            
            #mod_string = "slr"
            function = slr.self_consistency_equations
            
        case 'self-limiting gc c':
            
            ##module = slgc
            
            #mod_string = "slgc"
            function = slgc.self_consistency_equations
            
        case 'self-limiting gc c inf':
            
            #module = slgc
            
            #mod_string = "slgc"
            function = slgc.self_consistency_equations
            
        case 'self-limiting gc c M':
            
            #module = slgcM
            
            #mod_string = "slgcM"
            function = slgcM.self_consistency_equations
            
        case 'self-limiting g cg inf':
            
            #module = slcg
            
            #mod_string = "slcg"
            function = slcg.self_consistency_equations
            
        case 'externally supplied':
            
            #module = es
            
            #mod_string = "es"
            function = es.self_consistency_equations
            
    #function = module.self_consistency_equations
            
    if isinstance(x_init[0], list):
        
        if isinstance(solver_kwargs, list):
            
            fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                        bounds, x0, ls_kwargs,
                                                        **s_kwgs)
                                                  for ls_kwargs, x0, s_kwgs 
                                                  in tqdm(zip(parameters, x_init,
                                                              solver_kwargs),
                                                          position = 0,
                                                          leave = True)])
        
        else:
        
            fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                        bounds, x0, ls_kwargs,
                                                        **solver_kwargs)
                                                  for ls_kwargs, x0 in tqdm(zip(parameters,
                                                                                x_init),
                                                                        position = 0,
                                                                        leave = True)])
        
    else:
        
        fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                    bounds, x_init, ls_kwargs,
                                                    **solver_kwargs)
                                              for ls_kwargs in tqdm(parameters,
                                                                    position = 0,
                                                                    leave = True)])
    
    fitted_values_df = pd.DataFrame(fitted_values_final_loss, columns = solved_quantities + ['loss'])
    
    df = pd.concat([pd.DataFrame(parameters), fitted_values_df], axis = 1)
    
    return df

# %%

def solve_sces_2(parameters, equation_func, solved_quantities, bounds, x_init, solver,
                 solver_kwargs, include_multistability = False):
    
    match equation_func:
        
        case 'self-limiting':
            
            module = slr
            
        case 'self-limiting gc c':
            
            module = slgc
            
        case 'self-limiting gc c M':
            
            module = slgcM
            
        case 'self-limiting g cg inf':
            
            module = slcg
            
        case 'externally supplied':
            
            module = es
            
    function = module.self_consistency_equations
    
    if include_multistability == True:
        
        ms_function = module.instability_condition
        
    else:
        
        ms_function = None
            
    if isinstance(x_init[0], list):
        
        if isinstance(solver_kwargs, list):
            
            fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                        bounds, x0, ls_kwargs,
                                                        ms_function,
                                                        **s_kwgs)
                                                  for ls_kwargs, x0, s_kwgs 
                                                  in tqdm(zip(parameters, x_init,
                                                              solver_kwargs),
                                                          position = 0,
                                                          leave = True)])
        
        else:
        
            fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                        bounds, x0, ls_kwargs,
                                                        ms_function,
                                                        **solver_kwargs)
                                                  for ls_kwargs, x0 in tqdm(zip(parameters,
                                                                                x_init),
                                                                        position = 0,
                                                                        leave = True)])
        
    else:
        
        fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                    bounds, x_init, ls_kwargs,
                                                    ms_function,
                                                    **solver_kwargs)
                                              for ls_kwargs in tqdm(parameters,
                                                                    position = 0,
                                                                    leave = True)])
    
    fitted_values_df = pd.DataFrame(fitted_values_final_loss, columns = solved_quantities + ['loss'])
    
    df = pd.concat([pd.DataFrame(parameters), fitted_values_df], axis = 1)
    
    return df


# %%

def phase_boundary(parameters, equation_func, solved_quantities, bounds, x_init,
                   solver, solver_kwargs):
    
    match equation_func:
        
        case 'self-limiting':
        
            function = slr.instability_condition
            
        case 'self-limiting gc c':
            
            function = slgc.instability_condition
            
        case 'self-limiting gc c inf':
            
            function = slgc.instability_condition
            
        case 'self-limiting gc c M':
            
            function = slgcM.instability_condition
            
        case 'self-limiting g cg inf':
            
            function = slcg.instability_condition
            
        case 'externally supplied':
            
            function = es.instability_condition
            
    if isinstance(x_init, list):
        
        fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                    bounds, x0, ls_kwargs,
                                                    **solver_kwargs)
                                              for ls_kwargs, x0 in tqdm(zip(parameters,
                                                                            x_init),
                                                                    position = 0,
                                                                    leave = True)])
        
    else:
        
        fitted_values_final_loss = np.array([solver(function, solved_quantities,
                                                    bounds, x_init, ls_kwargs,
                                                    **solver_kwargs)
                                              for ls_kwargs in tqdm(parameters,
                                                                    position = 0,
                                                                    leave = True)])
    
    fitted_values_df = pd.DataFrame(fitted_values_final_loss, columns = solved_quantities + ['loss'])
    
    df = pd.concat([pd.DataFrame(parameters), fitted_values_df], axis = 1)
    
    return df

# %%

def solve_for_multistability(y, multistability_equation_func):
    
    bounds = ([-1e15, -1e15], [1e15, 1e15])
    x_init = [0, 0]
    
    match multistability_equation_func:
        
        case 'self-limiting':
            
            fun = slr.multistability_equations
            ls_kwarg_names = ['rho', 'gamma', 'sigma_c', 'sigma_g', 'phi_N', 'phi_R', 'v_N', 'chi_R']
            
        case 'self-limiting gc c':
            
            fun = slgc.multistability_equations
            ls_kwarg_names = ['M', 'gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R']
            
        case 'self-limiting gc c inf':
            
            fun = slgc.multistability_equations_inf
            ls_kwarg_names = ['gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R']
            
        case 'self-limiting gc c M':
            
            fun = slgcM.multistability_equations
            ls_kwarg_names = ['M', 'gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R', 'v_N', 'chi_R']
            
        case 'self-limiting g cg inf':
            
            fun = slcg.multistability_equations_inf
            ls_kwarg_names = ['gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R', 'chi_R', 'v_N']
            
        case 'externally supplied':
            
            fun = es.multistability_equations
            ls_kwarg_names = ['rho', 'gamma', 'mu_c', 'sigma_c', 'sigma_g', 'mu_K',
                              'mu_D', 'sigma_D', 'phi_N', 'N_mean', 'q_N', 'v_N',
                              'chi_R']
            
    ls_kwargs = {key : y[key] for key in ls_kwarg_names}
    
    sol = solve_equations_least_squares(fun, ['dNde', 'dRde'],  bounds, x_init,
                                        ls_kwargs, False)
    
    #print('loss = ', sol[-1])
    
    return sol
    
# %%

def solve_equations_least_squares(equation_func, solved_quantities, bounds, x_init,
                                  ls_kwargs, ms_function = None,
                                  return_all = False, **solver_kwargs):
    
    '''

    Parameters
    ----------
    ls_kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fitted_values : TYPE
        DESCRIPTION.

    '''
    
    if ms_function:
        
        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs) \
                         + ms_function(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
    
    else:

        fun = lambda x : equation_func(**{key: val for key, val in 
                                          zip(solved_quantities, x)}, **ls_kwargs)
      
    fitted_values = least_squares(fun, x_init, bounds = bounds,
                                  max_nfev = 10000, **solver_kwargs)
    #                              ftol = 1e-11, xtol = 1e-11, max_nfev = 10000)
    
    
    #print(fitted_values.message)
    
    if return_all is True:
        
        returned_values = fitted_values
        
    else: 
        
        returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values

# %%

def solve_equations_different_evolve(equation_func, solved_quantities, bounds, x_init,
                                     ls_kwargs,
                                     return_all = False):
    
    fun = lambda x : np.sum(equation_func(**{key: val for key, val in zip(solved_quantities, x)},
                                        **ls_kwargs)**2)
      
    fitted_values = differential_evolution(fun, x0 = x_init, bounds = bounds,
                                           atol = 1e-10, mutation = (0.1, 1.9), maxiter = 2000)
#                                           disp = True)

    breakpoint()
    
    if return_all is True:
        
        returned_values = fitted_values
        
    else: 
        
        returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values

# %%

def solve_equations_basinhopping(equation_func, solved_quantities, bounds, x_init,
                                 ls_kwargs, **bh_kwargs):
    
    fun = lambda x : np.sum(equation_func(**{key: val for key, val in zip(solved_quantities, x)},
                                          **ls_kwargs)**2)
      
    fitted_values = basinhopping(fun, x0 = x_init, niter = 100, 
                                 minimizer_kwargs = {"method" : "L-BFGS-B",
                                                     "bounds" : bounds,
                                                     "options" : {"maxiter" : 10000,
                                                                  "ftol" : 1e-12}},
                                 **bh_kwargs)

    returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    #print(fitted_values.message)
    
    return returned_values

# %%

def pickle_dump(filename,data):
    
    '''
    
    Pickle data.

    Parameters
    ----------
    filename : string
        Pickle file name. Should end with .pkl
    data : any
        Data to pickle.

    Returns
    -------
    None.

    '''
    
    with open(filename, 'wb') as fp:
        
        pickle.dump(data, fp)