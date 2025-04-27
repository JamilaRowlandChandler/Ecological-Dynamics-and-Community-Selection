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

# %% 

def parameter_combinations(parameter_ranges, n):
    
    variable_parameter_vals = np.meshgrid(*[np.linspace(*val_range, n)
                                            for val_range in parameter_ranges])
     
    v_p_v_flattened = np.array([v_p_v.flatten() for v_p_v in variable_parameter_vals])
    
    return v_p_v_flattened

# %%

def variable_fixed_parameters(variable_parameters,  v_names,
                              fixed_parameters):
    
    def variable_dict(v_p, v_p_names, fixed_parameters):
        
        return dict(zip(v_p_names, v_p)) | fixed_parameters 
    
    variable_list = np.apply_along_axis(variable_dict, 0, variable_parameters,
                                          v_p_names = v_names,
                                          fixed_parameters = fixed_parameters)
    
    return variable_list
    

# %%

def boundary(parameters, equation_func, solved_quantities, bounds, x_init, solver,
             infeasibility_condition = False):
            # multistability_condition = False):
    
    '''

    Parameters
    ----------
    variable_parameters : TYPE
        DESCRIPTION.
    fixed_parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
        
    fitted_values_final_loss = np.array([solver(equation_func, solved_quantities,
                                                bounds, x_init, ls_kwargs,
                                                infeasibility_condition)
                                          for ls_kwargs in tqdm(parameters,
                                                                position = 0,
                                                                leave = True)])
    
    fitted_values_df = pd.DataFrame(fitted_values_final_loss, columns = solved_quantities + ['loss'])
    df = pd.concat([pd.DataFrame(parameters.tolist()), fitted_values_df], axis = 1)
    
    return df

# %%

def solve_for_multistability(y, multistability_equation_func):
    
    bounds = ([-1e15, -1e15], [1e15, 1e15])
    x_init = [0.1, 0.1]
    
    match multistability_equation_func:
        
        case 'self-limiting':
            
            fun = multistability_equations_e
            ls_kwarg_names = ['rho', 'gamma', 'sigma_c', 'sigma_g', 'phi_N', 'phi_R', 'v_N', 'chi_R']
            
        case 'self-limiting gc c':
            
            fun = multistability_equations_sl_gc_c
            ls_kwarg_names = ['M', 'gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R']
            
        case 'self-limiting gc c inf':
            
            fun = multistability_equations_sl_gc_c_inf
            ls_kwarg_names = ['gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R']
            
        case 'self-limiting g cg inf':
            
            fun = multistability_equations_sl_g_cg_inf
            ls_kwarg_names = ['gamma', 'sigma_c', 'sigma_g', 'mu_c', 'mu_g',
                              'phi_N', 'phi_R', 'chi_R', 'v_N']
            
        case 'externally supplied':
            
            fun = multistability_equations
            ls_kwarg_names = ['rho', 'gamma', 'mu_c', 'sigma_c', 'sigma_g', 'mu_K',
                              'mu_D', 'sigma_D', 'phi_N', 'N_mean', 'q_N', 'v_N',
                              'chi_R']
            
    ls_kwargs = {key : y[key] for key in ls_kwarg_names}
    
    sol = solve_equations_least_squares(fun, ['dNde', 'dRde'],  bounds, x_init,
                                        ls_kwargs, False)
    
    #print('loss = ', sol[-1])
    
    return sol

# %%

def distance_from_multistability_threshold(y):
    
    eq_kwargs_names = ['mu_c', 'sigma_c', 'sigma_g', 'rho', 'gamma', 'mu_D',
                       'sigma_D', 'mu_K',
                       'chi_R', 'phi_N', 'N_mean', 'q_N']
    
    eq_kwargs = {key : y[key] for key in eq_kwargs_names}
    
    ms_condition = multistability_condition(**eq_kwargs)
    
    return ms_condition
    
# %%

def solve_equations_least_squares(equation_func, solved_quantities, bounds, x_init,
                                  ls_kwargs, infeasibility_condition,
                                  return_all = False):
    
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
    
    
    
    #if infeasibility_condition is True:
        
    #    chi_index = np.where(np.array(solved_quantities) == 'chi_R')[0][0]
        
    #    fun = lambda x : np.append(equation_func(**{key: val for key, val in zip(solved_quantities, x)},
    #                                             **ls_kwargs), 
    #                               infeasibility(x[chi_index]))
    
    #elif infeasibility_condition is False:
    
   #     fun = lambda x : equation_func(**{key: val for key, val in zip(solved_quantities, x)},
   #                                    **ls_kwargs)
   
    fun = lambda x : equation_func(**{key: val for key, val in zip(solved_quantities, x)},
                                        **ls_kwargs)
      
    fitted_values = least_squares(fun, x_init, bounds = bounds,
                                  ftol = 1e-11, xtol = 1e-11, max_nfev = 10000)
    
    #print(str(i) + '/' + str(n))
    
    if return_all is True:
        
        returned_values = fitted_values
        
    else: 
        
        returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values

# %%

def solve_equations_different_evolve(equation_func, solved_quantities, bounds, x_init,
                                     ls_kwargs, infeasibility_condition,
                                     i, n,
                                     return_all = False):
    
    fun = lambda x : np.sum(equation_func(**{key: val for key, val in zip(solved_quantities, x)},
                                        **ls_kwargs)**2)
      
    fitted_values = differential_evolution(fun, x0 = x_init, bounds = bounds,
                                           tol = 1e-11, atol = 1e-10, mutation = (0.1, 1.9), maxiter = 2000)
#                                           disp = True)

    print(str(i) + '/' + str(n))
    
    if return_all is True:
        
        returned_values = fitted_values
        
    else: 
        
        returned_values = np.append(fitted_values.x, np.log10(np.sum(fitted_values.fun**2)))
    
    return returned_values

# %%

def self_consistency_equations_e(rho, gamma, mu_c, mu_g, sigma_c, sigma_g, mu_m, sigma_m, mu_K, sigma_K,
                                 phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, chi_R):
    
    # average species growth rate
    kappa = (mu_g * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_K - (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt((sigma_g**2 * q_R) + sigma_m**2)
    
    # std. in resource growth rate
    sigma_omega = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_K**2)
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # delta omega
    delta_omega = omega/sigma_omega
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # resource gaussian error function (survival fraction)
    erf_do = 0.5*erfc(-delta_omega/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    # resource exponential tern
    exp_do = np.exp(-(delta_omega**2)/2)/np.sqrt(2 * np.pi)
    
    A = sigma_g * sigma_c * rho * chi_R
    B = 1 - (sigma_g * sigma_c * rho * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_phi_R = erf_do
    eq_R_mean = (sigma_omega/B) * (exp_do + delta_omega*erf_do)
    eq_q_R = (sigma_omega/B)**2 * (delta_omega*exp_do + (1 + delta_omega**2)*erf_do)
    eq_chi_R = phi_R/B
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         phi_R - eq_phi_R,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def infeasibility(chi_R):
        
    return chi_R - 0

# %%

def multistability_equations_e(dNde, dRde, 
                               rho, gamma, sigma_c, sigma_g,
                               phi_N, phi_R, v_N, chi_R):
          
    eq_dNde = (phi_R/(sigma_c * rho * chi_R)**2) * (dRde + 1)
    
    eq_dRde = ((sigma_c**2 * phi_N)/(1 - (rho*sigma_c*sigma_g*v_N)/gamma)**2) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def self_consistency_equations(rho, gamma, mu_c, mu_g, sigma_c, sigma_g, mu_m,
                               sigma_m, mu_K, sigma_K, mu_D, sigma_D,
                               phi_N, N_mean, q_N, v_N, R_mean, q_R, chi_R):
    
    #breakpoint()
    
    # average species growth rate
    kappa = (mu_g * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_D + (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt((sigma_g**2 * q_R) + sigma_m**2)
    
    # std. in resource growth rate
    sigma_omega = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_D**2)
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    A = sigma_g * sigma_c * rho * chi_R
    B = (sigma_g * sigma_c * rho * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_R_mean = (1/(2*B)) * (omega - np.sqrt(omega**2 - 4*B*mu_K) - \
                             (sigma_omega**2/(2 * np.sqrt(omega**2 - 4*B*mu_K))))
        
    eq_q_R = (1/(2*B))**2 * (2*omega**2 + 3*sigma_omega**2 - 4*B*mu_K - \
                             2*omega*np.sqrt(omega**2 - 4*B*mu_K) - \
                             (6*omega*sigma_omega**2)/np.sqrt(omega**2 - 4*B*mu_K) + \
                             (4*(omega*sigma_omega)**2 + 3*sigma_omega**4 + 16*(B*sigma_K)**2)/(omega**2 - 4*B*mu_K))
    #eq_q_R = R_mean**2 + (sigma_omega**4 + \
    #                      (8*(B*sigma_K)**2)/(8*B**2*(omega**2 - 4*B*mu_K)) + \
    #                      (sigma_omega**2*(np.sqrt(omega**2 - 4*B*mu_K) - omega**2))/(4*B**2*(omega**2 - 4*B*mu_K)))
    
    eq_chi_R = -(1/(2*B)) * (1 - omega/np.sqrt(omega**2 - 4*B*mu_K) + \
                            (omega*sigma_omega**2)/(2*(omega**2 - 4*B*mu_K)**(3/2)))
    #eq_chi_R = -(1/(2*B)) * (1 - omega/np.sqrt(omega**2 - 4*B*mu_K) + \
    #                        (3*omega*sigma_omega**2)/(2*(omega**2 - 4*B*mu_K)**(3/2)))
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations(dNde, dRde, 
                             rho, gamma, mu_c, sigma_c, sigma_g, mu_K, mu_D, sigma_D,
                             phi_N, N_mean, q_N, v_N, chi_R):
    
    #breakpoint()
     
    omega = mu_D + (mu_c * N_mean)/gamma
    B = (sigma_g * sigma_c * rho * v_N)/gamma
    x = (phi_N * sigma_c**2)/gamma
          
    eq_dNde = (dRde + 1) * (1/(sigma_c * rho * chi_R)**2) 
    
    eq_dRde = (dNde + 1) * (1/(2*B))**2 * (x*(1 - \
                                              (4*omega)/(np.sqrt(omega**2 - 4*B*mu_K)) + \
                                              4*(omega**2 + (sigma_c*sigma_D)**2)/(omega**2 - 4*B*mu_K)) + \
                                           (x**2)*(q_N/(omega**2 - 4*B*mu_K)))
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def multistability_condition(mu_c, sigma_c, sigma_g, rho, gamma, mu_D, sigma_D, 
                             mu_K,
                             chi_R, phi_N, N_mean, q_N):
    
    omega = mu_D + (mu_c * N_mean)/gamma
    
    x = phi_N*(sigma_c**2)/gamma
    
    A = (sigma_c * rho * chi_R)**2
    
    frac_1B = ((chi_R**2)/(4/gamma)**2) * (x*(1 - (omega/np.sqrt(omega**2 + (4*mu_K)/(gamma*chi_R))) + \
                                              4*(omega**2 + (sigma_c*sigma_D)**2)/(omega**2 + (4*mu_K)/(gamma*chi_R))) + \
                                           (x**2)*(q_N/(omega**2 + (4*mu_K)/(gamma*chi_R))))
    B = 1/frac_1B
    
    return A*B - 1

# %%

def self_consistency_equations_sl_gc_c(M, gamma, mu_c, mu_g, sigma_c, sigma_g,
                                       mu_m, sigma_m, mu_K, sigma_K,
                                       phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, chi_R):
    
    # average species growth rate
    kappa = (mu_g * mu_c * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_K - (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt(q_R*(((mu_c*sigma_g)**2)/M + (mu_g*sigma_c)**2 + \
                               (sigma_c*sigma_g)**2) + sigma_m**2)
    
    # std. in resource growth rate
    sigma_omega = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_K**2)
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # delta omega
    delta_omega = omega/sigma_omega
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # resource gaussian error function (survival fraction)
    erf_do = 0.5*erfc(-delta_omega/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    # resource exponential tern
    exp_do = np.exp(-(delta_omega**2)/2)/np.sqrt(2 * np.pi)
    
    A = mu_g * sigma_c**2 * chi_R
    B = 1 - (mu_g * sigma_c**2 * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_phi_R = erf_do
    eq_R_mean = (sigma_omega/B) * (exp_do + delta_omega*erf_do)
    eq_q_R = (sigma_omega/B)**2 * (delta_omega*exp_do + (1 + delta_omega**2)*erf_do)
    eq_chi_R = phi_R/B
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         phi_R - eq_phi_R,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations_sl_gc_c(dNde, dRde, 
                                     M, gamma, mu_c, mu_g, sigma_c, sigma_g,
                                     phi_N, phi_R):
    
    N_n = phi_R * (((mu_c*sigma_g)**2)/M + (mu_g*sigma_c)**2 + (sigma_c*sigma_g)**2)
    N_d = (mu_g * sigma_c**2 *(phi_N/gamma - phi_R))**2
          
    eq_dNde = (N_n/N_d) * (dRde + 1)
    
    R_n = sigma_c**2 * phi_N/gamma
    R_d = (1 - phi_N/(gamma*(phi_N/gamma - phi_R)))**2
    
    eq_dRde = (R_n/R_d) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def self_consistency_equations_sl_gc_c_inf(gamma, mu_c, mu_g, sigma_c, sigma_g,
                                           mu_m, sigma_m, mu_K, sigma_K,
                                           phi_N, N_mean, q_N, v_N, phi_R, R_mean,
                                           q_R, chi_R):
    
    # average species growth rate
    kappa = (mu_g * mu_c * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_K - (mu_c * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt(q_R*((mu_g*sigma_c)**2 + (sigma_c*sigma_g)**2) + sigma_m**2)
    
    # std. in resource growth rate
    sigma_omega = np.sqrt((sigma_c**2 * q_N)/gamma + sigma_K**2)
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # delta omega
    delta_omega = omega/sigma_omega
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # resource gaussian error function (survival fraction)
    erf_do = 0.5*erfc(-delta_omega/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    # resource exponential tern
    exp_do = np.exp(-(delta_omega**2)/2)/np.sqrt(2 * np.pi)
    
    A = mu_g * sigma_c**2 * chi_R
    B = 1 - (mu_g * sigma_c**2 * v_N)/gamma
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_phi_R = erf_do
    eq_R_mean = (sigma_omega/B) * (exp_do + delta_omega*erf_do)
    eq_q_R = (sigma_omega/B)**2 * (delta_omega*exp_do + (1 + delta_omega**2)*erf_do)
    eq_chi_R = phi_R/B
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         phi_R - eq_phi_R,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations_sl_gc_c_inf(dNde, dRde, 
                                         gamma, mu_c, mu_g, sigma_c, sigma_g,
                                         phi_N, phi_R):
    
    N_n = phi_R * ((mu_g*sigma_c)**2 + (sigma_c*sigma_g)**2)
    N_d = (mu_g * sigma_c**2 *(phi_N/gamma - phi_R))**2
          
    eq_dNde = (N_n/N_d) * (dRde + 1)
    
    R_n = sigma_c**2 * phi_N/gamma
    R_d = (1 - phi_N/(gamma*(phi_N/gamma - phi_R)))**2
    
    eq_dRde = (R_n/R_d) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

# %%

def self_consistency_equations_sl_g_cg(gamma, mu_c, mu_g, sigma_c, sigma_g,
                                       mu_m, sigma_m, mu_K, sigma_K,
                                       phi_N, N_mean, q_N, v_N, phi_R, R_mean, q_R, chi_R):
    
    # average species growth rate
    kappa = (mu_g * R_mean) - mu_m
    
    # average resource growth rate
    omega = mu_K - (mu_c * mu_g * N_mean)/gamma
    
    # std. in species growth rate
    sigma_kappa = np.sqrt(sigma_m**2 + (q_R * sigma_g**2))
     
    # std. in resource growth rate
    sigma_omega = np.sqrt(sigma_K**2 + (q_N/gamma)*((mu_c*sigma_g)**2 + (sigma_c*sigma_g)**2))
    
    # delta kappa
    delta_kappa = kappa/sigma_kappa
    
    # delta omega
    delta_omega = omega/sigma_omega
    
    # species gaussian error function (survival fraction)
    erf_dk = 0.5*erfc(-delta_kappa/np.sqrt(2))
    
    # resource gaussian error function (survival fraction)
    erf_do = 0.5*erfc(-delta_omega/np.sqrt(2))
    
    # species exponential term
    exp_dk = np.exp(-(delta_kappa**2)/2)/np.sqrt(2 * np.pi)
    
    # resource exponential tern
    exp_do = np.exp(-(delta_omega**2)/2)/np.sqrt(2 * np.pi)
    
    A = mu_c * sigma_g**2 * chi_R
    B = 1 - ((mu_c * sigma_g**2 * v_N)/gamma)
    
    ##### Species self consistency equations ####
    eq_phi_N = erf_dk
    eq_N_mean = (sigma_kappa/A) * (exp_dk + delta_kappa*erf_dk)
    eq_q_N = (sigma_kappa/A)**2 * (delta_kappa*exp_dk + (1 + delta_kappa**2)*erf_dk)
    eq_v_N = -phi_N/A
    
    ##### Resource self consistency equations ####
    eq_phi_R = erf_do
    eq_R_mean = (sigma_omega/B) * (exp_do + delta_omega*erf_do)
    eq_q_R = (sigma_omega/B)**2 * (delta_omega*exp_do + (1 + delta_omega**2)*erf_do)
    eq_chi_R = phi_R/B
    
    f_to_min = np.array([phi_N - eq_phi_N,
                         N_mean - eq_N_mean,
                         q_N - eq_q_N,
                         v_N - eq_v_N,
                         phi_R - eq_phi_R,
                         R_mean - eq_R_mean,
                         q_R - eq_q_R,
                         chi_R - eq_chi_R])
    
    return f_to_min

# %%

def multistability_equations_sl_g_cg_inf(dNde, dRde, 
                                         gamma, mu_c, mu_g, sigma_c, sigma_g,
                                         phi_N, phi_R, chi_R, v_N):
    
    N_n = phi_R * sigma_g**2
    N_d = (mu_g * sigma_c**2 * chi_R)**2
          
    eq_dNde = (N_n/N_d) * (dRde + 1)
    
    R_n = (phi_N/gamma) * ((mu_c * sigma_g)**2 + (sigma_c * sigma_g)**2)
    R_d = (1 - (mu_c * sigma_g**2 * v_N)/gamma)**2
    
    eq_dRde = (R_n/R_d) * (dNde + 1)
    
    f_to_min = np.array([dNde - eq_dNde, dRde - eq_dRde])
    
    return f_to_min

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