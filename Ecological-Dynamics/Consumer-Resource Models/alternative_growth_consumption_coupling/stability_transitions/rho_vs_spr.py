# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:31:03 2025

@author: jamil
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling/cavity_solutions_vs_simulations')

sys.path.insert(0, "C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
                    "Ecological-Dynamics/Consumer-Resource Models/alternative_growth_consumption_coupling")
from simulation_functions import CRMs_create_and_save, CRM_df

# %%

def rho_vs_spr(rhos, growth_consumption_rates_args, model_specific_rates_args, M):
    
    full_directory = "C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                        + "rho_vs_spr"
    
    if not os.path.exists(full_directory):
        
        os.makedirs(full_directory)
    
    gc_rates_args_list = [{"rho" : rho} | growth_consumption_rates_args
                           for rho in rhos]
    
    for growth_consumption_rates_args in tqdm(gc_rates_args_list, position = 0,
                                              leave = True,
                                              total = len(gc_rates_args_list)):
    
        CRMs_create_and_save("rho_vs_spr/simulations_" + str(growth_consumption_rates_args['rho']),
                             M, M, growth_consumption_rates_args,
                             model_specific_rates_args,
                             no_communities = 10, t_end = 7000)
    
# %%

def generate_rho_df():
    
    df = CRM_df("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                            + "rho_vs_spr", 
                ['no_species', 'no_resources', 'mu_c', 'sigma_c','mu_g',
                 'sigma_g', 'rho', 'd_val', 'b_val'])
        
    df.rename(columns = {'mu_c' : 'mu_c/M', 'sigma_c' : 'sigma_c/root_M',
                         'mu_g' : 'mu_g/M', 'sigma_g' : 'sigma_g/root_M',
                         'no_resources' : 'M', 'no_species' : 'S'},
                        inplace = True)
    
    df['mu_c'] = df['mu_c/M'] * df['M']
    df['sigma_c'] = df['sigma_c/root_M'] * np.sqrt(df['M'])
    df['mu_g'] = df['mu_g/M'] * df['M']
    df['sigma_g'] = df['sigma_g/root_M'] * np.sqrt(df['M'])
       
    df['Instability distance'] = df['rho']**2 - df['Species packing']
    df['Infeasibility distance'] = df['phi_R'] - df['phi_N']/(df['M']/df['S'])
    
    for var in ['rho', 'mu_c', 'mu_g', 'sigma_c', 'sigma_g', 'mu_c/M',
                'sigma_c/root_M', 'mu_g/M','sigma_g/root_M']:
        
        df[var] = np.round(df[var], 6)
    
    df['M'] = np.int32(df['M'])
    df['S'] = np.int32(df['S'])
    
    return df
        
# %%

mu = 200
sigma = 3.5
M = 150

rho_vs_spr(rhos = np.linspace(0.25, 1, 7),
           growth_consumption_rates_args = dict(method = 'coupled by rho',
                                                mu_c = mu/M,
                                                sigma_c = sigma/np.sqrt(M),
                                                mu_g = mu/M,
                                                sigma_g = sigma/np.sqrt(M)),
           model_specific_rates_args = dict(death_method = 'constant',
                                            death_args =  {'d' : 1},
                                            resource_growth_method = 'constant',
                                            resource_growth_args = {'b' : 1}),
           M = M)

# %%
        
simulation_df = generate_rho_df()    
    
# %%

sns.set_style('ticks')
    
sns.lineplot(simulation_df, x = 'rho', y = 'Species packing', linewidth = 1.5,
             color = 'black', err_style = "bars", errorbar = ("pi", 100))
    
plt.show()

# %%

simulations = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" \
                             + "rho_vs_spr/simulations_0.75.pkl")
    
plt.plot(simulations[0].ODE_sols[0].t, simulations[0].ODE_sols[0].y[:M].T)
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    