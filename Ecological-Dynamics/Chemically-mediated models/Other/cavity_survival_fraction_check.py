# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:04:52 2025

@author: jamil
"""

# %%

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\Chemically-mediated models

########################

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe
import pandas as pd
import seaborn as sns
import sys
from copy import deepcopy
import pickle
import colorsys
from scipy.special import erf

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules')

from models import Consumer_Resource_Model
from community_level_properties import max_le

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

# %%

def consumer_resource_model_dynamics(rho,
                                     mu_c, sigma_c, mu_g, sigma_g,
                                     no_species, no_resources):
    
    no_communities = 25
    no_lineages = 5
    
    def community_dynamics(i, lineages, rho, 
                           mu_c, sigma_c, mu_g, sigma_g,
                           no_species, no_resources):
        
        print({'mu': mu_c, 'rho' : rho, 'Community' : i}, '\n')
       
        community = Consumer_Resource_Model(no_species, no_resources, {'mu_g' : mu_g, 'sigma_g' : sigma_g},
                                            {'mu_c' : mu_c, 'sigma_c' : sigma_c}, rho = rho)
        community.generate_parameters(method = 'correlated')
        community.simulate_community(lineages, 3500, model_version = 'growth_consumption_uncoupled',
                                     assign = True)
        community.calculate_community_properties(lineages, 3000)
        community.lyapunov_exponent = \
            {'lineage ' + str(i): max_le(community, 1000, simulation.y[:,-1],
                                          1e-3, 'growth_consumption_uncoupled', dt = 20, separation = 1e-3)
             for i, simulation in enumerate(community.ODE_sols.values())}
        
        final_abundances = np.concatenate([simulation.y[:,-1] for simulation in community.ODE_sols.values()])
        
        if np.any(np.log(np.abs(final_abundances)) > 6) \
            or np.isnan(np.log(np.abs(final_abundances))).any():
                
                return None
            
        else:
            
            return community 

    messy_communities_list = [deepcopy(community_dynamics(i, np.arange(no_lineages),
                                                 rho, mu_c, sigma_c,
                                                 mu_g, sigma_g, no_species,
                                                 no_resources))
                              for i in range(no_communities)]
    communities_list = list(filter(lambda item: item is not None, messy_communities_list))
    
    return communities_list

# %%
    
def CR_dynamics_df(communities_list, no_species, mu_c, rho):
    
    simulation_data = {'Species Volatility' : [volatility for community in communities_list for volatility in community.species_volatility.values()],
                       'Resource Volatility' : [volatility for community in communities_list for volatility in community.resource_volatility.values()],
                       'Max. lyapunov exponent' : [le for community in communities_list for le in community.lyapunov_exponent.values()],
                       'Species Fluctuation CV' : [fluctuations for community in communities_list for fluctuations in community.species_fluctuations.values()],
                       'Resource Fluctuation CV' : [fluctuations for community in communities_list for fluctuations in community.resource_fluctuations.values()],
                       'Species diversity' : [diversity for community in communities_list for diversity in community.species_survival_fraction.values()],
                       'Resource diversity' : [diversity for community in communities_list for diversity in community.resource_survival_fraction.values()]}
    
    closeness_to_competitive_exclusion = \
        np.array(simulation_data['Species diversity'])/np.array(simulation_data['Resource diversity'])
    
    data_length = len(simulation_data['Resource diversity'])
    annot_no_species = np.repeat(no_species, data_length)
    annot_mu = np.repeat(mu_c, data_length)
    annot_rho = np.repeat(rho, data_length)
    
    data = pd.DataFrame([annot_mu, annot_rho, annot_no_species, simulation_data['Species Volatility'],
                         simulation_data['Resource Volatility'],  simulation_data['Max. lyapunov exponent'],
                         simulation_data['Species Fluctuation CV'], simulation_data['Resource Fluctuation CV'],
                         simulation_data['Species diversity'], simulation_data['Resource diversity'],
                         closeness_to_competitive_exclusion], 
                        index = ['Average consumption rate', 'Correlation', 'Number of species', 
                                 'Volatility (species)', 'Volatility (resources)', 'Max lyapunov exponent',
                                 'Fluctuation CV (species)', 'Fluctuation CV (resources)',
                                 'Diversity (species)', 'Diversity (resources)',
                                 'Closeness to competitive exclusion']).T
    
    return data

#%%

def create_and_delete_CR(filename,
                         rho, mu_c, sigma_c, mu_g, sigma_g,
                         no_species, no_resources):
    
    CR_communities = consumer_resource_model_dynamics(rho, mu_c, sigma_c, mu_g, sigma_g,
                                                      no_species, no_resources)
    
    pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl",
                CR_communities)
    del CR_communities
    
# %%

def create_df_and_delete_simulations(filename, no_species, mu_c, rho):
    
    CR_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
    
    df = CR_dynamics_df(CR_communities, no_species, mu_c, rho)
    
    return df

# %%

def prop_chaotic(x,
                instability_threshold = 0.004):
        
    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

# %%

rhos = [0.4]
rho_string = ['04']

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

no_species = 100
sigma = 0.1
sigma_s = '01'

for rho, rho_s in zip(rhos, rho_string):

    for mu, mu_s in zip(mu_cs, mu_string):
            
        filename_CR = "CR_growth_consumption_test_c_" + str(mu_s) + "_" + str(rho_s)
    
        create_and_delete_CR(filename_CR, 
                             rho, mu, sigma,
                             1, sigma, no_species, no_species)
        
for rho, rho_s in zip(rhos, rho_string):

    for mu, mu_s in zip(mu_cs, mu_string):
            
        filename_CR = "CR_growth_consumption_test_g_" + str(mu_s) + "_" + str(rho_s)
    
        create_and_delete_CR(filename_CR, 
                             rho, 1, sigma,
                             mu, sigma, no_species, no_species)
    
# %%

rhos = [0.4]
rho_string = ['04']

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

no_species = 100

cr_df_c = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_c_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

cr_df_g = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_g_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

# %%

for df in [cr_df_c, cr_df_g]:
    
    print(pd.pivot_table(df,
                         index = 'Correlation',
                         columns = 'Average consumption rate',
                         values = 'Max lyapunov exponent',
                         aggfunc = prop_chaotic))

# %%

def average_abundances(mu, rho, index):
    
    #filename = "CR_growth_consumption_underlying_coupling_" + str(mu) + "_" + str(rho)
    #filename = "CR_growth_consumption_test_c_" + str(mu) + "_" + str(rho)
    filename = "CR_growth_consumption_test_g_" + str(mu) + "_" + str(rho)
    
    communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
    
    avg_abundances = np.mean(np.array([np.mean(community.ODE_sols['lineage 0'].y[index, -1]) for community in communities]))
    
    return avg_abundances

# %%

# erf

mus = np.arange(0.3, 1.5, 0.2)
mu_string = ['03','05','07','09','11','13']
sigma = 0.1
K = 1
m = 1
gamma = 1
rho = 0.4
rho_string = '04'

avg_N = np.array([average_abundances(mu_s, rho_string, np.arange(100)) for mu_s in mu_string])
avg_R = np.array([average_abundances(mu_s, rho_string, np.arange(100, 200)) for mu_s in mu_string])

gs = 100*mus*avg_R - m
kappas = K - 100*mus*(1/gamma)*avg_N
delta_g = gs/np.std(gs)
delta_kappa = kappas/np.std(kappas)

erf_g, erf_kappa = erf(delta_g), erf(delta_kappa)

instability_bound = erf_kappa * ((rho**2) * erf_kappa - erf_g)
print(instability_bound)

plt.plot(mus, instability_bound)
    
# %%

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

rhos = [0.3, 0.4, 0.5, 0.6, 0.7]
rho_string = ['03', '04', '05', '06', '07']


no_species = 100

cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_underlying_coupling_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

