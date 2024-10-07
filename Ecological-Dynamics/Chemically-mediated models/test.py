# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:17:08 2024

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

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules')

from models import Consumer_Resource_Model
from community_level_properties import max_le

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules')

from model_classes import gLV

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

def consumer_resource_model_dynamics(model_version,
                                     mu_c, sigma_c, mu_g, sigma_g,
                                     no_species, no_resources):
    
    no_communities = 25
    no_lineages = 5
    
    def community_dynamics(i, lineages, model_version,
                           mu_c, sigma_c, mu_g, sigma_g,
                           no_species, no_resources):
        
        print({'mu': mu_c, 'sigma' : sigma_c, 'Community' : i}, '\n')
       
        community = Consumer_Resource_Model(no_species, no_resources, {'mu_g' : mu_g, 'sigma_g' : sigma_g},
                                            {'mu_c' : mu_c, 'sigma_c' : sigma_c})
        community.generate_parameters()
        community.simulate_community(lineages, 3500, model_version = model_version,
                                                    assign = True)
        community.calculate_community_properties(lineages, 3000)
        community.lyapunov_exponent = \
            {'lineage ' + str(i): max_le(community, 1000, simulation.y[:,-1],
                                          1e-3, model_version, dt = 20, separation = 1e-3)
             for i, simulation in enumerate(community.ODE_sols.values())}
        
        final_abundances = np.concatenate([simulation.y[:,-1] for simulation in community.ODE_sols.values()])
        
        if np.any(np.log(np.abs(final_abundances)) > 6) \
            or np.isnan(np.log(np.abs(final_abundances))).any():
                
                return None
            
        else:
            
            return community 

    messy_communities_list = [deepcopy(community_dynamics(i, np.arange(no_lineages),
                                                 model_version, mu_c, sigma_c,
                                                 mu_g, sigma_g, no_species,
                                                 no_resources))
                              for i in range(no_communities)]
    communities_list = list(filter(lambda item: item is not None, messy_communities_list))
    
    return communities_list

# %%
    
def CR_dynamics_df(communities_list, no_species, mu_c, sigma_c):
    
    simulation_data = {'Model objects' : communities_list,
                       'Species Volatility' : [volatility for community in communities_list for volatility in community.species_volatility.values()],
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
    annot_sigma = np.repeat(sigma_c, data_length)
    
    data = pd.DataFrame([annot_mu, annot_sigma, annot_no_species, simulation_data['Species Volatility'],
                         simulation_data['Resource Volatility'],  simulation_data['Max. lyapunov exponent'],
                         simulation_data['Species Fluctuation CV'], simulation_data['Resource Fluctuation CV'],
                         simulation_data['Species diversity'], simulation_data['Resource diversity'],
                         closeness_to_competitive_exclusion], 
                        index = ['Average consumption rate', 'Consumption rate std', 'Number of species', 
                                 'Volatility (species)', 'Volatility (resources)', 'Max lyapunov exponent',
                                 'Fluctuation CV (species)', 'Fluctuation CV (resources)',
                                 'Diversity (species)', 'Diversity (resources)',
                                 'Closeness to competitive exclusion']).T
    
    return data

# %%
    
def gLV_dynamics_df(communities_list, no_species, mu_a, sigma_a):
    
    simulation_data = {'Model objects' : communities_list,
                       'Species Volatility' : [volatility for community in communities_list for volatility in community.reinvadability.values()],
                       'Max. lyapunov exponent' : [le for community in communities_list for le in community.lyapunov_exponent.values()],
                       'Species Fluctuation CV' : [fluctuations for community in communities_list for fluctuations in community.fluctuation_coefficient.values()],
                       'Species diversity' : [diversity/no_species  for community in communities_list for diversity in community.final_diversity.values()]}
                        
    data_length = len(simulation_data['Species diversity'])
    annot_no_species = np.repeat(no_species, data_length)
    annot_mu = np.repeat(mu_a, data_length)
    annot_sigma = np.repeat(sigma_a, data_length)
    
    data = pd.DataFrame([annot_mu, annot_sigma, annot_no_species, simulation_data['Species Volatility'],
                         simulation_data['Max. lyapunov exponent'], simulation_data['Species Fluctuation CV'],
                         simulation_data['Species diversity']],
                        index = ['Average interaction strength', 'Interaction strength std', 'Number of species', 
                                 'Volatility (species)', 'Max lyapunov exponent',
                                 'Fluctuation CV (species)', 'Diversity (species)']).T
    
    return data

#%%

def create_and_delete_CR(filename,
                         model_version, mu_c, sigma_c, mu_g, sigma_g,
                         no_species, no_resources):
    
    CR_communities = consumer_resource_model_dynamics(model_version, mu_c, sigma_c, mu_g, sigma_g,
                                                      no_species, no_resources)
    
    pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl",
                CR_communities)
    del CR_communities
    
# %%

def create_df_and_delete_simulations(filename, no_species, mu_c, sigma_c):
    
    CR_communities = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
    
    df = CR_dynamics_df(CR_communities, no_species, mu_c, sigma_c)
    
    return df

# %%

def cmap_norm(base, data_len = 256):
    
    colourmap_base = mpl.colors.ColorConverter.to_rgb(base)
    
    def scale_lightness(rgb, scale_l):
        
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
    
    colours_list = [scale_lightness(colourmap_base, scale) 
                    for scale in np.linspace(4.5, 0, data_len)]
    
    cmap = mpl.colors.ListedColormap(colours_list)
    norm = mpl.colors.PowerNorm(0.25, vmin = 0, vmax = 1)
    #norm = mpl.colors.PowerNorm(1.3, vmin = -0.03, vmax = 0.02)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    return {'cmap' : cmap, 'sm' : sm}
        
# %%

# C-R model where growth does not scale with consumption, but does have the same variance

mu_string = ['01','03','05','07','09','11','13']
mu_cs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_growth_consumption_uncoupled_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'growth_consumption_uncoupled', mu_c, sigma_c,
                             mu_g = 1, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)
        
# %%

# with more species
        
mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
#sigma_string = ['002','004','006','008','01','015','02']
#sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
sigma_string = ['015','02']
sigma_cs = [0.15, 0.2]

no_species = 250

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_growth_consumption_uncoupled_more_species_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'growth_consumption_uncoupled', mu_c, sigma_c,
                             mu_g = 1, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)
        
        
# %%

# Uncoupled growth and consumption, but growth is sampled from same distribution as consumption

mu_string = ['01','03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02,0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_growth_consumption_uncoupled_scaled_copy_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'growth_consumption_uncoupled', mu_c, sigma_c,
                             mu_g = mu_c, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)
        
# %%

# more species

mu_string = ['01','03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
#sigma_string = ['002','004','006','008','01','015','02']
#sigma_cs = [0.02,0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
sigma_string = ['002','004']
sigma_cs = [0.02, 0.04]

no_species = 250

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        
        filename_CR = "CR_growth_consumption_uncoupled_scaled_more_species_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'growth_consumption_uncoupled', mu_c, sigma_c,
                             mu_g = mu_c, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)
        
# %%

# C-R model where consumption scales with growth 

mu_string = ['01','03','05','07','09','11','13']
mu_cs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_consumption_coupled_to_growth_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'consumption_funtion_of_growth', mu_c, sigma_c,
                             mu_g = 1, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)

# %%

# C-R model where growth scales with consumption 

mu_string = ['11','13','15','17','19','21']
mu_cs = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
sigma_string = ['01','015','02','025','03']
sigma_cs = [0.1, 0.15, 0.2, 0.25, 0.3]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_growth_coupled_to_consumption_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, 
                             'growth_function_of_consumption', mu_c, sigma_c,
                             mu_g = 1, sigma_g = sigma_c,
                             no_species = no_species, no_resources = no_species)
        
#%%

######################################### Dataframes #########################################

# Uncoupled growth and consumption

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

uncoupled_cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_uncoupled_" + str(mu_s) + "_" + str(sigma_s), no_species, mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

uncoupled_cr_df['Model'] = np.repeat('CR', uncoupled_cr_df.shape[0])
uncoupled_cr_df['Annotation'] = np.repeat('uncoupled', uncoupled_cr_df.shape[0])

# %%

# more species

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
#sigma_string = ['002','004','006','008','01','015','02']
#sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
sigma_string = ['002','004','006']
sigma_cs = [0.02, 0.04, 0.06]

no_species = 250

uncoupled_cr_df_2 = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_uncoupled_more_species_" + str(mu_s) + "_" + str(sigma_s), no_species, mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

uncoupled_cr_df_2['Model'] = np.repeat('CR', uncoupled_cr_df_2.shape[0])
uncoupled_cr_df_2['Annotation'] = np.repeat('uncoupled', uncoupled_cr_df_2.shape[0])

# %%

# Uncoupled but scaled growth and consumption

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

uncoupled_scaled_cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_uncoupled_scaled_copy_" + str(mu_s) + "_" + str(sigma_s), no_species, mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

uncoupled_scaled_cr_df['Model'] = np.repeat('CR', uncoupled_scaled_cr_df.shape[0])
uncoupled_scaled_cr_df['Annotation'] = np.repeat('uncoupled', uncoupled_scaled_cr_df.shape[0])


# %%

# consumption coupled to growth

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_string = ['002','004','006','008','01','015','02']
sigma_cs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]

no_species = 50

coupled_c_cr_df = pd.concat([create_df_and_delete_simulations("CR_consumption_coupled_to_growth_" + str(mu_s) + "_" + str(sigma_s), no_species, mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

coupled_c_cr_df['Model'] = np.repeat('CR', coupled_c_cr_df.shape[0])
coupled_c_cr_df['Annotation'] = np.repeat('coupled c', coupled_c_cr_df.shape[0])

# %%

# growth coupled to consumption

mu_string = ['11','13','15','17','19','21']
mu_cs = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
sigma_string = ['01','015','02','025','03']
sigma_cs = [0.1, 0.15, 0.2, 0.25, 0.3]

no_species = 50

coupled_g_cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_coupled_to_consumption_" + str(mu_s) + "_" + str(sigma_s), no_species, mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

coupled_g_cr_df['Model'] = np.repeat('CR', coupled_g_cr_df.shape[0])
coupled_g_cr_df['Annotation'] = np.repeat('coupled g', coupled_g_cr_df.shape[0])

# %%

##################### gLV

gLV_communities_fixed_growth = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_3.pkl")

mu_as = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
sigma_as = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
no_species = 50

data_gLV_mu_sigma_unscaled = pd.concat([gLV_dynamics_df(simulation_data, no_species, mu, sigma)
                                        for simulation_data, mu, sigma in 
                                        zip(gLV_communities_fixed_growth, np.tile(mu_as, len(sigma_as)),
                                            np.repeat(sigma_as, len(mu_as)))])
    
data_gLV_mu_sigma_unscaled['Model'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])
data_gLV_mu_sigma_unscaled['Annotation'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])

# %%

################################ Phase diagram with max. lyapunov exponents ################

def prop_chaotic(x, instability_threshold = 0.002):
        
    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

# %%

prop_le_cr = pd.pivot_table(uncoupled_cr_df[uncoupled_cr_df['Average consumption rate'] > 0.1],
                                        index = 'Consumption rate std',
                                        columns = 'Average consumption rate',
                                        values = 'Max lyapunov exponent',
                                        aggfunc = prop_chaotic)
#                                        aggfunc = 'mean')

prop_le_cr_c = pd.pivot_table(coupled_c_cr_df[coupled_c_cr_df['Average consumption rate'] > 0.1],
                                          index = 'Consumption rate std',
                                          columns = 'Average consumption rate',
                                          values = 'Max lyapunov exponent',
                                          aggfunc = prop_chaotic)
#                                        aggfunc = 'mean')

prop_le_cr_g = pd.pivot_table(coupled_g_cr_df[coupled_g_cr_df['Average consumption rate'] > 0.1],
                                          index = 'Consumption rate std',
                                          columns = 'Average consumption rate',
                                          values = 'Max lyapunov exponent',
                                          aggfunc = prop_chaotic)
#                                        aggfunc = 'mean')

prop_le_gLV = pd.pivot_table(data_gLV_mu_sigma_unscaled,
                                         index = 'Interaction strength std',
                                         columns = 'Average interaction strength',
                                         values = 'Max lyapunov exponent',
                                         aggfunc = prop_chaotic)
#                                        aggfunc = 'mean')

top_cr = prop_le_cr.iloc[-2:]
top_cr_c = prop_le_cr_c.iloc[-2:]
top_gLV = prop_le_gLV.iloc[-2:]

bottom_cr = prop_le_cr.iloc[:-2]
bottom_cr_c = prop_le_cr_c.iloc[:-2]
bottom_gLV = prop_le_gLV.iloc[:-2]

data_list = [top_gLV, top_cr, top_cr_c,
             bottom_gLV, bottom_cr, bottom_cr_c,
             prop_le_cr_g]

#%%

def cmap_norm_2(base, data_len = 256):
    
    colourmap_base = mpl.colors.ColorConverter.to_rgb(base)
    
    def scale_lightness(rgb, scale_l):
        
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
    
    colours_list = [scale_lightness(colourmap_base, scale) 
                    for scale in np.linspace(2.95, 0, data_len)]
    #            for scale in np.linspace(6, 0, data_len)]
    
    cmap = mpl.colors.ListedColormap(colours_list)
    norm = mpl.colors.PowerNorm(1, vmin = 0, vmax = 1)
    #norm = mpl.colors.PowerNorm(1, vmin = -0.03, vmax = 0.02)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    return {'cmap' : cmap, 'sm' : sm}

#%%

colourmap = cmap_norm_2(mpl.colormaps['viridis_r'](0.85))
    
mosaic = [['top1', 'top2', 'top3', '.'],['bottom1', 'bottom2', 'bottom3', 'bottom4']]

sns.set_style('white')
    
fig, axs = plt.subplot_mosaic(mosaic, layout = 'constrained', figsize = (14.5,4),
                              gridspec_kw = {'width_ratios' : [1, 1, 1, 1], 'wspace' : 0.1,
                                             'hspace' : 0.0, 'height_ratios' : [2, 5]})

for i, (data, ax) in enumerate(zip(data_list, axs.values())):
    
    subfig = sns.heatmap(data, ax=ax, vmin=0, vmax=1, cbar=False, cmap = colourmap['cmap'],
                         norm = colourmap['sm'].norm, cbar_kws={"ticks":[0,0.1,1]})
    
    subfig.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if i == len(data_list) - 1:
        
        mappable = subfig.get_children()[0]
        
    #######
    
    if i < 3:
        
        subfig.axhline(2, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(6, 0, 1, color = 'black', linewidth = 2)
        
    elif i == len(data_list) - 1:
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(6, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(5, 0, 1, color = 'black', linewidth = 2)
        
    else: 
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axvline(6, 0, 1, color = 'black', linewidth = 2)
        
cbar = plt.colorbar(mappable, ax = [axs['top3'], axs['bottom4']],
                    orientation = 'vertical')
cbar.set_label(label=r'$P( \lambda > 0.002)$',weight='bold', size='20')
cbar.ax.tick_params(labelsize=14)

axs['bottom1'].set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')
axs['bottom2'].set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')
axs['bottom3'].set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')
axs['bottom4'].set_xticklabels([r'$1.1$', r'$1.3$', r'$1.5$', r'$1.7$', r'$1.9$', r'$2.1$'],
                               size = '14')


axs['bottom1'].set_yticklabels([r'$0.02^2$', r'$0.04^2$', r'$0.06^2$', r'$0.08^2$', r'$0.1^2$'],
                               size = '14')
axs['top1'].set_yticklabels([r'$0.15^2$', r'$0.2^2$'],
                               size = '14')
axs['bottom1'].yaxis.set_tick_params(rotation=360)
axs['top1'].yaxis.set_tick_params(rotation=360)

axs['bottom4'].set_yticklabels([r'$0.1^2$', r'$0.15^2$', r'$0.2^2$', r'$0.25^2$', r'$0.3^2$'],
                               size = '14')
axs['bottom4'].yaxis.set_tick_params(rotation=360)

###

fig.text(0.57, 1.05, 'Consumer - Resource model', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=20, weight = 'bold',
                color = '#43bf6bff',
                path_effects= [pe.withStroke(linewidth=1, foreground="black")])
fig.text(0.13, 1.05, 'gLV', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=20, weight = 'bold',
                color = '#ab00d5ff',
                path_effects= [pe.withStroke(linewidth=1, foreground="black")])
fig.text(-0.01, 0.5, 'Variance ' + r'($\sigma_a^2$ or $\sigma_{\alpha}^2$)', verticalalignment = 'center',
                horizontalalignment = 'center', fontsize=20, weight = 'bold',
                rotation = 90, rotation_mode = 'anchor')
fig.text(0.57, -0.04, 'Average consumption rate ' + r'$(\mu_c)$', verticalalignment = 'center',
                horizontalalignment = 'center', fontsize=20, weight = 'bold')
fig.text(0.13, -0.04, 'Average interaction strength ' + r'$(\mu_{\alpha})$', verticalalignment = 'center',
                horizontalalignment = 'center', fontsize=20, weight = 'bold')

axs['top2'].set_title('growth and consumption\nare uncoupled', fontsize = 20, color = '#88d7a2ff',
                        path_effects= [pe.withStroke(linewidth=1, foreground="black")])
axs['top3'].set_title('consumption is\ncoupled to growth', fontsize = 20, color = '#2a7c44ff',
                      path_effects= [pe.withStroke(linewidth=1, foreground="black")])
fig.text(0.82, 0.92, 'growth is coupled\nto consumption', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=20,
                color = '#2a7c44ff', path_effects= [pe.withStroke(linewidth=1, foreground="black")])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_le.svg",
            bbox_inches='tight')

#%%

prop_le_s_cr = pd.pivot_table(uncoupled_scaled_cr_df[uncoupled_scaled_cr_df['Average consumption rate'] > 0.1],
                              index = 'Consumption rate std',
                              columns = 'Average consumption rate',
                              values = 'Max lyapunov exponent',
                              aggfunc = prop_chaotic)

top_s_cr = prop_le_s_cr.iloc[-2:]
bottom_s_cr = prop_le_s_cr.iloc[:-2]

######

colourmap = cmap_norm_2(mpl.colormaps['viridis_r'](0.85))
    
mosaic = [['top'],['bottom']]

sns.set_style('white')
    
fig, axs = plt.subplot_mosaic(mosaic, layout = 'constrained', figsize = (3.625,4),
                              gridspec_kw = {'width_ratios' : [1], 'wspace' : 0.1,
                                             'hspace' : 0.0, 'height_ratios' : [2, 5]})

subfig1 = sns.heatmap(top_s_cr, ax=axs['top'], vmin=0, vmax=1, cbar=False, cmap = colourmap['cmap'],
                     norm = colourmap['sm'].norm)
subfig1.invert_yaxis()
subfig1.axhline(2, 0, 1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 1, color = 'black', linewidth = 2)
subfig1.axvline(6, 0, 1, color = 'black', linewidth = 2)

mappable = subfig1.get_children()[0]

subfig2 = sns.heatmap(bottom_s_cr, ax=axs['bottom'], vmin=0, vmax=1, cbar=False, cmap = colourmap['cmap'],
                      norm = colourmap['sm'].norm)
subfig2.invert_yaxis()
subfig2.axhline(0, 0, 1, color = 'black', linewidth = 2)
subfig2.axvline(0, 0, 1, color = 'black', linewidth = 2)
subfig2.axvline(6, 0, 1, color = 'black', linewidth = 2)

for ax in axs.values():
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

axs['bottom'].set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')
axs['bottom'].set_xlabel('Average consumption rate ' + r'$(\mu_c)$', fontsize=20, weight = 'bold')
axs['bottom'].set_yticklabels([r'$0.02^2$', r'$0.04^2$', r'$0.06^2$', r'$0.08^2$', r'$0.1^2$'],
                               size = '14')
axs['top'].set_yticklabels([r'$0.15^2$', r'$0.2^2$'],
                               size = '14')
axs['bottom'].yaxis.set_tick_params(rotation=360)
axs['top'].yaxis.set_tick_params(rotation=360)
fig.text(-0.02, 0.5, 'Variance ' + r'($\sigma_a^2$)',
         verticalalignment = 'center', horizontalalignment = 'center',
         fontsize=20, weight = 'bold', rotation = 90, rotation_mode = 'anchor')

axs['top'].set_title('growth is uncoupled to\nbut increases with consumption', fontsize = 20, color = '#88d7a2ff',
                        path_effects= [pe.withStroke(linewidth=1, foreground="black")])

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gc_u_s_le.svg",
            bbox_inches='tight')

# %%

# more species

prop_le_cr = pd.pivot_table(uncoupled_cr_df_2,
                                        index = 'Consumption rate std',
                                        columns = 'Average consumption rate',
                                        values = 'Max lyapunov exponent',
                                        aggfunc = prop_chaotic)

colourmap = cmap_norm_2(mpl.colormaps['viridis_r'](0.85))

sns.set_style('white')
    
fig, axs = plt.subplots(1, 1, layout = 'constrained', figsize = (5,4))

subfig1 = sns.heatmap(prop_le_cr, ax=axs, vmin=0, vmax=1, cbar=False, cmap = colourmap['cmap'],
                     norm = colourmap['sm'].norm)
subfig1.invert_yaxis()
subfig1.axhline(0, 0, 1, color = 'black', linewidth = 2)
subfig1.axhline(3, 0, 1, color = 'black', linewidth = 2)
subfig1.axvline(0, 0, 1, color = 'black', linewidth = 2)
subfig1.axvline(6, 0, 1, color = 'black', linewidth = 2)

mappable = subfig1.get_children()[0]
    
axs.set_xticklabels([])
axs.set_yticklabels([])
axs.set_xlabel('')
axs.set_ylabel('')

axs.set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')
axs.set_xlabel('Average consumption rate ' + r'$(\mu_c)$', fontsize=20, weight = 'bold')
axs.set_yticklabels([r'$0.02^2$', r'$0.04^2$', r'$0.06^2$'], size = '14')
axs.yaxis.set_tick_params(rotation=360)
fig.text(-0.02, 0.5, 'Variance ' + r'($\sigma_a^2$)',
         verticalalignment = 'center', horizontalalignment = 'center',
         fontsize=20, weight = 'bold', rotation = 90, rotation_mode = 'anchor')

#%%

################### Species coexistence with max. lyapunov exponents #######################

data_list = [data_gLV_mu_sigma_unscaled, uncoupled_cr_df, coupled_c_cr_df, coupled_g_cr_df]
mus = [0.9, 0.7, 0.5, 1.5]
sigmas = [0.1, 0.1, 0.1, 0.2]
#mus = [0.9, 0.5, 0.5, 1.5]
#sigmas = [0.1, 0.1, 0.1, 0.2]

palette = ['#ab00d5ff', '#88d7a2ff', '#2a7c44ff', '#2a7c44ff']

def generate_subdata(data, mu, sigma):
    
    if np.any(data.columns == 'Average interaction strength'):
        
        subdata = data.iloc[np.where((data['Average interaction strength'] == mu) & \
                                     (data['Interaction strength std'] == sigma))]
        
    elif np.any(data.columns == 'Average consumption rate'):
        
        subdata = data.iloc[np.where((data['Average consumption rate'] == mu) & \
                                     (data['Consumption rate std'] == sigma))]
            
        print(subdata['Max lyapunov exponent'].min(), 
              subdata['Max lyapunov exponent'].max())
              
    return subdata
                      
plotting_data = [generate_subdata(data, mu, sigma) 
                 for data, mu, sigma in zip(data_list, mus, sigmas)]


sns.set_style('white')

fig, axs = plt.subplots(1,4,figsize=(16,4.3),layout='constrained')

for i, (data, ax) in enumerate(zip(plotting_data, axs.flatten())):
    
    if i == 0:
        
        #ax.axvline(0.002, 0, 1, color = 'grey', linewidth = 4, linestyle = '--')
        subfig = sns.scatterplot(data = data, x = 'Max lyapunov exponent',
                                 y = 'Diversity (species)', color = palette[i],
                                 ax = ax, s = 125,
                                 edgecolor = 'black', alpha = 1)
        subfig.set_ylabel('Species\nsurvival fraction',fontsize=24,weight='bold', multialignment='center')
        
    else:
        
        #ax.axhline(1, 0, 1, color = 'grey', linewidth = 4, linestyle = '--')
        #ax.axvline(0.002, 0, 1, color = 'grey', linewidth = 4, linestyle = '--')
        subfig = sns.scatterplot(data = data, x = 'Max lyapunov exponent',
                                 y = 'Closeness to competitive exclusion', color = palette[i],
                                 ax = ax, s = 125,
                                 edgecolor = 'black', alpha = 1)
        ax.set_ylabel('')
        
        if i == 1:
            
            ax.set_ylabel(r'$\frac{\text{Species surivival fraction}}{\text{Resource surivival fraction}}$',
                          fontsize=24,weight='bold', multialignment='center')
        
        
    subfig.set(xlabel=None)
    subfig.set_yticks(range(2))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlim([-0.015, 0.012])
    
fig.supxlabel('Max. lyapunov exponent ' + r'$(\lambda)$',fontsize=24,weight='bold', multialignment='center',
               color='#5f47aeff', path_effects= [pe.withStroke(linewidth=1, foreground="black")])

fig.text(0.66, 1.05, 'Consumer - Resource model', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=20, weight = 'bold',
                color = '#43bf6bff',
                path_effects= [pe.withStroke(linewidth=1, foreground="black")])
fig.text(0.17, 1.05, 'gLV', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=20, weight = 'bold',
                color = '#ab00d5ff',
                path_effects= [pe.withStroke(linewidth=1, foreground="black")])
axs[1].set_title('growth and consumption\nare uncoupled', fontsize = 20, color = '#88d7a2ff',
                        path_effects= [pe.withStroke(linewidth=1, foreground="black")])
axs[2].set_title('consumption is\ncoupled to growth', fontsize = 20, color = '#2a7c44ff',
                      path_effects= [pe.withStroke(linewidth=1, foreground="black")])
axs[3].set_title('growth is coupled\nto consumption', fontsize = 20, color = '#2a7c44ff',
                      path_effects= [pe.withStroke(linewidth=1, foreground="black")])

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/le_species_coexistence.svg",
            bbox_inches='tight')

#%%
# %%

############################## Community dynamics simulations ##################################

# stable dynamics
stable_phase_dynamics = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_consumption_coupled_to_growth_07_004.pkl")
competitive_exclusion_dynamics = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_growth_consumption_uncoupled_07_004.pkl")
s_sim_1 = stable_phase_dynamics[2].ODE_sols['lineage 1']
s_sim_2 = competitive_exclusion_dynamics[19].ODE_sols['lineage 0']
del stable_phase_dynamics
del competitive_exclusion_dynamics

# dynamic phase
dynamic_phase_dynamics_1 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_consumption_coupled_to_growth_05_01.pkl")
dynamic_phase_dynamics_2 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_growth_consumption_uncoupled_05_01.pkl")
d_sim_1 = dynamic_phase_dynamics_1[5].ODE_sols['lineage 4']
d_sim_2 = dynamic_phase_dynamics_2[14].ODE_sols['lineage 4']
del dynamic_phase_dynamics_1
del dynamic_phase_dynamics_2

#%%

simulations = [s_sim_1, s_sim_2, d_sim_1, d_sim_2]
no_species = 50

cmap = mpl.colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                    ['#e9a100ff','#1fb200ff',
                                                     '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                    N = 50)

fig, axs = plt.subplots(1, len(simulations),figsize=(22,4),layout='constrained')

for d_i, (ax, data) in enumerate(zip(axs.flatten(), simulations)):
    
    for i in range(no_species):
    
        ax.plot(data.t, data.y[i,:].T, color = 'black', linewidth = 5.75)
        ax.plot(data.t, data.y[i,:].T, color = cmap(i), linewidth = 5)
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

sns.despine()
 
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_community_dynamics.svg",
            bbox_inches='tight')

# %%

fig, ax = plt.subplots(1,1)

sns.scatterplot(data = uncoupled_cr_df.iloc[np.where((uncoupled_cr_df['Average consumption rate'] == 0.7) & \
                                                     (uncoupled_cr_df['Consumption rate std'] == 0.1))],
                x = 'Volatility (species)',
                y = 'Max lyapunov exponent',
                s = 125, ax = ax,
                edgecolor = 'black', alpha = 1)
ax.set_ylim([-0.02, 0.02])
