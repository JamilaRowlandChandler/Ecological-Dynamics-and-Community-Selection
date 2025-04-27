# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:29:41 2024

@author: jamil
"""

# %%

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
import os

os.chdir('C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Chemically-mediated models')

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
        
        print({'mu_c': mu_c, 'sigma_c' : sigma_c,
               'mu_g': mu_g, 'sigma_g' : sigma_g,
               'rho' : rho, 'Community' : i}, '\n')
       
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

#%%

def variance0(x):
    
    return np.std(x, ddof = 0)

# %%

def prop_chaotic(x,
                instability_threshold = 0.004):
        
    return 1 - np.count_nonzero(x < instability_threshold)/len(x)

# %%

rhos = [0.3, 0.4, 0.5, 0.6, 0.7]
rho_string = ['03', '04', '05', '06', '07']
#rhos = [0.4]
#rho_string = ['04']

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

no_species = 100
sigma = 0.1
sigma_s = '01'

for rho, rho_s in zip(rhos, rho_string):

    for mu, mu_s in zip(mu_cs, mu_string):
            
        filename_CR = "CR_growth_consumption_underlying_coupling_" + str(mu_s) + "_" + str(rho_s)
        #filename_CR = "CR_growth_consumption_underlying_coupling_2_" + str(mu_s) + "_" + str(rho_s)
    
        create_and_delete_CR(filename_CR, 
                             rho, mu, sigma,
                             mu, sigma, no_species, no_species)

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

# fixed variance in 1 variable, vary other

no_species = 100

#sigmas = np.linspace(0.2, 2, 10)/np.sqrt(no_species)
sigmas = np.round(np.linspace(2.2, 2.4, 2)/np.sqrt(no_species), 2)
sigma_strings =  np.char.mod('%s', sigmas)
sigma_strings = np.char.replace(sigma_strings, '.', '_').tolist()

fixed_sigma = 0.1

fixed_mu = 1
mu_s = '1'

rho = 0.4
rho_string = '04'

for sigma, sigma_s in zip(sigmas, sigma_strings):
        
    filename_CR = "CR_growth_consumption_test_c_v_" + str(sigma_s) + "_" + str(rho_string)

    create_and_delete_CR(filename_CR, 
                         rho, fixed_mu, sigma,
                         fixed_mu, fixed_sigma, no_species, no_species)
    
for sigma, sigma_s in zip(sigmas, sigma_strings):
            
    filename_CR = "CR_growth_consumption_test_g_v_" + str(sigma_s) + "_" + str(rho_string)

    create_and_delete_CR(filename_CR, 
                         rho, fixed_mu, fixed_sigma,
                         fixed_mu, sigma, no_species, no_species)
        
# %%

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

rhos = [0.3, 0.4, 0.5, 0.6, 0.7]
rho_string = ['03', '04', '05', '06', '07']


no_species = 100

cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_underlying_coupling_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

cr_df['Model'] = np.repeat('CR', cr_df.shape[0])
cr_df['Annotation'] = np.repeat('c_g_scaled', cr_df.shape[0])

# %%

#mu_string = ['03','05','07','09','11','13']
#mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

#rhos = [0.4]
#rho_string = ['04']


#no_species = 100

#cr_df = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_underlying_coupling_2_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
#                              for rho, rho_s in zip(rhos, rho_string) 
#                              for mu_c, mu_s in zip(mu_cs, mu_string)])

#cr_df['Model'] = np.repeat('CR', cr_df.shape[0])
#cr_df['Annotation'] = np.repeat('c_g_scaled', cr_df.shape[0])

# %%

rhos = [0.4]
rho_string = ['04']

mu_string = ['03','05','07','09','11','13']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

no_species = 100

cr_df_c = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_c_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])
cr_df_c['Annotation'] = np.repeat('c_scaled', cr_df_c.shape[0])

cr_df_g = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_g_" + str(mu_s) + "_" + str(rho_s), no_species, mu_c, rho)
                              for rho, rho_s in zip(rhos, rho_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])
cr_df_g['Annotation'] = np.repeat('g_scaled', cr_df_g.shape[0])

# %%

no_species = 100

sigmas = np.array([0.02, 0.04, 0.06000000000000001, 0.08, 0.1, 0.12, 0.14, 0.16,
                   0.18, 0.2, 0.22, 0.24])
sigma_strings =  np.char.mod('%s', sigmas)
sigma_strings = np.char.replace(sigma_strings, '.', '_').tolist()

fixed_mu = 1
mu_s = '1'

rho = 0.4
rho_string = '04'

no_species = 100

cr_df_c_v = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_c_v_" + str(sigma_string) + "_" + str(rho_string), no_species, sigma, rho)
                              for sigma, sigma_string in zip(sigmas, sigma_strings)])
cr_df_c_v['Annotation'] = np.repeat('c_scaled', cr_df_c_v.shape[0])

cr_df_c_v.rename(columns = {"Average consumption rate": "Variance in consumption rate"},
                 inplace = True)

cr_df_g_v = pd.concat([create_df_and_delete_simulations("CR_growth_consumption_test_g_v_" + str(sigma_string) + "_" + str(rho_string), no_species, sigma, rho)
                              for sigma, sigma_string in zip(sigmas, sigma_strings)])
cr_df_g_v['Annotation'] = np.repeat('c_scaled', cr_df_g_v.shape[0])
cr_df_g_v.rename(columns = {"Average consumption rate": "Variance in consumption rate"},
                 inplace = True)

# %%

prop_le_cr = pd.pivot_table(cr_df,
                            index = 'Correlation',
                            columns = 'Average consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)
#                            aggfunc = variance0)

#%%

def cmap_norm(base, N = 256):
    
    #breakpoint()
    
    colourmap_base = mpl.colors.ColorConverter.to_rgb(base)
    
    def scale_lightness(rgb, scale_l):
        
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h = h, l = scale_l, s = s)
    
    colours_list = [scale_lightness(colourmap_base, scale) 
                    for scale in np.linspace(1, 0, N)]
    
    cmap = mpl.colors.ListedColormap(colours_list)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #norm = mpl.colors.Normalize(vmin=np.sqrt(3e-07), vmax=np.sqrt(3e-05))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    return {'cmap' : cmap, 'sm' : sm}

#%%

colourmap = cmap_norm(mpl.colormaps['viridis_r'](0.85))
    
sns.set_style('white')
    
fig, ax = plt.subplots(1, 1)

subfig = sns.heatmap(prop_le_cr, ax=ax, vmin=0, vmax=1, cbar = False, cmap = colourmap['cmap'],
                     norm = colourmap['sm'].norm)
    
subfig.invert_yaxis()
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel('')
ax.set_ylabel('')

mappable = subfig.get_children()[0]
        
subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
subfig.axvline(6, 0, 1, color = 'black', linewidth = 2)
subfig.axhline(5, 0, 1, color = 'black', linewidth = 2)
        
cbar = plt.colorbar(mappable, ax = ax,
                    orientation = 'vertical')
cbar.set_label(label=r'$P( \lambda > 0.004)$',weight='bold', size='20')
cbar.ax.tick_params(labelsize=14)

ax.set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                   size = '14')
ax.set_yticklabels([r'$0.3$', r'$0.4$', r'$0.5$', r'$0.6$', r'$0.7$'],
                   size = '14')
ax.yaxis.set_tick_params(rotation=360)
 
ax.set_xlabel('Average consumption/growth rate\n' + r'$(\mu_c/S$, ' + r'$\mu_g/M)$',
              fontsize=20, weight = 'bold')
ax.set_ylabel('Correlation between\n' + 'consumption and growth ' + r'$(\rho)$',
              fontsize=20, weight = 'bold')

fig.suptitle('Increasing the mean consumption and growth\n rate, independent of their correlation, drives\n communities from the dynamic to stable phase',
             fontsize=24, weight = 'bold', y = 1.25)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_coupled_by_rho.svg",
            bbox_inches='tight')

# %%

prop_le_c = pd.pivot_table(cr_df_c,
                            index = 'Correlation',
                            columns = 'Average consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)
#                            aggfunc = variance0)

prop_le_g = pd.pivot_table(cr_df_g,
                            index = 'Correlation',
                            columns = 'Average consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)
#                            aggfunc = variance0)

prop_le_cr_03 = pd.pivot_table(cr_df[cr_df['Correlation'] == 0.3],
                            index = 'Correlation',
                            columns = 'Average consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)
#                            aggfunc = variance0)
prop_le_cr_04 = pd.pivot_table(cr_df[cr_df['Correlation'] == 0.4],
                            index = 'Correlation',
                            columns = 'Average consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)
#                            aggfunc = variance0)

# %%

top_cr = prop_le_cr.iloc[2:]

data_list = [top_cr, prop_le_cr_04, prop_le_c, prop_le_g, prop_le_cr_03]

colourmap = cmap_norm(mpl.colormaps['viridis_r'](0.85))
    
sns.set_style('white')

mosaic = [['top1', '.', '.'], 
          ['middle1', 'middle2', 'middle3'],
          ['bottom1', '.', '.']]

sns.set_style('white')
    
fig, axs = plt.subplot_mosaic(mosaic, figsize = (14.5,4),
                              gridspec_kw = {'width_ratios' : [1, 1, 1], 'wspace' : 0.1,
                                             'hspace' : 0.0, 'height_ratios' : [3, 1, 1]})
    
for data, (key, ax) in zip(data_list, axs.items()):
    
    subfig = sns.heatmap(data, ax=ax, vmin=0, vmax=1, cbar = False, cmap = colourmap['cmap'],
                         norm = colourmap['sm'].norm)
        
    subfig.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    
    subfig.axvline(0, 0, 1, color = 'black', linewidth = 2)
    subfig.axvline(6, 0, 1, color = 'black', linewidth = 2)
    
    if key == 'top1':
    
        mappable = subfig.get_children()[0]
            
        subfig.axhline(3, 0, 1, color = 'black', linewidth = 2)
        
    if key == 'middle2' or key == 'middle3':
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        subfig.axhline(1, 0, 1, color = 'black', linewidth = 2)
        
    
    if key == 'bottom1':
        
        subfig.axhline(0, 0, 1, color = 'black', linewidth = 2)
        


axs['bottom1'].set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                               size = '14')

for ax in [axs['middle2'], axs['middle3']]:
    
    ax.set_xticklabels([r'$0.3$', r'$0.5$', r'$0.7$', r'$0.9$', r'$1.1$', r'$1.3$'],
                       size = '14', y = -1)

axs['top1'].set_yticklabels([r'$0.5$', r'$0.6$', r'$0.7$'],
                   size = '14')
axs['middle1'].set_yticklabels([r'$0.4$'],
                   size = '14')
axs['bottom1'].set_yticklabels([r'$0.3$'],
                   size = '14')

for ax in [axs['top1'], axs['middle1'], axs['bottom1']]:

    ax.yaxis.set_tick_params(rotation=360)

axs['bottom1'].set_xlabel(r'$\mu_c/S$, ' + r'$\mu_g/M$' + '\n' + '(Avg. consumption, growth rate)',
              fontsize=20, weight = 'bold')
axs['middle2'].set_xlabel(r'$\mu_c/S$ ' + r'$(\mu_g/M = 1)$',
              fontsize=20, weight = 'bold')
axs['middle3'].set_xlabel(r'$\mu_g/M$ ' + r'$(\mu_c/S = 1)$',
              fontsize=20, weight = 'bold')

fig.text(0.07, 0.5, 'Correlation between\n' + 'consumption and growth ' + r'$(\rho)$',
         verticalalignment = 'center',
         horizontalalignment = 'center', fontsize=20, weight = 'bold',
         rotation = 90, rotation_mode = 'anchor')

fig.suptitle('Altering the average species-resource interaction drives regime shifts',
             fontsize=24, weight = 'bold', y = 1.05)

#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagrams_mu_rho.svg",
#            bbox_inches='tight')
#plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagrams_mu_rho.png",
#            bbox_inches='tight')

# %%

prop_le_c_v = pd.pivot_table(cr_df_c_v,
                            index = 'Correlation',
                            columns = 'Variance in consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)

prop_le_g_v = pd.pivot_table(cr_df_g_v,
                            index = 'Correlation',
                            columns = 'Variance in consumption rate',
                            values = 'Max lyapunov exponent',
                            aggfunc = prop_chaotic)

# %%

############################## Community dynamics simulations ##################################

# stable dynamics
stable_phase_dynamics = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_growth_consumption_underlying_coupling_11_07.pkl")
s_sim = stable_phase_dynamics[0].ODE_sols['lineage 0']
del stable_phase_dynamics

dynamic_phase_dynamics = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_growth_consumption_underlying_coupling_07_03.pkl")
d_sim = dynamic_phase_dynamics[0].ODE_sols['lineage 0']
del dynamic_phase_dynamics

simulations = np.tile([s_sim, d_sim], 2)
species = np.arange(100)
resources = np.arange(100, 200)
colour_index = np.arange(100)
np.random.shuffle(colour_index)

cmap = mpl.colors.LinearSegmentedColormap.from_list('custom YlGBl',
                                                    ['#e9a100ff','#1fb200ff',
                                                     '#1f5a00ff','#00e9e9ff','#001256fd'],
                                                    N = 100)

fig, axs = plt.subplots(2, 2,figsize=(16,10),layout='constrained')

for d_i, (ax, data) in enumerate(zip(axs.flatten(), simulations)):
    
    if d_i <= 1:
        
        indexer = species
        
    else:
        
        indexer = resources
    
    for i, spec in zip(colour_index, indexer):
    
        ax.plot(data.t, data.y[spec,:].T, color = 'black', linewidth = 3.75)
        ax.plot(data.t, data.y[spec,:].T, color = cmap(i), linewidth = 3)
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_community_dynamics_uc.svg",
            bbox_inches='tight')

# %%

############################ average growth and consumption vs species packing ####################

labels = [r'$\mu_c/S$, ' + r'$\mu_g/M$' + '\n' + '(Avg. consumption, growth rate)',
          r'$\mu_c/S$ ' + r'$(\mu_g/M = 1)$',
          r'$\mu_g/M$ ' + r'$(\mu_c/S = 1)$']

sns.set_style('white')

fig, axs = plt.subplots(1, 3, figsize=(8, 3.5), layout='constrained', sharex = True, sharey = True)

for data, label, ax in zip([cr_df[cr_df['Correlation'] == 0.4], cr_df_c, cr_df_g],
                                   labels, axs):
    
    sns.stripplot(data,
                  x = 'Average consumption rate',
                  y = 'Closeness to competitive exclusion',
                  color = '#2a7c44ff',
                  ax = ax,
                  alpha = 0.2, legend=False)

    sns.pointplot(data,
                  x = 'Average consumption rate',
                  y = 'Closeness to competitive exclusion',
                  color = 'black',
                  ax = ax,
                  linewidth = 3.5, errorbar=None,
                  marker="", markersize=5, markeredgewidth=3)
    sns.pointplot(data,
                  x = 'Average consumption rate',
                  y = 'Closeness to competitive exclusion',
                  color = '#2a7c44ff',
                  ax = ax,
                  linewidth = 2.75, errorbar=None,
                  marker="", markersize=5, markeredgewidth=3)
    
    ax.set_xlabel(label, fontsize = 12)
    ax.set_ylabel("")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_ylim([0, 1])
    
fig.supylabel(r'$\frac{\text{Species survival fraction}}{\text{Resource survival fraction}}$',
              fontsize = 16)
fig.suptitle('')

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/consumption_growth_vs_ce.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/consumption_growth_vs_ce.png",
            bbox_inches='tight')

# %%

labels = [r'$\mu_c/S$, ' + r'$\mu_g/M$' + '\n' + '(Avg. consumption, growth rate)',
          r'$\mu_c/S$ ' + r'$(\mu_g/M = 1)$',
          r'$\mu_g/M$ ' + r'$(\mu_c/S = 1)$']

sns.set_style('white')

fig, axs = plt.subplots(1, 3, figsize=(8, 3.5), layout='constrained', sharex = True, sharey = True)

for data, label, ax in zip([cr_df[cr_df['Correlation'] == 0.4], cr_df_c, cr_df_g],
                                   labels, axs):
    
    ax.axhline(y = 0.004, color = 'gray', linestyle = '--', linewidth = 1)
    
    sns.stripplot(data,
                  x = 'Average consumption rate',
                  y = 'Max lyapunov exponent',
                  color = '#2a7c44ff',
                  ax = ax,
                  alpha = 0.2, legend=False)

    sns.pointplot(data,
                  x = 'Average consumption rate',
                  y = 'Max lyapunov exponent',
                  color = 'black',
                  ax = ax,
                  linewidth = 3.5, errorbar=None,
                  marker="", markersize=5, markeredgewidth=3)
    sns.pointplot(data,
                  x = 'Average consumption rate',
                  y = 'Max lyapunov exponent',
                  color = '#2a7c44ff',
                  ax = ax,
                  linewidth = 2.75, errorbar=None,
                  marker="", markersize=5, markeredgewidth=3)
    
    ax.set_xlabel(label, fontsize = 12)
    ax.set_ylabel("")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
fig.supylabel('Maximum lyapunov exponent ' + r'($\lambda$)',
              fontsize = 12)
fig.suptitle('')

sns.despine()

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/consumption_growth_vs_le.svg",
            bbox_inches='tight')
plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/consumption_growth_vs_le.png",
            bbox_inches='tight')

# %%

labels = [r'$\mu_c/S$, ' + r'$\mu_g/M$' + '\n' + '(Avg. consumption, growth rate)',
          r'$\mu_c/S$ ' + r'$(\mu_g/M = 1)$',
          r'$\mu_g/M$ ' + r'$(\mu_c/S = 1)$']

dfs = [cr_df[(cr_df['Correlation'] == 0.4)], cr_df_c, cr_df_g]

mus = np.round(np.arange(0.3, 1.5, 0.2), 1)

fig = plt.figure(figsize = (15, 4))
subfigs = fig.subfigures(1, 3)

for subfig, df, label in zip(subfigs.flat, dfs, labels):

    sns.set_style('white', rc={"axes.facecolor": (0, 0, 0, 0), 'axes.spines.left': False,
                               'axes.spines.right': False, 'axes.spines.top': False})

    axs = subfig.subplots(len(mus), 1, sharex = True, sharey = True,
                        gridspec_kw = {'hspace' : -0.25})

    for mu, ax in zip(mus, axs):
        
        data = df[df['Average consumption rate'] == mu]
    
        sns.kdeplot(data,
                    x = 'Max lyapunov exponent',
                    color = '#2a7c44ff', fill = True,
                    ax = ax, bw_adjust = 0.5,
                    alpha = 1, linewidth = 1.5, legend=False)
        sns.kdeplot(data,
                    x = 'Max lyapunov exponent',
                    color="w", lw=2, bw_adjust = 0.5,
                    ax = ax, legend=False)
        
        #ax.set_xlabel(label, fontsize = 12)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim([-0.007, 0.015])
        
    fig.supylabel(r'$\frac{\text{Species survival fraction}}{\text{Resource survival fraction}}$',
                  fontsize = 16)
    fig.suptitle('')

#sns.despine()

# %%

def average_abundance_from_cavity_solutions(S, M,
                                            mu_g, mu_c, sigma_g, sigma_c,
                                            rho,
                                            phi_N, phi_R):
    
    '''
    

    Parameters
    ----------
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    phi_N : TYPE
        DESCRIPTION.
    phi_R : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    gamma = M/S
    
    # Suseptibilities 
    
    v = ((1/gamma)*(phi_N/phi_R))/((rho * sigma_g * sigma_g *(1/gamma)) * ((1/gamma)*(phi_N/phi_R) - 1))
    chi = phi_R - (1/gamma)*phi_N
    
    # Average abundances
    
    N_avg = (sigma_)
    
    

#%%

def total_interaction_scaling_check(n):

    mu = 100
    sigma = 20
    
    parm = np.random.randn(n, n)
    parm_n = (mu/n) + (sigma/np.sqrt(n))*parm
    new_parm = np.random.randn(n+1, n+1)
    new_parm[:n, :n] = parm
    
    parm_n_add_1 = (mu/(n+1)) + (sigma/np.sqrt(n+1))*new_parm
    parm_incorrect_scaling = (mu/n) + (sigma/np.sqrt(n))*new_parm
    
    correct_diff = np.sum(parm_n_add_1[:n, :], axis = 1) - np.sum(parm_n, axis = 1)
    incorrect_diff = np.sum(parm_incorrect_scaling[:n, :], axis = 1) - np.sum(parm_n, axis = 1)
    
    if np.sum(correct_diff) < np.sum(incorrect_diff):
        
        print("it's all good lads: ", np.mean(correct_diff))
    
    else:
        
        print("we're fucked")
    
#%%

for _ in range(20):
    
    total_interaction_scaling_check(n = 1000)
    

