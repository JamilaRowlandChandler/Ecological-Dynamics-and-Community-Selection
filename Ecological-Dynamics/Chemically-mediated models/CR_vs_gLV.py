# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:25:09 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import sys

from CR_vs_gLV_functions import *

sys.path.insert(0, 'C:/Users/jamil/Documents/PhD/Github Projects/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/Lotka-Volterra models/model_modules')
from utility_functions import pickle_dump

# %%

####################################### SIMULATIONS ##################################

############################ generalised Lotka-Volterra dynamics ##############################

# Classic gLV with fixed self-interactions and growth rate.

mu_as = [0.3,0.5,0.7,0.9,1.1]
sigma_as = [0.05,0.1,0.15,0.2]

mu_g = 1
sigma_g = 0

gLV_communities_fixed_growth = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 50) 
                                for sigma_a in sigma_as for mu_a in mu_as]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_2.pkl",
            gLV_communities_fixed_growth)

# %%

# gLV with self-interactions and growth scaled with species interactions and CR model

mu_c_from_cr = np.array([0.3,0.5,0.7,0.9,1.1])
mu_g_from_cr = mu_c_from_cr
sigma_from_cr = np.array([0.05,0.1,0.15,0.2])

mu_gs = mu_c_from_cr
sigma_gs = sigma_from_cr

mu_as = mu_c_from_cr * mu_g_from_cr
sigma_as = [[np.sqrt((sigma**2 + mu_c**2) * (sigma**2 + mu_g**2) \
                - (mu_c**2 * mu_g**2)) for sigma in sigma_from_cr]
            for mu_c, mu_g in zip(mu_c_from_cr, mu_gs)]

gLV_communities_scaled_self = [gLV_dynamics(mu_a, sigma_a, mu_g, sigma_g, no_species = 50,
                                            scale_self_inhibition = True) 
                               for mu_a, mu_g, sigma_mu_a in zip(mu_as, mu_gs, sigma_as) 
                               for sigma_a, sigma_g in zip(sigma_mu_a, sigma_gs)]
pickle_dump("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/gLV_d_s_scaled.pkl",
            gLV_communities_scaled_self)

# %%

###################################### Consumer - Resource dynamics ###############################

# C-R model where growth does not scale with consumption, but does have the same variance

mu_string = ['03','05','07','09','11']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_string = ['005','01','015','02']
sigma_cs = [0.05, 0.1, 0.15, 0.2]

no_species = 50

for sigma_c, sigma_s in zip(sigma_cs, sigma_string):
    
    for mu_c, mu_s in zip(mu_cs, mu_string):
        
        filename_CR = "CR_d_s_small_" + str(mu_s) + "_" + str(sigma_s)

        create_and_delete_CR(filename_CR, {'mu_c' : mu_c, 'sigma_c' : sigma_c,
                                           'mu_g' : mu_c, 'sigma_g' : sigma_c,
                                           'no_species' : no_species,
                                           'no_resources' : no_species})
        
# %%

# C-R model with growth scaled with consumption

mu_c = [0.3,0.5,0.7,0.9,1.1]
sigma_c = [0.05,0.1,0.15,0.2]

no_species = 50

for sigma in sigma_c :
    
    for mu in mu_c:
        
        filename_CR = "cr_growth_scaled_consumption_" + str(mu) + "_" + str(sigma)

        create_and_delete_CR(filename_CR, {'mu_c' : mu_c, 'sigma_c' : sigma_c,
                                           'mu_g' : mu_c, 'sigma_g' : sigma_c,
                                           'no_species' : no_species,
                                           'no_resources' : no_species})
        
# %%

#################################### Dataframes ############################################

################################## Consumer - Resource models ######################

# C-R model where growth does not scale with consumption, but does have the same variance

mu_string = ['03','05','07','09','11']
mu_cs = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_string = ['005','01','015','02']
sigma_cs = [0.05, 0.1, 0.15, 0.2]
        
data_mu_sigma_s = pd.concat([create_df_and_delete_simulations("CR_d_s_small_" + str(mu_s) + "_" + str(sigma_s), mu_c, sigma_c)
                              for sigma_c, sigma_s in zip(sigma_cs, sigma_string) 
                              for mu_c, mu_s in zip(mu_cs, mu_string)])

data_mu_sigma_s['Model'] = np.repeat('CR', data_mu_sigma_s.shape[0])

# %%

# C-R model with growth scaled with consumption

mu_c = [0.3,0.5,0.7,0.9,1.1]
sigma_c = [0.05,0.1,0.15,0.2]

data_mu_sigma_gc = pd.concat([create_df_and_delete_simulations("cr_growth_scaled_consumption_" + str(mu) + "_" + str(sigma),
                                                               mu, sigma)
                              for sigma in sigma_c for mu in mu_c])

data_mu_sigma_gc['Model'] = np.repeat('CR', data_mu_sigma_gc.shape[0])

# %%

########################## gLV ###########################

# Classic gLV with fixed self-interactions and growth rate.


mu_as = [0.3, 0.5, 0.7, 0.9, 1.1]
sigma_as = [0.05, 0.1, 0.15, 0.2]

data_gLV_mu_sigma_unscaled = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                                        for simulation_data, mu, sigma in 
                                        zip(gLV_communities_fixed_growth, np.tile(mu_as, len(sigma_as)),
                                            np.repeat(sigma_as, len(mu_as)))])
    
data_gLV_mu_sigma_unscaled['Model'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])

# %%

# gLV with self-interactions and growth scaled with species interactions and CR model

mu_c_from_cr = np.array([0.3,0.5,0.7,0.9,1.1])
mu_g_from_cr = mu_c_from_cr
sigma_from_cr = np.array([0.05,0.1,0.15,0.2])

mu_gs = mu_c_from_cr
sigma_gs = sigma_from_cr

mu_as = mu_c_from_cr
sigma_as = [[np.sqrt((sigma**2 + mu_c**2) * (sigma**2 + mu_g**2) \
                - (mu_c**2 * mu_g**2)) for sigma in sigma_from_cr]
            for mu_c, mu_g in zip(mu_c_from_cr, mu_gs)]
    
data_gLV_mu_sigma_scaled = pd.concat([gLV_dynamics_df(simulation_data, mu, sigma)
                                        for simulation_data, mu, sigma in 
                                        zip(gLV_communities_scaled_self, np.tile(mu_as, len(sigma_gs)),
                                            np.concatenate(sigma_as))])
    
data_gLV_mu_sigma_scaled['Model'] = np.repeat('gLV', data_gLV_mu_sigma_unscaled.shape[0])


# %%

################################### Phase diagrams ######################################

# Compare unscaled gLV to C-R with fixed growth 

prop_reinvadability_cr = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_cr = 1 - prop_reinvadability_cr

prop_reinvadability_gLV = pd.pivot_table(data_gLV_mu_sigma_unscaled,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV = 1 - prop_reinvadability_gLV

fig_unscaled, axs_unscaled = phase_diagram([prop_reinvadability_cr, prop_reinvadability_gLV])

fig_unscaled.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

axs_unscaled[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs_unscaled[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

axs_unscaled[0].set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
axs_unscaled[0].set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
axs_unscaled[1].set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
axs_unscaled[1].set_ylabel('')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gLV_CR.png",
            dpi=300,bbox_inches='tight')

# %%

# Compare unscaled gLV with growth-scaled CR

prop_reinvadability_gc = pd.pivot_table(data_mu_sigma_gc,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gc = 1 - prop_reinvadability_gc

prop_reinvadability_gLV = pd.pivot_table(data_gLV_mu_sigma_unscaled,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV = 1 - prop_reinvadability_gLV

fig_scaled, axs_scaled = phase_diagram([prop_reinvadability_gc, prop_reinvadability_gLV])

fig_scaled.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

axs_scaled[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs_scaled[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

axs_scaled[0].set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
axs_scaled[0].set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
axs_scaled[1].set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
axs_scaled[1].set_ylabel('')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gLV_CR_scaled.png",
            dpi=300,bbox_inches='tight')

# %%

# Compare scaled gLV with unscaled CR

prop_reinvadability_s = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_s = 1 - prop_reinvadability_s

data_gLV_mu_sigma_scaled['True interaction strength std'] = \
    data_gLV_mu_sigma_scaled['Interaction strength std']
    
data_gLV_mu_sigma_scaled['Interaction strength std'] = np.repeat([0.05, 0.1, 0.15, 0.2], 25*5*len(mu_as))

prop_reinvadability_gLV_s = pd.pivot_table(data_gLV_mu_sigma_scaled,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV_s = 1 - prop_reinvadability_gLV_s

fig_scaled2, axs_scaled2 = phase_diagram([prop_reinvadability_s, prop_reinvadability_gLV_s])

fig_scaled2.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.15)

axs_scaled2[1].set_title('gLV', fontsize = 16, weight = 'bold')
axs_scaled2[0].set_title('Consumer-Resource model', fontsize = 16, weight = 'bold')

axs_scaled2[0].set_xlabel('Avg. consumption rate',fontsize=16, weight = 'bold')
axs_scaled2[0].set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
axs_scaled2[1].set_xlabel('Avg. interaction strength',fontsize=16, weight = 'bold')
axs_scaled2[1].set_ylabel('')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gLV_scaled_CR.png",
            dpi=300,bbox_inches='tight')

#%%

# Compare unscaled gLV to C-R with fixed growth and variable growth 

prop_reinvadability_cr = pd.pivot_table(data_mu_sigma_s,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_cr = 1 - prop_reinvadability_cr

prop_reinvadability_gc = pd.pivot_table(data_mu_sigma_gc,
                                      index = 'Consumption rate std',
                                      columns = 'Average consumption rate',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gc = 1 - prop_reinvadability_gc

prop_reinvadability_gLV = pd.pivot_table(data_gLV_mu_sigma_unscaled,
                                      index = 'Interaction strength std',
                                      columns = 'Average interaction strength',
                                      values = 'Reinvadability (species)',
                                      aggfunc = prop_stable)
prop_reinvadability_gLV = 1 - prop_reinvadability_gLV

fig_3phase, axs_3phase = phase_diagram([prop_reinvadability_cr, prop_reinvadability_gc,
                                       prop_reinvadability_gLV])

fig_3phase.suptitle('The phase transition out of stability qualitatively\ndiffers between the gLV and C-R model',fontsize=20,
             weight='bold', y = 1.17)

axs_3phase[0].set_title('growth does not scale \nwith consumption',
                        fontsize = 14)
axs_3phase[1].set_title('growth scales\nwith consumption',
                        fontsize = 14)
fig_3phase.text(0.33, 0.93, 'Consumer - Resource model', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=16, weight = 'bold')
fig_3phase.text(0.75, 0.93, 'gLV', verticalalignment = 'center',
                horizontalalignment = 'center',fontsize=16, weight = 'bold')

axs_3phase[0].set_xlabel('',fontsize=14, weight = 'bold')
axs_3phase[1].set_xlabel('',fontsize=14, weight = 'bold')
axs_3phase[2].set_xlabel('',fontsize=14, weight = 'bold')
fig_3phase.text(0.33, 0.18, 'Avg. consumption rate', verticalalignment = 'center',
                horizontalalignment = 'center', fontsize=14, weight = 'bold')
fig_3phase.text(0.745, 0.18, 'Avg. interaction strength', verticalalignment = 'center',
                horizontalalignment = 'center', fontsize=14, weight = 'bold')

axs_3phase[0].set_ylabel(r'$\sigma$',fontsize=16, weight = 'bold')
axs_3phase[1].set_ylabel('')
axs_3phase[2].set_ylabel('')

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/phase_diagram_gLV_both_CR.png",
            dpi=300,bbox_inches='tight')

# %%

############################## Diversity - Stability Relationships ###############################

# Compare unscaled gLV to C-R with fixed growth 

fig_ds_unscaled, axs_ds_unscaled = diversity_stability_plot([data_mu_sigma_s, data_gLV_mu_sigma_unscaled],
                                                            'Reinvadability (species)',
                                                            'Diversity (species)',
                                                            'Model',
                                                            'Species\nsurvival fraction',
                                                            ['C-R', 'gLV'],
                                                            0.5, 0.1, 0.9, 0.1)

fig_ds_unscaled.suptitle('The gLV has a stronger negative diversity-stability\nrelationship in its dynamical phase than the C-R model.',
             fontsize=28,weight='bold',y=1.2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_gLV_CR.png",
            dpi=300,bbox_inches='tight')

#%%

#binned_reinvadability = pd.qcut(data_mu_sigma_s.iloc[np.where((data_mu_sigma_s['Average consumption rate'] == 0.3) & \
#                                      (data_mu_sigma_s['Consumption rate std'] == 0.1))]['Reinvadability (species)'], 10,
#                                duplicates = 'drop')
    
#median_reinvadability_per_bin = [(bin_r.left + bin_r.right)/2 for bin_r in binned_reinvadability]


# %%

# Compare unscaled gLV with growth-scaled CR

fig_ds_scaled, axs_ds_scaled = diversity_stability_plot(data_mu_sigma_gc, data_gLV_mu_sigma_unscaled,
                                                        'Reinvadability (species)',
                                                        'Diversity (species)',
                                                        'Model',
                                                        'Species\nsurvival fraction',
                                                        ['C-R', 'gLV'],
                                                        0.5, 0.2, 0.9, 0.1)

fig_ds_scaled.suptitle('The gLV and C-R with scaled growth have a\nsimilar diversity-stability relationship.',
             fontsize=28,weight='bold',y=1.2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_gLV_CR_scaled.png",
            dpi=300,bbox_inches='tight')

# %%

# Compare scaled gLV with growth-scaled CR

fig_ds_scaled_scaled, axs_ds_scaled_scaled = \
    diversity_stability_plot(data_mu_sigma_gc, data_gLV_mu_sigma_scaled,
                             'Reinvadability (species)',
                             'Diversity (species)',
                             'Model',
                             'Species\nsurvival fraction',
                             ['C-R', 'gLV'],
                             0.5, 0.2, 0.9, 0.2)

fig_ds_scaled_scaled.suptitle('The gLV and C-R with scaled growth have a\nsimilar diversity-stability relationship.',
             fontsize=28,weight='bold',y=1.2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_gLV_CR_both_scaled.png",
            dpi=300,bbox_inches='tight')

# %%

data_mu_sigma_s['annotation'] = 'cr unscaled'
data_mu_sigma_gc['annotation'] = 'cr scaled'
data_gLV_mu_sigma_unscaled['annotation'] = 'gLV unscaled'

fig_ds, axs_ds = diversity_stability_plot([data_mu_sigma_gc, data_mu_sigma_s, data_gLV_mu_sigma_unscaled],
                                          'Reinvadability (species)',
                                          'Diversity (species)',
                                          'annotation',
                                          'Species\nsurvival fraction',
                                          ['C-R (growth scaled\nwith consumption)',
                                           'C-R (unscaled growth)', 'gLV'],
                                          [0.5, 0.5, 0.9], [0.2, 0.1, 0.1],
                                          palette = ['#2a7c44ff','#88d7a2ff', '#440154ff'])

fig_ds.suptitle('The gLV has a stronger negative\ndiversity-stability relationship in its\ndynamical phase than the C-R model.',
                fontsize=28,weight='bold',y=1.3)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/diversity_stability_scatterplot_all.png",
            dpi=300,bbox_inches='tight')

# %%

############################# Closeness to competitive exclusion ################

data_mu_sigma_s['Conditions'] = 'Unscaled'
data_mu_sigma_gc['Conditions'] = 'Growth rate scaled\nwith consumption'

fig_ce, axs_ce = competitive_exclusion_plot([data_mu_sigma_gc, data_mu_sigma_s],
                                            'Reinvadability (species)',
                                            'Closeness to competitive exclusion',
                                            0.5, 0.2)

fig_ce.suptitle('Instability does not bring species diversity\ncloser to the competitive exclusion threshold',
                fontsize=28,weight='bold',y=1.2)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/competitive_exclusion_scatterplot_CR.png",
            dpi=300,bbox_inches='tight')

#%%

######################################## Other plots ##########################

# Plot chaotic vs stable C-R simulations, resources + species dynamics
# Compare with gLV
 
chaotic_gLV = gLV_communities_fixed_growth[8]['Simulations'][20]
stable_gLV = gLV_communities_fixed_growth[5]['Simulations'][0]

####

cr_05_01 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_05_01.pkl")
chaotic_cr_1 = cr_05_01['Simulations'][25]
del cr_05_01

cr_11_01 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/CR_d_s_small_11_01.pkl")
stable_cr_1 = cr_11_01['Simulations'][1]
del cr_11_01

#####

cr_05_02_2 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/cr_growth_scaled_consumption_0.5_0.2.pkl")
chaotic_cr_2 = cr_05_02_2['Simulations'][18]
del cr_05_02_2

cr_11_01_2 = pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/cr_growth_scaled_consumption_1.1_0.1.pkl")
stable_cr_2 = cr_11_01_2['Simulations'][3]
del cr_11_01_2

fig_p, axs_p = plot_community_dynamics(stable_cr_1,
                                       chaotic_cr_1,
                                       stable_gLV,
                                       chaotic_gLV)

plt.savefig("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/cr_glv_population_dynamics.png",
            dpi=300,bbox_inches='tight')

#%%

# Comparing total consumption in the fixed growth CR

def total_consumption_cr(filename, y_index = np.arange(50, 100)):
    
    simulations = \
            pd.read_pickle("C:/Users/jamil/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/" + filename + ".pkl")
   
    return total_consumption(simulations, y_index)

average_total_consumption_cr = np.concatenate([np.nanmean(np.array(total_consumption_cr("CR_d_s_small_" + mu_s + "_015")), axis = (1,2))
                                               for mu, mu_s in zip([0.3, 0.5, 0.7, 0.9, 1.1],
                                                                   ['03', '05', '07', '09', '11'])])

average_total_consumption_gLV = np.concatenate([np.nanmean(total_consumption(simulations, np.arange(50)), axis = (1, 2))
                                                for simulations in gLV_communities_fixed_growth[10:15]])

#%%

fig, axs = plt.subplots(1, 1, layout = 'constrained', sharex = True, figsize = (4,4))

sns.stripplot(y = average_total_consumption_gLV[::-1],
                x = [str(val) 
                     for val in data_gLV_mu_sigma_unscaled.iloc[np.where(data_gLV_mu_sigma_unscaled['Interaction strength std'] == 0.15)]['Average interaction strength']],
                hue = np.repeat('gLV', len(average_total_consumption_gLV)), ax = axs,
                palette = ['#440154ff'], s = 4, edgecolor = 'white', linewidth = 0.2)

plt.yscale('log')

ax2 = plt.twinx()

sns.stripplot(y = average_total_consumption_cr,
                x = [str(val) for val in data_mu_sigma_s.iloc[np.where(data_mu_sigma_s['Consumption rate std'] == 0.15)]['Average consumption rate']],
                hue = np.repeat('C-R', len(average_total_consumption_cr)), ax = ax2,
                palette = ['#349b55ff'], s = 4, edgecolor = 'black', linewidth = 0.2)

plt.yscale('log')

axs.set_xlabel("Average consumption rate ($c$)", fontsize = 14, weight = 'bold')
axs.set_ylabel("Total consumption ($\sum_{i=1}^S c_{i,m} N_i$)",
               fontsize=14,weight='bold', multialignment='center')
ax2.set_ylabel("Total competition ($\sum_{j=1}^S a_{i,j} N_j$)",
               fontsize=14,weight='bold', multialignment='center',
               rotation = 90)

axs.get_legend().remove()
ax2.get_legend().remove()

