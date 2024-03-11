# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:44:44 2024

@author: jamil
"""

######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler
import pandas as pd
import seaborn as sns
from ast import literal_eval

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

#################### Functions ##########################

def identify_community_phenotypes(data_groupby_ecodyn):
    
    breakpoint()
    
    total_comp_over_cycles = data_groupby_ecodyn.groupby(['Lineage'])['Final Composition'].apply(\
                                        lambda x : np.unique(np.concatenate(x)))

################################ Load Data Structures ##########################

le_across_interact_dists = pd.read_csv("testing_criteria_for_eco_dyn.csv",index_col=0)
community_selection_dynamics = pd.read_csv("community_selection_dynamics2.csv",index_col=0)

community_selection_chaos = pd.read_pickle('community_selection_chaotic.pkl')
community_selection_oscillations = pd.read_pickle('community_selection_oscillating.pkl')
community_selection_stable = pd.read_pickle('community_selection_stable.pkl')

###

################### Plotting community function over selection cycles ######

community_selection_dynamics.groupby(['Species pool','Predicted dynamics (cycle 5)']).apply(identify_community_phenotypes)

##########

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

####

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 1',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_chaos_speciespool.png", dpi=300)

#####

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             #,errorbar=('pi',100),err_style='bars')

sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

ax.legend([],[], frameon=False)
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 2',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_osc_speciespool.png", dpi=300)


#####

community_selection_dynamics.iloc[\
                                  np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                           & (community_selection_dynamics['Cycle'] == 10))][\
                                  ['Lineage','Community Function']]
# phenotype 1 = lineage 9
# phenotype 2 = else
                                                                                             
fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] != 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] == 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'stable')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
    
ax.legend([],[], frameon=False)
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 3',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_stable_speciespool.png", dpi=300)

########## Create Legend ##################

sns.set_style('white')
fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] == 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

plt.savefig("Figures/communityfunction_legend.png", dpi=300)

######################### Predicted dynamics at cycle 12 #########

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

osc_chaos_lin = community_selection_dynamics.iloc[\
                    np.where((community_selection_dynamics['Species pool'] == 'oscillations')
                             & (community_selection_dynamics['Community Function'] > 0) \
                             & (community_selection_dynamics['Cycle'] == 4))]['Lineage']

osc_osc_lin = community_selection_dynamics.iloc[\
                    np.where((community_selection_dynamics['Species pool'] == 'oscillations')
                             & (community_selection_dynamics['Community Function'] < 0) \
                             & (community_selection_dynamics['Cycle'] == 4))]['Lineage']

####

index1 = np.where(community_selection_dynamics['Species pool'] == 'oscillations')
index2 = np.argwhere(np.isin(community_selection_dynamics['Lineage'],osc_chaos_lin)).ravel()
correct_index_to_plot1 = index1[0][np.isin(index1,index2)[0]]

index3 = np.argwhere(np.isin(community_selection_dynamics['Lineage'],osc_osc_lin)).ravel()
correct_index_to_plot2 = index1[0][np.isin(index1,index3)[0]]

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[correct_index_to_plot1],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 12)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

sns.scatterplot(data=community_selection_dynamics.iloc[correct_index_to_plot1],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 12)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

sns.lineplot(data=community_selection_dynamics.iloc[correct_index_to_plot2],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 12)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

sns.scatterplot(data=community_selection_dynamics.iloc[correct_index_to_plot2],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 12)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

ax.legend([],[], frameon=False)
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 2',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_osc_speciespool12.png", dpi=300)

########## Create Legend ##################

sns.set_style('white')
fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] == 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 12)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

plt.savefig("Figures/communityfunction_legend12.png", dpi=300)

################ Zoom in on cycle 12 ##################

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

####

mean_std_deviation(community_selection_dynamics.iloc[\
                        np.where((community_selection_dynamics['Species pool'] == 'oscillations') \
                                 & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'chaotic-like') \
                                 & (community_selection_dynamics['Cycle'] == 12))][\
                                    'Community Function'])
                                                                                   
mean_std_deviation(community_selection_dynamics.iloc[\
                        np.where((community_selection_dynamics['Species pool'] == 'oscillations') \
                                 & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'chaotic-like') \
                                 & (community_selection_dynamics['Cycle'] == 11))][\
                                    'Community Function'])

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
ax.set_xticks(range(len(community_selection_dynamics['Cycle'].unique())),labels=range(15))
ax.legend([],[], frameon=False)
ax.set_xlim(10,14)
ax.set_ylim(900,1200)

plt.savefig("Figures/communityfunction_oscillations_zoominon12.png", dpi=300)

###

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
ax.set_xticks(range(len(community_selection_dynamics['Cycle'].unique())),labels=range(15))
ax.legend([],[], frameon=False)
ax.set_xlim(10,14)
ax.set_ylim(-450,-350)

plt.savefig("Figures/communityfunction_chaos_zoominon12.png", dpi=300)

               
##################### Variation #####################

variation_comm = community_selection_dynamics.groupby(["Species pool",
                                                       "Predicted dynamics (cycle 5)",
                                                       "Cycle"])["Community Function"].var()
variation_comm = variation_comm.to_frame()
variation_comm.reset_index(inplace=True)
variation_comm.rename(columns={"Community Function":"Variation in Community Function"},inplace=True)

variation_comm.replace({'Species pool':{'chaos':'1','oscillations':'2','stable':'3'}},
                       inplace=True)

#### Plotting ####

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

ax = sns.lineplot(data=variation_comm,
             x="Cycle",y="Variation in Community Function",hue="Predicted dynamics (cycle 5)",
             style='Species pool',
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Variation in Community Function',fontsize=14)
plt.title('Variation in community function across ecological dynamics',fontsize=18)
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.savefig("Figures/variation_in_communityfunction.png", dpi=300, bbox_inches='tight')

############################### Heritability ################################

no_cycles = 15

heritability_comm = community_selection_dynamics.groupby(["Species pool","Predicted dynamics (cycle 5)"])\
    ["Community Function"].apply(lambda x : heritability_during_selection(x,no_cycles))
heritability_comm = heritability_comm.to_frame()
heritability_comm.reset_index(inplace=True)

groups, g_counts = np.unique(heritability_comm["Species pool"],return_counts=True)

heritability_comm_df = pd.DataFrame( \
                                    np.vstack([\
                                    np.concatenate([np.repeat(group,(g_count*(no_cycles-1)).astype(int)) for group, g_count in zip(groups,g_counts)]),
                                    np.concatenate([np.repeat(dyn,no_cycles-1) for dyn in heritability_comm['Predicted dynamics (cycle 5)']]),
                                    np.concatenate(heritability_comm['Community Function']).T]).T,
                                    columns=["Species pool",'Predicted dynamics (cycle 5)',
                                             "Cycle","Community Function Heritability"])   

heritability_comm_df['Cycle'] = heritability_comm_df['Cycle'].astype(float).astype(int)
heritability_comm_df['Community Function Heritability'] = \
    heritability_comm_df['Community Function Heritability'].astype(float)
heritability_comm_df.replace({'Species pool':{'chaos':'1','oscillations':'2','stable':'3'}},
                       inplace=True)

#### Plotting ####

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

ax = sns.lineplot(data=heritability_comm_df,
             x="Cycle",y="Community Function Heritability",hue="Predicted dynamics (cycle 5)",
             style='Species pool',
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Heritability of Community Function',fontsize=14)
plt.title('Heritability community function across ecological dynamics',fontsize=18)
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.savefig("Figures/heritability_of_communityfunction.png", dpi=300, bbox_inches='tight')

################################ Plotting raw simulations #############################

colours = plt.cm.jet(np.linspace(0,1,50))

################

############################## Plotting them all together ######################################

fig = plt.figure(constrained_layout=True,figsize=(20,8))
fig.suptitle('Community dynamics during maturation (cycle 8)',fontsize=20)
fig.supxlabel('time (t)',fontsize=16)
fig.supylabel('Species abundance',fontsize=16)

subfigs = fig.subfigures(nrows=1, ncols=3)

subfigs[0].suptitle('Species pool 1',fontsize=16)
subfigs[1].suptitle('Species pool 2',fontsize=16)
subfigs[2].suptitle('Species pool 3',fontsize=16)

ax1 = subfigs[0].subplots(nrows=2, ncols=1)
ax2 = subfigs[1].subplots(nrows=2, ncols=1)
ax3 = subfigs[2].subplots(nrows=2, ncols=1)

for i in range(50):
    
    ax1[0].plot(community_selection_chaos['cycle 8'].ODE_sols['lineage 12'].t,
             community_selection_chaos['cycle 8'].ODE_sols['lineage 12'].y[i,:].T,
             color=colours[i])

ax1[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax1[0].set_xlim(-100,7100)

for i in range(50):
    
    ax1[1].plot(community_selection_chaos['cycle 8'].ODE_sols['lineage 1'].t,
             community_selection_chaos['cycle 8'].ODE_sols['lineage 1'].y[i,:].T,
             color=colours[i])

ax1[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax1[1].set_xlim(-100,7100)

for i in range(50):
    
    ax2[0].plot(community_selection_oscillations['cycle 8'].ODE_sols['lineage 18'].t,
             community_selection_oscillations['cycle 8'].ODE_sols['lineage 18'].y[i,:].T,
             color=colours[i])

ax2[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2[0].set_xlim(-100,7100)

for i in range(50):
    
    ax2[1].plot(community_selection_oscillations['cycle 8'].ODE_sols['lineage 1'].t,
             community_selection_oscillations['cycle 8'].ODE_sols['lineage 1'].y[i,:].T,
             color=colours[i])

ax2[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2[1].set_xlim(-100,7100)

for i in range(50):
    
    ax3[0].plot(community_selection_stable['cycle 8'].ODE_sols['lineage 9'].t,
             community_selection_stable['cycle 8'].ODE_sols['lineage 9'].y[i,:].T,
             color=colours[i])

ax3[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax3[0].set_xlim(-100,7100)

for i in range(50):
    
    ax3[1].plot(community_selection_stable['cycle 8'].ODE_sols['lineage 10'].t,
             community_selection_stable['cycle 8'].ODE_sols['lineage 10'].y[i,:].T,
             color=colours[i])

ax3[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax3[1].set_xlim(-100,7100)

plt.savefig("Figures/community_dynamics_cycle8.png", dpi=300, bbox_inches='tight')

##########################################################################################

################ Chaos/Species pool 1 ###########

np.unique(community_selection_dynamics.iloc[np.where((community_selection_dynamics['Species pool'] == 'chaos') \
                                  & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'chaotic-like'))]['Lineage'])

np.unique(community_selection_dynamics.iloc[np.where((community_selection_dynamics['Species pool'] == 'chaos') \
                                  & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'stable'))]['Lineage'])
    
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=False)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)
fig.suptitle('Community dynamics during maturation (Species pool 1, cycle 8)',fontsize=18)

for i in range(50):
    
    ax1.plot(community_selection_chaos['cycle 8'].ODE_sols['lineage 12'].t,
             community_selection_chaos['cycle 8'].ODE_sols['lineage 12'].y[i,:].T,
             color=colours[i])
    
ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax1.set_xlim(-100,7100)

for i in range(50):
    
    ax2.plot(community_selection_chaos['cycle 8'].ODE_sols['lineage 1'].t,
             community_selection_chaos['cycle 8'].ODE_sols['lineage 1'].y[i,:].T,
             color=colours[i])

ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2.set_xlim(-100,7100)

plt.savefig("Figures/community_dynamics_chaos_cycle8.png", dpi=300, bbox_inches='tight')

################ Oscillations/Species pool 2 ###########

np.unique(community_selection_dynamics.iloc[np.where((community_selection_dynamics['Species pool'] == 'oscillations') \
                                  & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'chaotic-like'))]['Lineage'])

np.unique(community_selection_dynamics.iloc[np.where((community_selection_dynamics['Species pool'] == 'oscillations') \
                                  & (community_selection_dynamics['Predicted dynamics (cycle 5)'] == 'oscillations'))]['Lineage'])
    
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=False)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)
fig.suptitle('Community dynamics during maturation (Species pool 2, cycle 8)',fontsize=18)

for i in range(50):
    
    ax1.plot(community_selection_oscillations['cycle 8'].ODE_sols['lineage 18'].t,
             community_selection_oscillations['cycle 8'].ODE_sols['lineage 18'].y[i,:].T,
             color=colours[i])
    
ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax1.set_xlim(-100,7100)

for i in range(50):
    
    ax2.plot(community_selection_oscillations['cycle 8'].ODE_sols['lineage 1'].t,
             community_selection_oscillations['cycle 8'].ODE_sols['lineage 1'].y[i,:].T,
             color=colours[i])

ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2.set_xlim(-100,7100)

plt.savefig("Figures/community_dynamics_oscillations_cycle8.png", dpi=300, bbox_inches='tight')


################ Stable/Species pool 3 ###########

# phenotype 1 = lineage 9
# phenotype 2 = else


fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=False)
fig.supxlabel('time (t)',fontsize=14)
fig.supylabel('Species abundance',fontsize=14)
fig.suptitle('Community dynamics during maturation (Species pool 3, cycle 8)',fontsize=18)

for i in range(50):
    
    ax1.plot(community_selection_stable['cycle 8'].ODE_sols['lineage 9'].t,
             community_selection_stable['cycle 8'].ODE_sols['lineage 9'].y[i,:].T,
             color=colours[i])
    
ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax1.set_xlim(-100,7100)


for i in range(50):
    
    ax2.plot(community_selection_stable['cycle 8'].ODE_sols['lineage 10'].t,
             community_selection_stable['cycle 8'].ODE_sols['lineage 10'].y[i,:].T,
             color=colours[i])

ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
ax2.set_xlim(-100,7100)

plt.savefig("Figures/community_dynamics_stable_cycle8.png", dpi=300, bbox_inches='tight')

