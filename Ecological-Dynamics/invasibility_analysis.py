# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:49:48 2024

@author: Jamila
"""

##############################

# Home - cd Documents/PhD/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/
# Work - cd "Documents/PhD for github/Ecological-Dynamics-and-Community-Selection/Ecological-Dynamics/" 

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import itertools

from community_dynamics_and_properties_v2 import *

#################################################

####### Home #######

#community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_2.pkl")
#community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_2.pkl")
#community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_2.pkl")
#community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_2.pkl")

####### Work #######

community_dynamics_invasibility_005 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_005_2.pkl")
community_dynamics_invasibility_01 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_01_2.pkl")
community_dynamics_invasibility_015 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_015_2.pkl")
community_dynamics_invasibility_02 = pd.read_pickle("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/community_dynamics_invasibility_011_02_2.pkl")

#################################################

community_df_list = [community_object_to_df(community_obj,no_unique_compositions=False) \
                     for community_dict in [community_dynamics_invasibility_005,community_dynamics_invasibility_01,
                                            community_dynamics_invasibility_015,community_dynamics_invasibility_02] \
                         for community_interaction_dist in community_dict.values() \
                             for communities_no_species in community_interaction_dist.values() \
                                 for community_obj in communities_no_species]
    
communities_df = pd.concat(community_df_list,ignore_index=True)

communities_df['no_species'] = communities_df['no_species'].astype(int)

communities_df['survival_fraction'] = communities_df['diversity']/communities_df['no_species']

communities_df.to_csv("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/communities_no_species_interact_dist.csv")

##################################################

######### Home #########

#communities_df = pd.read_csv("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/communities_no_species_interact_dist.csv")

######### Work #########

communities_df = pd.read_csv("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Data/communities_no_species_interact_dist.csv",
                             index_col=0)

####################################################

def proportion_communities_invadable(data):
    
    return np.count_nonzero(data > 0.6)/len(data)

groupby_invasibilities_fluctprop = communities_df.groupby(['mu_a','sigma_a','no_species'])['invasibilities'].apply(proportion_communities_invadable)
groupby_invasibilities_fluctprop = groupby_invasibilities_fluctprop.to_frame()
groupby_invasibilities_fluctprop.reset_index(inplace=True)

groupby_invasibilities_005 = groupby_invasibilities_fluctprop.iloc[np.where(groupby_invasibilities_fluctprop['sigma_a'] == 0.05)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')
groupby_invasibilities_01 = groupby_invasibilities_fluctprop.iloc[np.where(groupby_invasibilities_fluctprop['sigma_a'] == 0.1)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')
groupby_invasibilities_015 = groupby_invasibilities_fluctprop.iloc[np.where(groupby_invasibilities_fluctprop['sigma_a'] == 0.15)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')
groupby_invasibilities_02 = groupby_invasibilities_fluctprop.iloc[np.where(groupby_invasibilities_fluctprop['sigma_a'] == 0.2)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.set_style('white')
norm = plt.Normalize(0,1)
s_m = plt.cm.ScalarMappable(cmap="magma", norm=norm)

fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle('Effect of species pool size and species interactions \n on the prevelence of chaotic-like invadable communities.',
             fontsize=14)
plt.subplots_adjust(top=0.775)
fig.supxlabel('Initial number of species',fontsize=12)
fig.supylabel('Average interaction strength',fontsize=12)

clb = plt.colorbar(s_m, ax=axs)
clb.ax.set_title('Proportion of communities \n with invasibility > 0.6',fontsize=10,
                 pad=7.5)

fig1 = sns.heatmap(groupby_invasibilities_005,ax=axs[0,0],vmin=0,vmax=1,cmap='magma',
                   cbar=False)
fig1.set(xlabel=None,ylabel=None)
fig1.set_title('std = 0.05',fontsize=10,pad=4)

fig2 = sns.heatmap(groupby_invasibilities_01,ax=axs[0,1],vmin=0,vmax=1,cmap='magma',
                   cbar=False)
fig2.set(xlabel=None,ylabel=None,title='0.1')
fig2.set_title('std = 0.1',fontsize=10,pad=4)

fig3 = sns.heatmap(groupby_invasibilities_015,ax=axs[1,0],vmin=0,vmax=1,cmap='magma',
                   cbar=False)
fig3.set(xlabel=None,ylabel=None,title='0.15')
fig3.set_title('std = 0.15',fontsize=10,pad=3)

fig4 = sns.heatmap(groupby_invasibilities_02,ax=axs[1,1],vmin=0,vmax=1,cmap='magma',
                   cbar=False)
fig4.set(xlabel=None,ylabel=None,title='0.2')
fig4.set_title('std = 0.2',fontsize=10,pad=3)

plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/no_species_interaction_propinvadable.png",
            dpi=300, bbox_inches='tight')

##################################

groupby_invasibilities_avg = communities_df.groupby(['mu_a','sigma_a','no_species'])['invasibilities'].mean()
groupby_invasibilities_avg = groupby_invasibilities_avg.to_frame()
groupby_invasibilities_avg.reset_index(inplace=True)

groupby_invasibilities_avg_005 = groupby_invasibilities_avg.iloc[np.where(groupby_invasibilities_avg['sigma_a'] == 0.05)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

groupby_invasibilities_avg_01 = groupby_invasibilities_avg.iloc[np.where(groupby_invasibilities_avg['sigma_a'] == 0.1)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

groupby_invasibilities_avg_015 = groupby_invasibilities_avg.iloc[np.where(groupby_invasibilities_avg['sigma_a'] == 0.15)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

groupby_invasibilities_avg_02 = groupby_invasibilities_avg.iloc[np.where(groupby_invasibilities_avg['sigma_a'] == 0.2)].pivot(index='mu_a',
                                                                    columns='no_species',values='invasibilities')

sns.set_style('white')
norm = plt.Normalize(0,1)
s_m = plt.cm.ScalarMappable(cmap="rocket", norm=norm)

fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
fig.suptitle('Effect of species pool size and species interactions \n on the average invasibility of communities.',
             fontsize=14)
plt.subplots_adjust(top=0.775)
fig.supxlabel('Initial number of species',fontsize=12)
fig.supylabel('Average interaction strength',fontsize=12)

clb = plt.colorbar(s_m, ax=axs)
clb.ax.set_title('Average \n community invasibility',fontsize=10,
                 pad=7.5)

fig1 = sns.heatmap(groupby_invasibilities_avg_005,ax=axs[0,0],vmin=0,vmax=1,cmap='rocket',
                   cbar=False)
fig1.set(xlabel=None,ylabel=None)
fig1.set_title('std = 0.05',fontsize=10,pad=4)

fig2 = sns.heatmap(groupby_invasibilities_avg_01,ax=axs[0,1],vmin=0,vmax=1,cmap='rocket',
                   cbar=False)
fig2.set(xlabel=None,ylabel=None,title='0.1')
fig2.set_title('std = 0.1',fontsize=10,pad=4)

fig3 = sns.heatmap(groupby_invasibilities_avg_015,ax=axs[1,0],vmin=0,vmax=1,cmap='rocket',
                   cbar=False)
fig3.set(xlabel=None,ylabel=None,title='0.15')
fig3.set_title('std = 0.15',fontsize=10,pad=3)

fig4 = sns.heatmap(groupby_invasibilities_avg_02,ax=axs[1,1],vmin=0,vmax=1,cmap='rocket',
                   cbar=False)
fig4.set(xlabel=None,ylabel=None,title='0.2')
fig4.set_title('std = 0.2',fontsize=10,pad=3)

plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/no_species_interaction_averageinvasivibility.png",
            dpi=300, bbox_inches='tight')

#################################################

sns.set_style('white')

cmap = mpl.cm.plasma_r
bounds = np.append(np.sort(np.unique(communities_df['no_species'])),52)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(4,10,sharex=True,sharey=True,figsize=(30,12),layout='constrained')
fig.suptitle('Effect of invasibility on community diversity',fontsize=20)
fig.supxlabel('Invasibility',fontsize=18)
fig.supylabel('Survival fraction',fontsize=18)

plt.gcf().text(0.94, 0.15, '0.2', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.39, '0.15', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.63, '0.1', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.87, '0.05', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.95, 'Std. in \n interaction \n strength', fontsize=12,
               horizontalalignment='center',verticalalignment='center')

clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,shrink=0.8,
                   pad=0.06)
clb.ax.set_title('Initial number \n of species',fontsize=12,
                 pad=7.5)

mu_as = np.unique(communities_df['mu_a'])
no_species_test = len(np.unique(communities_df['no_species']))
sigma_as = np.unique(communities_df['sigma_a'])

sigma_a_plot = np.repeat(sigma_as,len(mu_as))
mu_a_plot = np.tile(mu_as,len(sigma_as))

for i, ax in enumerate(axs.flat):
    
    subfig = sns.scatterplot(data=communities_df.iloc[np.where((communities_df['sigma_a'] == sigma_a_plot[i]) & \
                                                               (communities_df['mu_a'] == mu_a_plot[i]))],
                          x='invasibilities',y='survival_fraction',hue='no_species',
                          ax=ax,palette='plasma_r',hue_norm=norm,s=100)
    subfig.set(xlabel=None,ylabel=None)
    
    if i < 10:
        
        subfig.set_title('avg. interaction = ' + str(mu_a_plot[i]),fontsize=12,pad=4)
        
    ax.get_legend().remove()

plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/invasibility_survivalfraction_2.png",
            dpi=300,bbox_inches='tight')
        
#plt.savefig("C:/Users/Jamila/Documents/PhD/Data files and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/invasibility_survivalfraction_2.png",
#            dpi=300, bbox_inches='tight')

################################################

sns.set_style('white')

fig, axs = plt.subplots(4,10,sharex=True,sharey=True,figsize=(30,12),layout='constrained')
fig.suptitle('Effect of invasibility on community diversity',fontsize=20)
fig.supxlabel('Initial number of species',fontsize=18)
fig.supylabel('Survival fraction',fontsize=18)

plt.gcf().text(0.94, 0.15, '0.2', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.39, '0.15', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.63, '0.1', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.87, '0.05', fontsize=12,horizontalalignment='center',
               verticalalignment='center')
plt.gcf().text(0.94, 0.95, 'Std. in \n interaction \n strength', fontsize=12,
               horizontalalignment='center',verticalalignment='center')

#norm = plt.Normalize(0,1)
s_m = plt.cm.ScalarMappable(cmap="viridis_r", norm=mpl.colors.PowerNorm(gamma=1.5,vmin=0,vmax=1))

clb = plt.colorbar(s_m, ax=axs)
clb.ax.set_title('Community \n invasibility',fontsize=12,
                 pad=7.5)

mu_as = np.unique(communities_df['mu_a'])
sigma_as = np.unique(communities_df['sigma_a'])

sigma_a_plot = np.repeat(sigma_as,len(mu_as))
mu_a_plot = np.tile(mu_as,len(sigma_as))

for i, ax in enumerate(axs.flat):
    
    ax.scatter(x=communities_df.loc[np.where((communities_df['sigma_a'] == sigma_a_plot[i]) & \
                                    (communities_df['mu_a'] == mu_a_plot[i])),'no_species'],
                         y = communities_df.loc[np.where((communities_df['sigma_a'] == sigma_a_plot[i]) & \
                                    (communities_df['mu_a'] == mu_a_plot[i])),'survival_fraction'],
                         c = communities_df.loc[np.where((communities_df['sigma_a'] == sigma_a_plot[i]) & \
                                    (communities_df['mu_a'] == mu_a_plot[i])),'invasibilities'],
                         norm = mpl.colors.PowerNorm(gamma=1.5,vmin=0,vmax=1),cmap='viridis_r',
                         s = 50)

    if i < 10:
        
        ax.set_title('avg. interaction = ' + str(mu_a_plot[i]),fontsize=12,pad=4)
 
plt.savefig("C:/Users/jamil/Documents/Data and figures/Ecological-Dynamics-and-Community-Selection/Ecological Dynamics/Figures/nospecies_survivalfraction.png",
            dpi=300,bbox_inches='tight')
