# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:41:56 2024

@author: Jamila
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
from scipy.stats import linregress
from scipy.stats import pearsonr
import itertools

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.discrete.discrete_model import MNLogit

from community_dynamics_and_properties_v2 import *

#################################################

community_dynamics_dict = pd.read_pickle('community_dynamics_different_no_species.pkl')
community_dynamics_df = pd.read_csv('community_dynamics_different_no_species.csv',index_col=False)

min_species = 4
max_species = 50
no_species_to_test = np.arange(min_species,max_species,3)

no_communities = 6
no_lineages = 5
interaction_distribution = {'mu_a':0.9,'sigma_a':0.15}
t_end = 10000

community_dynamics_with_invasibility = {}

for no_species, community_per_species in community_dynamics_dict.items():
    
    print(str(no_species),end='\n')
    community_no_spec = []
    
    for i, community_i in enumerate(community_per_species):
        
        community_extracted = deepcopy(community_i)
        new_community_simulations = community(community_extracted.no_species,
                                            'fixed',None,None,
                                            {'mu_a':community_extracted.mu_a,'sigma_a':community_extracted.sigma_a},
                                            usersupplied_interactmat=community_extracted.interaction_matrix)
        new_community_simulations.simulate_community(t_end,'Default',np.arange(no_lineages),
                                              init_cond_func_name='Mallmin')
        
        community_no_spec.append(new_community_simulations)
        print('Community ',str(i),' simulations are complete.',end='\n')
        
    community_dynamics_with_invasibility[str(no_species)] = community_no_spec

################################

no_species_column = np.repeat(no_species_to_test,no_communities*no_lineages)
community_column = np.tile(np.repeat(np.arange(no_communities),no_lineages),len(no_species_to_test))
lineage_column = np.tile(np.tile(np.arange(no_lineages),no_communities),len(no_species_to_test))
invasibility_column = []
diversity_column = []

#################### Dataframe construction and analysis ##################

for no_species, community_per_species in community_dynamics_with_invasibility.items():
    
    for i, community_i in enumerate(community_per_species):
        
        community_extracted = deepcopy(community_i)
        
        diversity_column.append(list(community_extracted.diversity.values()))
        invasibility_column.append(list(community_extracted.invasibilities.values()))

community_dynamics_invasibility_df = pd.DataFrame(np.stack((no_species_column,community_column,
                                               lineage_column,np.concatenate(invasibility_column),
                                               np.concatenate(diversity_column))).T,
                                     columns=['No_Species','Community','Lineage',
                                              'Invasibility','Diversity'])

community_dynamics_invasibility_df.to_csv('community_dynamics_with_invasibility.csv')
pickle_dump('community_dynamics_with_invasibility.pkl',community_dynamics_with_invasibility)

sns.set_style('white')
norm = plt.Normalize(community_dynamics_invasibility_df['Invasibility'].min(),
                     community_dynamics_invasibility_df['Invasibility'].max())
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
ax = sns.scatterplot(community_dynamics_invasibility_df,x='No_Species',y='Diversity',
                hue='Invasibility',palette=sns.color_palette("viridis_r",n_colors=100,as_cmap=True))
plt.xlabel('Initial number of species',fontsize=14)
plt.ylabel('Species diversity at the \n end of simulations',fontsize=14)
ax.get_legend().remove()
clb = plt.colorbar(sm, ax=ax)
clb.ax.set_title('Invasibility',pad=6)
plt.title('Effect of invasibility on species diversity',fontsize=16,pad=20)

###########################

community_dynamics_invasibility_df2 = pd.read_csv('community_dynamics_with_invasibility.csv',
                                                 index_col=False)
community_dynamics_with_invasibility2 = pd.read_pickle('community_dynamics_with_invasibility.pkl')

sns.set_style('white')
norm = plt.Normalize(community_dynamics_invasibility_df2['Invasibility'].min(),
                     community_dynamics_invasibility_df2['Invasibility'].max())
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
ax = sns.scatterplot(community_dynamics_invasibility_df2,x='No_Species',y='Diversity',
                hue='Invasibility',palette=sns.color_palette("viridis_r",n_colors=100,as_cmap=True))
plt.xlabel('Initial number of species',fontsize=14)
plt.ylabel('Species diversity at the \n end of simulations',fontsize=14)
ax.get_legend().remove()
clb = plt.colorbar(sm, ax=ax)
clb.ax.set_title('Invasibility',pad=6)
plt.title('Effect of invasibility on species diversity',fontsize=16,pad=20)

plt.savefig("Figures/diversity_no_species_invasibility.png", dpi=300, bbox_inches='tight')

#######################

sns.scatterplot(community_dynamics_invasibility_df,x='No_Species',y='Diversity',
                hue='Invasibility')

res_diversity = ols('Diversity ~ No_Species + Invasibility',data=community_dynamics_invasibility_df).fit()
print(res_diversity.summary())
print(anova_lm(res_diversity))

diversity_predicted = res_diversity.predict(community_dynamics_invasibility_df[['No_Species','Invasibility']])

sns.scatterplot(x=community_dynamics_invasibility_df['Diversity'],y=diversity_predicted)

pearsonr(community_dynamics_invasibility_df['Diversity'],diversity_predicted)

sns.scatterplot(community_dynamics_invasibility_df,x='No_Species',y='Diversity',
                hue='Invasibility')
plt.plot(community_dynamics_invasibility_df['No_Species'],diversity_predicted,'x')
