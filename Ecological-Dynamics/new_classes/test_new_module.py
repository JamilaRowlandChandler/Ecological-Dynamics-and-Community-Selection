# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:40:03 2024

@author: jamil
"""

# cd C:\Users\jamil\Documents\PhD\GitHub projects\Ecological-Dynamics-and-Community-Selection\Ecological-Dynamics\new_classes

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from model_classes import gLV

gLV_test = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = 'random', interact_args = {'mu_a':0.9,'sigma_a':0.15})
gLV_test.simulate_community(np.arange(5), t_end = 10000)
gLV_test.calculate_community_properties(np.arange(5))

gLV_test_sparse = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = 'sparse',
               interact_args = {'mu_a':0.9,'sigma_a':0.15,'connectance':0.7})

gLV_test_modular = gLV(no_species = 50, growth_func = 'fixed', growth_args = None,
               interact_func = 'modular',
               interact_args = {'no_modules':3,'p_mu_a':0.9,'p_sigma_a':0.15,
                                'p_connectance':1,'q_mu_a':0.6,'q_sigma_a':0.15,
                                'q_connectance':0.1})

w = (0.4 + 0.1*np.random.randn(10000))
weak_interaction_dist = ((w-0.5)**2)/w

s = (2 + 0.1*np.random.randn(10000))
strong_interaction_dist = ((s-2)**2)/s

label = np.repeat(['weak','strong'],10000)
df = pd.DataFrame([label,np.concatenate((weak_interaction_dist,strong_interaction_dist))]).T
df.columns = ['label','value']

sns.histplot(data=df,x='value',hue='label')
plt.xlim(-0.01,0.15)
plt.ylim(0,3000)

s1 = (2 + 0.1*np.random.randn(10000))
s2 = (2 + 0.1*np.random.randn(10000))

w1 =(0.4 + 0.1*np.random.randn(10000))
w2 = (0.4 + 0.1*np.random.randn(10000))

ss_i_dist = (s1 + s2)**2/s1
ss_j_dist = (s1 + s2)**2/s2

ww_i_dist = (w1 + w2)**2/w1
ww_j_dist = (w1 + w2)**2/w2

ws_i_dist = (w1 + s1)**2/w1
ws_j_dist = (w1 + s1)**2/s1

label2 = np.repeat(['weak (weak)','strong (strong)','weak (strong)','strong (weak)'],10000)
df2 = pd.DataFrame([label2,np.concatenate((ww_i_dist,ss_i_dist,ws_i_dist,ws_j_dist))]).T
df2.columns = ['label','value']
sns.histplot(data=df2,x='value',hue='label')
plt.xlim(-0.01,30)


###

ss_i_dist = (s1 - s2)/s1
ww_i_dist = (w1 - w2)/w1
ws_i_dist = (w1 - s1)/w1
ws_j_dist = (w1 - s1)/s1

label2 = np.repeat(['weak (weak)','strong (strong)','weak (strong)','strong (weak)'],10000)
df2 = pd.DataFrame([label2,np.concatenate((ww_i_dist,ss_i_dist,ws_i_dist,ws_j_dist))]).T
df2.columns = ['label','value']
sns.histplot(data=df2,x='value',hue='label')
plt.xlim(-10,1)

###################

growth_rates = np.linspace(0.01, 1.3, num = 50)

probability_effect = (growth_rates - np.min(growth_rates))/(np.max(growth_rates) - np.min(growth_rates))

are_species_interacting = \
    np.random.binomial(1,np.tile(probability_effect,50),size=50*50).reshape((50,50))

probability_cooperation = np.ones((50,50))

for i in range(50):
    
    for j in range(50):
        
        probability_cooperation[i,j] = ((growth_rates[i] - growth_rates[j])**2)/(growth_rates[i] * growth_rates[j])
        
probability_cooperation[probability_cooperation > 1] = 1
probability_cooperation = probability_cooperation.flatten()

cooperation_matrix = \
    np.random.binomial(1,probability_cooperation,size=50*50).reshape((50,50))
cooperation_matrix[np.where(cooperation_matrix == 0)] = -1

are_species_interacting[np.where(are_species_interacting == 1)] = \
    cooperation_matrix[np.where(are_species_interacting == 1)]

p_c = probability_cooperation.reshape((50,50))

#species = np.arange(1,self.no_species+1)

# calculate node weights, used to calculate the probability species i interacts with j.
#weights = \
#    self.average_degree*((beta-2)/(beta-1))*((self.no_species/species)**(1/(beta-1)))

# calculate the probability species i interacts with j.
#probability_of_interactions = \
#    (np.outer(weights,weights)/np.sum(weights)).flatten()

# set probabilities > 1 to 1.
#probability_of_interactions[probability_of_interactions > 1] = 1
