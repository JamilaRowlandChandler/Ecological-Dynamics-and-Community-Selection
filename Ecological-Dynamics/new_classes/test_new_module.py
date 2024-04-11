# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:40:03 2024

@author: jamil
"""

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

