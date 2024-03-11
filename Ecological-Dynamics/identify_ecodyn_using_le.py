# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:00:57 2024

@author: jamil
"""

######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from copy import deepcopy

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

###########################

def lyapunov_exponents_for_many_communities(no_lineages,no_species,
                                            growth_func_name,growth_args,
                                            interact_func_name,interact_args):
    
    lineages = np.arange(no_lineages)
    
    comm_dynamics = community(no_species,
                                  growth_func_name, growth_args,
                                  interact_func_name, interact_args)
    comm_dynamics.simulate_community("Default",lineages,
                                     init_cond_func_name="Mallmin")

    for cycle in range(2):
        
        initial_conditions = np.vstack([comm_dynamics.ODE_sols[lineage].y[:,-1] \
                  for lineage in comm_dynamics.ODE_sols.keys()]).T
        
        comm_dynamics.simulate_community("Supply initial conditions",
                                         lineages,
                                         array_of_init_conds=initial_conditions)
    
    comm_dynamics.repeat_lyapunov(lineages)
    
    le_df = pd.DataFrame(deepcopy(comm_dynamics.lyapunov_exponents)).transpose().reset_index()
    le_df.columns = ["Lineage","Lyapunov Exponent mean","Lypunov Exponent std error"]
    
    comm_simulations = deepcopy(comm_dynamics.ODE_sols)
    
    return le_df, comm_simulations

##############################

le_communities = pd.read_csv("True_ecological_dynamics_and_lyapunovs.csv")
community_selection_dynamics = pd.read_csv("community_selection_dynamics2.csv")

# Note - I initially could not load these. To resolve, in git bash, do: git lfs pull
community_selection_chaotic = pd.read_pickle("community_selection_chaotic.pkl")
community_selection_oscillating = pd.read_pickle("community_selection_oscillating.pkl")
community_selection_stable = pd.read_pickle("community_selection_stable.pkl")

#######################

# stable communities are easy to identify - mean(lambda) < -1
le_overlapping_communities = \
    le_communities.drop(le_communities[le_communities["True Dynamics"] == "stable"].index)

# leaves only oscillating communities
le_overlapping_communities.drop(le_overlapping_communities[\
        np.round(le_overlapping_communities['Lyapunov exponent std error'],1) >= 0.1].index)["True Dynamics"]

# leaves only chaotic communities
le_overlapping_communities.drop(le_overlapping_communities[\
        np.round(le_overlapping_communities['Lyapunov exponent std error'],1) < 0.1].index)["True Dynamics"]

#plt.plot(community_selection_oscillating["cycle 1"].ODE_sols["lineage 11"].t,
#         community_selection_oscillating["cycle 1"].ODE_sols["lineage 11"].y.T)

#####################

no_communities = 5
no_lineages = 5
no_species = 50

interact_dists = [{"mu_a":0.2,"sigma_a":0.05},
                  {"mu_a":0.3,"sigma_a":0.15},
                  {"mu_a":0.7,"sigma_a":0.15},
                  {"mu_a":0.9,"sigma_a":0.05},
                  {"mu_a":1.1,"sigma_a":0.1}]

###############

le_across_interact_dists = []
comm_sim_dict = {}

for count, interact_dist in enumerate(interact_dists):
    
    comm_sim_subdict = {"Interaction Distribution":interact_dist}

    for comm in range(no_communities):
    
        le_df, comm_simulations = lyapunov_exponents_for_many_communities(no_lineages,
                                                                          no_species,
                                                                          "fixed",
                                                                          None,
                                                                          "random",
                                                                          interact_dist)
        le_across_interact_dists.append(le_df)
        comm_sim_subdict["Community " + str(comm)] = comm_simulations
        
    comm_sim_dict[count] = comm_sim_subdict

le_across_interact_dists2 = pd.concat(le_across_interact_dists)
le_across_interact_dists2.rename(columns={"Lypunov Exponent std error":"Lyapunov Exponent std error"},
                                 inplace=True)

eco_dyn = ["stable","oscillations","chaos"]
eco_dyn_conditions = [(le_across_interact_dists2["Lyapunov Exponent mean"] <= -1),
                      (le_across_interact_dists2["Lyapunov Exponent mean"] > -1) & \
                      (np.round(le_across_interact_dists2["Lyapunov Exponent std error"],1) < 0.1),
                      (le_across_interact_dists2["Lyapunov Exponent mean"] > -1) & \
                      (np.round(le_across_interact_dists2["Lyapunov Exponent std error"],1) >= 0.1)]
      
le_across_interact_dists2["Predicted dynamics"] = np.select(eco_dyn_conditions,eco_dyn)
le_across_interact_dists2["True dynamics"] = np.nan
le_across_interact_dists2.reset_index(drop=True,inplace=True)

le_across_interact_dists2.to_csv("testing_criteria_for_eco_dyn.csv")
pickle_dump("testing_criteria_for_eco_dyn_sim.pkl", comm_sim_dict)

data_as_list = list(comm_sim_dict[3]["Community 4"].values())

fig, axs = plt.subplots(3,2,figsize=(5,2))

for i, ax in enumerate(axs.flatten()):
    
    ax.plot(data_as_list[i].t,data_as_list[i].y.T)
    ax.set_title("Lineage " + str(i))
    
plt.tight_layout()
plt.show()

plt.close()
plt.plot(data_as_list[3].t,data_as_list[3].y.T)

plt.close()

le_across_interact_dists2.loc[95:99,"True dynamics"] = ["stable",
                                                        "stable",
                                                        "stable",
                                                        "stable",
                                                        "stable"]

####################

le_filled = le_across_interact_dists2.iloc[0:100]
np.sum(le_filled["Predicted dynamics"] == le_filled["True dynamics"])/le_filled.shape[0]


    
