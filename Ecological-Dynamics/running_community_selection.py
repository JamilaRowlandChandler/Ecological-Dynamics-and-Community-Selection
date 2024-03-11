# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:13:50 2024

@author: Jamila
"""
##############################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

###########################


interact_dist = {"mu_a":0.9,"sigma_a":0.05}

chaotic_interactions = np.load("chaos_09_005.npy")
oscillating_interactions = np.load("oscillations_09_005.npy")
stable_interactions = np.load("stable_09_005.npy")

#######

no_communities = 20
no_selection_cycles = 15
no_species = 50

############

community_selection_chaotic = community_selection_protocal1(no_communities,
                                                            no_selection_cycles,
                                                            no_species,
                                                            chaotic_interactions,
                                                            interact_dist)
community_selection_oscillating = community_selection_protocal1(no_communities,
                                                            no_selection_cycles,
                                                            no_species,
                                                            oscillating_interactions,
                                                            interact_dist)
community_selection_stable = community_selection_protocal1(no_communities,
                                                            no_selection_cycles,
                                                            no_species,
                                                            stable_interactions,
                                                            interact_dist)

community_selection_dynamics = []

for eco_dyn, community in zip(["chaos","oscillations","stable"],
                              [community_selection_chaotic,community_selection_oscillating,community_selection_stable]):

    for cycle in community.keys():
        
        if cycle != "Lyapunov Exponents":
        
            df = pd.DataFrame([community[cycle].diversity,community[cycle].fluctuations,
                                        community[cycle].final_composition,
                                        community[cycle].community_functions,
                                        community["Lyapunov Exponents"]["cycle 5"],
                                        community["Lyapunov Exponents"]["cycle 12"]]).T
            df.columns = ["Diversity","Fluctuations","Final Composition",
                          "Community Function","Lyapunov Exponent (cycle 5)",
                          "Lyapunov Exponent (cycle 12)"]
            df['Cycle'] = np.repeat(cycle,no_communities)
            df['No. Unique Compositions'] = np.repeat(community[cycle].no_unique_compositions,no_communities)
            df['Ecological Dynamics'] = np.repeat(eco_dyn,no_communities)
            
            community_selection_dynamics.append(df)

community_selection_dynamics_df = pd.concat(community_selection_dynamics)

community_selection_dynamics_df.reset_index(inplace=True)
community_selection_dynamics_df = community_selection_dynamics_df.rename(columns={"index":"Lineage"})
community_selection_dynamics_df["Lineage"] = \
    community_selection_dynamics_df["Lineage"].str.replace('lineage ','').astype(int)
community_selection_dynamics_df["Cycle"] = \
    community_selection_dynamics_df["Cycle"].str.replace('cycle ','').astype(int)
community_selection_dynamics_df['Community Function'] = \
    community_selection_dynamics_df['Community Function'].astype(float)
community_selection_dynamics_df.sort_values(by=["Ecological Dynamics","Cycle"],inplace=True)

community_selection_dynamics_df.to_csv("community_selection_dynamics2.csv")
pickle_dump('community_selection_chaotic.pkl',community_selection_chaotic)
pickle_dump('community_selection_oscillating.pkl',community_selection_oscillating)
pickle_dump('community_selection_stable.pkl',community_selection_stable)

##################

le_chaos = np.array([*community_selection_chaotic["Lyapunov Exponents"]["cycle 5"].values()])
le_oscillations = np.array([*community_selection_oscillating["Lyapunov Exponents"]["cycle 5"].values()])
le_stable = np.array([*community_selection_stable["Lyapunov Exponents"]["cycle 5"].values()])
    
identify_dynamics = pd.DataFrame(np.vstack([np.vstack((np.unique(community_selection_dynamics_df["Lineage"]),
                                                       le_array.T,
                                                       np.log10(le_array[:,1]/np.abs(le_array[:,0])))).T \
                                            for le_array in [le_chaos,le_oscillations,le_stable]]),
                                 columns=["Lineages","Lyapunov exponent mean",
                                          "Lyapunov exponent std error",
                                          "Normalised Lyapunov exponent std error"])
    
true_dynamics_chaos = ["chaos","stable","chaos","chaos","stable",
                       "stable","chaos","chaos","chaos","chaos",
                       "stable","chaos","chaos","chaos","chaos",
                       "chaos","stable","stable","chaos","chaos"]

true_dynamics_oscillations = ["chaos","oscillations","oscillations","oscillations","oscillations",
                              "oscillations","oscillations","oscillations","oscillations","oscillations",
                              "oscillations","oscillations","chaos","oscillations","oscillations",
                              "oscillations","oscillations","oscillations","chaos","chaos"]

true_dynamics_stable = ["stable","stable","stable","stable","stable",
                        "stable","stable","stable","stable","stable",
                        "stable","stable","stable","stable","stable",
                        "stable","stable","stable","stable","stable"]

identify_dynamics["True Dynamics"] = true_dynamics_chaos + true_dynamics_oscillations + \
    true_dynamics_stable

identify_dynamics.to_csv("True_ecological_dynamics_and_lyapunovs.csv")

identify_dynamics.groupby("True Dynamics")["Lyapunov exponent mean"].min()
identify_dynamics.groupby("True Dynamics")["Lyapunov exponent mean"].max()


plt.plot(community_selection_stable["cycle 5"].ODE_sols["lineage 19"].t,
         community_selection_stable["cycle 5"].ODE_sols["lineage 19"].y.T)

###################

heritability_comm = community_selection_dynamics_df.groupby(["Ecological Dynamics"])["Community Function"].apply(lambda x : heritability_during_selection(x,no_communities))
heritability_comm = heritability_comm.to_frame()
heritability_comm.reset_index(inplace=True)
heritability_comm_df = pd.DataFrame(np.vstack([np.concatenate([np.repeat(dyn,no_selection_cycles-1) \
                                                  for dyn in heritability_comm['Ecological Dynamics']]),
                                  np.concatenate(heritability_comm['Community Function']).T]).T,
                                  columns=["Ecological Dynamics","Cycle","Community Function Heritability"])
heritability_comm_df['Cycle'] = heritability_comm_df['Cycle'].astype(float).astype(int)
heritability_comm_df['Community Function Heritability'] = \
    heritability_comm_df['Community Function Heritability'].astype(float)
        
variation_comm = community_selection_dynamics_df.groupby(["Ecological Dynamics","Cycle"])["Community Function"].var()
variation_comm = variation_comm.to_frame()
variation_comm.reset_index(inplace=True)
variation_comm.rename(columns={"Community Function":"Variation in Community Function"},inplace=True)

sns.lineplot(heritability_comm_df,x="Cycle",y="Community Function Heritability",
             hue="Ecological Dynamics")

sns.lineplot(variation_comm,
             x="Cycle",y="Variation in Community Function",
             hue="Ecological Dynamics")

plt.plot(community_selection_oscillating['cycle 11'].ODE_sols["lineage 2"].t,
         community_selection_oscillating['cycle 11'].ODE_sols["lineage 2"].y.T)