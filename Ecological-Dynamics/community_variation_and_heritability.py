# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:07:21 2024

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
from ast import literal_eval

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

################################## Functions ##########################

################################ Tidying data structures #####################

le_across_interact_dists = pd.read_csv("testing_criteria_for_eco_dyn.csv")
community_selection_dynamics = pd.read_csv("community_selection_dynamics2.csv")

community_selection_chaotic = pd.read_pickle("community_selection_chaotic.pkl")
community_selection_oscillating = pd.read_pickle("community_selection_oscillating.pkl")
community_selection_stable = pd.read_pickle("community_selection_stable.pkl")

##################

# convert standard error to standard deviation
le_across_interact_dists["Lyapunov Exponent std error"] *= np.sqrt(50)
le_across_interact_dists.rename(columns={"Lyapunov Exponent std error":"Lyapunov Exponent std deviation"},
                                inplace=True)

#############

community_selection_dynamics["Lyapunov Exponent (cycle 5)"] = \
    community_selection_dynamics['Lyapunov Exponent (cycle 5)'].apply(lambda x: literal_eval(x) if "[" in x else x)
community_selection_dynamics["Lyapunov Exponent (cycle 12)"] = \
    community_selection_dynamics['Lyapunov Exponent (cycle 12)'].apply(lambda x: literal_eval(x) if "[" in x else x)

community_selection_dynamics[["Lyapunov Exponent mean (cycle 5)",
                              "Lyapunov Exponent std deviation (cycle 5)"]] = \
    pd.DataFrame(community_selection_dynamics['Lyapunov Exponent (cycle 5)'].to_list())
community_selection_dynamics[["Lyapunov Exponent mean (cycle 12)",
                              "Lyapunov Exponent std deviation (cycle 12)"]] = \
    pd.DataFrame(community_selection_dynamics['Lyapunov Exponent (cycle 12)'].to_list())

community_selection_dynamics.drop(['Lyapunov Exponent (cycle 5)','Lyapunov Exponent (cycle 12)'],
                                  axis=1,inplace=True)

# convert standard error to standard deviation
community_selection_dynamics["Lyapunov Exponent std deviation (cycle 5)"] *= np.sqrt(50)
community_selection_dynamics["Lyapunov Exponent std deviation (cycle 12)"] *= np.sqrt(50)

############################# Assigning ecological dynamics ##################

identify_ecological_dynamics(community_selection_dynamics,
                             "Lyapunov Exponent mean (cycle 5)",
                             "Lyapunov Exponent std deviation (cycle 5)",
                             predicted_dynamics_col="Predicted dynamics (cycle 5)")

identify_ecological_dynamics(community_selection_dynamics,
                             "Lyapunov Exponent mean (cycle 12)",
                             "Lyapunov Exponent std deviation (cycle 12)",
                             predicted_dynamics_col="Predicted dynamics (cycle 12)")

############################## Assessing heritability and variation #############

############ Heritability ##########

no_cycles = 15

heritability_comm = community_selection_dynamics.groupby(["Ecological Dynamics","Predicted dynamics (cycle 5)"])\
    ["Community Function"].apply(lambda x : heritability_during_selection(x,no_cycles))
heritability_comm = heritability_comm.to_frame()
heritability_comm.reset_index(inplace=True)

groups, g_counts = np.unique(heritability_comm["Ecological Dynamics"],return_counts=True)

heritability_comm_df = pd.DataFrame( \
                                    np.vstack([\
                                    np.concatenate([np.repeat(group,(g_count*(no_cycles-1)).astype(int)) for group, g_count in zip(groups,g_counts)]),
                                    np.concatenate([np.repeat(dyn,no_cycles-1) for dyn in heritability_comm['Predicted dynamics (cycle 5)']]),
                                    np.concatenate(heritability_comm['Community Function']).T]).T,
                                    columns=["Species Pool","Ecological dynamics",
                                             "Cycle","Community Function Heritability"])   

heritability_comm_df['Cycle'] = heritability_comm_df['Cycle'].astype(float).astype(int)
heritability_comm_df['Community Function Heritability'] = \
    heritability_comm_df['Community Function Heritability'].astype(float)

#### Plotting ####

sns.lineplot(data=heritability_comm_df.iloc[np.where(heritability_comm_df["Species Pool"] == "stable")],
             x="Cycle",y="Community Function Heritability",hue="Ecological dynamics")

sns.lineplot(data=heritability_comm_df.iloc[np.where(heritability_comm_df["Species Pool"] == "oscillations")],
             x="Cycle",y="Community Function Heritability",hue="Ecological dynamics")

sns.lineplot(data=heritability_comm_df.iloc[np.where(heritability_comm_df["Species Pool"] == "chaos")],
             x="Cycle",y="Community Function Heritability",hue="Ecological dynamics")

########### Variation ########

variation_comm = community_selection_dynamics.groupby(["Ecological Dynamics",
                                                       "Predicted dynamics (cycle 5)",
                                                       "Cycle"])["Community Function"].var()
variation_comm = variation_comm.to_frame()
variation_comm.reset_index(inplace=True)
variation_comm.rename(columns={"Ecological Dynamics":"Species Pool",
                               "Predicted dynamics (cycle 5)":"Ecological dynamics",
                               "Community Function":"Variation in Community Function"},inplace=True)

#### Plotting ####

sns.lineplot(data=variation_comm.iloc[np.where(variation_comm["Species Pool"] == "stable")],
             x="Cycle",y="Variation in Community Function",hue="Ecological dynamics")

sns.lineplot(data=variation_comm.iloc[np.where(variation_comm["Species Pool"] == "oscillations")],
             x="Cycle",y="Variation in Community Function",hue="Ecological dynamics")

sns.lineplot(data=variation_comm.iloc[np.where(variation_comm["Species Pool"] == "chaos")],
             x="Cycle",y="Variation in Community Function",hue="Ecological dynamics")

############################ Tidy dataframe then re-write ######################

community_selection_dynamics.drop(community_selection_dynamics.columns[0],
                                  axis=1,
                                  inplace=True)

community_selection_dynamics.rename(columns={"Ecological Dynamics":"Species pool"},
                                    inplace=True)

list(community_selection_dynamics.columns.values)
new_cols = ["Species pool","Predicted dynamics (cycle 5)","Predicted dynamics (cycle 12)",
            "No. Unique Compositions","Cycle","Lineage",
            "Community Function","Diversity","Final Composition",
            'Lyapunov Exponent mean (cycle 5)','Lyapunov Exponent std deviation (cycle 5)',
            'Lyapunov Exponent mean (cycle 12)','Lyapunov Exponent std deviation (cycle 12)',
            "Fluctuations"]
community_selection_dynamics = community_selection_dynamics[new_cols]

community_selection_dynamics.to_csv("community_selection_dynamics2.csv")

####################### Other plotting #####################

sns.scatterplot(data=community_selection_dynamics.iloc[np.where(community_selection_dynamics["Species pool"] == "stable")],
                x="Cycle",y="Community Function",hue="Predicted dynamics (cycle 12)")

sns.scatterplot(data=community_selection_dynamics.iloc[np.where(community_selection_dynamics["Species pool"] == "oscillations")],
                x="Cycle",y="Community Function",hue="Predicted dynamics (cycle 12)")

sns.scatterplot(data=community_selection_dynamics.iloc[np.where(community_selection_dynamics["Species pool"] == "chaos")],
                x="Cycle",y="Community Function",hue="Predicted dynamics (cycle 12)")


