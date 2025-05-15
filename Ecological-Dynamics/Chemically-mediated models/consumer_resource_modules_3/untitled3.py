# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:08:45 2025

@author: jamil

"""

import os

os.chdir("C:/Users/jamil/Documents/PhD/GitHub projects/Ecological-Dynamics-and-Community-Selection/" + \
         "Ecological-Dynamics/Chemically-mediated models/consumer_resource_modules_3")

from new_models import Consumer_Resource_Model
from community_level_properties import max_le

# %%

community = Consumer_Resource_Model("Self-limiting resource supply", 100, 100)

community.growth_consumption_rates('coupled by rho',
                                   1, 0.1, 1, 0.1, rho = 1)
community.model_specific_rates('constant', {'death' : 1},
                               'constant', {'K' : 1})

community.simulate_community(3500, 1)

community.calculate_community_properties()
community.lyapunov_exponent = max_le(community, 500, community.ODE_sols[0].y[:, -1],
                                     1e-3, dt = 20, separation = 1e-3)
