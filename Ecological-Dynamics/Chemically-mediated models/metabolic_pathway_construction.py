# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:57:35 2024

@author: jamil
"""

import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

###########################################################

def metabolic_network(no_resources,no_external_resources,
                      expected_no_edges):
    
    metabolism_matrix = np.zeros((no_resources,no_resources))
    
    lower_triangular_no_diagonal = np.tril_indices_from(metabolism_matrix,-1)
    only_byproducts = lower_triangular_no_diagonal[0] > no_external_resources - 1
    triangle_byproducts = (lower_triangular_no_diagonal[0][only_byproducts],
                           lower_triangular_no_diagonal[1][only_byproducts])
    
    no_nodes_in_triangle = len(triangle_byproducts[0])
    p = expected_no_edges/triangle_byproducts
    edges = np.random.binomial(1, p, size = no_nodes_in_triangle)

    metabolism_matrix[triangle_byproducts] = edges
    
    return metabolism_matrix

def microbe_resource_interactions(no_resources,no_external_resources,no_species,
                                  metabolism_matrix,
                                  expected_consumption_per_resource):
    
    consumption_matrix = np.zeros((no_species,no_resources))
    production_matrix = np.zeros((no_species,no_resources))
    
    def species_metabolic_pathway(spec,i,
                                  metabolism_matrix,probability_consumption,
                                  no_external_resources):
         
        resource_consumed = np.random.binomial(1, probability_consumption)
        consumption_matrix[spec,i] = resource_consumed
        
        if resource_consumed == 1:
            
            producable_byproducts = \
                np.nonzero(metabolism_matrix[no_external_resources:,i])[0]
                
            if producable_byproducts.size > 0:
            
                produced_byproduct = np.random.choice(producable_byproducts)
                production_matrix[spec,no_external_resources+produced_byproduct] = 1
    
    for spec in range(no_species):
    
        for i in range(no_resources):
            
            if production_matrix[spec,i] == 0:
            
                probability_consumption = \
                    expected_consumption_per_resource[i]/no_species
                
                species_metabolic_pathway(spec,i,metabolism_matrix,probability_consumption,
                                          no_external_resources)
                
    return consumption_matrix, production_matrix


def growth_rate_per_resource_tradeoffs(max_total_growth,consumption_matrix):
    
    def decomposition(max_total_growth,no_consumed_resources):
        
        growth_left = max_total_growth
        
        for i in no_consumed_resources:
            
            g_r = np.random.uniform(0,growth_left)
            
            yield g_r
            growth_left -= g_r
        
    np.array([list(decomposition())])
    
    pass

def dSR_dt(t,var,
           no_spec,
           influx,dilution):
    
    S = var[:no_spec]
    R = var[no_spec:]
    
    growth = max_growth * R
    
    S * (np.sum(growth * R, axis = 1))
    
    dR = influx - dilution*R - consumption * R
    
    pass





