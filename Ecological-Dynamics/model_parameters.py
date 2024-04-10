# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:13:21 2024

@author: jamil
"""

import numpy as np
from scipy import linalg

############## Growth Rates ###########

def growth_rates_norm(parameters_object):
    
    '''
    
    Draw growth rates for n species from normal(mu,sigma) distribution

    Parameters
    ----------
    mu_g : float
        Mean growth rate.
    sigma_g : float
        Standard deviation in growth rate.
    no_species : int
        Number of species (n).

    Returns
    -------
    growth_r : np.array of float64.
        array of growth rates for each species drawn from normal(mu_g,sigma_g).

    '''
    
    growth_r = parameters_object.mu_g + parameters_object.sigma_g*np.random.rand(parameters_object.no_species)
    
    return growth_r

def growth_rates_fixed(parameters_object):
    
    '''
    
    Generate array of growth rates all fixed to 1.
    
    Parameters
    ----------
    no_species : int
        number of species.

    Returns
    -------
    growth_r : np.array of float64.
        array of growth rates, all entries = 1.0.

    '''
    
    growth_r = np.ones((parameters_object.no_species,))
    
    return growth_r

    
##################### Interaction Matrix #####################

########## Sparse and dense random interaction matrices #########
        
def random_interaction_matrix(mu_a,sigma_a,no_species):
    
    '''
    
    Generate a classic, dense, random interaction matrix.

    Parameters
    ----------
    mu_a : float
        mean interaction strength.
    sigma_a : float
        interaction strength standard deviation.
     no_species : int
         number of species (n).

    Returns
    -------
    interact_mat : np.array of size (n,n)
        Interaction matrix. 

    '''
    
    # generate interaction matrix drawn from normal(mu_a,sigma_a)
    interact_mat = mu_a + sigma_a*np.random.randn(no_species,no_species)
    # set a_ij = -1 for i = j/parameters_object-interaction to prevent divergence
    np.fill_diagonal(interact_mat, 1)
    
    return interact_mat

def sparse_interaction_matrix(mu_a,sigma_a,connectance,no_species):
    
    '''
    
    Generate a sparse random interaction matrix using a Erdős–Rnyi graph.
    
    See May (1972) for details. https://doi.org/10.1038/238413a0

    Returns
    -------
    interact_mat : np.array of size (n,n)
        Interaction matrix. 

    '''
    
    interact_mat =\
        interaction_matrix_with_connectance(no_species, mu_a, sigma_a, connectance)
    
    return interact_mat

def mixed_sparse_interaction_matrix(competitive_mu_a,competitive_sigma_a,
                                    cooperative_mu_a,cooperative_sigma_a,
                                    competitive_connectance,cooperative_connectance,
                                    probability_cooperative,
                                    no_species):
    
    # generate competitive interactions to start
    interact_mat = \
        interaction_matrix_with_connectance(no_species,competitive_mu_a,competitive_sigma_a,
                                                 competitive_connectance)
    
    cooperative_interaction_indices = \
        np.random.binomial(1,probability_cooperative,size=no_species*no_species).reshape((no_species,no_species))
    
    cooperative_interaction_matrix = \
        interaction_matrix_with_connectance(cooperative_interaction_indices.shape[0],
                                                 cooperative_mu_a,cooperative_sigma_a,
                                                 cooperative_connectance,
                                                 parameters_object_inhibition=False)
    
    interact_mat[np.where(interact_mat == cooperative_interaction_indices)] = \
        cooperative_interaction_matrix
        
    return interact_mat

############# Non-uniform structured interaction matrices ###########

def modular_interaction_matrix(no_species,no_modules,
                               p_mu_a,p_sigma_a,p_connectance,
                               q_mu_a,q_sigma_a,q_connectance,
                               module_probabilities):
    
    '''
    
    Generate a modular interaction matrix using a stochastic block model (SBM).
    
    See Akjouj et al. (2024) for details on how SBMs can be applied to gLVs.
    https://doi.org/10.1098/rspa.2023.0284

    Returns
    -------
    interact_mat : np.array of size (n,n)
        Interaction matrix. 

    '''
    
    ########### Cluster species into modules ##########

    clustered_species = np.random.multinomial(no_species,module_probabilities,
                                              size=1)[0]
  
    # create the interaction matrices for each module
    module_interactions = \
        [interaction_matrix_with_connectance(nodes,p_mu_a,p_sigma_a,p_connectance)
         for nodes in clustered_species]
    
    # combine module interactions into a community interaction matrix
    interact_mat = linalg.block_diag(*module_interactions)
    
    ####### Assign interactions between members of different groups ######
    
    # get indices of interaction matrix where species from different modules interact
    non_group_interaction_indices = np.where(interact_mat == 0)
    
    # generate the interactions between species from different modules
    non_group_interactions = \
        interaction_matrix_with_connectance(no_species,q_mu_a,q_sigma_a,q_connectance)    
    
    # add between-module interactions to the interaction matrix
    interact_mat[non_group_interaction_indices] = \
        non_group_interactions[non_group_interaction_indices]
    
    return interact_mat

def nested_interaction_matrix(mu_a,sigma_a,average_degree,no_species,beta=7):
    
    '''
    
    Created a nested, or scale-free, interaction matrix using the Chung-Lu model.
    
    See Akjouj et al. (2024) for details on how the Chung-Lu model can be
    applied to gLVs. https://doi.org/10.1098/rspa.2023.0284

    Parameters
    ----------
    beta : float, optional
        Scale parameter used to describe the probabiltiy node n has k nodes.
        The default is 7.

    Returns
    -------
    interact_mat : np.array of size (n,n)
        Interaction matrix. 

    '''
    
    # create i's
    species = np.arange(1,no_species+1)
    
    # calculate node weights, used to calculate the probability species i interacts with j.
    weights = \
        average_degree*((beta-2)/(beta-1))*((no_species/species)**(1/(beta-1)))
    
    # calculate the probability species i interacts with j.
    probability_of_interactions = \
        (np.outer(weights,weights)/np.sum(weights)).flatten()
    
    # set probabilities > 1 to 1.
    probability_of_interactions[probability_of_interactions > 1] = 1
    
    interact_mat = \
        interaction_matrix_with_connectance(no_species,mu_a,sigma_a,probability_of_interactions)
    
    return interact_mat

def modular_mixed_interaction_matrix():
    
    pass

def nested_mixed_interaction_matrix():
    
    pass

########### Extra functions for generating interaction matrices #####

def interaction_matrix_with_connectance(n,mu_a,sigma_a,connectance,
                                        self_inhibition=True):
    
    '''
    
    Generate a random interaction matric with connectance c.

    Parameters
    ----------
    n : int
        Number of n. 
        (The interaction matrix describes interaction/edges between n.)
    mu_a : float
        Average interaction strength.
    sigma_a : float
        Standard deviation in interaction strength.
    connectance : float
        Probability of node i and j interacting (c).

    Returns
    -------
    interaction_matrix : np.ndarray of size (n,n).
        Interaction matrix.

    '''
    # create the connectance matrix (whether n are interacting or not)
    are_species_interacting = \
        np.random.binomial(1,connectance,size=n*n).reshape((n,n))
    
    # create the interaction strength matrix
    interaction_strengths = mu_a + sigma_a*np.random.randn(n,n)
    
    # create the interaction matrix
    interaction_matrix = interaction_strengths * are_species_interacting
    
    if self_inhibition == True:
        
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
        np.fill_diagonal(interaction_matrix, 1)
    
    return interaction_matrix

############################### Community Function #######################
   
def species_contribution_to_community_function(no_species,mu_contribution,sigma_contribution):
    
    '''
    
    Generate parameters for species contribution to community function, or species function.
        Inspired by Chang et al. (2021), "Engineering complex communities by directed evolution".
        All species had a fixed species function, rather than community function
        being emergent from dynamic mechanistic interactions.
        Species contribution to community function is drawn from 
        normal(mu_contribution,sigma_contribution)
        
    Parameters
    ----------
    no_species : int
        Number of species.
    mean_contribution : float
        Mean species function.
    function_std : float
        Standard deviation for species function.
    
    Returns
    -------
    species_function : np.array of floats, size (no_species,)
        Array of individual species functions, drawn from distribution normal(0,function_std).
    
    '''
    
    species_function = mu_contribution + sigma_contribution*np.random.randn(no_species)
    
    return species_function