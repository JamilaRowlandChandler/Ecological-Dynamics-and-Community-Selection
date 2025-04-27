# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:13:21 2024

@author: jamil
"""

import numpy as np
from scipy import linalg

############## Growth Rates ###########

class ParametersInterface:

    def growth_rates_norm(self):
        
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
        
        growth_r = self.mu_g + self.sigma_g*np.random.rand(self.no_species)
        
        return growth_r
    
    def growth_rates_fixed(self):
        
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
        
        growth_r = np.ones((self.no_species,))
        
        return growth_r
    
        
    ##################### Interaction Matrix #####################
    
    ########## Sparse and dense random interaction matrices #########
            
    def random_interaction_matrix(self,mu_a,sigma_a,self_interaction=1):
        
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
        interact_mat = mu_a + sigma_a*np.random.randn(self.no_species,self.no_species)
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
        
        np.fill_diagonal(interact_mat,self_interaction)
        
        return interact_mat
    
    def sparse_interaction_matrix(self,mu_a,sigma_a,connectance,self_interaction=1):
        
        '''
        
        Generate a sparse random interaction matrix using a Erdős–Rnyi graph.
        
        See May (1972) for details. https://doi.org/10.1038/238413a0
    
        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        interact_mat =\
            self.interaction_matrix_with_connectance(self.no_species, mu_a,
                                                sigma_a, connectance,self_interaction)
        
        return interact_mat
    
    ############# Non-uniform structured interaction matrices ###########
    
    def modular_interaction_matrix(self,p_mu_a,p_sigma_a,p_connectance,
                                   q_mu_a,q_sigma_a,q_connectance,self_interaction=1):
        
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
        
        if hasattr(self, 'module_probabilities'):
    
            clustered_species = np.random.multinomial(self.no_species,
                                                      self.module_probabilities,
                                                      size=1)[0]
        
        else:
            
            clustered_species = np.random.multinomial(self.no_species,
                                                      np.repeat(1/self.no_modules,self.no_modules),
                                                      size=1)[0]
      
        # create the interaction matrices for each module
        module_interactions = \
            [self.interaction_matrix_with_connectance(nodes,p_mu_a,p_sigma_a,p_connectance,self_interaction)
             for nodes in clustered_species]
        
        # combine module interactions into a community interaction matrix
        interact_mat = linalg.block_diag(*module_interactions)
        
        ####### Assign interactions between members of different groups ######
        
        # get indices of interaction matrix where species from different modules interact
        non_group_interaction_indices = np.where(interact_mat == 0)
        
        # generate the interactions between species from different modules
        non_group_interactions = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                     q_mu_a,q_sigma_a,q_connectance,self_interaction)    
        
        # add between-module interactions to the interaction matrix
        interact_mat[non_group_interaction_indices] = \
            non_group_interactions[non_group_interaction_indices]
        
        return interact_mat
    
    def nested_interaction_matrix(self,mu_a,sigma_a,average_degree,
                                  beta=7,self_interaction=1):
        
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
        species = np.arange(1,self.no_species+1)
        
        # calculate node weights, used to calculate the probability species i interacts with j.
        weights = \
            average_degree*((beta-2)/(beta-1))*((self.no_species/species)**(1/(beta-1)))
        
        # calculate the probability species i interacts with j.
        probability_of_interactions = \
            (np.outer(weights,weights)/np.sum(weights)).flatten()
        
        # set probabilities > 1 to 1.
        probability_of_interactions[probability_of_interactions > 1] = 1
        
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,mu_a,sigma_a,
                                                     probability_of_interactions,self_interaction)
        
        return interact_mat
    
        
    ########### Extra functions for generating interaction matrices #####
    
    def interaction_matrix_with_connectance(self,n,mu_a,sigma_a,connectance,
                                            self_interaction):
        
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
        
        np.fill_diagonal(interaction_matrix, self_interaction)
        
        return interaction_matrix
    
