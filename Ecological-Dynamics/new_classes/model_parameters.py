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
            
    def random_interaction_matrix(self,mu_a,sigma_a):
        
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
        np.fill_diagonal(interact_mat, 1)
        
        return interact_mat
    
    def sparse_interaction_matrix(self,mu_a,sigma_a,connectance):
        
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
                                                sigma_a, connectance)
        
        return interact_mat
    
    def mixed_sparse_interaction_matrix(self):
        
        # generate competitive interactions to start
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                     self.competitive_mu_a,
                                                     self.competitive_sigma_a,
                                                     self.competitive_connectance)
        
        cooperative_interaction_indices = \
            np.random.binomial(1,self.probability_cooperative,
                               size=self.no_species*self.no_species).reshape((self.no_species,self.no_species))
        
        cooperative_interaction_matrix = \
            self.interaction_matrix_with_connectance(self.cooperative_interaction_indices.shape[0],
                                                     self.cooperative_mu_a,
                                                     self.cooperative_sigma_a,
                                                     self.cooperative_connectance,
                                                     self_inhibition=False)
        
        interact_mat[np.where(interact_mat == cooperative_interaction_indices)] = \
            cooperative_interaction_matrix
            
        return interact_mat
    
    ############# Non-uniform structured interaction matrices ###########
    
    def modular_interaction_matrix(self,p_mu_a,p_sigma_a,p_connectance,
                                   q_mu_a,q_sigma_a,q_connectance):
        
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
            [self.interaction_matrix_with_connectance(nodes,p_mu_a,p_sigma_a,p_connectance)
             for nodes in clustered_species]
        
        # combine module interactions into a community interaction matrix
        interact_mat = linalg.block_diag(*module_interactions)
        
        ####### Assign interactions between members of different groups ######
        
        # get indices of interaction matrix where species from different modules interact
        non_group_interaction_indices = np.where(interact_mat == 0)
        
        # generate the interactions between species from different modules
        non_group_interactions = \
            self.interaction_matrix_with_connectance(self.no_species,
                                                     q_mu_a,q_sigma_a,q_connectance)    
        
        # add between-module interactions to the interaction matrix
        interact_mat[non_group_interaction_indices] = \
            non_group_interactions[non_group_interaction_indices]
        
        return interact_mat
    
    def nested_interaction_matrix(self,mu_a,sigma_a,beta=7):
        
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
            self.average_degree*((beta-2)/(beta-1))*((self.no_species/species)**(1/(beta-1)))
        
        # calculate the probability species i interacts with j.
        probability_of_interactions = \
            (np.outer(weights,weights)/np.sum(weights)).flatten()
        
        # set probabilities > 1 to 1.
        probability_of_interactions[probability_of_interactions > 1] = 1
        
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,mu_a,sigma_a,
                                                     probability_of_interactions)
        
        return interact_mat
    
    def modular_mixed_interaction_matrix():
        
        pass
    
    def nested_mixed_interaction_matrix():
        
        pass
    
    def strong_compete_weak_parasitise(self):
        
        min_growth = np.min(self.growth_rates)
        max_growth = np.max(self.growth_rates)
        
        probability_interaction = \
            (self.growth_rates - min_growth)/(max_growth-min_growth)
        
        are_species_interacting = \
            np.random.binomial(1,
                               np.tile(probability_interaction,self.no_species),
                               size=self.no_species*self.no_species).reshape((self.no_species,self.no_species))
            
        probability_cooperation = np.zeros((self.no_species,self.no_species))

        for i in range(50):
            for j in range(50):
                probability_cooperation[i,j] = \
                    ((self.growth_rates[i] - self.growth_rates[j])**2)/(self.growth_rates[i] * self.growth_rates[j])
        
        probability_cooperation = probability_cooperation.flatten()
        probability_cooperation[probability_cooperation > 1] = 1
        
        interaction_type_matrix = \
            np.random.binomial(1,
                               probability_cooperation,
                               size=self.no_species*self.no_species).reshape((self.no_species,self.no_species))
    
        interaction_type_matrix[np.where(interaction_type_matrix == 0)] = -1
    
        are_species_interacting[np.where(are_species_interacting == 1)] = \
            interaction_type_matrix[np.where(are_species_interacting == 1)]

        interact_mat = are_species_interacting
        
        competitive_interactions = np.where(interact_mat == -1)
        cooperative_interactions = np.where(interact_mat == 1)
        
        interact_mat[competitive_interactions] = \
            self.mu_comp + self.sigma_comp*np.random.randn(len(competitive_interactions))
        
        interact_mat[cooperative_interactions] = \
            self.mu_coop + self.sigma_coop*np.random.randn(len(competitive_interactions))
        
        return interact_mat
    
    def competition_scaled_with_growth(self,mu_a,sigma_a,connectance):
        
        min_growth = np.min(self.growth_rates)
        max_growth = np.max(self.growth_rates)
        
        weights = self.growth_rates # maybe come up with a better function
        
        # calculate the probability species i interacts with j.
        probability_of_interactions = \
            (np.outer(weights,weights)/np.sum(weights)).flatten()
        
        # set probabilities > 1 to 1.
        probability_of_interactions[probability_of_interactions > 1] = 1
            
        interact_mat = \
            self.interaction_matrix_with_connectance(self.no_species,mu_a,sigma_a,
                                                     probability_of_interactions)
        
        return interact_mat
    
    def cooperation_scaled_with_growth(self,mu_a,sigma_a,connectance):
        
        min_growth = np.min(self.growth_rates)
        max_growth = np.max(self.growth_rates)
        
        pass
        
        
    ########### Extra functions for generating interaction matrices #####
    
    def interaction_matrix_with_connectance(self,n,mu_a,sigma_a,connectance,
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
    
    def generate_community_function(self,func_name='Generate community function',
                                    community_func_args={'mu_contribution':0,'sigma_contribution':1},
                                    usersupplied_community_function=None):
        
        '''
        
        Generate or assign community function.
        
        Parameters
        ----------
        usersupplied_community_function : None or np.array, size (no_species,).
            User-supplied array of species contribution to community function, default None.
            
        Returns
        -------
        None
        '''
        
        match func_name:
            
            case 'Generate community function':
                
                self.species_contribution_community_function = \
                    self.species_contribution_to_community_function(self.no_species,
                                                                    **community_func_args)
                
            case None: 
                
                self.species_contribution_community_function = usersupplied_community_function
       
    def species_contribution_to_community_function(self,mu_contribution,
                                                   sigma_contribution):
        
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
        
        species_function = \
            mu_contribution + sigma_contribution*np.random.randn(self.no_species)
        
        return species_function