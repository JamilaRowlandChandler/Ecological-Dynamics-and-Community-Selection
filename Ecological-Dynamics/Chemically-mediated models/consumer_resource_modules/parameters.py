# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:06 2024

@author: jamil
"""

# %%
import numpy as np
from scipy import linalg

# %%

class ParametersInterface:
    
    def generate_parameters(self, method = 'dense', **kwargs):
        
        match method:
            
            case 'dense':
                
                self.growth = self.growth_norm()
                self.consumption = self.consumption_norm()
                
            case 'sparse':
                
                growth = self.growth_norm()
                consumption = self.consumption_norm()
                
                sparse_method = kwargs.get('sparse_method', None)
                
                if sparse_method is None:
                    
                    raise Exception("No sparse method given.")
                
                match sparse_method:
                    
                    case 'same':
                        
                        connectance = kwargs.get('connectance', None)
                        
                        species_resource_interactions = self.sparsity(connectance,
                                                                      (self.no_species,
                                                                       self.no_resources))
                        
                        self.growth = growth * species_resource_interactions
                        self.consumption = consumption * species_resource_interactions.T
                    
                    case 'different':
                        
                        connectance = kwargs.get('connectance', None)
                        
                        species_resource_interactions = self.sparsity(connectance,
                                                                      (self.no_species,
                                                                       self.no_resources))
                        resource_species_interactions = self.sparsity(connectance,
                                                                      (self.no_resource,
                                                                       self.no_species))
                        
                        self.growth = growth * species_resource_interactions
                        self.consumption = consumption * resource_species_interactions

                    case 'user supplied':
                        
                        sparse_interactions = kwargs.get('sparse_interactions', None)
                        
                        self.growth = growth * sparse_interactions[0]
                        self.consumption = consumption * sparse_interactions[1]
                        
            case 'user supplied':
                    
                self.growth = kwargs.get('growth', None)
                self.consumption = kwargs.get('consumption', None)
                
        self.death = np.ones(self.no_species)
        self.influx = np.ones(self.no_resources)
                        
    def growth_norm(self):
        
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
        growth : np.array of float64.
            array of growth rates for each species drawn from normal(mu_g,sigma_g).
    
        '''
        
        growth = np.abs(self.mu_g + self.sigma_g*np.random.randn(self.no_species, self.no_resources))
        
        return growth
    
    def consumption_norm(self, method = 'dense', **kwargs):
        
        '''
        
        Draw growth rates for n species from normal(mu,sigma) distribution
    
        Parameters
        ----------
        mu_c : float
            Mean consumption rate.
        sigma_c : float
            Standard deviation in consumption rate.
        no_species : int
            Number of species (n).
    
        Returns
        -------
        growth : np.array of float64.
            array of consumption rates for each species drawn from normal(mu_c,sigma_c).
    
        '''
        
        consumption = np.abs(self.mu_c + self.sigma_c*np.random.randn(self.no_resources, self.no_species))
        
        return consumption
    
    def sparsity(connectance, dims):
        
        species_variable_interactions = \
            np.random.binomial(1, connectance, 
                               size = dims[0] * dims[1]).reshape(dims)
        
        return species_variable_interactions
                