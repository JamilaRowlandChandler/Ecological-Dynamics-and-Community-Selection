# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:06 2024

@author: jamil
"""

# %%
import numpy as np

# %%

class ParametersInterface:
    
    def generate_parameters(self, growth_consumption_method,
                            other_parameter_methods = {'death' : 'normal',
                                                       'influx' : 'normal'},
                            **kwargs):
        
        match growth_consumption_method:
            
            case 'coupled by rho':
                
                self.consumption, self.growth = \
                    self.growth_consumption_underlying_correlation()
                    
                self.eff_consumption, self.eff_growth = self.consumption, self.growth
                
            case 'growth function of consumption':
                
                self.growth = self.normal_parameters(self.mu_g, self.sigma_g,
                                                     (self.no_species, self.no_resources))
                self.consumption = self.normal_parameters(self.mu_c, self.sigma_c,
                                                     (self.no_resources, self.no_species))
                
                self.eff_consumption, self.eff_growth = \
                    self.consumption, self.growth * self.consumption.T
                    
            case 'consumption funtion of growth':
                
                self.growth = self.normal_parameters(self.mu_g, self.sigma_g,
                                                     (self.no_species, self.no_resources))
                self.consumption = self.normal_parameters(self.mu_c, self.sigma_c,
                                                     (self.no_resources, self.no_species))
                
                self.eff_consumption, self.eff_growth = \
                    self.consumption * self.growth.T, self.growth
                    
            case 'user supplied':
                    
                self.growth = kwargs.get('growth matrix', None)
                self.consumption = kwargs.get('consumption matrix', None)
                
                self.eff_consumption, self.eff_growth = self.consumption, self.growth
                
        for p_type, p_method in other_parameter_methods.items():
            
            match p_type:
                
                case 'death':
                    
                    suffix = 'm'
                    dims =  (self.no_species, )
            
                case 'influx':
                    
                    suffix = 'K'
                    dims = (self.no_resources, )
            
                case 'outflux':
                    
                    suffix = 'D'
                    dims = (self.no_resources, )
            
            match p_method:
                
                case 'normal':
                    
                    parameters = self.normal_parameters(getattr(self, 'mu_' + suffix),
                                                        getattr(self, 'sigma_' + suffix),
                                                        dims)
                    
                case 'constant':
                    
                    parameters = getattr(self, suffix) * np.ones(dims)
                    
            setattr(self, p_type, parameters)
        
    def normal_parameters(self, mu, sigma, dims):
        
        return mu + sigma*np.random.randn(*dims)
    
    def growth_consumption_underlying_correlation(self):
        
        X_c = np.random.randn(self.no_resources, self.no_species)
        X_g = np.random.randn(self.no_species, self.no_resources)
        
        consumption = self.mu_c + self.sigma_c*X_c
        growth = self.mu_g + self.sigma_g*(self.rho*X_c.T + np.sqrt(1 - self.rho**2)*X_g)
        
        return consumption, growth   