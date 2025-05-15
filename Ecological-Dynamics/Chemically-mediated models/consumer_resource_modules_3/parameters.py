# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:06 2024

@author: jamil
"""

# %%
import numpy as np

# %%

class ParametersInterface:
    
    # Public methods
            
    def growth_consumption_rates(self, method, mu_c, sigma_c, mu_g, sigma_g,
                                 **kwargs):
        
        '''
        
        Parameters
        ----------
        method : TYPE
            DESCRIPTION.
        mu_c : TYPE
            DESCRIPTION.
        mu_g : TYPE
            DESCRIPTION.
        sigma_c : TYPE
            DESCRIPTION.
        sigma_g : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        for name, statistic in zip(['mu_c', 'mu_g', 'sigma_c', 'sigma_g'],
                                   [mu_c, mu_g, sigma_c, sigma_g]): 
            
            setattr(self, name, statistic)
        
        X_c, X_g = np.random.randn(self.no_resources, self.no_species),\
                    np.random.randn(self.no_species, self.no_resources)
        
        match method:
            
            case 'coupled by rho':
                
                rho = kwargs.get('rho', None)
                
                if rho is None:
                    
                    raise Exception('Please supply a value for rho.\n' + \
                                    '(In growth_consumption_method(), add a rho = x argument.)')
                
                else: 
                
                    self.rho = rho
                    
                self.consumption = self.mu_c + self.sigma_c*X_c
                self.growth = self.mu_g + self.sigma_g*(self.rho*X_c.T + np.sqrt(1 - self.rho**2)*X_g)
            
            case 'growth function of consumption':
                
                self.consumption = self.mu_c + self.sigma_c*X_c
                self.rue = self.mu_g + self.sigma_g*X_g
                self.growth = self.rue * self.consumption.T
            
            case 'consumption function of growth':
                
                self.growth = self.mu_g + self.sigma_g*X_g
                self.rue = self.mu_c + self.sigma_c*X_c
                self.consumption = self.rue * self.growth.T
                
            case 'user supplied':
                
                consumption, growth = kwargs.get('consumption', None), kwargs.get('growth', None)
                
                if consumption is None or growth is None:
                        
                    raise Exception('Please supply your growth or consumption rates.\n'
                                    '(In growth_consumption_method(), add the arguments ' 
                                    'consumption = <some np.array>, growth = <some np.array>')
                            
                else:
                
                    self.consumption = consumption
                    self.growth = growth 
                    
            case _:
                
                raise Exception('You have not selected an exisiting method.\n' + \
                      'Please chose from either "coupled by rho", ' + \
                          '"growth function of consumption", ' + \
                              '"consumption function of growth", or ' + \
                                  '"user supplied".')
    
    def other_parameter_methods(self, parameter_method, parameter_args,
                                  p_label, dims):
        
        match parameter_method:
            
            case 'normal':
                
                try:
                
                    mu, sigma = parameter_args['mu'], parameter_args['sigma']
                    
                    setattr(self, 'mu_' + p_label, mu)
                    setattr(self, 'sigma_' + p_label, sigma)
                    
                    parameters = self.__normal_parameters(mu, sigma, dims)
                    
                    setattr(self, p_label, parameters)
                    
                except KeyError as e:
                    
                    print("You need to supply a value for 'mu' and 'sigma' in your dictionary argument.")
                
            case 'constant':
                
                try:
                
                    setattr(self, p_label, parameter_args[p_label] * np.ones(dims))
                
                except KeyError as e:
                    
                    print("You need to supply a value for " + p_label + " in your dictionary argument.")
        
    def __normal_parameters(self, mu, sigma, dims):
        
        return mu + sigma*np.random.randn(*dims)
    