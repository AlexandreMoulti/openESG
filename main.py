import numpy as np
import pandas as pd

class BlackScholes(object):
    """ Black Scholes model
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, initial_value, maturity, nb_simulations, nb_steps):
        """ generates the scenarios
            initial_value:  initial starting point of the simulations
            maturity: horizon of the simulation
            nb_simulations: number of simulations
            nb_steps: number of steps per year
            result: matrix of simulations
            dimension: (nb_simulations, nb_steps*maturity+1) as the first column represent the initial point
        """
        #brownian increments
        dB        = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = initial_value
        dt        = 1.0/nb_steps
        root_dt   = np.sqrt(dt)
        #loop over columns
        for j in range(res.shape[1]-1):
            res[:,j+1] = res[:,j]*np.exp(self.mu*dt+self.sigma*root_dt*dB[:,j]-0.5*self.sigma*self.sigma*dt)
        return(res)

    @classmethod
    def from_history(cls, historical_values, time_interval=1.0):
        """ instantiates the black scholes model from from historical values
            historical_values: historical observations of the variable
            time_interval: time betwen two observations (by default :1 year)
        """
        hv = pd.Series(historical_values)
        hv = np.log(hv).diff()
        mu = hv.mean()/time_interval
        sigma = hv.std()/time_interval
        return(cls(mu, sigma))
    
    @classmethod
    def from_call(cls, call_contracts):
        """ to be done
        """
        return(cls(mu=0, sigma=1)
    
    @classmethod
    def from_put(cls, put_contracts):
        """ to be done
        """
        return(cls(mu=0, sigma=1)
