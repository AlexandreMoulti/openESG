import numpy as np
import scipy as sc
from scipy.stats import norm


class CIR(object):
    """ CIR mean reverting process
    """
    def __init__(self, init, target, speed, vol_of_vol):
        self.init       = init
        self.target     = target
        self.speed      = speed
        self.vol_of_vol = vol_of_vol

    def mean(self, init, dt):
        """ mean of the process at time t
        """
        return(self.target+(init-self.taget)*np.exp(-self.speed*dt))

    def variance(self, init, dt):
        """ variance of the process at time t
        """
        res = init*self.vol_of_vol*self.vol_of_vol*np.exp(-self.speed*dt)*(1-np.exp(-self.speed*dt))/self.speed
        res = res + 0.5*self.target*self.vol_of_vol*self.vol_of_vol*(1-np.exp(-self.speed*dt))**2/self.speed
        return(res)

    def simulate(self, maturity, nb_simulations, nb_steps):
        #non optimal QE algo
        #check if works
        urand     = np.random.uniform(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        norm_rand = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        root_dt   = np.sqrt(dt)
        for j in range(res.shape[1]-1):
            for i in range(res.shape[0])
                m  = self.mean(res[i,j], dt)
                s2 = self.variance(res[i,j], dt)
                psi = s2/(m*m)  
                if (psi<1.5):
                    b2  = 2/psi-1+np.sqrt(2/psi*(1/psi-1))
                    a   = m/(1+b2)
                    b   = np.sqrt(b)
                    res[i,j+1] = a*(b+norm_rand[i,j])**2
                else:
                    beta = 2/(m*(psi+1))
                    p    = (psi-1)/(psi+1)
                    u    = urand[i,j]
                    res[i,j+1]  = self.__inverse_psi__(u,p, beta)
        

    def simulate_integral(self, maturity, nb_simulations, nb_steps):
        #simulates the integrated variance process as described in Andersen paper
        #useful for heston
        return(0)
        
    #help tools for CIR class
    def __inverse_psi__(self, u, p, beta):
        if 0<=u<=p:
            return(0)
        else:
            return(1/beta*np.log((1-p)/(1-u)))

class HestonModel(object):
    """ Heston Model
    """

    def __init__(self, init, rate, dividend, vol_init, target, speed, vol_of_vol ):
        self.init       = init
        self.rate       = rate
        self.dividend   = dividend
        self.volatility = CIR(vol_init, target, speed, vol_of_vol)

    def simulate(self, maturity, nb_simulations, nb_steps):
        #to do
        return(0)

    def call_price(self, maturity, strike):
        return(0)
        
    def put_price(self, maturity, strike):
        return(0)
    
    
    
#functions from the matlab code

def chfun_norm(s0, v, r, t, w):
    """ Characteristic function of the Black Scholes model
        INPUTS
        ------
        s0 : stock price
        v: volatility
        r: risk-free rate
        t: time to maturity
        w: point at which to evaluate the functiono
    """
    mean = np.log(s0)+(r-0.5*v*v)*t
    var = v*v*t
    return(np.exp(1j*w*mean-0.5*w*w*var))
    
def bsm_integrand1(w, s0, k, v, r, t):
    res = np.exp(-1j*w*np.log(k))*chfun_norm(s0, v, r, t, w-1j)/ (1j*w*chfun_norm(s0, v, r, t, -1j))
    res = float(np.real(res))
    return(res)

def bsm_integrand2(w, s0, k, v, r, t):
    res = np.exp(-1j*w*np.log(k))*chfun_norm(s0, v, r, t, w)/ (1j*w)
    res = float(np.real(res))
    return(res)

def call_bsm_cf(s0, v, r, t, k):
    #first integral
    integ1 = sc.integrate.quad(bsm_integrand1, 0,100, args=(s0,k,v,r,t))[0]
    integ1 = integ1/np.pi+0.5
    #second integral
    integ2 = sc.integrate.quad(bsm_integrand2, 0,100, args=(s0,k,v,r,t))[0]
    integ2 = integ2/np.pi+0.5
    #result
    res = s0*integ1-np.exp(-r*t)*k*integ2
    return(res)
    

call_bsm_cf(100, 0.2, 0.02, 1.0,100)

def chfun_heston(s0,v0,vbar, a, vvol, r, rho, t, w):
    """ Characteristic function of the heston model      
    """
    alpha = -0.5*w*w-0.5j*w
    beta = a - 1j*rho*vvol*w
    gamma = 0.5*vvol*vvol
    h    = np.sqrt(beta*beta-4*alpha*gamma+0j)
    rplus= (beta+h)/(vvol*vvol)
    rminus = (beta-h)/(vvol*vvol)
    g=rminus/rplus
    
    big_c = a*(rminus*t-(2/(vvol*vvol)*np.log((1-g*np.exp(-h*t))/(1-g))))
    big_d = rminus*(1-np.exp(-h*t))/(1-g*np.exp(-h*t))
    res= np.exp(big_c*vbar+big_d*v0+1j*w*np.log(s0*np.exp(r*t)))
    return(res)
    
def heston_integ1(w,s0, v0, vbar, a, vvol, r, rho, t, k ):
    res = np.exp(-1j*w*np.log(k))*chfun_heston(s0,v0,vbar, a, vvol, r, rho, t, w-1j)/(1j*w*chfun_heston(s0,v0,vbar, a, vvol, r, rho, t, -1j))
    res = float(np.real(res))
    return(res)
    
def heston_integ2(w,s0, v0, vbar, a, vvol, r, rho, t, k ):
    res = np.exp(-1j*w*np.log(k))*chfun_heston(s0,v0,vbar, a, vvol, r, rho, t, w)/(1j*w)
    res = float(np.real(res))
    return(res)

def call_heston_cf(s0, v0, vbar, a, vvol, r, rho, t, k):
    #first integral
    integ1 = sc.integrate.quad(heston_integ1, 0,100, args=(s0, v0, vbar, a, vvol, r, rho, t, k))[0]
    integ1 = integ1/np.pi+0.5
    #second integral
    integ2 = sc.integrate.quad(heston_integ2, 0,100, args=(s0, v0, vbar, a, vvol, r, rho, t, k))[0]
    integ2 = integ2/np.pi+0.5   
    #res
    res = s0*integ1-np.exp(-r*t)*k*integ2
    return(res)

print(call_heston_cf(1, 0.16, 0.16, 1, 2, 0, -0.8, 10, 2))
