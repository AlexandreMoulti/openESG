import numpy as np
import scipy as sc


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
