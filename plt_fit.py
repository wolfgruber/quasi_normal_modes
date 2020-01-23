import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def PlotLog(rfix):
    '''Plotts the log(abs(phi)) at r = rfix.'''
    plt.figure(figsize = (12,4))
    plt.subplot(121)
    plt.plot(tp, np.log(np.abs(phip[:,rfix]+0.000001)))
    plt.ylim(-10,5)
    
    plt.subplot(122)
    plt.plot(tp,phip[:,rfix])
    plt.ylim(-20,20)
    plt.show()
    
def PlotLog3(rfix):
    '''Plotts the log(abs(phi)) at r = rfix for three different phi.'''
    plt.plot(tp, np.log(np.abs(phip1[:,rfix])))
    plt.plot(tp, np.log(np.abs(phip2[:,rfix])))
    plt.plot(tp, np.log(np.abs(phip3[:,rfix])))
    plt.ylim(-15,2.5)
    plt.show()


def PlotMovie(t):
    '''Plotting phi animated.'''
    plt.figure(figsize = (12,4))
    plt.subplot(121)
    plt.plot(rp, np.log(np.abs(phip[t,:]+0.000001)))
    plt.ylim(-10,5)
    
    plt.subplot(122)
    plt.plot(rp, phip[t,:])
    plt.ylim(-2,2)
    plt.show()
    
def PlotMovie3(t):
    '''Plotting 3 different phi animated.'''
    plt.plot(rp, phip1[t,:])
    plt.plot(rp, phip2[t,:])
    plt.plot(rp, phip3[t,:])
    plt.ylim(-2,2)
    plt.show()


def export(phi, l, loc):
    '''Saves a field "phi" in the location "loc".'''
    np.save(loc+"/phi_{:02d}.npy".format(l), phi, allow_pickle=False)
    return
    
def imp(l_max, loc):
    '''Loads a field "phi" with "l_max" l-s from the location "loc".'''
    t = np.load(loc+"/t.npy", allow_pickle=False)
    r = np.load(loc+"/r.npy", allow_pickle=False)
    phi = np.empty((t.size, r.size, l_max))
    for l in range(l_max):
        phi[:,:,l] = np.load(loc+"/phi_{:02d}.npy".format(l), allow_pickle=False)
    
    return phi, t, r


def QNM(t, Omega, Gamma, A, B):
    '''Function to fit a quasi normal mode with.'''
    f = A * np.sin(Omega * t + B) * np.exp(-Gamma * t)
    return f


def QNM_fit(phi, t, t_min, t_max):
    '''Function for fitting a quasi normal mode.'''
    dt = t[1] - t[0]
    lower = int(t_min/dt)
    upper = int(t_max/dt)
    phi_fit = phi[lower:upper]
    t_fit = t[lower:upper]
    
    bounds = (np.array([0.001,0,1,0]),np.array([0.3,0.5,100,200]))
    guess = np.array([0.2,0.2,30,np.pi*1.3])

    P, Mp = opt.curve_fit(QNM, t_fit, phi_fit, p0=guess, method='trf', bounds=bounds)
    
    return P
    
def analyze(phi, t, r, rfix):
    '''Goes through different l-s and asks for border to fit quasi normal
    modes on phi.'''
    # some params
    l_max = phi[0,0,:].size
    omega = np.empty((l_max,2))
    dt = t[1] - t[0]
    dr = r[1] - r[0]
    r_fix = int(rfix/dr)
    
    # looping over all l's
    for l in range(l_max):
        plt.figure(figsize=(12.8, 4.8))
        plt.subplot(121)
        plt.plot(t, np.log(np.abs(phi[:,r_fix,l])))
        plt.grid(True)
        plt.ylim(-15,0)
        
        plt.subplot(122)
        plt.plot(t, phi[:,r_fix,l])
        plt.grid(True)
        plt.show()
        print('l = {}'.format(l))
        t_min = float(input("lower limit: "))
        t_max = float(input("upper limit: "))
        
        phi_a = phi[int(t_min/dt):int(t_max/dt),r_fix,l]
        t_a = t[int(t_min/dt):int(t_max/dt)]
        bounds = (np.array([0.001,0.01,0.000001,0]),np.array([l_max,0.5,1000,200]))
        #guess = np.array([l/2,0.2,30,np.pi*1.3])
        #P,Mp = opt.curve_fit(QNM, lt, lphi, p0=guess, method='trf', bounds=bounds)
        P, Mp = opt.curve_fit(QNM, t_a, phi_a, method='trf', bounds=bounds)
        omega[l,:] = P[0:2]
        
        fit = QNM(t, *P)
        
        plt.figure(figsize=(12.8, 4.8))
        plt.subplot(121)
        plt.plot(t, np.log(np.abs(phi[:,r_fix,l])))
        plt.plot(t, np.log(np.abs(fit)))
        plt.ylim(-15,0)
        
        plt.subplot(122)
        plt.plot(t, phi[:,r_fix,l])
        plt.plot(t, fit)
        plt.show()
    
    for l in range(l_max):
        print('{:2d}: {:1.4f} - {:1.4f} i'.format(l, omega[l,0], omega[l,1]))

    return omega
    
