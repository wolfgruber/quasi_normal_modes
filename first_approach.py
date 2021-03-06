# 23.01.20
# Bachelor Thesis: "Scattering of scalar waves on a Schwarzschild black hole"
# Ludwig Wolfgruber

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as sp

from plt_fit import *


# functions:
def GaussWave(t, r, c, m, shift, sigma):
    '''Gaussian wave signal for initial data.'''
    e1 = (r-shift+c*t)**2/(2*sigma)
    return np.exp(-e1)/(2*np.sqrt(np.pi*sigma))


def Initialize(t, r, c, m, shift, sigma, l):
    ''' Initializes a signal as initial data.'''
    dt = t[1] - t[0]
    initphi = GaussWave(t[0],r,c,m,shift, sigma)
    '''init1 = GaussWave(t[0]+dt,r,c,m,shift, sigma)
    init2 = GaussWave(t[0]-dt,r,c,m,shift, sigma)
    initpsi = (init1-init2)/(2*dt)'''
    initpsi = initphi * (r-shift+c*t[0]) * c
    return initphi, initpsi

    
def Lu_neu(r, dr, i, m, psi, phi, l): 
    '''Evaluates the differential operator with artificial dissipation.'''
    ri = r[i]
    N = r.size
    eps = 0.1 # factor for smoothing out phi
    newphi = psi[i]

    if (i == 0):
        a1 = (ri-m)/(dr*(ri**2+2*m*ri)) * (- 3*phi[0] + 4*phi[1] - phi[2])
        a2 = 2*m/(ri**2+2*m*ri) * psi[0]
        a3 = 2*m/(dr*(ri+2*m)) * (- 3*psi[0] + 4*psi[1] - psi[2])
        a4 = (ri-2*m)/(dr**2*(ri+2*m)) * (2*phi[0] - 5*phi[1] + 4*phi[2] - phi[3])
        Lu = (a1 + a2 + a3 + a4)

    elif (i == N-1):
        Lu = 0
        newphi = 0

    else:
        a1 = (ri-m)/(dr*(ri**2+2*m*ri)) * (phi[i+1] - phi[i-1])
        a2 = 2*m/(ri**2+2*m*ri) * psi[i]
        a3 = 2*m/(dr*(ri+2*m)) * (psi[i+1] - psi[i-1])
        a4 = (ri-2*m)/(dr**2*(ri+2*m)) * (phi[i+1] - 2*phi[i] + phi[i-1])

        Lu = (a1 + a2 + a3 + a4)

        if (i > 1) and (i < N-2):
            Lu = Lu - eps * (phi[i+2] - 4*phi[i+1] + 6*phi[i] - 4*phi[i-1] + phi[i-2]) / 16
            newphi = newphi - eps * (phi[i+2] - 4*phi[i+1] + 6*phi[i] - 4*phi[i-1] + phi[i-2]) / 16

    return Lu - l * (l+1) * phi[i] / ri**2, newphi  # sph_harm(m=0, n=l, theta=0, phi=0)


def TimeStepRK_neu(dt, r, dr, m, phi, psi, l):
    '''Computes a timestep with the Runge-Kutta 2 method.'''
    N = phi.size
    newphi = np.zeros(N)
    newpsi = np.zeros(N)

    for i in range(N):
        k1 = Lu_neu(r, dr, i, m, psi, phi, l)
        k2 = Lu_neu(r, dr, i, m, psi + k1[0]*dt/2, phi + k1[1]*dt/2, l)
        k3 = Lu_neu(r, dr, i, m, psi + k2[0]*dt/2, phi + k2[1]*dt/2, l)
        k4 = Lu_neu(r, dr, i, m, psi + k3[0]*dt, phi + k3[1]*dt, l)

        newpsi[i] = psi[i] + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        newphi[i] = phi[i] + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

    newpsi[N-1] = 0

    return newphi, newpsi


def Run(dr, R, T, shift, sigma, l_max, mode, f):
    '''Runs a simulation for the given values of l. "mode" is set equal to "c"
    if computing the convergence.'''
    # some constants:
    G = 1
    c = 1
    M = 0.5
    m = G * M / c**2 #the mass with included constants

    r = np.arange(2*m, R, dr)
    Nr = r.size

    #dt = dr**2/2
    dt = dr*f
    t = np.arange(0, T, dt)
    Nt = t.size

    phi_l = np.empty((Nt,Nr))
    psi = np.empty((Nt,Nr))
    
    start=0
    if mode == 'c':
        start = l_max - 1

    for l in range(start, l_max):
        print('computing time evolution for l = '+str(l))
        phi_l[0,:], psi[0,:] = Initialize(t, r, c, m, shift, sigma, l) # computing the initial data
        
        for j in range(1,Nt): # going forward in time
            if j % 1000 == 0:
                print('    at time = {}'.format(t[j]))
                
            phi_l[j,:], psi[j,:] = TimeStepRK_neu(dt, r, dr, m, phi_l[j-1,:], psi[j-1,:], l)
                
        #phi_l[:,:] = sph_harm(0, l, 0, 0).real * phi_l[:,:]

        if mode != 'c':
            export(phi=phi_l, l=l, loc='phi_fir')

            if l == 0:
                np.save('phi_fir/t.npy', t, allow_pickle=False)
                np.save('phi_fir/r.npy', r, allow_pickle=False)

    return phi_l, t, r


def convergence(dr, R, T, shift, sigma, f, l):
    '''Compute the convergence as a function of time for the given value of l.'''
    print('computing convergence')
    h = dr/4
    h2 = dr/2
    h4 = dr
    
    phi, t, r = Run(h, R, T, shift, sigma, l, 'c', f)
    phi2, t2, r2 = Run(h2, R, T, shift, sigma, l, 'c', f)
    phi4, t4, r4 = Run(h4, R, T, shift, sigma, l, 'c', f)
    
    
    plt.plot(t4, np.linalg.norm(phi4, axis=1), label='phi 4')
    plt.plot(t2, np.linalg.norm(phi2, axis=1), label='phi 2')
    plt.plot(t, np.linalg.norm(phi, axis=1), label='phi')
    
    plt.legend()
    plt.show()
        
    phi = np.delete(phi, np.arange(1, phi[:,0].size, 2), axis=0)
    phi = np.delete(phi, np.arange(1, phi[:,0].size, 2), axis=0)
    phi = np.delete(phi, np.arange(1, phi[0,:].size, 2), axis=1)
    phi = np.delete(phi, np.arange(1, phi[0,:].size, 2), axis=1)

    phi2_small = np.delete(phi2, np.arange(1, phi2[:,0].size, 2), axis=0)
    phi2_small = np.delete(phi2_small, np.arange(1, phi2[0,:].size, 2), axis=1)
    
    np.save('phi_fir/conv_phi4.npy', phi4)
    np.save('phi_fir/conv_phi2.npy', phi2_small)
    np.save('phi_fir/conv_phi.npy', phi)
    np.save('phi_fir/conv_t.npy', t4, allow_pickle=False)

    if (phi4.shape != phi2_small.shape) or (phi2_small.shape != phi.shape):
        print("Shapes do not match!")
        print(phi4.shape, phi2_small.shape)
        print(phi2_small.shape, phi.shape)
        
    num = np.linalg.norm(phi4-phi2_small, axis=1)
    denom = np.linalg.norm(phi2_small-phi, axis=1)
    Q = num / denom

    np.save('phi_fir/conv.npy', Q, allow_pickle=False)
    np.save('phi_fir/t_conv.npy', t4, allow_pickle=False)
    
    plt.plot(t4, Q)
    plt.grid(True)

    plt.xlabel('$t_*/m$')
    plt.ylabel('$Q(t)$')
    plt.ylim(-0.5, 5.5)

    plt.show()
    

    return Q[-1]
    

def conv_calc(start, ratio, tratio):
    
    phi = np.load('phi_fir/conv_phi.npy')
    phi2 = np.load('phi_fir/conv_phi2.npy')
    phi4 = np.load('phi_fir/conv_phi4.npy')
    t = np.load('phi_fir/conv_t.npy')
    
    idx = int(phi.shape[1] * ratio)
    Tidx = int(len(t) * tratio)
    start = int(phi.shape[1] * start)
    
    phi = phi[:Tidx,start:idx]
    phi2 = phi2[:Tidx,start:idx]
    phi4 = phi4[:Tidx,start:idx]
    t = t[:Tidx]
    
    if (phi4.shape != phi2.shape) or (phi2.shape != phi.shape):
        print("Shapes do not match!")
        print(phi4.shape, phi2.shape)
        print(phi2.shape, phi.shape)


    num = np.linalg.norm(phi4-phi2, ord=None, axis=1)
    print(num)
    denom = np.linalg.norm(phi2-phi, ord=None, axis=1)
    Q = num / denom

    np.save('phi_fir/conv.npy', Q, allow_pickle=False)
    np.save('phi_fir/t_conv.npy', t, allow_pickle=False)
    
    plt.plot(t, Q)
    plt.grid(True)

    plt.xlabel('$t_*/m$')
    plt.ylabel('$Q(t)$')
    plt.ylim(-0.5, 5.5)

    plt.savefig('convergence_first.jpg', dpi=250)
    plt.show()
    

    return Q[-1]
    

