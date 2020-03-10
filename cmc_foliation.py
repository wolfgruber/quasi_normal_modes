# 23.01.20
# Bachelor Thesis: "Scattering of scalar waves on a Schwarzschild black hole"
# Ludwig Wolfgruber

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from ipywidgets import interactive
import ipywidgets as widgets

from scipy.special import sph_harm

from plt_fit import *
from time import time


# Setting global constants. Notice that C is a integration constant and C != c == speed of light
K = 0.5
C = 1

# functions:

def Initialize(t, r, c, m, shift, sigma, l):
    '''Initializes a gaussian pulse and its derivatives for the gives parameters.'''
    ampl = 1 #amplitude

    initphi = ampl * np.exp(-(r-shift)**2/sigma**2)
    initpsi = -2*(r-shift)/sigma**2 * initphi
    initpi = -initphi - initpsi/r**2 * (1 - beta(r,m)*gamma(r,m)/alpha(r,m))

    return initphi, initpsi, initpi


# three parameters alpha, beta and gamma:
def alpha(r, m):
    r1 = r - 1 #  \Omega
    a = r1**2 * r**3 * (2*m*r1 + r) + 1/9 * (3*C*r1**3 + K*r**3)**2
    return np.sqrt(a)/r**2


def beta(r, m):
    b = C*(1-r)**3/r**2 - K*r/3
    return b * alpha(r, m)


def gamma(r, m):
    return 1/alpha(r, m)


def Lu(r, i, m, phi, psi, pi,  l):
    '''Evaluates the left hand sider for all three differential equations.'''
    dr = r[1] - r[0]
    ri = r[i]
    N = r.size

    if i == 0:
        # eq_phi:
        newphi = alga[0] * pi[0] + bet[0] * psi[0]

        # eq_psi:
        newpsi1 = newphi
        newpsi2 = alga[1] * pi[1] + bet[1] * psi[1]
        newpsi3 = alga[2] * pi[2] + bet[2] * psi[2]
        newpsi = (-3*newpsi1 + 4*newpsi2 - newpsi3) / (2 * dr)

        # eq_pi:
        newpi1 = r[0]**2 * (alga[0] * psi[0] + bet[0] * pi[0])
        newpi2 = r[1]**2 * (alga[1] * psi[1] + bet[1] * pi[1])
        newpi3 = r[2]**2 * (alga[2] * psi[2] + bet[2] * pi[2])
        newpi = (-3*newpi1 + 4*newpi2 - newpi3) / (2 * dr * ri**2)
        newpi = newpi - (12 * (1-ri)*(ri+m * (2*ri-1))/(6*ri**2) + l*(l+1)/ri**2) * phi[i]

    elif i == N-1:
        # eq_phi:
        newphi = alga[N-1] * pi[N-1] + bet[N-1] * psi[N-1]

        # eq_psi:
        newpsi1 = newphi
        newpsi2 = alga[N-2] * pi[N-2] + bet[N-2] * psi[N-2]
        newpsi3 = alga[N-3] * pi[N-3] + bet[N-3] * psi[N-3]
        newpsi = (3*newpsi1 - 4*newpsi2 + newpsi3) / (2 * dr)

        # eq_pi:
        newpi1 = r[N-1]**2 * (alga[N-1] * psi[N-1] + bet[N-1] * pi[N-1])
        newpi2 = r[N-2]**2 * (alga[N-2] * psi[N-2] + bet[N-2] * pi[N-2])
        newpi3 = r[N-3]**2 * (alga[N-3] * psi[N-3] + bet[N-3] * pi[N-3])
        newpi = (3*newpi1 - 4*newpi2 + newpi3) / (2 * dr * ri**2)
        newpi = newpi - (12 * (1-ri)*(ri+m * (2*ri-1))/(6*ri**2) + l*(l+1)/ri**2) * phi[i]

    else:
        # eq_phi:
        newphi = alga[i] * pi[i] + bet[i] * psi[i]

        # eq_psi:
        newpsi1 = alga[i+1] * pi[i+1] + bet[i+1] * psi[i+1]
        newpsi2 = alga[i-1] * pi[i-1] + bet[i-1] * psi[i-1]
        newpsi = (newpsi1 - newpsi2) / (2 * dr)

        # eq_pi:
        newpi1 = r[i+1]**2 * (alga[i+1] * psi[i+1] + bet[i+1] * pi[i+1])
        newpi2 = r[i-1]**2 * (alga[i-1] * psi[i-1] + bet[i-1] * pi[i-1])
        newpi = (newpi1 - newpi2) / (2 * dr * ri**2)
        newpi = newpi - (12 * (1-ri)*(ri+m * (2*ri-1))/(6*ri**2) + l*(l+1)/ri**2) * phi[i]

    return newphi, newpsi, newpi


def TimeStepRK(dt, r, dr, m, phi, psi, pi, l): 
    '''Computes a timestep with the Runge-Kutta 2 method'''
    eps = 0.5
    N = phi.size
    newphi = np.zeros(N)
    newpsi = np.zeros(N)
    newpi = np.zeros(N)

    for i in range(N):
        k1 = Lu(r, i, m, phi, psi, pi,  l)
        k2 = Lu(r, i, m, phi + k1[0]*dt/2, psi + k1[1]*dt/2, pi + k1[2]*dt/2, l)
        k3 = Lu(r, i, m, phi + k2[0]*dt/2, psi + k2[1]*dt/2, pi + k2[2]*dt/2, l)
        k4 = Lu(r, i, m, phi + k3[0]*dt, psi + k3[1]*dt, pi + k3[2]*dt, l)

        if (i > 1) and (i < N-2):
            newphi[i] = phi[i] + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) - eps * (phi[i+2] - 4*phi[i+1] + 6*phi[i] - 4*phi[i-1] + phi[i-2]) / 16
            newpsi[i] = psi[i] + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) - eps * (psi[i+2] - 4*psi[i+1] + 6*psi[i] - 4*psi[i-1] + psi[i-2]) / 16
            newpi[i]  = pi[i]  + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) - eps * (pi[i+2] - 4*pi[i+1] + 6*pi[i] - 4*pi[i-1] + pi[i-2]) / 16

        else:
            newphi[i] = phi[i] + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            newpsi[i] = psi[i] + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            newpi[i]  = pi[i]  + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    return newphi, newpsi, newpi

def Run(dr, f, T, shift, sigma, l_max, mode):
    '''Runs a simulation for the given values of l. "mode" is set equal to "c"
    if computing the convergence.'''
    print('Initializing')
    # some constants:
    G = 1
    c = 1
    M = 0.5
    m = G * M / c**2 #the mass with included constants

    r = np.arange(2*m/(1+2*m), 1, dr)
    Nr = r.size

    dt = dr * f
    t = np.arange(0, T, dt)
    Nt = t.size

    # computing dictionaries for faster access to alpha/gamma & beta
    global alga #  alpha(r,m)/gamma(r,m)
    alga = {}
    global bet #  beta(r,m)
    bet = {}

    for i in range(Nr): #  filling the dictionaries with values
        alga[i] = alpha(r[i],m)/gamma(r[i],m)
        bet[i] = beta(r[i],m)

    phi_l = np.empty((Nt,Nr))
    psi = np.empty((Nt,Nr))
    pi = np.empty((Nt,Nr))

    start = 0

    if mode == 'c':
        start = l_max - 1

    for l in range(start, l_max):
        print('Time evolution for l = {:2d}'.format(l))
        phi_l[0,:], psi[0,:], pi[0,:] = Initialize(t, r, c, m, shift, sigma, l) # computing the initial data

        for j in range(1,Nt): # going forward in time
            if t[j] % 2 == 0:
                print('    at time = {}'.format(t[j]))
            phi_l[j,:], psi[j,:], pi[j,:] = TimeStepRK(dt, r, dr, m, phi_l[j-1,:], psi[j-1,:], pi[j-1,:], l)

        phi_l[:,:] = sph_harm(0, l, 0, 0).real * phi_l[:,:]
        
        if mode != 'c':
            export(phi=phi_l, l=l, loc='phi_cmc2')
        
            if l == 0:
                np.save('phi_cmc2/t.npy', t, allow_pickle=False)
                np.save('phi_cmc2/r.npy', r, allow_pickle=False)

    return phi_l, t, r


def convergence(dr, f, T, shift, sigma, l):
    '''Compute the convergence as a function of time for the given value of l.'''
    h = dr/4
    h2 = dr/2
    h4 = dr

    ti0 = time()
    phi, t, r = Run(h, f, T, shift, sigma, l+1, 'c')
    ti1 = time() - ti0
    print("took "+str(ti1)+" time\n")
    phi2, t2, r2 = Run(h2, f, T, shift, sigma, l+1, 'c')
    ti2 = time() - ti1 - ti0
    print("took "+str(ti2)+" time\n")
    phi4, t4, r4 = Run(h4, f, T, shift, sigma, l+1, 'c')
    ti3 = time() - ti2 - ti1 - ti0
    print("took "+str(ti3)+" time\n")
    
    plt.plot(t4, np.linalg.norm(phi4, axis=1), label='phi 4')
    plt.plot(t2, np.linalg.norm(phi2, axis=1), label='phi 2')
    plt.plot(t, np.linalg.norm(phi, axis=1), label='phi')
    
    plt.legend()
    plt.show()

    # reducing the sizes of the arrays, to make a subtraction possible
    phi = np.delete(phi, np.arange(1, phi[:,0].size, 2), axis=0)
    phi = np.delete(phi, np.arange(1, phi[:,0].size, 2), axis=0)
    phi = np.delete(phi, np.arange(1, phi[0,:].size, 2), axis=1)
    phi = np.delete(phi, np.arange(1, phi[0,:].size, 2), axis=1)

    phi2 = np.delete(phi2, np.arange(1, phi2[:,0].size, 2), axis=0)
    phi2 = np.delete(phi2, np.arange(1, phi2[0,:].size, 2), axis=1)

    if (phi4.shape != phi2.shape) or (phi2.shape != phi.shape):
        print("Shapes do not match!")
        print(phi4.shape, phi2.shape)
        print(phi2.shape, phi.shape)
    #else:
    #    print("Seems alright.")
    #    print(phi4.shape, phi2_small.shape)
    #    print(phi2.shape, phi.shape)
    

    num = np.linalg.norm(phi4-phi2, axis=1)
    denom = np.linalg.norm(phi2-phi, axis=1)
    Q = num / denom
    
    np.save('phi_cmc2/conv.npy', Q, allow_pickle=False)
    np.save('phi_cmc2/t_conv.npy', t4, allow_pickle=False)
    
    plt.plot(t4, Q)
    plt.grid(True)

    plt.xlabel('t/m')
    plt.ylabel('$Q(t)$')
    plt.ylim(-0.5, 5.5)

    plt.show()

    return

