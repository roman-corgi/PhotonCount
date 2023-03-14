# -*- coding: utf-8 -*-
"""Get histograms of mean and noise for photon-counted frames, corrected
and uncorrected."""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.special import gamma, factorial
from scipy.stats import chisquare, chi2


from emccd_detect.emccd_detect import EMCCDDetect
from PhotonCount.corr_photon_count import (get_count_rate,
                                        get_counts_uncorrected)

if __name__ == '__main__':
    pix_row = 50 #number of rows and number of columns
    fluxmap = .5*np.ones((pix_row,pix_row)) #photon flux map, photons/s
    frametime = .1  # s (adjust lambda by adjusting this)
    em_gain = 5000. # e-

    emccd = EMCCDDetect(
        em_gain=em_gain,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=8.33e-4,  # e-/pix/s
        cic=0.01,  # e-/pix/frame
        read_noise=100.,  # e-/pix/frame
        bias=0,  # e-; 0 for simplicity
        qe=0.9,  # quantum efficiency, e-/photon
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=1,  # for simplicity, set this to 1 so there's no data loss when converting back to e-
        nbits=64, # number of ADU bits
        numel_gain_register=604 #number of gain register elements
        )

    thresh = emccd.em_gain/10 # threshold
    N = 25 #number of frames

    if np.average(frametime*fluxmap) > 0.5:
        warnings.warn('average # of photons/pixel is > 0.5.  Decrease frame '
        'time to get lower average # of photons/pixel.')

    if emccd.read_noise <=0:
       warnings.warn('read noise should be greater than 0 for effective '
       'photon counting')
    if thresh < 4*emccd.read_noise:
       warnings.warn('thresh should be at least 4 or 5 times read_noise for '
       'accurate photon counting')

    avg_ph_flux = np.mean(fluxmap)
    L = avg_ph_flux*emccd.qe*frametime + emccd.dark_current*frametime\
        +emccd.cic
    L_dark = emccd.dark_current*frametime + emccd.cic
    g = em_gain
    T = thresh
    # uncorrected case
    def e_coinloss(L):
        return (1 - np.exp(-L)) / L
    def e_thresh(g, L, T):
        return (np.exp(-T/g)* (
            T**2 * L**2
            + 2*g * T * L * (3 + L)
            + 2*g**2 * (6 + 3*L + L**2)
            )
            / (2*g**2 * (6 + 3*L + L**2))
            )

    # corrected case
    # assuming lambda relatively small, expansion to 3rd order
    def mean(g, L, T, N):
        return (((-1 + np.e**L)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*
            (8*np.e**(3*(L + T/g))*g**6*(6 + L*(3 + L))**3*(6*g - 5*T)*(g**2 - 2*g*T + 2*T**2) -
            28*np.e**(2*(L + T/g))*g**4*N*(6 + L*(3 + L))**2*(6*g - 5*T)*(g**2 - 2*g*T + 2*T**2)*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2) + 12*np.e**(L + T/g)*g**2*N**2*(6 + L*(3 + L))*
                (6*g - 5*T)*(g**2 - 2*g*T + 2*T**2)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2 -
            N**3*(6*g - 5*T)*(g**2 - 2*g*T + 2*T**2)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**3 +
            4*np.e**(3*L + (2*T)/g)*g**4*N*(6 + L*(3 + L))**2*(100*g**5*(6 + L*(3 + L)) -
                6*g**4*(270 + 31*L*(3 + L))*T + 6*g**3*(328 + L*(45 + 22*L))*T**2 +
                7*g**2*(-120 + L*(72 + 7*L))*T**3 + 14*g*(-30 + L)*L*T**4 - 70*L**2*T**5) +
            np.e**L*N**3*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2*
                (52*g**5*(6 + L*(3 + L)) - 2*g**4*(402 + 49*L*(3 + L))*T + 2*g**3*(456 + L*(75 + 34*L))*T**2 +
                3*g**2*(-120 + L*(72 + 7*L))*T**3 + 6*g*(-30 + L)*L*T**4 - 30*L**2*T**5) -
            24*np.e**(2*L + T/g)*g**2*N**2*(6 + L*(3 + L))*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*
                (16*g**5*(6 + L*(3 + L)) - 6*g**4*(42 + 5*L*(3 + L))*T + 3*g**3*(98 + L*(15 + 7*L))*T**2 +
                g**2*(-120 + L*(72 + 7*L))*T**3 + 2*g*(-30 + L)*L*T**4 - 10*L**2*T**5) -
            np.e**(2*L)*N**3*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*
                (184*g**7*(6 + L*(3 + L))**2 - 4*g**6*(6 + L*(3 + L))*(570 + 43*L*(3 + L))*T -
                8*g**5*(-1548 + (-3 + L)*L*(114 + L*(57 + 2*L)))*T**2 +
                4*g**4*(-1080 + L*(1656 + L*(789 + L*(279 + 22*L))))*T**3 +
                2*g**3*L*(-2160 + L*(300 + L*(222 + 71*L)))*T**4 - 9*g**2*L**2*(200 + L*(32 + 3*L))*T**5 -
                18*g*L**3*(20 + 3*L)*T**6 - 30*L**4*T**7) + 12*np.e**(3*L + T/g)*g**2*N**2*(6 + L*(3 + L))*
                (48*g**7*(6 + L*(3 + L))**2 - 4*g**6*(6 + L*(3 + L))*(162 + 11*L*(3 + L))*T -
                4*g**5*(-972 + L*(-216 + L*(-39 + L*(30 + L))))*T**2 +
                4*g**4*(-360 + L*(522 + L*(246 + L*(87 + 7*L))))*T**3 + 2*g**3*L*(-720 + L*(90 + L*(69 + 22*L)))*
                T**4 - 3*g**2*L**2*(200 + L*(32 + 3*L))*T**5 - 6*g*L**3*(20 + 3*L)*T**6 - 10*L**4*T**7) +
            np.e**(3*L)*N**3*(400*g**9*(6 + L*(3 + L))**3 + 8*g**8*(-15 + L)*(18 + L)*(6 + L*(3 + L))**2*T -
                8*g**7*(6 + L*(3 + L))*(-1152 + L*(558 + L*(381 + L*(153 + 14*L))))*T**2 +
                4*g**6*(-4320 + L*(12096 + L*(8712 + L*(3726 + L*(495 + (12 - 13*L)*L)))))*T**3 +
                4*g**5*L*(-6480 + L*(2664 + L*(3402 + L*(1794 + L*(381 + 41*L)))))*T**4 +
                2*g**4*L**2*(-8640 + L*(-1872 + L*(396 + L*(417 + 61*L))))*T**5 - 6*g**3*L**3*(1080 + L*(412 + 71*L))*
                T**6 - g**2*L**4*(1440 + L*(504 + 65*L))*T**7 - 2*g*L**5*(90 + 19*L)*T**8 - 10*L**6*T**9)))/
                (384*np.e**(4*L)*g**11*N**3*(6 + L*(3 + L))**4))

    def var(g, L, T, N):
        return (np.e**(-5*L + T/g)*(-1 + np.e**L)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*
            (16*np.e**(4*(L + T/g))*g**8*(6 + L*(3 + L))**4*(4*g**2 - 8*g*T + 5*T**2)**2 - 240*np.e**(3*(L + T/g))*g**6*(6 + L*(3 + L))**3*N*(4*g**2 - 8*g*T + 5*T**2)**2*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2) + 336*np.e**(2*(L + T/g))*g**4*(6 + L*(3 + L))**2*N**2*(4*g**2 - 8*g*T + 5*T**2)**2*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2 - 108*np.e**(L + T/g)*g**2*(6 + L*(3 + L))*N**3*(4*g**2 - 8*g*T + 5*T**2)**2*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**3 + 9*N**4*(4*g**2 - 8*g*T + 5*T**2)**2*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**4 - 36*np.e**L*N**4*(4*g**2 - 8*g*T + 5*T**2)*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**3*(12*g**4*(6 + L*(3 + L)) - 12*g**3*(10 + L*(3 + L))*T - 2*g**2*(-30 + L*(9 + L))*T**2 +
                2*g*L*(15 + L)*T**3 + 5*L**2*T**4) - 672*np.e**(3*L + (2*T)/g)*g**4*(6 + L*(3 + L))**2*N**2*(4*g**2 - 8*g*T + 5*T**2)*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(10*g**4*(6 + L*(3 + L)) - 2*g**3*(54 + 5*L*(3 + L))*T - 2*g**2*(-30 + L*(9 + L))*T**2 +
                2*g*L*(15 + L)*T**3 + 5*L**2*T**4) + 48*np.e**(4*L + (3*T)/g)*g**6*(6 + L*(3 + L))**3*N*(4*g**2 - 8*g*T + 5*T**2)*
                (44*g**4*(6 + L*(3 + L)) - 4*g**3*(126 + 11*L*(3 + L))*T - 10*g**2*(-30 + L*(9 + L))*T**2 + 10*g*L*(15 + L)*T**3 + 25*L**2*T**4) +
            36*np.e**(2*L + T/g)*g**2*(6 + L*(3 + L))*N**3*(4*g**2 - 8*g*T + 5*T**2)*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2*
                (100*g**4*(6 + L*(3 + L)) - 4*g**3*(258 + 25*L*(3 + L))*T - 18*g**2*(-30 + L*(9 + L))*T**2 + 18*g*L*(15 + L)*T**3 + 45*L**2*T**4) -
            36*np.e**(3*L)*N**4*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(12*g**4*(6 + L*(3 + L)) - 12*g**3*(10 + L*(3 + L))*T -
                2*g**2*(-30 + L*(9 + L))*T**2 + 2*g*L*(15 + L)*T**3 + 5*L**2*T**4)*(48*g**6*(6 + L*(3 + L))**2 - 288*g**5*(6 + L*(3 + L))*T -
                4*g**4*(-180 + L*(180 + L*(123 + L*(48 + 5*L))))*T**2 - 8*g**3*L*(-90 + L*(3 + L)*(-3 + 2*L))*T**3 + 12*g**2*L**2*(25 + L*(7 + L))*T**4 +
                12*g*L**3*(5 + L)*T**5 + 5*L**4*T**6) + 9*np.e**(4*L)*N**4*(48*g**6*(6 + L*(3 + L))**2 - 288*g**5*(6 + L*(3 + L))*T -
                4*g**4*(-180 + L*(180 + L*(123 + L*(48 + 5*L))))*T**2 - 8*g**3*L*(-90 + L*(3 + L)*(-3 + 2*L))*T**3 + 12*g**2*L**2*(25 + L*(7 + L))*T**4 +
                12*g*L**3*(5 + L)*T**5 + 5*L**4*T**6)**2 + 18*np.e**(2*L)*N**4*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)**2*
                (480*g**8*(6 + L*(3 + L))**2 - 192*g**7*(6 + L*(3 + L))*(48 + 5*L*(3 + L))*T + 32*g**6*(2232 + L*(1044 + L*(420 + L*(39 + 11*L))))*T**2 +
                288*g**5*(-150 + L*(45 + L*(29 + L*(15 + L))))*T**3 + 12*g**4*(900 + L*(-2340 + L*(-459 + L*(-108 + 19*L))))*T**4 -
                24*g**3*L*(-450 + L*(255 + L*(69 + 16*L)))*T**5 - 12*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 60*g*L**3*(15 + L)*T**7 + 75*L**4*T**8) +
            48*np.e**(4*L + (2*T)/g)*g**4*(6 + L*(3 + L))**2*N**2*(716*g**8*(6 + L*(3 + L))**2 - 8*g**7*(6 + L*(3 + L))*(1914 + 179*L*(3 + L))*T +
                12*g**6*(11076 + L*(4692 + L*(1757 + L*(82 + 37*L))))*T**2 + 112*g**5*(-810 + L*(243 + L*(147 + 5*L*(15 + L))))*T**3 +
                28*g**4*(900 + L*(-2160 + L*(-387 + L*(-87 + 16*L))))*T**4 - 84*g**3*L*(-300 + L*(160 + L*(41 + 9*L)))*T**5 -
                28*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 140*g*L**3*(15 + L)*T**7 + 175*L**4*T**8) - 36*np.e**(3*L + T/g)*g**2*(6 + L*(3 + L))*N**3*
                (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T + L**2*T**2)*(1200*g**8*(6 + L*(3 + L))**2 - 2400*g**7*(6 + L*(3 + L))*(10 + L*(3 + L))*T +
                32*g**6*(6084 + L*(2736 + L*(1071 + L*(81 + 26*L))))*T**2 + 32*g**5*(-3870 + L*(1161 + L*(729 + 25*L*(15 + L))))*T**3 +
                12*g**4*(2700 + L*(-6780 + L*(-1281 + L*(-296 + 53*L))))*T**4 - 8*g**3*L*(-4050 + L*(2235 + L*(591 + 134*L)))*T**5 -
                36*g**2*L**2*(-375 + L*(15 + 4*L))*T**6 + 180*g*L**3*(15 + L)*T**7 + 225*L**4*T**8) + 36*np.e**(4*L + T/g)*g**2*(6 + L*(3 + L))*N**3*
                (1248*g**10*(6 + L*(3 + L))**3 - 96*g**9*(6 + L*(3 + L))**2*(228 + 13*L*(3 + L))*T -
                16*g**8*(6 + L*(3 + L))*(-9828 + L*(-828 + L*(417 + L*(537 + 52*L))))*T**2 +
                16*g**7*(-33480 + L*(22788 + L*(26406 + L*(16011 + L*(4011 + L*(657 + 23*L))))))*T**3 +
                24*g**6*(5400 + L*(-22860 + L*(-6294 + L*(393 + L*(1843 + 13*L*(37 + 5*L))))))*T**4 -
                24*g**5*L*(-8100 + L*(8580 + L*(4809 + L*(1613 + L*(109 + L)))))*T**5 - 4*g**4*L**2*(-32400 + L*(4950 + L*(5307 + L*(1971 + 166*L))))*T**6 +
                4*g**3*L**3*(12150 + L*(1950 - L*(51 + 95*L)))*T**7 + 6*g**2*L**4*(1800 + L*(405 + 37*L))*T**8 + 30*g*L**5*(45 + 7*L)*T**9 + 75*L**6*T**10)))/(4608*g**14*(6 + L*(3 + L))**5*N**5)

    def mean(g, L, T, N):
        return (L * (4 * np.exp((2 * T) / g) * g**2 + 6 * np.exp(T / g) * g**2 * N + 12 * g**2 * N**2 -
                 8 * np.exp((2 * T) / g) * g * T - 6 * np.exp(T / g) * g * N * T +
                 5 * np.exp((2 * T) / g) * T**2)) / (12 * g**2 * N**2) + 1 / (24 * g**3 * N**2) * \
           np.exp(-(T / g)) * L**2 * (-28 * np.exp((2 * T) / g) * g**3 - 18 * np.exp(T / g) * g**3 * N +
                                   24 * np.exp((2 * T) / g) * g**3 * N - 12 * g**3 * N**2 + 12 * np.exp(T / g) * g**3 * N**2 +
                                   56 * np.exp((2 * T) / g) * g**2 * T + 4 * np.exp((3 * T) / g) * g**2 * T + 18 * np.exp(T / g) * g**2 * N * T -
                                   42 * np.exp((2 * T) / g) * g**2 * N * T - 35 * np.exp((2 * T) / g) * g * T**2 -
                                   8 * np.exp((3 * T) / g) * g * T**2 + 24 * np.exp((2 * T) / g) * g * N * T**2 +
                                   5 * np.exp((3 * T) / g) * T**3) + 1 / (144 * g**4 * N**2) * \
           np.exp(-((2 * T) / g)) * L**3 * (248 * np.exp((2 * T) / g) * g**4 + 84 * np.exp(T / g) * g**4 * N -
                                            288 * np.exp((2 * T) / g) * g**4 * N + 24 * g**4 * N**2 - 72 * np.exp(T / g) * g**4 * N**2 +
                                            48 * np.exp((2 * T) / g) * g**4 * N**2 -496 * np.exp((2 * T) / g) * g**3 * T - \
           168 * np.exp((3 * T) / g) * g**3 * T - 4 * np.exp((4 * T) / g) * g**3 * T - 84 * np.exp(T / g) * g**3 * N * T + \
           468 * np.exp((2 * T) / g) * g**3 * N * T + 138 * np.exp((3 * T) / g) * g**3 * N * T - \
           36 * np.exp((2 * T) / g) * g**3 * N**2 * T + 310 * np.exp((2 * T) / g) * g**2 * T**2 + \
           336 * np.exp((3 * T) / g) * g**2 * T**2 + 12 * np.exp((4 * T) / g) * g**2 * T**2 - \
           252 * np.exp((2 * T) / g) * g**2 * N * T**2 - 276 * np.exp((3 * T) / g) * g**2 * N * T**2 - \
           210 * np.exp((3 * T) / g) * g * T**3 - 13 * np.exp((4 * T) / g) * g * T**3 + \
           174 * np.exp((3 * T) / g) * g * N * T**3 + 5 * np.exp((4 * T) / g) * T**4)

    def var2(g, L, T, N):
        return (L*(-576*g**8*L**5*N**5 +
        576*np.e**(T/g)*g**7*L**4*N**4*
         (-7*g*L + 6*g*(1 + L)*N + 7*L*T) +
        12*np.e**((9*T)/g)*g**2*(4*g**2 - 8*g*T + 5*T**2)**2*
         (12*g**2 - g*(-6 + L)*L*T + L**2*T**2) -
        48*np.e**((2*T)/g)*g**6*L**3*N**3*
         (g**2*(395*L**2 - 180*L*(2 + 3*L)*N +
              12*(21 + 24*L + 13*L**2)*N**2) +
           2*g*L*(-395*L + 180*(1 + 2*L)*N - 18*L*N**2)*T +
           L**2*(457 - 252*N)*T**2) +
        48*np.e**((3*T)/g)*g**5*L**2*N**2*
         (2*g**3*(-434*L**3 + 3*L**2*(215 + 292*L)*N -
              6*(-42 + 75*L + 132*L**2 + 86*L**3)*N**2 +
              36*L*(9 + 5*L + 2*L**2)*N**3) -
           6*g**2*L*(-434*L**2 + L*(430 + 661*L)*N -
              2*(75 + 186*L + 130*L**2)*N**2 +
              18*L*(1 + L)*N**3)*T +
           g*L**2*(-2821*L + 6*(253 + 519*L)*N -
              6*(150 + 101*L)*N**2)*T**2 +
           L**3*(1085 - 672*N - 174*N**2)*T**3) -
        4*np.e**((4*T)/g)*g**4*L*N*
         (4*g**4*(3844*L**4 - 72*L**3*(71 + 124*L)*N +
              3*L*(-2232 + 2535*L + 3048*L**2 + 2224*L**3)*
               N**2 - 108*
               (-18 - 48*L + 75*L**2 + 42*L**3 + 16*L**4)*
               N**3 + 36*L**2*(33 + 12*L + 4*L**2)*N**4) -
           4*g**3*L*(15376*L**3 - 72*L**2*(213 + 425*L)*N +
              9*(-744 + 1690*L + 2252*L**2 + 1999*L**3)*
               N**2 - 54*(-60 + 228*L + 118*L**2 + 53*L**3)*
               N**3 + 108*L*(6 + 3*L + 2*L**2)*N**4)*T +
           4*g**2*L**2*
            (24986*L**2 - 18*L*(923 + 2372*L)*N +
              3*(3045 + 5130*L + 6506*L**2)*N**2 -
              54*(108 + 29*L + 21*L**2)*N**3 + 81*L**2*N**4)*
            T**2 - 4*g*L**3*
            (19220*L - 18*(355 + 1557*L)*N +
              3*(978 + 3187*L)*N**2 + 54*(29 + 8*L)*N**3)*
            T**3 + L**4*(24025 - 30240*N + 8628*N**2)*T**4)\
         - 24*np.e**((5*T)/g)*g**3*N*
         (4*g**5*(-868*L**4 +
              3*L**2*(-1265 + 278*L + 584*L**2)*N -
              12*L*(-63 - 351*L + 90*L**2 + 86*L**3)*N**2 +
              72*(-3 - 6*L - 9*L**2 + 4*L**3 + 2*L**4)*N**3)\
            + 2*g**4*L*
            (-1736*(-4 + L)*L**3 +
              4*L*(3795 - 1251*L - 3012*L**2 + 857*L**3)*N -
              9*(168 + 1704*L - 474*L**2 - 619*L**3 +
                 220*L**4)*N**2 +
              6*(108 + 294*L - 30*L**2 - 51*L**3 + 46*L**4)*
               N**3)*T -
           2*g**3*L**2*
            (-868*L**2*(-13 + 8*L) +
              (8970 - 5421*L - 16872*L**2 + 12956*L**3)*N +
              (-8424 + 2592*L + 6099*L**2 - 6891*L**3)*
               N**2 + 3*(228 + 372*L + 54*L**2 + 253*L**3)*
               N**3)*T**2 +
           2*g**2*L**3*
            (868*(10 - 13*L)*L +
              (-2085 - 11124*L + 20033*L**2)*N +
              (54 + 3063*L - 9804*L**2)*N**2 +
              6*(174 + 51*L + 127*L**2)*N**3)*T**3 +
           g*L**4*(1085*(-5 + 16*L) + (6030 - 29442*L)*N +
              6*(-248 + 2197*L)*N**2 - 522*L*N**3)*T**4 +
           L**5*(-5425 + 8870*N - 3654*N**2)*T**5) +
        np.e**((8*T)/g)*(4*g**2 - 8*g*T + 5*T**2)*
         (864*g**6*(-21*L + (2 + 20*L)*N) +
           48*g**5*(-378*(-2 + L)*L +
              (-36 - 702*L + 357*L**2 + 2*L**3)*N)*T -
           4*g**4*L*(1134*(5 - 8*L) +
              (-5184 + 8568*L + 108*L**2 - 12*L**3 + L**4)*N)
             *T**2 + 8*g**3*L**2*
            (-2835 + (2682 + 75*L - 18*L**2 + 2*L**3)*N)*T**3
             + g**2*L**3*(-300 + 156*L - 25*L**2)*N*T**4 +
           6*g*L**4*(-10 + 3*L)*N*T**5 - 5*L**5*N*T**6) -
        4*np.e**((6*T)/g)*g**2*N*
         (24*g**6*(2*L**2*(-2346 + 271*L) -
              9*L*(-165 - 680*L + 88*L**2)*N +
              12*(-18 - 99*L - 126*L**2 + 22*L**3)*N**2) -
           8*g**5*(2*L**2*
               (-21114 + 3252*L - 1068*L**2 + 31*L**3) -
              9*L*(-990 - 5625*L + 876*L**2 - 374*L**3 +
                 8*L**4)*N +
              3*(-216 - 2268*L - 3366*L**2 + 657*L**3 -
                 432*L**4 + 4*L**5)*N**2)*T +
           g**4*L*(8*L*
               (-45747 + 10569*L - 8544*L**2 + 1192*L**3) -
              72*(-585 - 5640*L + 1176*L**2 - 1436*L**3 +
                 198*L**4)*N +
              9*(-3456 - 6864*L + 1516*L**2 - 4268*L**3 +
                 569*L**4)*N**2)*T**2 -
           12*g**3*L**2*
            (-11730 + 5420*L - 9256*L**2 + 2786*L**3 -
              3*(-3930 + 1470*L - 4507*L**2 + 1432*L**3)*N +
              (-1008 + 240*L - 4863*L**2 + 1631*L**3)*N**2)*
            T**3 + g**2*L**3*
            (20325 - 85440*L + 51568*L**2 -
              36*(375 - 3360*L + 2243*L**2)*N +
              12*(-60 - 3513*L + 2617*L**2)*N**2)*T**4 -
           3*g*L**4*(-8900 + 12845*L - 60*(-205 + 339*L)*N +
              6*(-696 + 1339*L)*N**2)*T**5 +
           L**5*(11800 - 18900*N + 7569*N**2)*T**6) +
        12*np.e**((7*T)/g)*g*
         (16*g**7*(2534*L**2 - 18*L*(31 + 196*L)*N +
              9*(11 + 56*L + 112*L**2)*N**2) +
           4*g**6*(-40544*L**2 +
              4*L*(1674 + 13554*L + 126*L**2 - 7*L**3)*N +
              3*(-264 - 1950*L - 4715*L**2 - 136*L**3 +
                 8*L**4)*N**2)*T +
           4*g**5*(65884*L**2 -
              2*L*(3627 + 42516*L + 1008*L**2 - 154*L**3 +
                 14*L**4)*N +
              (468 + 6156*L + 20259*L**2 + 1596*L**3 -
                 252*L**4 + 23*L**5)*N**2)*T**2 -
           4*g**4*L*(50680*L -
              2*(1395 + 31653*L + 1638*L**2 - 483*L**3 +
                 70*L**4)*N +
              (2286 + 13713*L + 2550*L**2 - 780*L**3 +
                 115*L**4)*N**2)*T**3 +
           g**3*L**2*(63350 -
              8*(9630 + 1260*L - 707*L**2 + 147*L**3)*N +
              (15276 + 7740*L - 4548*L**2 + 967*L**3)*N**2)*
            T**4 + g**2*L**3*N*
            (7*(450 - 585*L + 184*L**2) +
              (-2400 + 3294*L - 1061*L**2)*N)*T**5 +
           g*L**4*N*(245*(5 - 3*L) + (-990 + 607*L)*N)*
            T**6 + 5*L**5*(35 - 29*N)*N*T**7)))/(20736.*np.e**((4*T)/g)*g**8*N**5)

    # now from p.22 of Remarkable 2 note:
    # def mean(g, L, T, N):
    #     return (L*(-36*g**3*(1 + 2*g)*L**2*N**2 -
    #        18*np.e**(T/g)*g**2*(1 + 2*g)*L**2*N*(g - T) -
    #        3*np.e**((2*T)/g)*g*(1 + 2*g)*L**2*
    #         (4*g**2 - 8*g*T + 5*T**2) +
    #        np.e**((1 + 4*T)/(2.*g))*(4*g**2 - 8*g*T + 5*T**2)*
    #         (2*g**2*(6 + L*(-3 + 4*L)) + 2*g*(3 - 2*L)*L*T +
    #           L**2*T**2) +
    #        12*np.e**(1/(2.*g))*g**2*
    #         (2*g**2*(-3*L*N + 6*N**2 +
    #              L**2*(4 + 3*(-1 + N)*N)) +
    #           2*g*L*(-8*L + 3*N + 6*L*N)*T +
    #           L**2*(10 - 9*N)*T**2) -
    #        6*np.e**((1 + 2*T)/(2.*g))*g*
    #         (2*g**3*(-12*(-1 + L)*L + (-6 + L*(-9 + 8*L))*N) +
    #           12*g**2*(2*L*(-2 + 3*L) + N + (3 - 5*L)*L*N)*T +
    #           g*L*(30 - 78*L + (-24 + 73*L)*N)*T**2 +
    #           L**2*(30 - 29*N)*T**3)))/(144.*np.e**(1/(2.*g))*g**4*N**2)

    # def var2(g, L, T, N):
    #     return (L*(-1296*g**6*(1 + 2*g)**2*L**5*N**5 -
    #         1296*np.e**(T/g)*g**5*(1 + 2*g)**2*L**5*N**4*(g - T) -
    #         108*np.e**((3*T)/g)*g**3*L**5*(N + 2*g*N)**2*(g - T)*
    #          (4*g**2 - 8*g*T + 5*T**2) -
    #         36*np.e**((1 + 10*T)/(2.*g))*g**3*(1 + 2*g)*L**2*
    #          (4*g**2 - 8*g*T + 5*T**2)**2 -
    #         9*np.e**((4*T)/g)*(g + 2*g**2)**2*L**5*N*
    #          (4*g**2 - 8*g*T + 5*T**2)**2 -
    #         108*np.e**((2*T)/g)*g**4*(1 + 2*g)**2*L**5*N**3*
    #          (11*g**2 - 22*g*T + 13*T**2) +
    #         12*np.e**((1 + 5*T)/g)*g**2*
    #          (4*g**2 - 8*g*T + 5*T**2)**2*
    #          (2*g**2*(6 + L*(-3 + 4*L)) + 2*g*(3 - 2*L)*L*T +
    #            L**2*T**2) +
    #         864*np.e**(1/(2.*g))*g**5*(1 + 2*g)*L**3*N**3*
    #          (2*g**2*(-3*L*N + 6*N**2 +
    #               L**2*(4 + 3*(-1 + N)*N)) +
    #            2*g*L*(-8*L + 3*N + 6*L*N)*T +
    #            L**2*(10 - 9*N)*T**2) -
    #         432*np.e**((1 + 2*T)/(2.*g))*g**4*(1 + 2*g)*L**2*N**2*
    #          (2*g**3*(-4*L**3 + 3*(5 - 3*L)*L**2*N +
    #               (6 + L*(-12 + L*(-9 + 5*L)))*N**2) +
    #            6*g**2*L*(4*N**2 + 2*L*N*(-5 + 3*N) +
    #               L**2*(4 - 9*(-1 + N)*N))*T +
    #            g*L**2*(-26*L + N*(36 - 57*L + (-24 + 73*L)*N))*
    #             T**2 + L**3*(10 + (21 - 29*N)*N)*T**3) +
    #         6*np.e**((1 + 8*T)/(2.*g))*g*(1 + 2*g)*L**2*N*
    #          (4*g**2 - 8*g*T + 5*T**2)*
    #          (8*g**4*(-9 + L*(6 + L*(-3 + 4*L))) +
    #            8*g**3*(9 + L*(-12 + (9 - 10*L)*L))*T +
    #            2*g**2*L*(30 + L*(-39 + 38*L))*T**2 +
    #            2*g*(15 - 14*L)*L**2*T**3 + 5*L**3*T**4) -
    #         144*np.e**(1/g)*g**4*L*N*
    #          (4*g**4*(36*L*(-2 + N)*N**2 + 36*N**3 -
    #               6*L**3*N*(4 + 3*(-1 + N)*N) +
    #               L**4*(4 + 3*(-1 + N)*N)**2 +
    #               3*L**2*N**2*(19 + 12*(-1 + N)*N)) +
    #            8*g**3*L*(-18*(-2 + N)*N**2 +
    #               3*L*N**2*(-19 + 12*N) +
    #               9*L**2*N*(4 + (-3 + N)*N) +
    #               2*L**3*(-4 + 3*N)*(4 + 3*(-1 + N)*N))*T -
    #            4*g**2*L**2*
    #             (-104*L**2 + 6*L*(13 + 27*L)*N -
    #               3*(23 + L*(21 + 31*L))*N**2 +
    #               27*(2 + L**2)*N**3)*T**2 -
    #            4*g*L**3*(-10 + 9*N)*(-8*L + (3 + 6*L)*N)*T**3 +
    #            L**4*(10 - 9*N)**2*T**4) +
    #         72*np.e**((1 + 4*T)/(2.*g))*g**3*(1 + 2*g)*L**2*N*
    #          (2*g**4*(16*L**3 + 24*(-2 + L)*L**2*N +
    #               (-36 + L*(66 + L*(15 + 4*L)))*N**2) +
    #            2*g**3*(-64*L**3 + 48*(3 - 2*L)*L**2*N +
    #               (36 + L*(-132 + 5*L*(-9 + 10*L)))*N**2)*T +
    #            g**2*L*(156*N**2 + 6*L*N*(-52 + 17*N) +
    #               L**2*(208 + (288 - 293*N)*N))*T**2 +
    #            2*g*L**2*(-80*L +
    #               N*(60 - 96*L + (-21 + 139*L)*N))*T**3 +
    #            L**3*(50 + (45 - 82*N)*N)*T**4) -
    #         72*np.e**((1 + 6*T)/(2.*g))*g**2*(1 + 2*g)*L**2*N*
    #          (2*g**5*(-24*(-1 + L)*L**2 +
    #               (33 + 4*L*(-6 + L*(-3 + 2*L)))*N) -
    #            4*g**4*(12*(4 - 5*L)*L**2 +
    #               (33 + 4*L*(-3 + 2*L)*(3 + 4*L))*N)*T +
    #            6*g**3*(4*(13 - 21*L)*L**2 +
    #               (13 + L*(-26 + L*(-27 + 58*L)))*N)*T**2 +
    #            4*g**2*L*(6*L*(-10 + 23*L) +
    #               (15 + (33 - 112*L)*L)*N)*T**3 +
    #            3*g*L**2*(25 - 105*L + (-15 + 94*L)*N)*T**4 +
    #            5*L**3*(15 - 14*N)*T**5) +
    #         144*np.e**((1 + T)/g)*g**3*N*
    #          (4*g**5*(-48*(-1 + L)*L**4 +
    #               4*L**2*(99 + L*(-15 + L*(-9 + 17*L)))*N -
    #               3*L*(36 + L*(132 + L*(9 + L*(-13 + 20*L))))*
    #                N**2 +
    #               3*(12 +
    #                  L*(18 + L*(20 + L*(10 + L*(-9 + 8*L)))))*
    #                N**3) -
    #            4*g**4*L*(48*(4 - 5*L)*L**3 +
    #               4*L*(198 + L*(-45 + L*(-36 + 91*L)))*N -
    #               3*(36 + L*(264 + L*(27 - 22*L + 82*L**2)))*
    #                N**2 +
    #               6*(9 + L*(14 + 3*L*(9 + L*(-3 + 5*L))))*N**3)*
    #             T + 6*g**3*L**2*
    #             (16*(13 - 21*L)*L**2 +
    #               2*(156 + L*(-65 + L*(-85 + 271*L)))*N -
    #               (324 + L*(54 + 5*L*(-4 + 59*L)))*N**2 +
    #               (38 + L*(146 + L*(-24 + 73*L)))*N**3)*T**2 +
    #            2*g**2*L**3*
    #             (48*L*(-10 + 23*L) +
    #               2*(75 + (222 - 941*L)*L)*N +
    #               (54 + 885*L**2)*N**2 - 87*(2 + L**2)*N**3)*
    #             T**3 + 3*g*L**4*
    #             (100 - 420*L +
    #               N*(-110 + 752*L + (14 - 335*L)*N))*T**4 +
    #            L**5*(-10 + 9*N)*(-30 + 29*N)*T**5) -
    #         np.e**((1 + 4*T)/g)*(4*g**2 - 8*g*T + 5*T**2)*
    #          (16*g**6*(-1116*(-1 + L)*L +
    #               (-108 +
    #                  L*(-1026 +
    #                     L*(1008 + L*(57 + 8*L*(-3 + 2*L)))))*N)\
    #             - 192*g**5*
    #             (93*(2 - 3*L)*L +
    #               (-9 + L*
    #                   (-171 + L*(261 + L*(13 + L*(-7 + 4*L)))))*
    #                N)*T +
    #            12*g**4*L*(372*(5 - 13*L) +
    #               (-1728 +
    #                  L*(4620 + L*(227 + 4*L*(-39 + 20*L))))*N)*
    #             T**2 - 8*g**3*L**2*
    #             (-2790 + (2682 + L*(165 + 4*L*(-39 + 19*L)))*N)*
    #             T**3 + 12*g**2*L**3*(25 + L*(-33 + 19*L))*N*
    #             T**4 + 12*g*(5 - 4*L)*L**4*N*T**5 +
    #            5*L**5*N*T**6) -
    #         12*np.e**((1 + 2*T)/g)*g**2*N*
    #          (4*g**6*(16*L**2*
    #                (-450 + L*(39 + 5*L*(-12 + 7*L))) +
    #               12*L*(231 +
    #                  L*(621 - 4*L*(-3 + 2*L)*(-2 + 7*L)))*N +
    #               3*(-144 +
    #                  L*(-720 +
    #                     L*(-312 + L*(97 + 24*L*(-7 + 4*L)))))*
    #                N**2) -
    #            48*g**5*(8*L**2*
    #                (-225 + L*(26 + L*(-50 + 33*L))) +
    #               L*(462 +
    #                  L*(1863 - 4*L*(24 + L*(-137 + 90*L))))*N +
    #               (-36 + L*
    #                   (-360 +
    #                     L*(-198 + L*(31 + 5*L*(-45 + 28*L)))))*
    #                N**2)*T +
    #            12*g**4*L*(8*L*
    #                (-975 + L*(169 + 5*L*(-84 + 65*L))) +
    #               4*(273 +
    #                  2*L*(1017 + L*(-87 + (618 - 493*L)*L)))*N\
    #                + (-864 +
    #                  L*(-816 + L*(-46 + 3*L*(-669 + 520*L))))*
    #                N**2)*T**2 -
    #            4*g**3*L**2*
    #             (-9000 + 16*L*(195 - 690*L + 664*L**2) -
    #               48*(-198 + L*(39 - 358*L + 366*L**2))*N +
    #               3*(-336 + L*(-124 + 25*L*(-93 + 98*L)))*N**2)*
    #             T**3 + 3*g**2*L**3*
    #             (1300 - 80*N*(12 + N) -
    #               12*L*(700 + N*(-1137 + 466*N)) +
    #               L**2*(11236 + N*(-19864 + 8829*N)))*T**4 -
    #            6*g*L**4*(20*(-50 + 123*L) +
    #               N*(1670 - 4556*L + 29*(-24 + 73*L)*N))*T**5 +
    #            L**5*(2800 + 3*N*(-1770 + 841*N))*T**6) +
    #         12*np.e**((1 + 3*T)/g)*g*
    #          (8*g**7*(4320*L**2 -
    #               24*L*(45 + L*(219 + L*(9 + L*(-7 + 4*L))))*
    #                N + (198 +
    #                  L*(909 +
    #                     2*L*(570 + L*(51 + 4*L*(-15 + 8*L)))))*
    #                N**2) -
    #            8*g**6*(17280*L**2 -
    #               24*L*(135 +
    #                  L*(876 + L*(45 - 36*L + 22*L**2)))*N +
    #               (396 + L*
    #                   (2727 +
    #                     2*L*(2181 + L*(327 + 4*L*(-81 + 50*L))))
    #                  )*N**2)*T +
    #            12*g**5*(18720*L**2 -
    #               12*L*(195 +
    #                  L*(1913 + 2*L*(63 + L*(-54 + 35*L))))*N +
    #               (156 + L*
    #                   (1974 +
    #                     L*(4739 + L*(1077 + 4*L*(-257 + 174*L)))
    #                     ))*N**2)*T**2 -
    #            4*g**4*L*(43200*L +
    #               N*(-2700 -
    #                  24*L*(2235 + L*(207 + L*(-202 + 139*L))) +
    #                  2286*N +
    #                  L*(11310 + L*(3957 + 4*L*(-1017 + 736*L)))*
    #                   N))*T**3 +
    #            4*g**3*L**2*
    #             (13500 - 3*(5700 + L*(945 + L*(-1179 + 872*L)))*
    #                N + (3819 + L*(2430 + L*(-3126 + 2407*L)))*
    #                N**2)*T**4 -
    #            12*g**2*L**3*N*
    #             (-225 + 6*(80 - 67*L)*L +
    #               (200 + L*(-442 + 379*L))*N)*T**5 +
    #            g*L**4*N*(30*(35 - 41*L) + 11*(-90 + 107*L)*N)*
    #             T**6 + 5*L**5*(30 - 29*N)*N*T**7)))/(20736.*np.e**(1/g)*g**8*N**5)

    def prob_dist(x, g, L, T, N):
        return -(((L*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T +
                L**2*T**2))/
            (np.e**(T/g)*g**2*(6 + L*(6 + L*(3 + L)))))**x*
         (1 - (L*(2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T +
                 L**2*T**2))/
             (2.*np.e**(T/g)*g**2*(6 + L*(6 + L*(3 + L)))))**
          (N - x)*gamma(1 + N))/(2**x*(-1 + (1 - (L*
                 (2*g**2*(6 + L*(3 + L)) + 2*g*L*(3 + L)*T +
                   L**2*T**2))/
               (2.*np.e**(T/g)*g**2*(6 + L*(6 + L*(3 + L)))))**N)*
         gamma(1 + N - x)*gamma(1 + x))

    counts = 0 # intializing number of photo-electron counts
    # number of iterations to run photon-counting algorithm (for statistics):
    ntimes = 100
    pc_list = [] #initializing
    pc_dark_list = []
    grand_e_list = []
    grand_e_dark_list = []
    for time in range(ntimes):
        frame_e_list = []
        frame_e_dark_list = []
        frame_e_dark_sub_list = []
        for i in range(N):
            # Simulate bright
            frame_dn = emccd.sim_sub_frame(fluxmap, frametime)
            # Simulate dark
            frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap), frametime)

            # Convert from dn to e- and bias subtract
            frame_e = frame_dn * emccd.eperdn - emccd.bias
            frame_e_dark = frame_dn_dark * emccd.eperdn - emccd.bias

            frame_e_list.append(frame_e)
            frame_e_dark_list.append(frame_e_dark)
            #frame_e_dark_sub_list.append(frame_e - frame_e_dark)

        # can also make stack of dark-subtracted if desired:
        #frame_e_cube = np.stack(frame_e_dark_sub_list)
        frame_e_cube = np.stack(frame_e_list)
        frame_e_dark_cube = np.stack(frame_e_dark_list)

        # grand stack for uncorrected case b/c averaging each iteration in
        # ntimes would affect the standard deviation
        grand_e_list += frame_e_list
        grand_e_dark_list += frame_e_dark_list

        if np.mean(np.mean(frame_e_cube))/emccd.em_gain < 4* \
        np.max(np.array([emccd.cic, emccd.dark_current])):
            warnings.warn('# of electrons/pixel needs to be bigger than about 4 '
            'times the noise (due to CIC and dark current)')

        mean_rate = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
        mean_dark_rate = get_count_rate(frame_e_dark_cube, thresh, emccd.em_gain)

        pc_list.append(mean_rate)
        pc_dark_list.append(mean_dark_rate)

    #uncorrected frames
    grand_e_cube = np.stack(grand_e_list)
    grand_e_dark_cube = np.stack(grand_e_dark_list)
    frames_unc = get_counts_uncorrected(grand_e_cube,
                        thresh, emccd.em_gain)
    frames_dark_unc = get_counts_uncorrected(grand_e_dark_cube,
                        thresh, emccd.em_gain)
    #to get avg # of '1's per pixel
    mean_num_counts = np.sum(np.sum(frames_unc,axis=0))/(ntimes*N*pix_row**2)
    mean_num_dark_counts = np.sum(np.sum(frames_dark_unc,axis=0))/(ntimes*N*pix_row**2)

    # corrected frames
    pc_cube = np.stack(pc_list)
    pc_dark_cube = np.stack(pc_dark_list)
    pc_cube_unc = frames_unc
    pc_dark_cube_unc = frames_dark_unc

    # these should be the same on average (relevant for uncorrected)
    print('mean number of 1-designated counts = ', mean_num_counts)
    print('e_thresh*L*e_coinloss = ', e_thresh(g, L, T)*L*e_coinloss(L))

    # plotting uncorrected case, no dark subtraction
    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(pc_cube_unc,axis=0).flatten(), bins=20)
    #ax[0].axvline(L, color='black')
    ax[0].axvline(e_thresh(g, L, T)*L*e_coinloss(L), color='blue')
    ax[0].axvline(np.mean(np.mean(pc_cube_unc,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[0].set_title('PC pixel mean, uncorrected')
    ax[1].hist(np.std(pc_cube_unc,axis=0).flatten(), bins=20)
    uncorrected_std_dev = np.sqrt(L*e_coinloss(L)*e_thresh(g, L, T))*e_coinloss(L)
    ax[1].axvline(np.sqrt(L*e_coinloss(L)*e_thresh(g, L, T))*e_coinloss(L),color='blue')
    #ax[1].axvline(np.sqrt(L*e_coinloss(L)*e_thresh(g, L, T))*np.exp(-L/2),color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube_unc,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, uncorrected')
    plt.tight_layout()
    plt.show()

    # plotting corrected case, no dark subtraction
    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(pc_cube,axis=0).flatten(), bins=20)
    ax[0].axvline(mean(g, L, T, N), color='blue')
    ax[0].axvline(np.mean(np.mean(pc_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[0].set_title('PC pixel mean, corrected')
    ax[1].hist(np.std(pc_cube,axis=0).flatten(), bins=20)
    std_dev = np.sqrt(var(g, L, T, N))*e_coinloss(L)
    std_dev_2 = np.sqrt(var2(g, L, T, N))
    ax[1].axvline(std_dev, color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, corrected')
    plt.tight_layout()
    plt.show()

    print('difference b/w expected and actual std dev:  ', np.mean(np.std(pc_cube,axis=0).flatten()) - std_dev)
    print('difference b/w expected and actual std dev, 2:  ', np.mean(np.std(pc_cube,axis=0).flatten()) - std_dev_2)

    # plotting corrected case where photon-counted darks have been subtracted
    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(pc_cube - pc_dark_cube,axis=0).flatten(), bins=20)
    # mean of the difference is the difference of the means
    ax[0].axvline(mean(g, L, T, N) - mean(g, L_dark, T, N), color='blue')
    ax[0].axvline(np.mean(np.mean(pc_cube-pc_dark_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[0].set_title('PC pixel mean, corrected, d-s')
    ax[1].hist(np.std(pc_cube - pc_dark_cube,axis=0).flatten(), bins=20)
    # variance of the difference is the sum of the variances
    # standard deviation is the square root of this sum
    std_dev_subtracted = np.sqrt(var(g, L, T, N)*e_coinloss(L)**2 +
        var(g, L_dark, T, N)*e_coinloss(L_dark)**2)
    std_dev_subtracted_2 = np.sqrt(var2(g, L, T, N) +
        var2(g, L_dark, T, N))
    ax[1].axvline(std_dev_subtracted,color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube-pc_dark_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, corrected, d-s')
    plt.tight_layout()
    plt.show()

    print('difference b/w d-s expected and actual std dev:  ',np.mean(np.std(pc_cube-pc_dark_cube,axis=0).flatten()) - std_dev_subtracted)
    print('difference b/w d-s expected and actual std dev, 2:  ',np.mean(np.std(pc_cube-pc_dark_cube,axis=0).flatten()) - std_dev_subtracted_2)

    def poisson_dist(x, L):
        return L**x*np.e**(-L)/factorial(x)

    x_arr = np.linspace(0, N+1, 500)
    #x_arr = np.arange(0, N+1)
    co_added_unc = []
    for i in range(ntimes-1):
        co_added_unc.append(np.sum(pc_cube_unc[int(N*i):int(N*(i+1))],axis=0))
    co_added_unc = np.stack(co_added_unc).flatten()

    # plotting histogram of coadded frames (to check the probability distribution)
    f, ax = plt.subplots()
    ax.set_title('PC pixel sum over frames (Nbr), with expected prob dist')
    # to get integer-valued x values for reliable chi2 analysis, we choose # of bins so that the x values are integer, as they should be
    y_vals, x_vals, _ = ax.hist(co_added_unc, bins=co_added_unc.max())
    #scale_old = np.max(y_vals)/np.max(poisson_dist(x_arr, N*L*e_coinloss(L)*e_thresh(g, L, T)))
    #scale_new = np.max(y_vals)/np.max(prob_dist(x_arr, g, L, T, N))
    # x_vals is the boundaries of the bins, so the length of x_vals is 1 more than y_vals, so we take x_vals[:-1]
    # this scale ensures the data and the expected distribution values are normalized the same, basically
    scale_new = np.sum(y_vals)/np.sum(prob_dist(x_vals[:-1], g,L,T,N))
    chisquare_value, pvalue = chisquare(y_vals, scale_new*prob_dist(x_vals[:-1], g, L, T, N))
    print('chi square value:  ', chisquare_value, 'p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=co_added_unc.max()))
    # null hypothesis accepted (i.e., good fit) when chi square value less than critical value
    #plt.plot(x_arr, scale_old*poisson_dist(x_arr, N*L*e_coinloss(L)*e_thresh(g, L, T)))
    plt.plot(x_arr, scale_new*prob_dist(x_arr, g, L, T, N))
    plt.show()

