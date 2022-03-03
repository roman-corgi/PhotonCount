# -*- coding: utf-8 -*-
"""Get histograms of noise."""
import numpy as np
import matplotlib.pyplot as plt
import warnings

from emccd_detect.emccd_detect import EMCCDDetect
from PhotonCount.corr_photon_count import (get_count_rate,
                                        get_counts_uncorrected)

def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
    """Plot a scaled colormap."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == '__main__':
    pix_row = 50 #number of rows and number of columns
    fluxmap = np.ones((pix_row,pix_row)) #photon flux map, photons/s
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

    counts = 0 # intializing number of photo-electron counts
    # number of iterations to run photon-counting algorithm (for statistics):
    ntimes = 100
    pc_list = [] #initializing
    pc_dark_list = []
    grand_e_list = []
    grand_e_dark_list = []
    for x in range(ntimes):
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

    pc_cube = np.stack(pc_list)
    pc_dark_cube = np.stack(pc_dark_list)
    pc_cube_unc = frames_unc
    pc_dark_cube_unc = frames_dark_unc

    # these should be the same on average (relevant for uncorrected)
    print('mean number of 1-designated counts = ', mean_num_counts)
    print('e_thresh*L*e_coinloss = ', e_thresh(g, L, T)*L*e_coinloss(L))

    # plotting uncorrected case
    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(pc_cube_unc,axis=0).flatten(), bins=20)
    #ax[0].axvline(L, color='black')
    ax[0].axvline(e_thresh(g, L, T)*L*e_coinloss(L), color='blue')
    ax[0].axvline(np.mean(np.mean(pc_cube_unc,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[0].set_title('PC pixel mean, uncorrected')
    ax[1].hist(np.std(pc_cube_unc,axis=0).flatten(), bins=20)
    ax[1].axvline(np.sqrt(L*e_coinloss(L)*e_thresh(g, L, T))*e_coinloss(L),color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube_unc,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, uncorrected')
    plt.tight_layout()
    plt.show()

    # plotting corrected case
    f, ax = plt.subplots(1,2)
    ax[0].hist(np.mean(pc_cube,axis=0).flatten(), bins=20)
    ax[0].axvline(mean(g, L, T, N), color='blue')
    ax[0].axvline(np.mean(np.mean(pc_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[0].set_title('PC pixel mean, corrected')
    ax[1].hist(np.std(pc_cube,axis=0).flatten(), bins=20)
    ax[1].axvline(np.sqrt(var(g, L, T, N))*e_coinloss(L),color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, corrected')
    plt.tight_layout()
    plt.show()

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
    ax[1].axvline(std_dev_subtracted,color='blue')
    ax[1].axvline(np.mean(np.std(pc_cube-pc_dark_cube,axis=0).flatten()), color='green', linestyle= 'dotted')
    ax[1].set_title('PC pixel sdev, corrected, d-s')
    plt.tight_layout()
    plt.show()

