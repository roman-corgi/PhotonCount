# -*- coding: utf-8 -*-
"""Calculate observational (from simulation) and theoretical signal and noise for SNR
for photon-counted frames, "corrected"
and "uncorrected".  See paper for details."""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.special import comb
from scipy.stats import chisquare, chi2
from mpmath import hyper

# for simulating detector frames
from emccd_detect.emccd_detect import EMCCDDetect
# for perform the photon-counting
from PhotonCount.corr_photon_count import (get_count_rate,
                                        get_counts_uncorrected)

if __name__ == '__main__':
    pix_row = 50 #number of rows and number of columns
    fluxmap = np.ones((pix_row,pix_row)) #photon flux map, photons/s
    frametime = 0.05  # s (adjust lambda by adjusting this)

    # In order for the uncertainty of standard deviation to be accurate (i.e.,
    # assumed to be what it is for normal distribution), N > 9*(1-eThresh)/eThresh
    # and N > 9*eThresh/(1-eThresh).  For our input parameters, this means N > 176.
    N = 200 # number of frames per trial
    N2 = 300 # number of frames per trial for darks
    # number of iterations to run photon-counting algorithm (for statistics):
    ntimes = 500
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
        cr_rate=0.,  # cosmic rays incidence, hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=1,  # for simplicity, set this to 1 so there's no data loss when converting back to e-
        nbits=64, # number of ADU bits
        numel_gain_register=604 #number of gain register elements
        )

    thresh = emccd.em_gain/10 # threshold

    if np.average(frametime*fluxmap) > 0.1:
        warnings.warn('average # of photons/pixel is > 0.1.  Decrease frame '
        'time to get lower average # of photons/pixel.')

    if emccd.read_noise <=0:
       warnings.warn('read noise should be greater than 0 for effective '
       'photon counting')
    if thresh < 4*emccd.read_noise:
       warnings.warn('thresh should be at least 4 or 5 times read_noise for '
       'accurate photon counting')

    avg_ph_flux = np.mean(fluxmap)
    # theoretical electron flux for brights
    L = avg_ph_flux*emccd.qe*frametime + emccd.dark_current*frametime\
        +emccd.cic
    # theoretical electron flux for darks
    L_dark = emccd.dark_current*frametime + emccd.cic

    g = em_gain
    T = thresh

    # assuming lambda << 1, expansion to 3rd order

    def Const(L):
        return (6*np.e**L)/(6 + L*(6 + L*(3 + L)))

    def eThresh(g, L, T):
        return Const(L)*(np.e**(-((g*L + T)/g))*L*(2*g**2*(6 + L*(3 + L)) +
            2*g*L*(3 + L)*T + L**2*T**2))/(12*g**2)

    def prob_dist(x, g, L, T, N):
        return (comb(N-1, x)+comb(N-1, x-1))*((1 - eThresh(g, L, T))**(N - x)*
                 eThresh(g, L, T)**x)

    def std_dev(g, L, T): #for one frame or for an average of frames
        return np.sqrt(eThresh(g, L, T) * (1-eThresh(g, L, T)))

    pc_list = []
    pc_dark_list = []
    pc_unc_sub_avg_list = []
    pc_unc_std_bright_list = []
    pc_unc_std_dark_list = []
    pc_unc_sub_std_list = []
    pc_unc_counts_list = []

    for time in range(ntimes):
        frame_e_list = []
        frame_e_dark_list = []
        frame_e_dark_sub_list = []
        for i in range(N):
            # Simulate bright
            frame_dn = emccd.sim_sub_frame(fluxmap, frametime)

            # Convert from dn to e- and bias subtract
            frame_e = frame_dn * emccd.eperdn - emccd.bias

            frame_e_list.append(frame_e)

        for i in range(N2):
            # Simulate dark
            frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap), frametime)

            # Convert from dn to e- and bias subtract
            frame_e_dark = frame_dn_dark * emccd.eperdn - emccd.bias

            frame_e_dark_list.append(frame_e_dark)

        frame_e_cube = np.stack(frame_e_list)
        frame_e_dark_cube = np.stack(frame_e_dark_list)

        # "corrected" frames
        # L_{2,br} in terms of paper in doc folder
        mean_rate = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
        # L_{2,dk} in terms of paper in doc folder
        mean_dark_rate = get_count_rate(frame_e_dark_cube, thresh, emccd.em_gain)

        pc_list.append(mean_rate)
        pc_dark_list.append(mean_dark_rate)

        # "uncorrected" frames
        # N_{br} in terms of paper in doc folder
        frames_unc = get_counts_uncorrected(frame_e_cube,
                            thresh, emccd.em_gain)
        # N_{dk} in terms of paper in doc folder
        frames_dark_unc = get_counts_uncorrected(frame_e_dark_cube,
                            thresh, emccd.em_gain)

        # for histogram at the end of script
        pc_unc_counts_list.append(np.sum(frames_unc,axis=0))

        # finding mean of brights minus mean of darks
        pc_unc_sub_avg_list.append(np.mean(frames_unc, axis=0) - np.mean(frames_dark_unc, axis=0))
        # finding standard deviation of when these frames are subtracted
        std_bright_frames = np.std(frames_unc, axis=0)
        std_dark_frames = np.std(frames_dark_unc, axis=0)
        pc_unc_std_bright_list.append(std_bright_frames)
        pc_unc_std_dark_list.append(std_dark_frames)

    # for histogram at end:
    pc_unc_counts_cube = np.stack(pc_unc_counts_list)

    # uncorrected frames
    pc_unc_avg_cube = np.stack(pc_unc_sub_avg_list)
    pc_unc_std_bright_cube = np.stack(pc_unc_std_bright_list)
    pc_unc_std_dark_cube = np.stack(pc_unc_std_dark_list)

    # SIGNAL
    # mean # of '1's per pixel per frame for brights - darks, averaged over ntimes trials and npix*npix pixels
    mean_num_counts = np.mean(pc_unc_avg_cube)
    # uncertainty of mean is standard deviation of the mean, averaged over ntimes trials and npix*npix pixels
    uncertainty_mean_num_counts = np.mean(np.sqrt((pc_unc_std_bright_cube/np.sqrt(N))**2 + (pc_unc_std_dark_cube/np.sqrt(N2))**2))

    # NOISE
    # standard deviation of '1's per pixel per frame for brights - darks, averaged over ntimes trials and npix*npix pixels
    std_num_counts = np.mean(np.sqrt(pc_unc_std_bright_cube**2 + pc_unc_std_dark_cube**2))
    # uncertainty of standard deviation is averaged over ntimes trials and npix*npix pixels
    uncertainty_std_num_counts = np.mean(np.sqrt((pc_unc_std_bright_cube/np.sqrt(2*(N-1)))**2 + (pc_unc_std_dark_cube/np.sqrt(2*(N2-1)))**2))

    print("UNCORRECTED-----------------------------")
    print('average signal (averaged over trials and all pixels) = ', mean_num_counts)
    print('uncertainty of the signal (over trials and all pixels) = ', uncertainty_mean_num_counts)
    print('signal range: (', mean_num_counts-uncertainty_mean_num_counts, ', ',mean_num_counts+uncertainty_mean_num_counts, ')')
    print('signal expected from probability distribution = ', eThresh(g,L,T) - eThresh(g,L_dark,T))

    print('average noise (averaged over trials and all pixels) = ', std_num_counts)
    print('uncertainty of the noise (over trials and all pixels) = ', uncertainty_std_num_counts)
    print('noise range: (', std_num_counts-uncertainty_std_num_counts, ', ',std_num_counts+uncertainty_std_num_counts, ')')
    print('noise expected from probability distribution = ', np.sqrt(std_dev(g, L, T)**2 + std_dev(g,L_dark,T)**2))

    print('SNR using average signal over average noise: ', mean_num_counts/std_num_counts)
    print('SNR range:  (', (mean_num_counts-uncertainty_mean_num_counts)/(std_num_counts+uncertainty_std_num_counts), ', ', (mean_num_counts+uncertainty_mean_num_counts)/(std_num_counts-uncertainty_std_num_counts), ')')
    print('Theoretical SNR:  ', (eThresh(g,L,T) - eThresh(g,L_dark,T))/np.sqrt(std_dev(g, L, T)**2 + std_dev(g,L_dark,T)**2))

    # now for corrected SNR calculations

    def exp1(g, L, T, N):
        return N*eThresh(g,L,T)

    def exp2(g, L, T, N):
        return N*eThresh(g,L,T)*(1 + (-1 + N)*eThresh(g,L,T))

    def exp3(g, L, T, N):
        return float((eThresh(g,L,T)*(-((-1 + N)*(-1 + eThresh(g,L,T))**2*
        (1 + (-2 + N)*eThresh(g,L,T)*(3 + (-3 + N)*eThresh(g,L,T)))) -
        (1 - eThresh(g,L,T))**N*
        hyper([2,2,2,1 - N],[1,1,1],1 + 1/(-1 + eThresh(g,L,T)))
        ))/(-1 + eThresh(g,L,T)))

    # "corrected" mean
    def meanL23(g, L, T, N):
        return np.e**(T/g)*exp1(g,L,T,N)/N + np.e**(2*T/g)*(g-T)*exp2(g,L,T,N)/(2*g*N**2) + \
        np.e**(3*T/g)*(4*g**2-8*g*T+5*T**2)*exp3(g,L,T,N)/(12*g**2*N**3)

    # "corrected" variance
    def varL23(g, L, T, N):
        return N*(std_dev(g,L,T))**2*(((np.e**((T/g)))/N) +
        2*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))*(N*eThresh(g,L,T)) +
        3*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
        12*g**2*N**3))*(N*eThresh(g,L,T))**2)**2

    # corrected frames
    pc_cube = np.stack(pc_list)
    pc_dark_cube = np.stack(pc_dark_list)

    # SIGNAL
    # mean number of signal photoelectrons per pixel per frame for brights - darks, averaged over npix*npix pixels
    mean_num_counts_corr = np.mean(np.mean(pc_cube - pc_dark_cube, axis=0))
    # uncertainty of mean is standard deviation of mean, averaged over npix*npix pixels
    uncertainty_mean_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0)/np.sqrt(ntimes))

    # NOISE
    # standard deviation per pixel per frame for brights - darks, averaged over npix*npix pixels
    std_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0))
    # uncertainty of standard deviation, averaged over npix*npix pixels
    uncertainty_std_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0)/np.sqrt(2*(ntimes-1)))

    print("\n")
    print("CORRECTED-------------------------------")
    print('average signal (averaged over all pixels) = ', mean_num_counts_corr)
    print('uncertainty of the signal (over all pixels) = ', uncertainty_mean_num_counts_corr)
    print('signal range: (', mean_num_counts_corr-uncertainty_mean_num_counts_corr, ', ',mean_num_counts_corr+uncertainty_mean_num_counts_corr, ')')
    print('signal expected from probability distribution = ', meanL23(g,L,T,N)-meanL23(g,L_dark,T,N2))

    print('average noise (averaged over all pixels) = ', std_num_counts_corr)
    print('uncertainty of the noise (over all pixels) = ', uncertainty_std_num_counts_corr)
    print('noise range: (', std_num_counts_corr-uncertainty_std_num_counts_corr, ', ',std_num_counts_corr+uncertainty_std_num_counts_corr, ')')
    print('noise expected from probability distribution = ', np.sqrt(varL23(g,L,T,N)+varL23(g,L_dark,T,N2)))

    print('SNR using average signal over average noise: ', mean_num_counts_corr/std_num_counts_corr)
    print('SNR range:  (', (mean_num_counts_corr-uncertainty_mean_num_counts_corr)/(std_num_counts_corr+uncertainty_std_num_counts_corr), ', ', (mean_num_counts_corr+uncertainty_mean_num_counts_corr)/(std_num_counts_corr-uncertainty_std_num_counts_corr), ')')
    print('Theoretical SNR:  ', (meanL23(g,L,T,N)-meanL23(g,L_dark,T,N2))/np.sqrt(varL23(g,L,T,N)+varL23(g,L_dark,T,N2)))


    co_added_unc = []

    # the stats of (brights - darks) is not really relevant; I just show that the brights follow the probability distribution
    # by the same token, the photon-counted darks would also follow the probability distribution.

    # variates for number of bright frames
    co_added_unc = pc_unc_counts_cube.flatten()

    # plotting histogram of coadded frames (to check the probability distribution)
    f, ax = plt.subplots()
    ax.set_title('PC pixel sum over frames (Nbr), with expected prob dist')
    # to get integer-valued x values for reliable chi2 analysis, we choose # of bins so that the x values are integer, as they should be.
    # If N is set very high and ntimes is not high enough, then a better visual fit to the plot made below would be for fewer bins than what is specified in the next line
    y_vals, x_vals = np.histogram(co_added_unc, bins=int(np.round(co_added_unc.max())))

    # x_vals is the boundaries of the bins, so the length of x_vals is 1 more than y_vals, so we take x_vals[:-1].
    # this scale below ensures the data and the expected distribution values are normalized the same
    scale = np.sum(y_vals)/np.sum(prob_dist(x_vals[:-1], g,L,T,N))
    # perform chi square analysis
    chisquare_value, pvalue = chisquare(y_vals/scale, prob_dist(x_vals[:-1], g, L, T, N))
    print('\n')
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=co_added_unc.max()))
    # p value close to 1:  data and trial probability distribution not statistically distinct
    # null hypothesis accepted (i.e., good fit) when chi square value less than critical value

    plt.scatter(x_vals[:-1], y_vals/scale)
    plt.xlabel('c (in electrons)')
    plt.ylabel('probability distribution')
    plt.title(r'Histogram of counts for $N_{br}$ photon-counted frames')
    x_arr = np.linspace(0, N+1, 1000)
    plt.plot(x_arr, prob_dist(x_arr, g, L, T, N))
    plt.show()

