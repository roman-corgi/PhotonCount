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
    N = 100 # number of frames per trial
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


    # if True, assumes a master dark is subtracted from each frame before
    # photon-counting them; if False, then a set of simulated darks are
    # photon-counted instead
    md = True

    if md:
        # number of frames to simulate master dark; must be much bigger than N
        N_dark = 1300
        if N_dark <= N:
            raise Exception('N_dark must be much bigger than N')

        master_dark_list = []
        for i in range(N_dark):
            frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap), frametime)
            # convet from counts to units of electrons and subtract the bias
            frame_e_dark = frame_dn_dark * emccd.eperdn - emccd.bias
            master_dark_list.append(frame_e_dark)
        master_dark_basis  = np.mean(master_dark_list, axis=0)
        master_dark = np.stack([master_dark_basis]*N)

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

    # assuming lambda relatively small, expansion to 3rd order

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

    pc_list = [] #initializing
    pc_dark_list = []
    pc_unc_avg_list = []
    pc_unc_sub_avg_list = []
    pc_unc_sub_std_list = []
    pc_unc_SNR_list = []
    pc_unc_sub_SNR_list = []
    grand_e_list = []
    grand_e_dark_list = []
    #pc_unc_nomask_list = []
    pc_unc_counts_list = []
    combined_mask = np.zeros_like(fluxmap)
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
            if md:
                frame_e_dark_list.append(master_dark_basis)
            else:
                frame_e_dark_list.append(frame_e_dark)
            #frame_e_dark_sub_list.append(frame_e - frame_e_dark)

        # can also make stack of dark-subtracted if desired:
        #frame_e_cube = np.stack(frame_e_dark_sub_list)
        frame_e_cube = np.stack(frame_e_list)
        frame_e_dark_cube = np.stack(frame_e_dark_list)

        # grand stack for uncorrected case b/c averaging each iteration in
        # ntimes would affect the standard deviation
        # grand_e_list += np.mean(frame_e_list) #frame_e_list
        # grand_e_dark_list += np.mean(frame_e_dark_list) # frame_e_dark_list

        # not needed since we are subtracting photon-counted darks
        # if np.mean(np.mean(frame_e_cube))/emccd.em_gain < 4* \
        # np.max(np.array([emccd.cic, emccd.dark_current])):
        #     warnings.warn('# of electrons/pixel needs to be bigger than about 4 '
        #     'times the noise (due to CIC and dark current)')

        mean_rate = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
        mean_dark_rate = get_count_rate(frame_e_dark_cube, thresh, emccd.em_gain)

        pc_list.append(mean_rate)
        pc_dark_list.append(mean_dark_rate)

        #uncorrected frames
        # grand_e_cube = np.stack(grand_e_list)
        # grand_e_dark_cube = np.stack(grand_e_dark_list)

        frames_unc = get_counts_uncorrected(frame_e_cube,
                            thresh, emccd.em_gain)
        frames_dark_unc = get_counts_uncorrected(frame_e_dark_cube,
                            thresh, emccd.em_gain)

        # no mask needed for prob dist plot
        # pc_unc_nomask_list.append(np.mean(frames_unc, axis=0))
        # # need to mask out any pixels with no counts in any of the N frames, which would lead to a std dev of 0
        # # This can be avoided if N is larger and frametime is larger
        # frames_unc_mask = np.zeros_like(frames_unc)

        # mask out those pixels that give 0 std dev for all N frames, for both the no-subtract and the subtract cases:
        # frames_unc_mask[:, np.where(np.std(frames_unc, axis=0)==0)] = 1
        # frames_unc_mask[:, np.where(np.std(frames_unc - frames_dark_unc, axis=0)==0)] = 1
        # frames_unc = np.ma.masked_array(frames_unc, frames_unc_mask)
        # frames_dark_unc = np.ma.masked_array(frames_dark_unc, frames_unc_mask)
        # combined_mask = np.logical_or(combined_mask, np.mean(frames_unc_mask, axis=0))

        # pc_unc_sub_avg_list.append(np.mean(frames_unc - frames_dark_unc, axis=0) - np.mean(frames_dark_unc, axis=0))
        # pc_unc_sub_std_list.append(np.std(frames_unc - frames_dark_unc, axis=0))

        # pc_unc_sub_avg_list.append(np.mean(frames_unc) - np.mean(frames_dark_unc))
        # pc_unc_sub_std_list.append(np.sqrt(np.std(frames_unc)**2 + np.std(frames_dark_unc)**2))

        pc_unc_counts_list.append(np.sum(frames_unc,axis=0))

        pc_unc_sub_avg_list.append(np.mean(frames_unc, axis=0) - np.mean(frames_dark_unc, axis=0))
        pc_unc_sub_std_list.append(np.sqrt(np.std(frames_unc, axis=0)**2 + np.std(frames_dark_unc, axis=0)**2))
        # pc_unc_sub_var_list.append(np.std(frames_unc - frames_dark_unc, axis=0)**2)

        # pc_unc_avg_list.append(np.ma.mean(frames_unc, axis=0))
        # pc_unc_sub_avg_list.append(np.ma.mean(frames_unc - frames_dark_unc, axis=0))

        # pc_unc_SNR_list.append(np.divide(np.mean(frames_unc, axis=0), np.std(frames_unc, axis=0), out=np.zeros_like(frames_unc[0]).astype(float), where= np.std(frames_unc,axis=0)!=0 ))
        # pc_unc_sub_SNR_list.append(np.divide(np.mean(frames_unc - frames_dark_unc, axis=0), np.std(frames_unc - frames_dark_unc, axis=0), out=np.zeros_like(frames_dark_unc[0]).astype(float), where=np.std(frames_unc - frames_dark_unc, axis=0)!=0))

        # pc_unc_SNR_list.append(np.ma.mean(frames_unc, axis=0)/np.ma.std(frames_unc, axis=0))
        # pc_unc_sub_SNR_list.append(np.ma.mean(frames_unc - frames_dark_unc, axis=0)/np.ma.std(frames_unc - frames_dark_unc, axis=0))

    # uncorrected frames
    pc_unc_avg_cube = np.stack(pc_unc_sub_avg_list)
    pc_unc_std_cube = np.stack(pc_unc_sub_std_list)
    pc_unc_counts_cube = np.stack(pc_unc_counts_list)

    # pc_unc_nomask_cube = np.stack(pc_unc_nomask_list)
    # pc_unc_cube = np.ma.stack(pc_unc_avg_list)
    # pc_unc_sub_avg_cube = np.ma.stack(pc_unc_sub_avg_list)
    # pc_unc_SNR_cube = np.ma.stack(pc_unc_SNR_list)
    # pc_unc_sub_SNR_cube = np.ma.stack(pc_unc_sub_SNR_list)
    # apply the combined mask:
    # overall_mask = np.zeros((ntimes, pix_row, pix_row))
    # overall_mask[:] = combined_mask
    # pc_unc_cube = np.ma.masked_array(pc_unc_cube, overall_mask)
    # pc_unc_sub_avg_cube = np.ma.masked_array(pc_unc_sub_avg_cube, overall_mask)
    # pc_unc_SNR_cube = np.ma.masked_array(pc_unc_SNR_cube, overall_mask)
    # pc_unc_sub_SNR_cube = np.ma.masked_array(pc_unc_sub_SNR_cube, overall_mask)

    # averaged over ntimes trials and npix*npix pixels:
    # avg # of '1's per pixel
    mean_num_counts = np.mean(pc_unc_avg_cube)
    uncertainty_mean_num_counts = np.mean(pc_unc_std_cube/np.sqrt(N)) #np.std(pc_unc_avg_cube)/np.sqrt(pc_unc_avg_cube.size)

    # standard deviation of '1's per pixel
    std_num_counts = np.mean(pc_unc_std_cube)
    uncertainty_std_num_counts = np.mean(pc_unc_std_cube/np.sqrt(2*(N-1)))  #np.std(pc_unc_std_cube)/np.sqrt(pc_unc_std_cube.size)

    # mean_num_counts = np.ma.mean(pc_unc_cube)
    # #mean_num_counts = np.sum(np.sum(frames_unc,axis=0))/(ntimes*N*pix_row**2)
    # #standard deviation of the mean for that
    # SDOM_num_counts = np.ma.std(pc_unc_cube)/np.sqrt(pc_unc_cube.size)
    # #SDOM_num_counts = (np.std(np.sum(frames_unc,axis=0))/(ntimes*N*pix_row**2))/np.sqrt(ntimes*N*pix_row**2)
    # #mean_num_dark_counts = np.mean(pc_unc_sub_avg_cube)
    # #mean_num_dark_counts = np.sum(np.sum(frames_dark_unc,axis=0))/(ntimes*N*pix_row**2)

    # mean_num_counts_minus = np.ma.mean(pc_unc_sub_avg_cube)
    # #mean_num_counts_minus = np.mean(frames_unc - frames_dark_unc) #np.sum(np.sum(frames_unc - frames_dark_unc,axis=0))/(ntimes*N*pix_row**2)
    # #standard deviation of the mean for that
    # SDOM_num_counts_minus = np.ma.std(pc_unc_sub_avg_cube)/np.sqrt(pc_unc_sub_avg_cube.size)
    # #SDOM_num_counts_minus = np.std(frames_unc - frames_dark_unc)/frames_unc.size #(np.std(np.sum(frames_unc - frames_dark_unc,axis=0))/(ntimes*N*pix_row**2))/(ntimes*N*pix_row**2)

    # pc_cube_unc = frames_unc.copy()
    # pc_dark_cube_unc = frames_dark_unc.copy()

    # these should be the same on average (relevant for uncorrected)
    # print("UNCORRECTED-----------------------------")
    # print('mean number of 1-designated counts = ', mean_num_counts)
    # print('standard deviation of the mean for the mean number of 1-designated counts = ', SDOM_num_counts)
    # print('mean expected from probability distribution = ', eThresh(g,L,T))
    # (minus)
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


    # mean_SNR_unc = np.ma.mean(pc_unc_SNR_cube)
    # SDOM_SNR_unc = np.ma.std(pc_unc_SNR_cube)/np.sqrt(pc_unc_SNR_cube.size)
    # # mean_SNR_unc = np.mean(np.mean(pc_unc_cube,axis=0)/np.std(pc_unc_cube, axis=0))
    # # SDOM_SNR_unc = np.std(np.mean(pc_unc_cube,axis=0)/np.std(pc_unc_cube, axis=0))
    # print('mean SNR per pixel = ', mean_SNR_unc)
    # print('standard deviation of the mean for SNR per pixel = ', SDOM_SNR_unc)
    # print('mean SNR range: (', mean_SNR_unc-SDOM_SNR_unc, ', ',mean_SNR_unc+SDOM_SNR_unc, ')')
    # print('SNR expected from probability distribution = ', eThresh(g, L, T)/std_dev(g, L, T))


    # mean_sub_SNR_unc = np.ma.mean(pc_unc_sub_SNR_cube)
    # SDOM_sub_SNR_unc = np.ma.std(pc_unc_sub_SNR_cube)/np.sqrt(pc_unc_sub_SNR_cube.size)
    # # mean_SNR_unc_minus = np.mean(np.mean(pc_cube_unc-pc_dark_cube_unc,axis=0)/np.std(pc_cube_unc-pc_dark_cube_unc, axis=0))
    # # SDOM_SNR_unc_minus = np.std(np.mean(pc_cube_unc-pc_dark_cube_unc,axis=0)/np.std(pc_cube_unc-pc_dark_cube_unc, axis=0))
    # print('mean SNR per pixel for subtracted frames = ', mean_sub_SNR_unc)
    # print('standard deviation of the mean for SNR per pixel for subtracted frames = ', SDOM_sub_SNR_unc)
    # print('mean SNR range for subtracted frames: ', mean_sub_SNR_unc-SDOM_sub_SNR_unc, ', ',mean_sub_SNR_unc+SDOM_sub_SNR_unc, ')')
    # print('SNR expected from probability distribution = ', (eThresh(g, L, T)-eThresh(g, L_dark, T))/np.sqrt(std_dev(g, L, T)**2+std_dev(g,L_dark,T)**2))


    # now for corrected SNR:

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

    def meanL23(g, L, T, N):
        return np.e**(T/g)*exp1(g,L,T,N)/N + np.e**(2*T/g)*(g-T)*exp2(g,L,T,N)/(2*g*N**2) + \
        np.e**(3*T/g)*(4*g**2-8*g*T+5*T**2)*exp3(g,L,T,N)/(12*g**2*N**3)

    # def varL23(g, L, T, N):
    #     return N*(std_dev(g,L,T))**2*(((np.e**((T/g)))/N)**2 +
    #     4*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))**2*(N*eThresh(g,L,T))**2 +
    #     9*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
    #     12*g**2*N**3))**2*(N*eThresh(g,L,T))**4)

    def varL23(g, L, T, N):
        return N*(std_dev(g,L,T))**2*(((np.e**((T/g)))/N) +
        2*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))*(N*eThresh(g,L,T)) +
        3*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
        12*g**2*N**3))*(N*eThresh(g,L,T))**2)**2

    # corrected frames
    pc_cube = np.stack(pc_list)
    pc_dark_cube = np.stack(pc_dark_list)

    # need to mask out any pixels with no counts in any of the N frames, which would lead to a std dev of 0
    # This can be avoided if N is larger and frametime is larger
    # pc_cube_mask = np.zeros_like(pc_cube)
    # pc_cube_mask[:, np.where(np.std(pc_cube, axis=0)==0)] = 1
    # pc_cube_mask[:, np.where(np.std(pc_cube - pc_dark_cube, axis=0)==0)] = 1
    # pc_cube = np.ma.masked_array(pc_cube, pc_cube_mask)
    # pc_dark_cube = np.ma.masked_array(pc_dark_cube, pc_cube_mask)
    # mask out those pixels that give 0 std dev for all N frames, for both the no-subtract and the subtract cases:

    mean_num_counts_corr = np.mean(np.mean(pc_cube - pc_dark_cube, axis=0))
    uncertainty_mean_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0)/np.sqrt(ntimes))   #np.std(np.mean(pc_cube - pc_dark_cube, axis=0))/np.sqrt(pix_row*pix_row)

    std_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0))
    uncertainty_std_num_counts_corr = np.mean(np.std(pc_cube - pc_dark_cube, axis=0)/np.sqrt(2*(ntimes-1)))   #np.std(np.std(pc_cube - pc_dark_cube, axis=0)**2)/np.sqrt(pix_row*pix_row)

    # mean_SNR = np.ma.mean(np.divide(np.ma.mean(pc_cube,axis=0), np.ma.std(pc_cube, axis=0), out=np.zeros_like(pc_cube[0]), where=np.std(pc_cube,axis=0)!=0))
    # SDOM_SNR = np.ma.std(np.divide(np.ma.mean(pc_cube,axis=0), np.ma.std(pc_cube, axis=0), out=np.zeros_like(pc_cube[0]), where=np.std(pc_cube,axis=0)!=0))/np.sqrt(pc_cube.size)

    print("\n")
    print("CORRECTED-------------------------------")
    print('average signal (averaged over trials and all pixels) = ', mean_num_counts_corr)
    print('uncertainty of the signal (over trials and all pixels) = ', uncertainty_mean_num_counts_corr)
    print('signal range: (', mean_num_counts_corr-uncertainty_mean_num_counts_corr, ', ',mean_num_counts_corr+uncertainty_mean_num_counts_corr, ')')
    print('signal expected from probability distribution = ', meanL23(g,L,T,N)-meanL23(g,L_dark,T,N))

    print('average noise (averaged over trials and all pixels) = ', std_num_counts_corr)
    print('uncertainty of the noise (over trials and all pixels) = ', uncertainty_std_num_counts_corr)
    print('noise range: (', std_num_counts_corr-uncertainty_std_num_counts_corr, ', ',std_num_counts_corr+uncertainty_std_num_counts_corr, ')')
    print('noise expected from probability distribution = ', np.sqrt(varL23(g,L,T,N)+varL23(g,L_dark,T,N)))
    # print('mean SNR per pixel = ', mean_SNR)
    # print('standard deviation of the mean for SNR per pixel = ', SDOM_SNR)
    # print('mean SNR range: (',mean_SNR-SDOM_SNR, ', ',mean_SNR+SDOM_SNR, ')')
    # print('SNR expected from probability distribution = ', meanL23(g,L,T,N)/np.sqrt(varL23(g,L,T,N)))


    print('SNR using average signal over average noise: ', mean_num_counts_corr/std_num_counts_corr)
    print('SNR range:  (', (mean_num_counts_corr-uncertainty_mean_num_counts_corr)/(std_num_counts_corr+uncertainty_std_num_counts_corr), ', ', (mean_num_counts_corr+uncertainty_mean_num_counts_corr)/(std_num_counts_corr-uncertainty_std_num_counts_corr), ')')
    print('Theoretical SNR:  ', (meanL23(g,L,T,N)-meanL23(g,L_dark,T,N))/np.sqrt(varL23(g,L,T,N)+varL23(g,L_dark,T,N)))

    #(minus)
    # mean_SNR_minus = np.ma.mean(np.ma.mean(pc_cube - pc_dark_cube,axis=0)/np.ma.std(pc_cube-pc_dark_cube, axis=0))
    # SDOM_SNR_minus = np.ma.std(np.ma.mean(pc_cube - pc_dark_cube,axis=0)/np.ma.std(pc_cube-pc_dark_cube, axis=0))/np.sqrt(np.where(np.logical_or(pc_dark_cube.mask,pc_cube.mask)==False)[0].size)
    # print('mean SNR per pixel for subtracted frames = ', mean_SNR_minus)
    # print('standard deviation of the mean for SNR per pixel for subtracted frames = ', SDOM_SNR_minus)
    # print('mean SNR range (minus): (',mean_SNR_minus-SDOM_SNR_minus, ', ',mean_SNR_minus+SDOM_SNR_minus, ')')
    # print('SNR expected from probability distribution = ', (meanL23(g,L,T,N)-meanL23(g,L_dark,T,N))/np.sqrt(varL23(g,L,T,N)+varL23(g,L_dark,T,N)))


    x_arr = np.linspace(0, N+1, 1000)
    co_added_unc = []

    # the stats of brights - darks is not really relevant; I just show that the stats of brights follows prob dist
    # by the same token, the photon-counted darks would also follow the prob dist.

    # if minus:
    #     pc_cube_unc = pc_cube_unc-pc_dark_cube_unc
    # for i in range(ntimes):
    #     co_added_unc.append(np.sum(pc_cube_unc[int(N*i):int(N*(i+1))],axis=0))

    # multiply by N to get the variates for N frames
    co_added_unc = pc_unc_counts_cube.flatten() #N*pc_unc_nomask_cube.flatten()
    # if minus:
    #     co_added_unc = co_added_unc[co_added_unc>=0]
    # plotting histogram of coadded frames (to check the probability distribution)
    f, ax = plt.subplots()
    ax.set_title('PC pixel sum over frames (Nbr), with expected prob dist')
    # to get integer-valued x values for reliable chi2 analysis, we choose # of bins so that the x values are integer, as they should be
    # Line below gave bar graph histogram, and the fit looked funny with that visually since all I needed was the left edge of each bar
    #y_vals, x_vals, _ = ax.hist(co_added_unc, bins=co_added_unc.max())
    # If N is set very high and ntimes is not high enough, then a better visual fit to the plot made below would be for fewer bins than what is specified in the next line
    y_vals, x_vals = np.histogram(co_added_unc, bins=int(np.round(co_added_unc.max())))

    #scale_old = np.max(y_vals)/np.max(poisson_dist(x_arr, N*L*e_coinloss(L)*e_thresh(g, L, T)))
    #scale_new = np.max(y_vals)/np.max(prob_dist(x_arr, g, L, T, N))
    # x_vals is the boundaries of the bins, so the length of x_vals is 1 more than y_vals, so we take x_vals[:-1]
    # this scale ensures the data and the expected distribution values are normalized the same, basically
    #scale_old = np.sum(y_vals)/np.sum(poisson_dist(x_vals[:-1], N*L*e_coinloss(L)*e_thresh(g,L,T)))
    scale_new = np.sum(y_vals)/np.sum(prob_dist(x_vals[:-1], g,L,T,N))
    #chisquare_value, pvalue = chisquare(y_vals, scale_new*prob_dist(x_vals[:-1], g, L, T, N))
    chisquare_value, pvalue = chisquare(y_vals/scale_new, prob_dist(x_vals[:-1], g, L, T, N))
    #chisquare_value2, pvalue2 = chisquare(y_vals, scale_old*poisson_dist(x_vals[:-1], N*L*e_coinloss(L)*e_thresh(g,L,T)))
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=co_added_unc.max()))
    # p value close to 1:  data and trial probability distribution not statistically distinct
    # null hypothesis accepted (i.e., good fit) when chi square value less than critical value
    #plt.plot(x_arr, scale_old*poisson_dist(x_arr, N*L*e_coinloss(L)*e_thresh(g, L, T)))
    plt.scatter(x_vals[:-1], y_vals/scale_new)
    plt.xlabel('x (in electrons)')
    plt.ylabel('probability distribution')
    plt.title(r'Histogram of counts for $N_{br}$ photon-counted frames')
    plt.plot(x_arr, prob_dist(x_arr, g, L, T, N))
    plt.show()

