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
    #full_fluxmap = np.ones((1024, 1024))
    pix_row = 150 #number of rows and number of columns
    fluxmap = np.ones((pix_row,pix_row))
    frametime = .1  # s (adjust lambda by adjust this)
    em_gain = 5000.

    emccd = EMCCDDetect(
        em_gain=em_gain,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=8.33e-4,  # e-/pix/s
        cic=0.02,  # e-/pix/frame
        read_noise=100.,  # e-/pix/frame
        bias=10, # 10000.,  # e-
        qe=.9,  # set this to 1 so it doesn't affect lambda
        cr_rate=0.,  # hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=2.,  # set this to 1 so there's no data loss when converting back to e-
        nbits=64,
        numel_gain_register=604
        )

    # Simulate several full frames
    frames_l = []
    pc_inputs = []
    nframes = 100
    thresh = emccd.em_gain/10
    counts = 0

    if np.average(frametime*fluxmap) > 0.5:
        warnings.warn('average # of photons/pixel is > 0.5.  Decrease frame '
        'time to get lower average # of photons/pixel.')

    # Photon count, co-add, and correct for photometric error
    if emccd.read_noise <=0:
       warnings.warn('read noise should be greater than 0 for effective '
       'photon counting')
    if thresh < 4*emccd.read_noise:
       warnings.warn('thresh should be at least 4 or 5 times read_noise for '
       'accurate photon counting')

    #investigating case where number of frames is 1:
    # for i in range(nframes):
    #     #sim_full_frame = emccd.sim_full_frame(full_fluxmap, frametime)
    #     sim_sub_frame = emccd.sim_sub_frame(fluxmap,frametime)
    #     pc_frame = get_count_rate(sim_sub_frame*emccd.eperdn-emccd.bias,
    #                              thresh, emccd.em_gain)
    #     pc_frame_unc = get_counts_uncorrected(sim_sub_frame*emccd.eperdn-emccd.bias,
    #                             thresh, emccd.em_gain)
    #     counts += np.sum(pc_frame_unc)/pix_row**2  #to get avg # of '1's per pixel
    #     #pc_inputs.append(sim_sub_frame*emccd.eperdn-emccd.bias)
    #     pc_inputs.append(pc_frame)
    #     e_frame = emccd.get_e_frame(sim_sub_frame)
    #     frames_l.append(e_frame)

    # mean_num_counts = counts/nframes

    # frames = np.stack(frames_l)

    # frame_e_cube = np.stack(pc_inputs)

    # Plot images
    #imagesc(emccd.get_e_frame(frames[0]), 'Output Full Frame')

    # f, ax = plt.subplots(1,2)
    # ax[0].hist(np.mean(frames,axis=0).flatten(), bins=20)
    # ax[0].axvline(np.mean(fluxmap)*frametime, color='black')
    # ax[0].set_title('Pixel mean')
    # ax[1].hist(np.std(frames,axis=0).flatten(), bins=20)
    # ax[1].axvline(np.sqrt(np.mean(fluxmap)*frametime),color='black')
    # ax[1].axvline(np.sqrt(2*np.mean(fluxmap)*frametime),color='red')
    # ax[1].set_title('Pixel sdev')
    # plt.tight_layout()
    # plt.show()

    exp_lambda = np.mean(fluxmap)*frametime
    #exp_lambda = mean_num_counts
    #pc_frame = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
    e_coinloss = (1 - np.exp(-exp_lambda)) / exp_lambda
    e_thresh = (
        np.exp(-thresh/em_gain)
        * (
            thresh**2 * exp_lambda**2
            + 2*em_gain * thresh * exp_lambda * (3 + exp_lambda)
            + 2*em_gain**2 * (6 + 3*exp_lambda + exp_lambda**2)
        )
        / (2*em_gain**2 * (6 + 3*exp_lambda + exp_lambda**2))
    )
    e_thresh1 = np.exp(-thresh/em_gain)

    #plotting images of photon-counted frames
    # f, ax = plt.subplots(1,2)
    # #ax[0].hist(np.mean(pc_frame).flatten(), bins=20)
    # ax[0].hist(np.mean(frame_e_cube,axis=0).flatten(), bins=20)
    # ax[0].axvline(np.mean(fluxmap)*frametime, color='black')
    # ax[0].axvline(mean_num_counts, color='red')
    # ax[0].axvline(e_thresh*exp_lambda*e_coinloss, color='green')
    # ax[0].axvline(np.mean(np.mean(frame_e_cube,axis=0).flatten()), color='brown')
    # ax[0].set_title('PC pixel mean')
    # #ax[1].hist(np.std(pc_frame).flatten(), bins=20)
    # ax[1].hist(np.std(frame_e_cube,axis=0).flatten(), bins=20)
    # ax[1].axvline(np.sqrt(mean_num_counts),color='black')
    # ax[1].axvline(np.sqrt(e_thresh*exp_lambda*e_coinloss), color='red')
    # ax[1].axvline(np.mean(np.std(frame_e_cube,axis=0).flatten()), color='green')
    # #ax[1].axvline(np.sqrt(exp_lambda*e_coinloss*e_thresh),color='red')
    # ax[1].set_title('PC pixel sdev')
    # plt.tight_layout()
    # plt.show()

    counts = 0
    ntimes = 100
    pc_list = []
    nframes = 70
    for x in range(ntimes):
        frame_e_list = []
        frame_e_dark_list = []
        for i in range(nframes):
            # Simulate bright
            frame_dn = emccd.sim_sub_frame(fluxmap, frametime)
            # Simulate dark
            frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap), frametime)

            # Convert from dn to e- and bias subtract
            frame_e = frame_dn * emccd.eperdn - emccd.bias
            frame_e_dark = frame_dn_dark * emccd.eperdn - emccd.bias

            frame_e_list.append(frame_e)
            frame_e_dark_list.append(frame_e_dark)

        frame_e_cube = np.stack(frame_e_list)
        mean_rate = get_count_rate(frame_e_cube, thresh, emccd.em_gain)
        pc_frame_unc = np.stack(get_counts_uncorrected(frame_e_cube,
                                thresh, emccd.em_gain))
        #counts += np.sum(np.sum(pc_frame_unc,axis=0))/(nframes*pix_row**2)  #to get avg # of '1's per pixel
        counts += np.sum(np.sum(pc_frame_unc,axis=0))/(nframes*pix_row**2)
        pc_list.append(mean_rate)
        #pc_list.append(pc_frame_unc)
        #pc_list.append(np.sum(pc_frame_unc,axis=0))
    mean_num_counts = counts/ntimes
    pc_cube = np.stack(pc_list)
    if emccd.qe*frametime* \
        np.average(np.sum(frame_e_cube,axis=0)/nframes)/emccd.em_gain < 5* \
        np.max(np.array([emccd.cic, emccd.dark_current])):
        warnings.warn('# of electrons/pixel needs to be bigger than about 5 '
        'times the noise (due to CIC and dark current)')



    f, ax = plt.subplots(1,2)
    #ax[0].hist(np.mean(pc_frame).flatten(), bins=20)
    ax[0].hist(np.mean(pc_cube,axis=0).flatten(), bins=20)
    #ax[0].axvline(np.mean(fluxmap)*frametime, color='black')
    #ax[0].axvline(mean_num_counts, color='red')
    #ax[0].axvline(e_thresh*exp_lambda*e_coinloss, color='green')
    ax[0].axvline(np.mean(np.mean(pc_cube,axis=0).flatten()), color='green')
    ax[0].set_title('PC pixel mean')
    #ax[1].hist(np.std(pc_frame).flatten(), bins=20)
    ax[1].hist(np.std(pc_cube,axis=0).flatten(), bins=20)
    #ax[1].axvline(np.sqrt(mean_num_counts),color='black')
    #ax[1].axvline(np.sqrt(exp_lambda*e_coinloss*e_thresh),color='red')
    ax[1].axvline(np.mean(np.std(pc_cube,axis=0).flatten()), color='green')
    ax[1].set_title('PC pixel sdev')
    plt.tight_layout()
    plt.show()