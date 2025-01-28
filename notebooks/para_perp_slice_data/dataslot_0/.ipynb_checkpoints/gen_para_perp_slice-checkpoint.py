#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from itertools import combinations
from functools import partial
from pycbc.filter import match
import p_tqdm
from calcwf import *

def eta2q(eta):
    return 1/(2*eta) - 1 + np.sqrt(1-4*eta)/(2*eta)

def q2eta(q):
    return q/(1+q)**2

def match_harms(wf_h, wf_t, f_low, psd, f_match=20, dominant_mode=0):

    # Perform match dominant mode
    wf_len = 2*len(wf_h[dominant_mode])
    for mode in wf_h.keys():
        wf_t[mode].resize(wf_len)
        wf_h[mode].resize(wf_len)

    m0_amp, m0_index, m0_phase = match(wf_t[dominant_mode].real(), wf_h[dominant_mode].real(), psd=psd,
                                       low_frequency_cutoff=f_match, subsample_interpolation=True, return_phase=True)

    matches = {}
    for k in wf_h.keys():
        h = wf_t[k].real()
        matches[k] = abs(overlap_cplx(wf_h[k].real(), h.cyclic_time_shift(m0_index * wf_h[dominant_mode].delta_t), psd=psd, low_frequency_cutoff=f_match))

    return matches

def single_match(slice_params, wf_hjs, f_low, harm_ids, n_gen, psd, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    e, chirp, q, chi1 = slice_params['e'], slice_params['chirp'], slice_params['q'], slice_params['chi1']
    s = {}
    all_wfs = list(get_h([1]*n_gen, f_low, e, chirp2total(chirp, q), q, sample_rate, chi1=chi1, approximant=approximant))
    for i, id in enumerate(harm_ids):
        s[id] = all_wfs[i+1]

    # Calculate matches
    match_cplx = match_harms(wf_hjs, s, f_low, psd)

    return list(match_cplx.values())

def save_slice_matches(match_arr, p1_len, p2_len, harm_ids):

    # Matches for single slice
    matches = {}
    for i, id in enumerate(harm_ids):
        matches[f'h{id}'] = np.abs(match_arr[:,i].reshape(p1_len, p2_len))
        matches[f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(p1_len, p2_len))

    return matches

def para_perp_slice_data(para_grad, perp_grad, fid_e, fid_chirp, fid_q, fid_chi1, e_vals, chirp_vals_dict, n, n_gen, f_low, approximant='TEOBResumS'):

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Loop over parallel then perpendicular direction
    all_matches = {}
    for direct, grad in zip(['para', 'perp'], [para_grad, perp_grad]):

        # Generate list of all grid points
        chirp_vals = chirp_vals_dict[direct]
        fid_eta = q2eta(fid_q)
        eta_vals = fid_eta + (chirp_vals - fid_chirp)*grad
        e_mesh, chirp_mesh = np.meshgrid(e_vals, chirp_vals, indexing='ij')
        _, q_mesh = np.meshgrid(e_vals, eta2q(eta_vals), indexing='ij')
        e_flat, chirp_flat, q_flat = e_mesh.flatten(), chirp_mesh.flatten(), q_mesh.flatten()
        grid_len = len(e_flat)
        chi1_flat = np.full(grid_len, fid_chi1)
        param_list = []
        for trial_e, trial_chirp, trial_q, trial_chi1 in zip(e_flat, chirp_flat, q_flat, chi1_flat):
            trial_params = {'e': trial_e, 'chirp': trial_chirp, 'q': trial_q, 'chi1': trial_chi1}
            param_list.append(trial_params)

        # Generate fiducial waveform and psd if parallel direction
        if direct == 'para':
            wf_hjs = {}
            long_wf = gen_wf(f_low, np.min(e_vals), chirp2total(np.min(chirp_vals), eta2q(np.min(eta_vals))), eta2q(np.min(eta_vals)), sample_rate, chi1=fid_chi1, approximant=approximant)
            all_wfs = list(get_h([1]*n_gen, f_low, fid_e, chirp2total(fid_chirp, fid_q), fid_q, sample_rate, chi1=fid_chi1, approximant=approximant))
            wf_hjs_raw = resize_wfs(all_wfs[1:n+1], tlen=ceiltwo(len(long_wf)))
            for i, id in enumerate(harm_ids):
                wf_hjs[id] = wf_hjs_raw[i]
            h_psd = timeseries.TimeSeries(list(wf_hjs[0].copy())+[0], wf_hjs[0].delta_t, epoch=wf_hjs[0].start_time)
            psd = gen_psd(h_psd, f_low)

        # Calculate all matches in parallel
        start = time.time()
        partial_single_match = partial(single_match, wf_hjs=wf_hjs, f_low=f_low, harm_ids=harm_ids, n_gen=n_gen, psd=psd, approximant=approximant)
        match_arr = np.array(p_tqdm.p_map(partial_single_match, param_list))

        # Save slices to dictionary
        all_matches[direct] = save_slice_matches(match_arr, len(e_vals), len(chirp_vals), harm_ids)
        end = time.time()
        print(f'{direct} slice finished in {end-start} seconds')

    # Save input parameters to dictionary
    all_matches['input_params'] = {}
    all_matches['input_params']['para_gradient'] = para_grad
    all_matches['input_params']['perp_gradient'] = perp_grad
    all_matches['input_params']['fid_e'] = fid_e
    all_matches['input_params']['fid_chirp'] = fid_chirp
    all_matches['input_params']['fid_q'] = fid_q
    all_matches['input_params']['fid_chi1'] = fid_chi1
    all_matches['input_params']['e_vals'] = e_vals
    all_matches['input_params']['chirp_vals_dict'] = chirp_vals_dict
    all_matches['input_params']['n'] = n
    all_matches['input_params']['n_gen'] = n_gen
    all_matches['input_params']['f_low'] = f_low
    all_matches['input_params']['approximant'] = approximant

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    para_perp_slice_data(0.035, -1/(625*0.035) , 0.1, 24, 2, 0, np.linspace(0, 0.4, 31), {'para': np.linspace(22.5, 24.793650793650794, 31), 'perp': np.linspace(23.39236111111111, 24.7, 31)}, 4, 20, 10, approximant='TEOBResumS')
