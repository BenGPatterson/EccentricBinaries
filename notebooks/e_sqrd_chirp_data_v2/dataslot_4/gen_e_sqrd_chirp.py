#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from calcwf import *
from interpolating_match import *

def single_match(e_chirp, wf_hjs_list, q, f_low, harm_ids, psd, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    e, chirp = e_chirp
    s = gen_wf(f_low, e, chirp2total(chirp, q), q, sample_rate, approximant=approximant)

    # Calculate matches
    match_quantities_list = []
    for wf_hjs in wf_hjs_list:

        match_cplx = match_hn(wf_hjs, s, f_low, psd=psd)

        # Process matches
        match_quantities = match_cplx.copy()
        match_quantities.append(np.linalg.norm(match_cplx))
        for i in range(1, len(wf_hjs)):
            match_quantities.append(np.abs(match_cplx[i])/np.abs(match_cplx[0])) # Single harmonic
            if i > 1:
                num = 0
                for j in range(1,i+1):
                    num += np.abs(match_cplx[j])**2
                match_quantities.append(np.sqrt(num/np.abs(match_cplx[0]**2))) # Naive multiple harmonics
                pc_frac = comb_harm_consistent(np.abs(match_cplx[:i+1]), np.angle(match_cplx[:i+1]), harms=harm_ids[:i+1])
                match_quantities.append(pc_frac) # Phase consistent multiple harmonics

        match_quantities_list.append(match_quantities)

    # Save memory
    del s

    return match_quantities_list

def e_sqrd_chirp_grid_data(e_vals, chirp_vals, n, fid_e_vals, fid_chirp, q, f_low, harm_ids, approximant='TEOBResumS'):

    # Generate fiducial waveform, resize to longest possible, and calculate psd
    wf_hjs_list = []
    long_wf = gen_wf(f_low, e_vals[0], chirp2total(chirp_vals[0], q), q, sample_rate, approximant)
    for fid_e in fid_e_vals:
        all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant=approximant))
        wf_hjs = resize_wfs(all_wfs[1:n+1], tlen=ceiltwo(len(long_wf)))
        del all_wfs
        wf_hjs_list.append(wf_hjs)
    h_psd = timeseries.TimeSeries(list(wf_hjs[0].copy())+[0], wf_hjs[0].delta_t, epoch=wf_hjs[0].start_time)
    psd = gen_psd(h_psd, f_low)

    # Generate list of all grid points
    e_vals_, chirp_vals_ = np.meshgrid(e_vals,chirp_vals,indexing='ij')
    e_chirp_vals = np.array([e_vals_.flatten(), chirp_vals_.flatten()]).T

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs_list=wf_hjs_list, q=q, f_low=f_low, harm_ids=harm_ids, psd=psd, approximant=approximant)
    match_arr_list = np.array(p_tqdm.p_map(partial_single_match, e_chirp_vals))
    match_arr_list = np.swapaxes(match_arr_list, 0, 1)

    # Save memory
    del wf_hjs_list

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for fid_e, match_arr in zip(fid_e_vals, match_arr_list):
        matches[fid_e] = {}
        for i in range(n):
            matches[fid_e][f'h{harm_ids[i]}'] = np.abs(match_arr[:,i].reshape(len(e_vals), len(chirp_vals)))
            matches[fid_e][f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(len(e_vals), len(chirp_vals)))
        matches[fid_e]['quad'] = match_arr[:,n].reshape(len(e_vals), len(chirp_vals))
        count = n+1
        for i in range(1,n):
            matches[fid_e][f'h{harm_ids[i]}_h0'] = match_arr[:,count].reshape(len(e_vals), len(chirp_vals))
            count += 1
            if i > 1:
                str_combo = ''
                for j in range(1, i+1):
                    str_combo += f'h{harm_ids[j]}_'
                matches[fid_e][f'{str_combo}h0'] = match_arr[:,count].reshape(len(e_vals), len(chirp_vals))
                count += 1
                matches[fid_e][f'{str_combo}h0_pc'] = match_arr[:,count].reshape(len(e_vals), len(chirp_vals))
                count += 1

        # Add parameter keys
        matches[fid_e]['fid_params'] = {}
        matches[fid_e]['fid_params']['f_low'] = f_low
        matches[fid_e]['fid_params']['e'] = fid_e
        matches[fid_e]['fid_params']['M'] = chirp2total(fid_chirp, q)
        matches[fid_e]['fid_params']['q'] = q
        matches[fid_e]['fid_params']['n'] = n
        matches[fid_e]['fid_params']['sample_rate'] = sample_rate
        matches[fid_e]['fid_params']['approximant'] = approximant

        # Add grid size keys
        matches[fid_e]['e_vals'] = e_vals
        matches[fid_e]['chirp_vals'] = chirp_vals

    return matches

def gen_e_sqrd_chirp_data(fid_e_vals, fid_chirp_vals, e_vals, chirp_vals, n, q, f_low, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Calculate grid for each chirp mass at all fiducial eccentricities
    for i, fid_chirp in enumerate(fid_chirp_vals):
        start = time.time()
        all_matches[fid_chirp] = e_sqrd_chirp_grid_data(e_vals, chirp_vals[i], n, fid_e_vals, fid_chirp, q, f_low, harm_ids, approximant=approximant)
        end = time.time()
        print(f'Chirp mass {fid_chirp}, eccentricities {fid_e_vals}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_e_sqrd_chirp_data([0, 0.05, 0.1, 0.15, 0.2], [24, 50], np.sqrt(np.linspace(0, 0.5**2, 301)), [np.linspace(21,25,101), np.linspace(44,56,101)], 4, 2, 10, approximant='TEOBResumS')
