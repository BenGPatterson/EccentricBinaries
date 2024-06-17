#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from calcwf import *

def single_match(s_f_e, e_chirp, wf_hjs, q, f_low, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    s_f, s_e = s_f_e
    e, chirp = e_chirp
    s = gen_wf(s_f, s_e, chirp2total(chirp, q), q, sample_rate, approximant=approximant)

    # Calculate matches
    match_cplx = match_hn(wf_hjs, s, f_low)

    # Process matches
    match_quad_sqrd = 0
    wf_matches = []
    for i in range(len(wf_hjs)):
        wf_matches.append(abs(match_cplx[i]))
        wf_matches.append(np.angle(match_cplx[i]))
        match_quad_sqrd += abs(match_cplx[i])**2
    phase_diff = np.angle(match_cplx[1])-np.angle(match_cplx[0])
    if phase_diff > 0:
        phase_diff -= 2*np.pi
    elif phase_diff < -2*np.pi:
        phase_diff += 2*np.pi

    return *wf_matches, phase_diff, np.sqrt(match_quad_sqrd)

def chirp_match_MA_grid_data(param_vals, MA_vals, n, fid_e, zero_ecc_chirp, q, f_low, approximant='TEOBResumS'):

    # Generate param values along line of degeneracy
    all_e_vals = np.array([fid_e, *param_vals]).flatten()
    all_chirp_vals = chirp_degeneracy_line(zero_ecc_chirp, all_e_vals, sample_rate, f_low=f_low, q=q)
    fid_e, e_vals = all_e_vals[0], all_e_vals[1:]
    fid_chirp, chirp_vals = all_chirp_vals[0], all_chirp_vals[1:]
    e_chirp_vals = list(map(list, zip(e_vals, chirp_vals)))
    e_chirp_vals = list(np.repeat(e_chirp_vals, len(MA_vals), axis=0))

    # Generate fiducial waveform
    all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant=approximant))
    wf_hjs = all_wfs[1:n+1]

    s_f_e_vals = []
    # Loop over chirp mass values
    for e, chirp in zip(e_vals, chirp_vals):

        # Find shifted_e, shifted_f for all MA values
        s_f_2pi = f_low - shifted_f(f_low, e, chirp2total(chirp, q), q)
        s_f_vals = f_low - MA_vals*s_f_2pi/(2*np.pi)
        s_e_vals = shifted_e(s_f_vals, f_low, e)
        s_f_e_vals += list(map(list, zip(s_f_vals, s_e_vals)))

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs=wf_hjs, q=q, f_low=f_low, approximant=approximant)
    match_arr = np.array(p_tqdm.p_map(partial_single_match, s_f_e_vals, e_chirp_vals))

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for i in range(n):
        matches[f'h{i}'] = match_arr[:,2*i].reshape(-1, len(MA_vals))
        matches[f'h{i}_phase'] = match_arr[:,2*i+1].reshape(-1, len(MA_vals))
    matches['diff_phase'] = match_arr[:,2*n].reshape(-1, len(MA_vals))
    matches['quad'] = match_arr[:,2*n+1].reshape(-1, len(MA_vals))
    matches['h1_h0'] = np.array(matches['h1'])/np.array(matches['h0'])
    matches['e_vals'] = e_vals
    matches['fid_e'] = fid_e

    return matches

def gen_grid_data(scaling_norms, param_vals, MA_vals, n, fid_e_vals, q, f_low, max_e, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate grid for each chirp mass
    for fid_e in fid_e_vals:
        start = time.time()
        zero_ecc_chirp = fid_e**(6/5)*scaling_norms[0]/(scaling_norms[1]**(6/5))
        e_vals = param_vals*fid_e
        if np.max(e_vals) > max_e:
            e_vals *= max_e/np.max(e_vals)
        all_matches[zero_ecc_chirp] = chirp_match_MA_grid_data(e_vals, MA_vals, n, fid_e, zero_ecc_chirp, q, f_low, approximant=approximant)
        end = time.time()
        print(f'Non-eccentric chirp mass: {zero_ecc_chirp}, fiducial eccentricity: {fid_e}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_grid_data([10, 0.0325], np.linspace(0, 20/3, 201), np.linspace(0, 2*np.pi, 32, endpoint=False), 4, np.linspace(0.15, 0.02, 14), 2, 10, 0.5, approximant='TEOBResumS')
