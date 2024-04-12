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

def chirp_match_MA_grid_data(param_vals, MA_vals, n, fid_e, fid_M, q, f_low, approximant='TEOBResumS'):

    # Generate fiducial waveform
    all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_M, q), q, sample_rate, approximant=approximant))
    wf_hjs = all_wfs[1:n+1]

    # Generate param values along line of degeneracy
    e_vals = param_vals
    chirp_vals = favata_et_al_avg(fid_e, fid_M, e_vals, sample_rate, f_low=f_low, q=q)
    e_chirp_vals = list(map(list, zip(e_vals, chirp_vals)))
    e_chirp_vals = list(np.repeat(e_chirp_vals, len(MA_vals), axis=0))

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
        matches[f'h{i+1}'] = match_arr[:,2*i].reshape(-1, len(MA_vals))
        matches[f'h{i+1}_phase'] = match_arr[:,2*i+1].reshape(-1, len(MA_vals))
    matches['diff_phase'] = match_arr[:,2*n].reshape(-1, len(MA_vals))
    matches['quad'] = match_arr[:,2*n+1].reshape(-1, len(MA_vals))
    matches['h2_h1'] = np.array(matches['h2'])/np.array(matches['h1'])
    matches['e_vals'] = e_vals

    return matches

def gen_grid_data(chirp_vals, param_vals, MA_vals, n, fid_e, q, f_low, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate grid for each chirp mass
    for chirp in chirp_vals:
        start = time.time()
        all_matches[chirp] = chirp_match_MA_grid_data(param_vals, MA_vals, n, fid_e, chirp, q, f_low, approximant=approximant)
        end = time.time()
        print(f'Chirp mass {chirp}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_grid_data(np.linspace(20, 30, 2), np.linspace(0,0.5,201), np.linspace(0, 3*np.pi, 48, endpoint=False), 4, 0.1, 2, 10, approximant='TEOBResumS')
