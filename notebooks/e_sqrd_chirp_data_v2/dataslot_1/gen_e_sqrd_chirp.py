#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from calcwf import *

def single_match(e_chirp, wf_hjs, q, f_low, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    e, chirp = e_chirp
    s = gen_wf(f_low, e, chirp2total(chirp, q), q, sample_rate, approximant=approximant)

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

def e_sqrd_chirp_grid_data(e_vals, chirp_vals, n, fid_e, fid_chirp, q, f_low, approximant='TEOBResumS'):

    # Generate fiducial waveform
    all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant=approximant))
    wf_hjs = all_wfs[1:n+1]

    # Generate list of all grid points
    e_vals_, chirp_vals_ = np.meshgrid(e_vals,chirp_vals,indexing='ij')
    e_chirp_vals = np.array([e_vals_.flatten(), chirp_vals_.flatten()]).T

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs=wf_hjs, q=q, f_low=f_low, approximant=approximant)
    match_arr = np.array(p_tqdm.p_map(partial_single_match, e_chirp_vals))

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for i in range(n):
        k=int((i+1)/2)*(i%2*2-1)
        matches[f'h{k}'] = match_arr[:,2*i].reshape(len(e_vals), len(chirp_vals))
        matches[f'h{k}_phase'] = match_arr[:,2*i+1].reshape(len(e_vals), len(chirp_vals))
    matches['diff_phase'] = match_arr[:,2*n].reshape(len(e_vals), len(chirp_vals))
    matches['quad'] = match_arr[:,2*n+1].reshape(len(e_vals), len(chirp_vals))
    matches['h1_h0'] = np.array(matches['h1'])/np.array(matches['h0'])
    matches['e_vals'] = e_vals
    matches['chirp_vals'] = chirp_vals

    return matches

def gen_e_sqrd_chirp_data(fid_e_vals, fid_chirp_vals, e_vals, chirp_vals, n, q, f_low, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate grid for each fiducial eccentricity, chirp mass
    for fid_e in fid_e_vals:
        all_matches[fid_e] = {}
        for i, fid_chirp in enumerate(fid_chirp_vals):
            start = time.time()
            all_matches[fid_e][fid_chirp] = e_sqrd_chirp_grid_data(e_vals, chirp_vals[i], n, fid_e, fid_chirp, q, f_low, approximant=approximant)
        end = time.time()
        print(f'Eccentricity: {fid_e}, chirp mass {fid_chirp}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_e_sqrd_chirp_data([0.05, 0.1, 0.2], [10, 24, 50], np.sqrt(np.linspace(0, 0.5**2, 301)), [np.linspace(9.1,10.1,101), np.linspace(21,25,101), np.linspace(44,56,101)], 4, 2, 10, approximant='TEOBResumS')
