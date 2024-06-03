#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from calcwf import *

def single_match(e_chirp, fid_h, q, f_low, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    e, chirp = e_chirp
    s = gen_wf(f_low, e, chirp2total(chirp, q), q, sample_rate, approximant)

    # Calculate matches
    match_cplx = match_wfs(fid_h, s, f_low, True)

    return abs(match_cplx), np.angle(match_cplx)

def e_sqrd_chirp_grid_data(e_vals, chirp_vals, n, fid_e, fid_chirp, q, f_low, approximant='TEOBResumS'):

    # Generate fiducial waveform
    fid_h = gen_wf(f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant)

    # Generate list of all grid points
    e_vals_, chirp_vals_ = np.meshgrid(e_vals,chirp_vals,indexing='ij')
    e_chirp_vals = np.array([e_vals_.flatten(), chirp_vals_.flatten()]).T

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, fid_h=fid_h, q=q, f_low=f_low, approximant=approximant)
    match_arr = np.array(p_tqdm.p_map(partial_single_match, e_chirp_vals))

    # Put match arrays into appropriate dictionary keys
    matches = {}
    matches[f'teob_match'] = match_arr[:,0].reshape(len(e_vals), len(chirp_vals))
    matches[f'teob_phase'] = match_arr[:,1].reshape(len(e_vals), len(chirp_vals))
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
    gen_e_sqrd_chirp_data([0], [24], np.sqrt(np.linspace(0, 0.5**2, 301)), [np.linspace(21,25,101)], 4, 2, 10, approximant='TEOBResumS')
