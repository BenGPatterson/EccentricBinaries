#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from calcwf import *

def favata_et_al_avg(given_e, given_chirp, e_vals, f_low=10, q=2):

    # Generate waveform at given point to use in sigmasq
    h = gen_wf(f_low, given_e, chirp2total(given_chirp, q), q, sample_rate, 'TEOBResumS')
    h.resize(ceiltwo(len(h)))

    # Generate the aLIGO ZDHP PSD
    psd, _ = gen_psd(h, f_low)

    # Calculate both integrals using sigmasq
    h = h.real().to_frequencyseries()
    ss = sigmasq(h, psd=psd, low_frequency_cutoff=f_low+3)
    ssf = sigmasq(h*h.sample_frequencies**(-7), psd=psd, low_frequency_cutoff=f_low+3)

    # Use average frequency to evolve eccentricities
    avg_f = (ssf/ss)**(-1/14)
    s_given_e = shifted_e(avg_f, f_low, given_e)
    s_e_vals = shifted_e(avg_f, f_low, e_vals)

    # Find effective chirp mass of given point
    eff_chirp = given_chirp/(1-(157/24)*s_given_e**2)**(3/5)

    # Convert to chirp mass values
    chirp_vals = eff_chirp*(1-(157/24)*s_e_vals**2)**(3/5)

    return chirp_vals

def chirp_match_MA_grid_data(param_vals, MA_vals, n, fid_e, fid_M, q, f_low, approximant='TEOBResumS'):

    # Setup match dict and calculate conversion from chirp to total mass
    matches = {'diff_phase':[], 'quad':[]}
    for i in range(n):
            matches[f'h{i+1}'] = []
            matches[f'h{i+1}_phase'] = []

    # Generate fiducial waveform
    all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_M, q), q, sample_rate, approximant=approximant))
    wf_hjs = all_wfs[1:n+1]

    # Generate param values
    e_vals = param_vals
    chirp_vals = favata_et_al_avg(fid_e, fid_M, e_vals, f_low=f_low, q=q)

    # Progress bar setup
    progress = 0
    gridsize = len(chirp_vals)*len(MA_vals)

    # Loop over chirp mass values
    for e, chirp in zip(e_vals, chirp_vals):
        for i in range(n):
            matches[f'h{i+1}'].append([])
            matches[f'h{i+1}_phase'].append([])
        matches['diff_phase'].append([])
        matches['quad'].append([])

        # Find shifted_e, shifted_f for all MA values
        s_f_2pi = f_low - shifted_f(f_low, e, chirp2total(chirp, q), q)
        s_f_vals = f_low - MA_vals*s_f_2pi/(2*np.pi)
        s_e_vals = shifted_e(s_f_vals, f_low, e)

        # Loop over MA values
        for s_f, s_e in zip(s_f_vals, s_e_vals):

            s = gen_wf(s_f, s_e, chirp2total(chirp, q), q, sample_rate, approximant=approximant)

            # Calculate matches with chosen method
            match_cplx = match_hn(wf_hjs, s, f_low)

            # Save matches
            match_quad_sqrd = 0
            for i in range(n):
                matches[f'h{i+1}'][-1].append(abs(match_cplx[i]))
                matches[f'h{i+1}_phase'][-1].append(np.angle(match_cplx[i]))
                match_quad_sqrd += abs(match_cplx[i])**2
            phase_diff = np.angle(match_cplx[1])-np.angle(match_cplx[0])
            if phase_diff > 0:
                phase_diff -= 2*np.pi
            elif phase_diff < -2*np.pi:
                phase_diff += 2*np.pi
            matches['diff_phase'][-1].append(phase_diff)
            matches['quad'][-1].append(np.sqrt(match_quad_sqrd))

            # Progress bar
            progress += 1
            if progress % 10 == 0:
                print(f'Chirp mass {fid_M}: {progress} done out of {gridsize}.')

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
    gen_grid_data(np.linspace(24, 50, 1), np.linspace(0,0.2,101), np.linspace(0, 2*np.pi, 32, endpoint=False), 4, 0.1, 2, 10, approximant='TEOBResumS')
