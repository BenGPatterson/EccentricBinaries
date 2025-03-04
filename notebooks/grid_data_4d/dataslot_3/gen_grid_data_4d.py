#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import logging
import time
from functools import partial
from scipy.optimize import minimize
from simple_pe.waveforms import generate_eccentric_waveform, calculate_eccentric_harmonics, calc_f_gen
from simple_pe.param_est import find_metric_and_eigendirections
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from pycbc import psd as psd_func
from calcwf import *
from interpolating_match import *

def degen_line_best_point(base_dict, fid_dict, data_wf, max_e, n_ecc_harms, n_ecc_gen, f_low, sample_rate):

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n_ecc_harms):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Get point at start and end of line
    e2_vals = np.array([0, max_e**2])
    se_dists = (e2_vals-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    se_vals = {'ecc10': np.sqrt(e2_vals), 'ecc10sqrd': e2_vals}
    for param in ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']:
        se_vals[param] = se_dists*(fid_dict[param]-base_dict[param])+base_dict[param]
    f_gen = calc_f_gen(f_low, n_ecc_harms)
    se_vals['ecc_gen'] = shifted_e(f_gen, 10, se_vals['ecc10'])

    # Generate longest possible waveform and psd
    long_q = 1/q_from_eta(np.max(se_vals['symmetric_mass_ratio']))
    long_M = np.sum(component_masses_from_mchirp_q(np.min(se_vals['chirp_mass']), 1/long_q))
    long_e = np.min(se_vals['ecc_gen'])
    long_chi_eff = np.max(se_vals['chi_eff'])
    long_f = shifted_f(f_gen, long_e, long_M, long_q)
    long_wf = generate_eccentric_waveform(long_M, long_q, long_e, long_chi_eff, long_chi_eff, long_f,
                                          sample_rate, f_ref_e=f_gen, to_fs=False)[0]
    tlen = ceiltwo(len(long_wf))
    long_wf.resize(tlen)
    data_wf.resize(tlen)
    psd = gen_psd(long_wf, f_gen)
    tlen = int(tlen/sample_rate)

    # Iterate until found eccentricity with close enough match
    target_tolerance = 1e-2
    cur_tolerance = 1
    test_dict = fid_dict.copy()
    counter = 0
    while cur_tolerance > target_tolerance:

        # Convert parameters
        q = 1/q_from_eta(test_dict['symmetric_mass_ratio'])
        M = np.sum(component_masses_from_mchirp_q(test_dict['chirp_mass'], 1/q))

        # Generate harmonics
        harms = calculate_eccentric_harmonics(M, q, test_dict['ecc10sqrd']**0.5, test_dict['chi_eff'], test_dict['chi_eff'], f_low,
                                              sample_rate, tlen=tlen, f_ref_e=10, n_ecc_harms=n_ecc_harms, n_ecc_gen=n_ecc_gen)
        wf_hjs = []
        for id in harm_ids:
            wf_hjs.append(harms[id].to_timeseries())
        del harms

        # Generate waveform at same point and perform matches
        f_gen = calc_f_gen(f_low, n_ecc_harms)
        e_gen = shifted_e(f_gen, 10, test_dict['ecc10sqrd']**0.5)
        hp, _ = generate_eccentric_waveform(M, q, e_gen, test_dict['chi_eff'], test_dict['chi_eff'], f_gen,
                                            sample_rate, tlen=tlen, f_ref_e=f_gen, to_fs=False)
        true_match_cplx = match_hn(wf_hjs, hp, f_gen, psd=psd, f_match=f_low)
        true_frac = comb_harm_consistent(np.abs(true_match_cplx[:3]), np.angle(true_match_cplx[:3]), harms=[0,1,-1])
        test_match_cplx = match_hn(wf_hjs, data_wf.real(), f_gen, psd=psd, f_match=f_low)
        test_frac = comb_harm_consistent(np.abs(test_match_cplx[:3]), np.angle(test_match_cplx[:3]), harms=[0,1,-1])

        # Compare matches and compute new test point
        test_dict['ecc10sqrd'] = (test_frac/true_frac)**2*test_dict['ecc10sqrd']
        if test_dict['ecc10sqrd'] <= 0:
            test_dict['ecc10sqrd'] = 0.001
        elif test_dict['ecc10sqrd'] > max_e**2:
            test_dict['ecc10sqrd'] = max_e**2
        test_dist = (test_dict['ecc10sqrd']-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
        for param in ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']:
            test_dict[param] = test_dist*(fid_dict[param]-base_dict[param])+base_dict[param]

        # Calculate relative distance between matches
        cur_tolerance = np.abs(test_frac-true_frac)/true_frac
        counter += 1
        print(f'true: {true_frac}, test: {test_frac}, moving to e={test_dict["ecc10sqrd"]**0.5}')
        print(f'Tolerance down to {cur_tolerance} after {counter} iteration(s).')

    return test_dict

def const_mm_point(metric, mm, target_par, base_vals, direction=1):

    # Find A
    keys = list(metric.dxs.keys())
    target_ind = keys.index(target_par)
    A = metric.metric[target_ind][target_ind]

    # Find B
    param_keys = [key for key in keys if key != target_par]
    B = np.array([metric.metric[target_ind][k] for k in range(len(keys)) if k != target_ind])

    # Find inverse of C
    C = np.delete(metric.metric, (target_ind), axis=0)
    C = np.delete(C, (target_ind), axis=1)
    C_inv = np.linalg.inv(C)

    # Find change in target parameter to reach mismatch
    BC_term = -np.matmul(B, np.matmul(C_inv, B))
    d_target = direction*np.sqrt(mm/(A+BC_term))

    # Find extreme point at specified mismatch
    dxs = -1*np.matmul(C_inv, B) * d_target
    extreme_point = {target_par: base_vals[target_par]+d_target}
    for i, key in enumerate(param_keys):
        extreme_point[key] = dxs[i] + base_vals[key]

    return extreme_point

def find_fid_point(pars, mismatch, f_low):

    # Disable pesummary warnings
    _logger = logging.getLogger('PESummary')
    _logger.setLevel(logging.CRITICAL + 10)

    # Defines PSD settings
    ifos = ['H1']
    psds = {'H1': 'aLIGOZeroDetHighPower',
            'f_low': f_low,
            'f_high': 8192,
            'length': 32
            }
    snr = 18
    psds['delta_f'] = 1. / psds['length']
    approximant = 'TEOBResumS-Dali-Harms'

    # Calculates PSD
    pycbc_psd = {}
    for ifo in ifos:
        pycbc_psd[ifo] = psd_func.analytical.from_string(psds[ifo], psds['length'] * psds['f_high'] + 1, psds['delta_f'],
                                                         psds['f_low'])
    pycbc_psd['harm'] = 1. / sum([1. / pycbc_psd[ifo] for ifo in ifos])

    # Uses simple-pe to calculate approx. of posterior dist. using metric, eigendirections
    pars['f_ref'] = 20
    par_dirs = ['ecc10sqrd', 'chirp_mass']

    # Calculate metric and find fiducial point
    metric = find_metric_and_eigendirections(pars, par_dirs, snr=snr, f_low=psds['f_low'],
                                             psd=pycbc_psd['harm'], approximant=approximant)
    fid_dict = const_mm_point(metric, mismatch, 'ecc10sqrd', pars)

    # Add in base values of parameters not included
    for key in pars.keys():
        if key not in fid_dict.keys() and key != 'f_ref':
            fid_dict[key] = pars[key]

    return fid_dict

def pipeline(base_dict, mismatch, data_e, max_e, n_ecc_harms, n_ecc_gen, f_low, sample_rate=4096):

    # # Find fiducial point along degeneracy line
    # start = time.time()
    # fid_dict = find_fid_point(base_dict, mismatch, f_low)
    # end = time.time()
    # print(f'Fiducial point found in {end-start} seconds.')
    # print(fid_dict)

    fid_dict = {'ecc10sqrd': 0.009961434775270773, 'chirp_mass': 23.952359125409874,
                'symmetric_mass_ratio': 0.2222222222222222, 'chi_eff': 0}

    # Generate data waveform
    data_dict = {'ecc10sqrd': data_e**2}
    data_dist = (data_dict['ecc10sqrd']-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    for param in ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']:
        data_dict[param] = data_dist*(fid_dict[param]-base_dict[param])+base_dict[param]
    f_gen = calc_f_gen(f_low, n_ecc_harms)
    e_gen = shifted_e(f_gen, 10, data_e)
    q = 1/q_from_eta(data_dict['symmetric_mass_ratio'])
    M = np.sum(component_masses_from_mchirp_q(data_dict['chirp_mass'], 1/q))
    hp, hc = generate_eccentric_waveform(M, q, e_gen, data_dict['chi_eff'], data_dict['chi_eff'], f_gen,
                                        sample_rate, tlen=32, f_ref_e=f_gen, to_fs=False)
    data_wf = hp - 1j*hc
    print('Generate data waveform.')
    print(data_dict)

    # Generate grid data
    start = time.time()
    best_point = degen_line_best_point(base_dict, fid_dict, data_wf, max_e, n_ecc_harms, n_ecc_gen, f_low, sample_rate)
    end = time.time()
    print(f'Found best point in {end-start} seconds.')
    print(best_point)

if __name__ == "__main__":

    sample_rate = 4096

    # Run pipeline
    base_dict = {'chirp_mass': 24, 'symmetric_mass_ratio': 2/9, 'ecc10sqrd': 0, 'chi_eff': 0}
    all_matches = pipeline(base_dict, 8.76*10**-5, 0.2, 0.5, 4, 6, 20)
