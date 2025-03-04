#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import logging
import time
import pickle
from functools import partial
from multiprocessing import get_context
import tqdm
from simple_pe.waveforms import generate_eccentric_waveform, calculate_eccentric_harmonics, calc_f_gen
from simple_pe.param_est import find_metric_and_eigendirections
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from pycbc import psd as psd_func
from calcwf import *
from interpolating_match import *

def single_match(params, wf_hjs, f_gen, f_low, psd, tlen, sample_rate):

    # Unpack values and generate waveform
    s_f = params['s_f']
    e = params['ecc_gen']
    M = params['total_mass']
    q = params['inverted_mass_ratio']
    chi_eff = params['chi_eff']
    s = generate_eccentric_waveform(M, q, e, chi_eff, chi_eff, s_f, sample_rate, f_ref_e=f_gen, tlen=tlen, to_fs=False)[0]

    # Calculate matches
    match_cplx = match_hn(wf_hjs, s, f_gen, psd=psd, f_match=f_low)

    # Get match quantities
    match_quantities = []
    for i in range(1, len(wf_hjs)):
        match_quantities.append(np.abs(match_cplx[i])/np.abs(match_cplx[0])) # Single harmonic
        if i > 1:
            num = 0
            for j in range(1,i+1):
                num += np.abs(match_cplx[j])**2
            match_quantities.append(np.sqrt(num/np.abs(match_cplx[0]**2))) # Naive multiple harmonics
            if i == 2:
                pc_frac = comb_harm_consistent(np.abs(match_cplx[:i+1]), np.angle(match_cplx[:i+1]), harms=[0,1,-1])
                match_quantities.append(pc_frac) # Phase consistent multiple harmonics

    # Save memory
    del s

    return *match_cplx, np.linalg.norm(match_cplx), *match_quantities

def degen_line_grid_data(base_dict, fid_dict, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, sample_rate):

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n_ecc_harms):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Generate param values along line of degeneracy
    e2_vals = e_vals**2
    param_dists = (e2_vals-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    param_vals = {'ecc10': e_vals, 'ecc10sqrd': e2_vals}
    for param in ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']:
        param_vals[param] = param_dists*(fid_dict[param]-base_dict[param])+base_dict[param]
    f_gen = calc_f_gen(f_low, n_ecc_harms)
    param_vals['ecc_gen'] = shifted_e(f_gen, 10, param_vals['ecc10'])

    # Convert mass parameters to total mass and inverted mass ratio
    param_vals['mass_ratio'] = q_from_eta(param_vals['symmetric_mass_ratio'])
    param_vals['inverted_mass_ratio'] = 1/param_vals['mass_ratio']
    param_vals['total_mass'] = np.sum(component_masses_from_mchirp_q(param_vals['chirp_mass'], param_vals['mass_ratio']), axis=0)
    fid_q = 1/q_from_eta(fid_dict['symmetric_mass_ratio'])
    fid_M = np.sum(component_masses_from_mchirp_q(fid_dict['chirp_mass'], fid_q))

    # Get all possible parameter values
    param_list = []
    for i in range(len(param_vals['ecc_gen'])):
        s_f_2pi = f_gen - shifted_f(f_gen, param_vals['ecc_gen'][i], param_vals['total_mass'][i], param_vals['inverted_mass_ratio'][i])
        s_f_vals = f_gen - MA_vals*s_f_2pi/(2*np.pi)
        for s_f in s_f_vals:
            param_list.append({'s_f': s_f, 'total_mass': param_vals['total_mass'][i],
                               'ecc_gen': param_vals['ecc_gen'][i],
                               'inverted_mass_ratio': param_vals['inverted_mass_ratio'][i],
                               'chi_eff': param_vals['chi_eff'][i]})

    # Generate longest possible waveform and psd
    long_M = np.min(param_vals['total_mass'])
    long_q = np.min(param_vals['inverted_mass_ratio'])
    long_e = np.min(param_vals['ecc_gen'])
    long_chi_eff = np.max(param_vals['chi_eff'])
    long_f = shifted_f(f_gen, long_e, long_M, long_q)
    long_wf = generate_eccentric_waveform(long_M, long_q, long_e, long_chi_eff, long_chi_eff, long_f,
                                          sample_rate, f_ref_e=f_gen, to_fs=False)[0]
    tlen = ceiltwo(len(long_wf))
    long_wf.resize(tlen)
    psd = gen_psd(long_wf, f_gen)
    tlen = int(tlen/sample_rate)

    # Generate fiducial harmonics
    wf_dict = calculate_eccentric_harmonics(fid_M, fid_q, fid_dict['ecc10sqrd']**0.5, fid_dict['chi_eff'], fid_dict['chi_eff'], f_low,
                                           sample_rate, tlen=tlen, f_ref_e=10, n_ecc_harms=n_ecc_harms, n_ecc_gen=n_ecc_gen)
    wf_hjs = []
    for id in harm_ids:
        wf_hjs.append(wf_dict[id].to_timeseries())
    del wf_dict

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs=wf_hjs, f_gen=f_gen, f_low=f_low, psd=psd, tlen=tlen, sample_rate=sample_rate)
    with get_context('spawn').Pool() as pool:
        match_arr = np.array(list(tqdm.tqdm(pool.imap(partial_single_match, param_list), total=len(param_list))))

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for i in range(n_ecc_harms):
        matches[f'h{harm_ids[i]}'] = np.abs(match_arr[:,i].reshape(-1, len(MA_vals)))
        matches[f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(-1, len(MA_vals)))
    matches['quad'] = np.abs(match_arr[:,n_ecc_harms].reshape(-1, len(MA_vals)))
    count = n_ecc_harms+1
    for i in range(1,n_ecc_harms):
        matches[f'h{harm_ids[i]}_h0'] = np.abs(match_arr[:,count].reshape(-1, len(MA_vals)))
        count += 1
        if i > 1:
            str_combo = ''
            for j in range(1, i+1):
                str_combo += f'h{harm_ids[j]}_'
            matches[f'{str_combo}h0'] = np.abs(match_arr[:,count].reshape(-1, len(MA_vals)))
            count += 1
            if i == 2:
                matches[f'{str_combo}h0_pc'] = np.abs(match_arr[:,count].reshape(-1, len(MA_vals)))
                count += 1

    # Add parameter keys
    matches['metadata'] = {}
    matches['metadata']['base_params'] = base_dict
    matches['metadata']['fid_params'] = fid_dict
    matches['metadata']['degen_params'] = param_vals
    matches['metadata']['MA_vals'] = MA_vals
    matches['metadata']['f_low'] = f_low
    matches['metadata']['f_gen'] = f_gen
    matches['metadata']['n_ecc_harms'] = n_ecc_harms
    matches['metadata']['n_ecc_gen'] = n_ecc_gen
    matches['metadata']['sample_rate'] = sample_rate

    return matches

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
    par_dirs = ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio']

    # Calculate metric and find fiducial point
    metric = find_metric_and_eigendirections(pars, par_dirs, snr=snr, f_low=psds['f_low'],
                                             psd=pycbc_psd['harm'], approximant=approximant)
    fid_dict = const_mm_point(metric, mismatch, 'ecc10sqrd', pars)

    # Add in base values of parameters not included
    for key in pars.keys():
        if key not in fid_dict.keys() and key != 'f_ref':
            fid_dict[key] = pars[key]

    return fid_dict

def pipeline(base_dict, mismatch, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, sample_rate=4096):

    # Find fiducial point along degeneracy line
    start = time.time()
    fid_dict = find_fid_point(base_dict, mismatch, f_low)
    end = time.time()
    print(f'Fiducial point found in {end-start} seconds.')
    print(fid_dict)

    # Generate grid data
    start = time.time()
    all_matches = degen_line_grid_data(base_dict, fid_dict, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, sample_rate)
    end = time.time()
    print(f'Generated grid in {end-start} seconds.')

    # Save grid data
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Run pipeline
    base_dict = {'chirp_mass': 24, 'symmetric_mass_ratio': 2/9, 'ecc10sqrd': 0, 'chi_eff': 0}
    all_matches = pipeline(base_dict, 8.76*10**-5, np.linspace(0, 0.5, 151), np.linspace(0, 2*np.pi, 32), 4, 6, 20)
