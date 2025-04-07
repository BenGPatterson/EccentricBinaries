#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import time
import pickle
from functools import partial
from multiprocessing import get_context
import tqdm
from simple_pe.waveforms import (generate_eccentric_waveform, calculate_eccentric_harmonics, calc_f_gen,
                                 calculate_mode_snr, network_mode_snr)
from simple_pe.param_est import find_metric_and_eigendirections, find_peak_snr
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from pycbc import psd as psd_func
from pycbc.filter import sigma
from calcwf import *
from interpolating_match import *
from plot_funcs import plot_SNR2ecc

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
                pc_frac, MA = comb_harm_consistent(np.abs(match_cplx[:i+1]), np.angle(match_cplx[:i+1]), harms=[0,1,-1])
                match_quantities.append(pc_frac) # Phase consistent multiple harmonics
                match_quantities.append(MA)

    # Save memory
    del s

    return *match_cplx, np.linalg.norm(match_cplx), *match_quantities

def degen_line_grid_data(base_dict, fid_dict, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, sample_rate, psd):

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

    # Generate fiducial harmonics
    tlen = int(1/psd.delta_f)
    wf_dict = calculate_eccentric_harmonics(fid_M, fid_q, fid_dict['ecc10sqrd']**0.5, fid_dict['chi_eff'], fid_dict['chi_eff'], f_low,
                                           sample_rate, tlen=tlen, f_ref_e=10, n_ecc_harms=n_ecc_harms, n_ecc_gen=n_ecc_gen)
    wf_hjs = []
    for id in harm_ids:
        wf_hjs.append(wf_dict[id].to_timeseries())
    del wf_dict

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs=wf_hjs, f_gen=f_gen, f_low=f_low, psd=psd, tlen=tlen, sample_rate=sample_rate)
    multi_ctx = get_context('forkserver')
    preload = ['numpy', 'calcwf', 'interpolating_match', 'simple_pe']
    multi_ctx.set_forkserver_preload(preload)
    with multi_ctx.Pool() as pool:
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
                matches[f'{str_combo}h0_pc_phase'] = np.abs(match_arr[:,count+1].reshape(-1, len(MA_vals)))
                count += 2

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

    return matches, wf_hjs

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

def find_fid_point(pars, mismatch, snr, approximant, f_low, psd):

    # Uses simple-pe to calculate approx. of posterior dist. using metric, eigendirections
    pars['f_ref'] = f_low
    par_dirs = ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']

    # Calculate metric and find fiducial point
    metric = find_metric_and_eigendirections(pars, par_dirs, snr=snr, f_low=f_low, psd=psd['harm'],
                                             approximant=approximant, max_iter=2, multiprocessing=True)
    fid_dict = const_mm_point(metric, mismatch, 'ecc10sqrd', pars)

    # Add in base values of parameters not included
    for key in pars.keys():
        if key not in fid_dict.keys() and key != 'f_ref':
            fid_dict[key] = pars[key]

    return fid_dict

def pipeline(data, init_guess, t_bounds, mismatch, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, psd,
             ifos, prior_e, prior_MA, true_e=None, sample_rate=4096):

    # Disable pesummary warnings
    _logger = logging.getLogger('PESummary')
    _logger.setLevel(logging.CRITICAL + 10)

    # Otherwise plt.savefig crashes with latex errors
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Find non-eccentric point with maximum SNR
    start = time.time()
    t_start, t_end = t_bounds
    dx_directions = ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    bounds = [(10,100), (0.1,0.25), (-0.99,0.99)]
    peak_dict, snr = find_peak_snr(ifos, data, psd, t_start, t_end, init_guess, dx_directions, f_low,
                                   bounds=bounds, approximant="TEOBResumS", method='scipy')
    base_dict = {key: peak_dict[key] for key in dx_directions}
    base_dict['ecc10sqrd'] = 0
    end = time.time()
    print(f'Peak SNR of {snr} found in {end-start} seconds.')
    print(base_dict)

    # Find fiducial point along degeneracy line
    start = time.time()
    fid_dict = find_fid_point(base_dict, mismatch, snr, 'TEOBResumS-Dali-Harms', f_low, psd)
    end = time.time()
    print(f'Fiducial point found in {end-start} seconds.')
    print(fid_dict)

    # Generate grid data
    start = time.time()
    all_matches, wf_hjs = degen_line_grid_data(base_dict, fid_dict, e_vals, MA_vals, n_ecc_harms, n_ecc_gen, f_low, sample_rate, psd['harm'])
    end = time.time()
    print(f'Generated grid in {end-start} seconds.')

    # Save grid data
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

    # Match filter harmonics against data
    fid_harms = {0: wf_hjs[0], 1: wf_hjs[1]}
    if n_ecc_harms > 2:
        fid_harms[-1] = wf_hjs[2]
    z = {}
    assert len(ifos) == 1
    fid_perp = {}
    for key in fid_harms.keys():
        fid_perp[key] = fid_harms[key] / sigma(fid_harms[key], psd[ifos[0]], low_frequency_cutoff=f_low,
                                                high_frequency_cutoff=psd[ifos[0]].sample_frequencies[-1])
    mode_SNRs, _ = calculate_mode_snr(data[ifos[0]], psd[ifos[0]], fid_perp, data[ifos[0]].sample_times[0],
                                      data[ifos[0]].sample_times[-1], f_low, fid_perp.keys(), dominant_mode=0)
    z[ifos[0]] = mode_SNRs
    print(f'Single detector SNRs calculated:')
    for key in z[ifos[0]].keys():
        print(f'rho_{key}: {np.abs(z[ifos[0]][key]):.2f}')
        print(f'phi_{key}: {np.angle(z[ifos[0]][key]):.2f}')

    # Create parameter samples
    start = time.time()
    map_e, map_MA, map_SNR = create_map(all_matches)
    param_samples = get_param_samples(z[ifos[0]], prior_e, prior_MA, map_e, map_MA, map_SNR)
    end = time.time()
    print(f'Created parameter samples in {end-start} seconds.')

    # Save parameter samples
    with open('param_samples', 'wb') as fp:
        pickle.dump(param_samples, fp)

    # Plot parameter samples
    plot_SNR2ecc(all_matches, param_samples['samples']['ecc10'], param_samples['samples']['SNR'],
                 param_samples['prior']['ecc10'], param_samples['prior']['SNR'], param_samples['likeL']['SNR'],
                 true_e=true_e, meas_SNR=param_samples['SNR']['SNR'])
    plt.savefig('param_samples.png', dpi=450)

if __name__ == "__main__":

    # Data settings
    true_dict = {'ecc10sqrd': 0.2**2, 'chirp_mass': 24, 'symmetric_mass_ratio': 2/9, 'chi_eff': 0}
    init_guess = {'ecc10sqrd': 0, 'chirp_mass': 25, 'symmetric_mass_ratio': 0.20, 'chi_eff': 0.1}
    target_snr = 20
    t_bounds = [-0.05,0.05]
    f_low = 20
    sample_rate = 4096
    tlen = 32
    print(f'Analysing data with SNR of {target_snr}.')
    print(true_dict)

    # Create psd
    ifos = ['H1']
    psds = {'H1': 'aLIGOZeroDetHighPower',
            'f_low': f_low,
            'f_high': int(sample_rate/2),
            'length': tlen,
            'delta_f': 1. / tlen
            }
    psd = {}
    for ifo in ifos:
        psd[ifo] = psd_func.analytical.from_string(psds[ifo], psds['length'] * psds['f_high'] + 1, psds['delta_f'],
                                                         psds['f_low'])
    psd['harm'] = 1. / sum([1. / psd[ifo] for ifo in ifos])

    # Generate data
    true_dict['mass_ratio'] = q_from_eta(true_dict['symmetric_mass_ratio'])
    true_dict['inverted_mass_ratio'] = 1/true_dict['mass_ratio']
    true_dict['total_mass'] = np.sum(component_masses_from_mchirp_q(true_dict['chirp_mass'], true_dict['mass_ratio']), axis=0)
    data, _ = generate_eccentric_waveform(true_dict['total_mass'], true_dict['inverted_mass_ratio'], true_dict['ecc10sqrd']**0.5,
                                                   true_dict['chi_eff'], true_dict['chi_eff'], 10, sample_rate, f_ref_e=10, tlen=tlen, to_fs=False)
    raw_snr = sigma(data, psd['H1'], low_frequency_cutoff=f_low, high_frequency_cutoff=psds['f_high'])
    data = {'H1': data*target_snr/raw_snr}
    init_guess['distance'] = raw_snr/target_snr

    # Create prior
    prior_e = np.random.uniform(0, 0.5, 10**6)
    prior_MA = np.random.uniform(0, 2*np.pi, 10**6)

    # Run pipeline
    all_matches = pipeline(data, init_guess, t_bounds, 8.76*10**-5, np.linspace(0, 0.5, 151), np.linspace(0, 2*np.pi, 32), 4, 6,
                           f_low, psd, ifos, prior_e, prior_MA, true_e=np.sqrt(true_dict['ecc10sqrd']), sample_rate=sample_rate)
