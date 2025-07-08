#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import time
import pickle
from multiprocessing import cpu_count
from scipy.optimize import minimize
from simple_pe.waveforms import calculate_mode_snr, make_waveform, two_ecc_harms_SNR, generate_eccentric_waveform_MA
from simple_pe.param_est import find_metric_and_eigendirections, find_peak_snr, pe
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from pycbc import psd as psd_func
from pycbc.filter import sigma, matched_filter
from pycbc.filter.matchedfilter import quadratic_interpolate_peak
from ecc4dpe import *

def test_ecc_point(e10, data, base_dict, fid_dict, ecc_harms, target_MA, f_low, psd, two_ecc_harms=True):

    # Make param dictionary
    dist = (e10**2-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    params = {'distance': 1}
    param_keys = ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    for param in param_keys:
        params[param] = dist*(fid_dict[param]-base_dict[param])+base_dict[param]

    # Create  trial waveform
    params['inverted_mass_ratio'] = 1/q_from_eta(params['symmetric_mass_ratio'])
    params['total_mass'] = np.sum(component_masses_from_mchirp_q(params['chirp_mass'], params['inverted_mass_ratio']))
    trial_wf = generate_eccentric_waveform_MA(params['total_mass'], params['inverted_mass_ratio'],
                                              params['ecc10sqrd']**0.5, params['chi_eff'], params['chi_eff'],
                                              psd, f_low, target_MA, ecc_harms, two_ecc_harms=two_ecc_harms)
    data_snr = matched_filter(trial_wf, data, psd, low_frequency_cutoff=f_low,
                              high_frequency_cutoff=psd.sample_frequencies[-1])
    maxsnr, max_id = data_snr.abs_max_loc()
    left = abs(data_snr[-1]) if max_id == 0 else abs(data_snr[max_id - 1])
    right = abs(data_snr[0]) if max_id == (len(data_snr) - 1) else abs(data_snr[max_id + 1])
    _, maxsnr = quadratic_interpolate_peak(left, maxsnr, right)

    return -maxsnr

def find_ecc_peak(data, base_dict, fid_dict, f_low, psd, n_ecc_harms, two_ecc_harms=True):

    # Find SNR of fiducial harmonics in data
    first_params = {key: fid_dict[key] for key in ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']}
    first_params['distance'] = 1
    ecc_harms = make_waveform(first_params, psd.delta_f, f_low, len(psd), approximant='TEOBResumS-Dali-Harms', n_ecc_harms=n_ecc_harms)
    mode_SNRs, _ = calculate_mode_snr(data, psd, ecc_harms, data.sample_times[0],
                                      data.sample_times[-1], f_low, ecc_harms.keys(), dominant_mode=0, subsample_interpolation=False)
    if two_ecc_harms:
        data_ecc_SNR, target_MA = two_ecc_harms_SNR({k: np.abs(mode_SNRs[k]) for k in [0,1,-1]},
                                                    {k: np.angle(mode_SNRs[k]) for k in [0,1,-1]})
    else:
        data_ecc_SNR = np.abs(mode_SNRs[1])/np.abs(mode_SNRs[0])
        target_MA = (np.angle(mode_SNRs[1])-np.angle(mode_SNRs[0])) % (2*np.pi)

    # Find SNR of waveform at fiducial parameters in data
    first_params['inverted_mass_ratio'] = 1/q_from_eta(first_params['symmetric_mass_ratio'])
    first_params['total_mass'] = np.sum(component_masses_from_mchirp_q(first_params['chirp_mass'], first_params['inverted_mass_ratio']))
    first_trial_wf = generate_eccentric_waveform_MA(first_params['total_mass'], first_params['inverted_mass_ratio'],
                                                    first_params['ecc10sqrd']**0.5, first_params['chi_eff'], first_params['chi_eff'],
                                                    psd, f_low, target_MA, ecc_harms, two_ecc_harms=two_ecc_harms)
    trial_z, _ = calculate_mode_snr(first_trial_wf, psd, ecc_harms, first_trial_wf.sample_times[0],
                                    first_trial_wf.sample_times[-1], f_low, ecc_harms.keys(),
                                    dominant_mode=0, subsample_interpolation=False)
    if two_ecc_harms:
        ecc_SNR_abs, _ = two_ecc_harms_SNR(
            {k: np.abs(trial_z[k]) for k in [0, 1, -1]},
            {k: np.angle(trial_z[k]) for k in [0, 1, -1]})
    else:
        ecc_SNR_abs = np.abs(trial_z[1]/trial_z[0])

    # Use scipy optimise to find maximum snr
    bounds = [(0, 0.8)]
    init_e10 = (data_ecc_SNR/ecc_SNR_abs)*(fid_dict['ecc10sqrd']**0.5)
    peak_e10_result = minimize(lambda x: test_ecc_point(x, data, base_dict, fid_dict, ecc_harms, target_MA, f_low, psd, two_ecc_harms=two_ecc_harms),
                               init_e10, bounds=bounds, method='Nelder-Mead', options={'xatol': 0.003, 'fatol': np.inf})
    peak_e10, peak_snr = peak_e10_result['x'][0], -peak_e10_result['fun']

    # Create dictionary of parameters at peak point
    dist = (peak_e10**2-base_dict['ecc10sqrd'])/(fid_dict['ecc10sqrd']-base_dict['ecc10sqrd'])
    peak_dict = {'distance': 1}
    param_keys = ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    for param in param_keys:
        peak_dict[param] = dist*(fid_dict[param]-base_dict[param])+base_dict[param]

    return peak_dict, peak_snr

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
    eta_max = 0.24
    base_pars = pars.copy()
    if pars['symmetric_mass_ratio'] > eta_max:
        print(f'Calculating metric at eta={eta_max} instead of {pars["symmetric_mass_ratio"]}')
        eta_diff = pars['symmetric_mass_ratio'] - eta_max
        base_pars['symmetric_mass_ratio'] = eta_max
    else:
        eta_diff = 0
    par_dirs = ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']

    # Calculate metric and find fiducial point
    metric = find_metric_and_eigendirections(base_pars, par_dirs, snr=snr, f_low=f_low, psd=psd['harm'],
                                             approximant=approximant, max_iter=0, multiprocessing=True)
    fid_dict = const_mm_point(metric, mismatch, 'ecc10sqrd', pars)

    # Add in base values of parameters not included
    for key in pars.keys():
        if key not in fid_dict.keys() and key != 'f_ref':
            fid_dict[key] = pars[key]
    fid_dict['symmetric_mass_ratio'] += eta_diff

    # Ensure degeneracy line does not give eta>0.25
    dist = 1/fid_dict['ecc10sqrd']
    max_e_eta = (fid_dict['symmetric_mass_ratio']-pars['symmetric_mass_ratio'])*dist+pars['symmetric_mass_ratio']
    if max_e_eta > 0.25:
        new_eta = (0.25-pars['symmetric_mass_ratio'])/dist + pars['symmetric_mass_ratio']
        print(f'Adjusting fiducial eta from {fid_dict["symmetric_mass_ratio"]} to {new_eta} to prevent '
        + 'unphysical degeneracy line')
        fid_dict['symmetric_mass_ratio'] = new_eta

    return fid_dict

def pipeline(data, init_guess, t_bounds, mismatch, n_ecc_harms, n_ecc_gen, f_low, psd,
             ifos, ecc_reweight_fn, two_ecc_harms=True, truth_dict=None, sample_rate=4096, make_plots=False):

    # Find non-eccentric point with maximum SNR
    start = time.time()
    t_start, t_end = t_bounds
    dx_directions = ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    bounds = [(10,100), (0.1,0.2499), (-0.99,0.99)]
    non_ecc_peak_dict, snr = find_peak_snr(ifos, data, psd, t_start, t_end, init_guess, dx_directions, f_low,
                                           bounds=bounds, approximant="IMRPhenomXPHM", method='scipy')
    base_dict = {key: non_ecc_peak_dict[key] for key in dx_directions}
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

    # Find eccentric peak along degeneracy line
    start = time.time()
    peak_dict, peak_SNR = find_ecc_peak(data['H1'], base_dict, fid_dict, f_low, psd['H1'], n_ecc_harms, two_ecc_harms=two_ecc_harms)
    end = time.time()
    print(f'Eccentric peak SNR of {peak_SNR} found in {end-start} seconds.')

    # Calculate peak harmonics and thus peak MA
    peak_dict, harm_dict, peak_SNRs = find_peak_MA(data['H1'], peak_dict, f_low, psd['H1'], n_ecc_harms, n_ecc_gen, two_ecc_harms=two_ecc_harms)
    print('Peak parameters:')
    print(peak_dict)
    print('Harmonic SNR of')
    for key in harm_dict.keys():
        print(f'rho_{key}: {np.abs(peak_SNRs[key]):.2f}, phi_{key}: {np.angle(peak_SNRs[key]):.2f}')

    # Save information about peak
    peak_info = {}
    peak_info['base_params'] = base_dict
    peak_info['fid_params'] = fid_dict
    peak_info['peak_params'] = peak_dict
    peak_info['peak_harm_dict'] = harm_dict
    peak_info['peak_SNRs'] = peak_SNRs
    peak_info['f_low'] = f_low
    peak_info['n_ecc_harms'] = n_ecc_harms
    peak_info['n_ecc_gen'] = n_ecc_gen
    peak_info['sample_rate'] = sample_rate
    with open('peak_info', 'wb') as fp:
        pickle.dump(peak_info, fp)

    # Create result object, generate metric, and samples
    start = time.time()
    pe_result = create_pe_result(peak_dict, peak_SNRs, f_low, psd, ifos)
    npts = int(1e7)
    _ = pe_result.generate_samples_from_metric(npts=npts)
    pe_result.reweight_samples(ecc_reweight_fn,
                               dx_directions=pe_result.metric.dx_directions)
    pe_result.reweight_samples(pe.component_mass_prior_weight,
                               dx_directions=pe_result.metric.dx_directions)
    end=time.time()
    print(f'Generated new metric and samples in {end-start} seconds.')
    print(f'Enforcing prior reduces samples from {npts} to {len(pe_result.samples)}.')

    # Create interpolation grid and perform rejection sampling
    start = time.time()
    ecc_SNR_grid = calc_ecc_SNR_grid(pe_result, peak_dict, harm_dict, f_low, psd['harm'], two_ecc_harms=two_ecc_harms, ncpus=cpu_count())
    final_samples, ecc_SNR_samples, ecc_SNR_weights = interpolate_ecc_SNR_samples(pe_result, ecc_SNR_grid,
                                                                                  peak_SNRs, two_ecc_harms=True)
    end = time.time()
    print(f'Calculated interpolation grid and performed rejection sampling in {end-start} seconds.')

    # Save final samples and related information
    metric_prior_samples = pe.SimplePESamples(pe_result.samples_dict)
    metric_prior_samples.generate_ecc()
    samples_info = {'metric_prior_samples': metric_prior_samples, 'final_samples': final_samples,
                    'ecc_SNR_grid': ecc_SNR_grid, 'ecc_SNR_samples': ecc_SNR_samples, 'ecc_SNR_weights': ecc_SNR_weights}
    with open('samples_info', 'wb') as fp:
        pickle.dump(samples_info, fp)
    quantiles = np.quantile(final_samples['ecc10'], [0.05, 0.5, 0.95])
    print(f'Mean eccentricity found to be {quantiles[1]:.3f} with 90% credible interval of {quantiles[0]:.3f} - {quantiles[2]:.3f}.')
    print(f'{len(final_samples.samples[0])} samples remaining after rejection sampling.')

    # Setup plots
    if make_plots:
        corner_params = ['chirp_mass', 'chi_eff', 'symmetric_mass_ratio', 'ecc10']
        if truth_dict is not None:
            if 'ecc10' not in truth_dict.keys():
                truth_dict['ecc10'] = truth_dict['ecc10sqrd']**0.5
            truth_vals = [truth_dict[key] for key in corner_params]
        else:
            truth_vals = None

        # Make subplots of weights for rejection sampling
        plt.figure(figsize=(6.4*2.5, 4.8*2))
        subplot_count = 0
        for i in range(len(corner_params)):
            for j in range(i+1, len(corner_params)):
                subplot_count += 1
                plt.subplot(2, 3, subplot_count)
                plt.scatter(metric_prior_samples[corner_params[i]], metric_prior_samples[corner_params[j]], s=0.1,
                            c=ecc_SNR_weights, cmap='viridis', vmin=0, vmax=1)
                plt.xlabel(corner_params[i])
                plt.ylabel(corner_params[j])
                if truth_dict is not None:
                    plt.axvline(truth_dict[corner_params[i]], color='C3')
                    plt.axhline(truth_dict[corner_params[j]], color='C3')
        plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
        cax = plt.axes([0.95, 0.1, 0.025, 0.8])
        plt.colorbar(cax=cax, label='weights')
        plt.savefig('ecc_SNR_weights.png', dpi=450)

        # Make and save 4d corner plot
        final_samples.plot(parameters=corner_params, truths=truth_vals, type="corner",
                           quantiles=[0.05, 0.5, 0.95], show_titles=True, truth_color='C3')
        plt.savefig('final_samples.png', dpi=450)

if __name__ == "__main__":

    # Disable pesummary warnings
    _logger = logging.getLogger('PESummary')
    _logger.setLevel(logging.CRITICAL + 10)

    # Otherwise plt.savefig crashes with latex errors
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Data settings
    true_dict = {'ecc10sqrd': 0.15**2, 'chirp_mass': 40, 'symmetric_mass_ratio': 0.25, 'chi_eff': 0.3}
    init_guess = {'ecc10sqrd': 0, 'chirp_mass': 43, 'symmetric_mass_ratio': 0.10, 'chi_eff': 0.1}
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
    true_dict['distance'] = 1
    data = make_waveform(true_dict.copy(), psd['harm'].delta_f, f_low, len(psd['harm']), approximant='TEOBResumS-Dali')
    raw_snr = sigma(data, psd['H1'], low_frequency_cutoff=f_low, high_frequency_cutoff=psds['f_high'])
    data = {'H1': data.to_timeseries()*target_snr/raw_snr}
    true_dict['distance'] = raw_snr/target_snr
    init_guess['distance'] = true_dict['distance']

    # Run pipeline
    pipeline(data, init_guess, t_bounds, 8.76*10**-5, 4, 6, f_low, psd, ifos,
             uniform_e_prior_weight, truth_dict=true_dict, sample_rate=sample_rate, make_plots=True)
