import time
import numpy as np
from scipy.optimize import minimize
from pycbc import psd as psd_func
from pycbc.filter import sigma, matched_filter
from pycbc.filter.matchedfilter import quadratic_interpolate_peak
from pesummary.gw.conversions.mass import component_masses_from_mchirp_q, q_from_eta
from simple_pe.waveforms import make_waveform, check_physical
from simple_pe.param_est import find_peak_snr, find_metric_and_eigendirections

def test_ecc_point(dist, evec_dict, params, data, approximant, f_low, psd, two_ecc_harms=True):

    # Create parameters
    test_params = params.copy()
    for key in evec_dict.keys():
        test_params[key] += dist*evec_dict[key]

    # Create  trial waveform
    trial_wf = make_waveform(test_params.copy(), psd.delta_f, f_low, len(psd), approximant=approximant)
    data_snr = matched_filter(trial_wf, data, psd, low_frequency_cutoff=f_low,
                              high_frequency_cutoff=psd.sample_frequencies[-1])
    maxsnr, max_id = data_snr.abs_max_loc()
    left = abs(data_snr[-1]) if max_id == 0 else abs(data_snr[max_id - 1])
    right = abs(data_snr[0]) if max_id == (len(data_snr) - 1) else abs(data_snr[max_id + 1])
    _, maxsnr = quadratic_interpolate_peak(left, maxsnr, right)

    return -maxsnr

def refine_evecs(metric, peak_guess, approximant, data, f_low, psd):

    # Remove eccentricity from metric
    dx_dirs = metric.dx_directions.copy()

    # Find scaled eigenvectors
    evals, evecs = np.linalg.eig(metric.metric)
    norm_evecs = (evecs * np.sqrt(metric.mismatch/evals)).T
    evec_dicts = []
    for norm_evec in norm_evecs:
        evec_dict = {key: norm_evec[i] for i, key in enumerate(dx_dirs)}
        evec_dicts.append(evec_dict)

    # Optimise each eigenvector at a time
    evec_coords = []
    new_peak = peak_guess.copy()
    while len(evec_dicts) > 0:

        # Decide which eigenvector based on physical parameter bounds
        bounds = []
        for evec_dict in evec_dicts:
            alpha_pos = check_physical(new_peak, evec_dict, 1)
            alpha_neg = -check_physical(new_peak, evec_dict, -1)
            bounds.append((alpha_neg, alpha_pos))
        bound_sum = -np.sum(np.abs(np.array(bounds)), axis=1)
        bound_inds = np.argsort(bound_sum)
        evec_dicts = np.array(evec_dicts)[bound_inds]

        # Perform optimisation
        x0 = 0
        bounds = [bounds[bound_inds[0]]]
        best_result = minimize(test_ecc_point, x0,
                               args=(evec_dicts[0], peak_guess, data['H1'], approximant, f_low, psd['H1']),
                               bounds=bounds, method='Powell')
        evec_coords.append(best_result['x'][0])

        # Update peak
        for key in evec_dicts[0]:
            new_peak[key] += evec_coords[-1]*evec_dicts[0][key]
        evec_dicts = np.delete(evec_dicts, 0, axis=0)

    return new_peak, -best_result['fun']

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

# Function to test each approximant
def test_approximant(approximant, base_dict, true_dict, data, psd, f_low, snr):

    # Perform refinement of non-eccentric peak
    start = time.time()
    par_dirs = ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    metric = find_metric_and_eigendirections(base_dict.copy(), par_dirs, snr=snr, f_low=f_low, psd=psd['harm'],
                                             approximant=approximant, max_iter=0, multiprocessing=True)
    new_params, snr = refine_evecs(metric, base_dict, approximant, data, f_low, psd)
    end = time.time()

    # Test how good new peak is
    par_dirs = ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff', 'ecc10sqrd']
    metric_2 = find_metric_and_eigendirections(base_dict.copy(), par_dirs, snr=snr, f_low=f_low, psd=psd['harm'],
                                               approximant='TEOBResumS-Dali-Harms', max_iter=0, multiprocessing=True)
    fid_point = const_mm_point(metric_2, 8.76*10**-5, 'ecc10sqrd', new_params)
    degen_dist = (true_dict['ecc10sqrd']-new_params['ecc10sqrd'])/(fid_point['ecc10sqrd'] - new_params['ecc10sqrd'])
    ecc_point = {}
    for key in par_dirs:
        ecc_point[key] = new_params[key] + degen_dist * (fid_point[key] - new_params[key])

    # Output results
    print('')
    print('--------------------------------------------------------------------------')
    print('')
    print(f'Approximant: {approximant}')
    print(f'Found non-ecc peak in {end-start:.2f} seconds with SNR: {snr:.4f}')
    print(f'Non-ecc parameters: {new_params}')
    print(f'Ecc parameters: {ecc_point}')
    print(f'Percentage errors:')
    for key in ['ecc10sqrd', 'chirp_mass', 'symmetric_mass_ratio', 'chi_eff']:
        if key == 'chi_eff':
            denom = 1
        else:
            denom = true_dict[key]
        print(f'{key}: {(ecc_point[key]-true_dict[key])*100/denom:.3g}%')

def run_code():

    # Disable pesummary warnings
    import logging
    _logger = logging.getLogger('PESummary')
    _logger.setLevel(logging.CRITICAL + 10)

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
    true_dict['distance'] = 1
    data = make_waveform(true_dict.copy(), psd['harm'].delta_f, f_low, len(psd['harm']), approximant='TEOBResumS-Dali')
    raw_snr = sigma(data, psd['H1'], low_frequency_cutoff=f_low, high_frequency_cutoff=psds['f_high'])
    data = {'H1': data.to_timeseries()*target_snr/raw_snr}
    true_dict['distance'] = raw_snr/target_snr
    init_guess['distance'] = true_dict['distance']

    # Find initial non-ecc guess
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

    # Calculate first metric to get rid of multiprocessing overhead
    start = time.time()
    par_dirs = ['chirp_mass', 'symmetric_mass_ratio', 'chi_eff']
    metric = find_metric_and_eigendirections(base_dict.copy(), par_dirs, snr=snr, f_low=f_low, psd=psd['harm'],
                                             approximant='IMRPhenomXPHM', max_iter=0, multiprocessing=True)
    end = time.time()
    print(f'Calculated control test metric in {end-start} seconds.')

    app_list = ['IMRPhenomXPHM', 'TEOBResumS', 'TEOBResumS-Dali', 'TEOBResumS-Dali-Harms']
    for approximant in app_list:
        test_approximant(approximant, base_dict, true_dict, data, psd, f_low, snr)

if __name__ == "__main__":
    run_code()


