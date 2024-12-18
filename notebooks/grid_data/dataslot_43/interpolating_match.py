import itertools
import time
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.optimize import curve_fit, minimize
from scipy.stats import ncx2, sampling
from pycbc.filter import match, optimized_match, sigma
from calcwf import chirp2total, chirp_degeneracy_line, gen_wf, shifted_f, shifted_e, gen_psd, resize_wfs, get_h
from simple_pe.waveforms import calculate_mode_snr, network_mode_snr

def estimate_coeffs(rhos, ovlps, ovlps_perp):
    """
    Estimate coefficients of harmonics in data from match filter SNR and overlaps
    between harmonics.

    Parameters:
        rhos: Match filter SNR of each harmonic.
        ovlps: Overlaps of unorthogonalised harmonics.
        ovlps_perp: Overlaps of orthogonalised harmonics with themselves.

    Returns:
        est_coeffs: Coefficient estimates.
    """  
    n = len(rhos)
    adjust = {}
    est_coeffs = {}
    for i in range(n-1, -1, -1):
        adjust[i] = 0
        for j in range(1,n-i):
            for comb in itertools.combinations(np.arange(i+1,n), j):
                comb = [i] + list(comb)
                prod = est_coeffs[comb[-1]]
                for k in range(1,len(comb)):
                    prod *= ovlps[comb[k]][comb[k-1]]
                adjust[i] += prod
        est_coeffs[i] = np.conj(rhos[i])/ovlps_perp[i] - adjust[i]

    return est_coeffs

def comb_log_L(params, A_primes, phi_primes, harms):
    """
    Calculate log likelihood of a set of harmonics in a phase consistent way.

    Parameters:
        params: Free parameters describing estimated amplitudes and phases of matches.
        A_primes: Magnitudes of matches with each harmonic.
        phi_primes: Phases of matches with each harmonic.
        harms: Which harmonics are included.

    Returns:
        tot: Total SNR squared.
    """

    tot = 0
    As = params[:-2]
    alpha, beta = params[-2:]

    # Add each harmonic in turn
    for i in range(len(harms)):
        tot += A_primes[i]*As[i]*np.cos(alpha+harms[i]*beta-phi_primes[i]) - 0.5*As[i]**2

    return tot

def comb_harm_consistent(A_primes, phi_primes, harms=[0,1,-1], return_denom=False):
    """
    Combine match of higher harmonics in phase consistent way for 
    a single point.

    Parameters:
        A_primes: Magnitudes of matches with each harmonic.
        phi_primes: Phases of matches with each harmonic.
        harms: Which harmonics to include.
        return_denom: Whether to return the denominator of the fraction.

    Returns:
        frac: Combined match relative to h0.
    """

    # Add fundamental to harmonic list
    if 0 not in harms:
        harms.insert(0,0)

    # Maximise total SNR
    bounds = [(0, None)]*len(harms) + [(-np.pi, np.pi), (-np.pi, np.pi)]
    init_guess = list(A_primes) + [phi_primes[harms.index(0)], phi_primes[harms.index(1)]-phi_primes[harms.index(0)]]
    init_guess[-1] = (init_guess[-1]+np.pi)%(2*np.pi) - np.pi
    best_fit = minimize(lambda x: -comb_log_L(x, A_primes, phi_primes, harms), init_guess, bounds=bounds)

    # Compute combined SNR of higher harmonics
    As = best_fit['x'][:-2]
    alpha, beta = best_fit['x'][-2:]
    num_sqrd = 0
    for i in range(len(harms)):
        if harms[i] == 0:
            #denom_sqrd = A_primes[i]*As[i]*np.cos(alpha+phi_primes[i])
            denom_sqrd = As[i]**2
            continue
        #num_sqrd += A_primes[i]*As[i]*np.cos(alpha+harms[i]*beta-phi_primes[i])
        num_sqrd += As[i]**2
    frac = np.sqrt(num_sqrd/denom_sqrd)

    # Returns denominator if requested
    if return_denom:
        return frac, np.sqrt(denom_sqrd)
    else:
        return frac


def comb_harm_consistent_grid(data, harms=[0,1,-1]):
    """
    Combine match of higher harmonics in phase consistent way for 
    grid of points.

    Parameters:
        data: Dictionary containing matches for given chirp mass.
        harms: Which harmonics to include.

    Returns:
        fracs: Combined match relative to h0.
    """

    # Add fundamental to harmonic list
    if 0 not in harms:
        harms.insert(0,0)

    # Get all magnitudes and phases of matches
    all_A_primes = []
    all_phi_primes = []
    for harm in harms:
        all_A_primes.append(data[f'h{harm}'])
        all_phi_primes.append(data[f'h{harm}_phase'])
    all_A_primes = np.rollaxis(np.array(all_A_primes),0,3)
    all_phi_primes = np.rollaxis(np.array(all_phi_primes),0,3)

    # Find num for each grid point
    fracs = np.zeros(np.shape(all_A_primes)[:2])
    for iy, ix in np.ndindex(np.shape(fracs)):
        fracs[iy, ix] = comb_harm_consistent(all_A_primes[iy][ix], all_phi_primes[iy][ix], harms)

    return fracs

def find_min_max(data, extra_keys=['h1_h0', 'h-1_h0', 'h2_h0', 'h1_h-1_h0', 'h1_h-1_h0_pca'], ovlps=None):
    """
    Finds minimum and maximum match of various match quantities across varying mean anomaly.

    Parameters:
        data: Dictionary containing matches.
        extra_keys: Extra match-related quantities to compute.
        ovlps: Optionally use overlaps between harmonics to improve SNR estimate.

    Returns:
        data: Dictionary containing matches with min/max matches added.
    """

    # Loop over each chirp mass grid
    for chirp in data.keys():

        # Calculate extra keys if not already present
        for key in extra_keys:
            if key not in list(data[chirp].keys()):
                if key == 'h1_h0':
                    data[chirp]['h1_h0'] = data[chirp]['h1']/data[chirp]['h0']
                elif key == 'h-1_h0':
                    data[chirp]['h-1_h0'] = data[chirp]['h-1']/data[chirp]['h0']
                elif key == 'h2_h0':
                    data[chirp]['h2_h0'] = data[chirp]['h2']/data[chirp]['h0']
                elif key == 'h1_h-1_h0':
                    num = np.sqrt(data[chirp]['h1']**2+data[chirp]['h-1']**2)
                    data[chirp]['h1_h-1_h0'] = num/data[chirp]['h0']
                elif key == 'h1_h-1_h2_h0':
                    num = np.sqrt(data[chirp]['h1']**2+data[chirp]['h-1']**2+data[chirp]['h2']**2)
                    data[chirp]['h1_h-1_h2_h0'] = num/data[chirp]['h0']
                elif key == 'h1_h-1_h0_pca':
                    angle = 2*data[chirp]['h0_phase']-data[chirp]['h1_phase']-data[chirp]['h-1_phase']
                    num = np.sqrt(data[chirp]['h1']**2+np.cos(angle)*data[chirp]['h-1']**2)
                    data[chirp]['h1_h-1_h0_pca'] = num/data[chirp]['h0']
                elif key == 'h1_h-1_h0_pcn':
                    data[chirp]['h1_h-1_h0_pcn'] = comb_harm_consistent_grid(data[chirp], harms=[1,-1])
                elif key == 'h1_h-1_h2_h0_pcn':
                    data[chirp]['h1_h-1_h2_h0_pcn'] = comb_harm_consistent_grid(data[chirp], harms=[1,-1,2])
                
        # Calculate min and max of each key
        for key in list(data[chirp].keys()):
            
            # Find min/max vals and add to grid
            if key[0] == 'h' or key=='quad':
                data[chirp][f'{key}_max'] = np.nanmax(np.array(data[chirp][key]), axis=1)
                data[chirp][f'{key}_min'] = np.nanmin(np.array(data[chirp][key]), axis=1)

    return data

def create_min_max_interp(data, chirp, key):
    """
    Create interpolation objects which give the min and max ecc value for 
    a given match value on line of degeneracy.

    Parameters:
        data: Dictionary containing matches.
        chirp: Chirp mass to calculate chirp mass for
        param_vals: Array of eccentricity values used to create data.

    Returns:
        max_interp, min_interp: Created interpolation objects.
    """

    max_match_arr = data[chirp][f'{key}_max']
    min_match_arr = data[chirp][f'{key}_min']
    e_vals = data[chirp]['e_vals']

    max_interp = interp1d(e_vals, max_match_arr, bounds_error=False)
    min_interp = interp1d(e_vals, min_match_arr, bounds_error=False)

    return max_interp, min_interp

def fid_e2zero_ecc_chirp(fid_e, scaling_norms=[10, 0.035]):
    """
    Convert a fiducial eccentricity to corresponding non-eccentric chirp
    mass.

    Parameters:
        fid_e: Fiducial eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        zero_ecc_chirp: Non-eccentric chirp mass.
    """
    
    zero_ecc_chirp = fid_e**(6/5)*scaling_norms[0]/(scaling_norms[1]**(6/5))
    
    return zero_ecc_chirp

def zero_ecc_chirp2fid_e(zero_ecc_chirp, scaling_norms=[10, 0.035]):
    """
    Convert a non-eccentric chirp mass to a corresponding fiducial eccentricity.

    Parameters:
        zero_ecc_chirp: Non-eccentric chirp mass.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        fid_e: Fiducial eccentricity.
    """
    
    fid_e = zero_ecc_chirp**(5/6)*scaling_norms[1]/(scaling_norms[0]**(5/6))
    
    return fid_e

def scaled_2D_interps(data, key):
    """
    Create interpolation objects which give the min and max match value at 
    arbitrary chirp mass and point in parameter space on line of degeneracy.
    These are normalised to account for different fiducial eccentricities.

    Parameters:
        data: Dictionary containing matches.
        key: Key of dictionary (e.g. h1_h0) to calculate interpolation object for.

    Returns:
        max_interp, min_interp: Created interpolation objects.
    """

    max_vals_arr = []
    min_vals_arr = []
    ecc_vals_arr = []
    fid_e_vals_arr = []

    common_e_vals = np.arange(0, 1, 0.001)
    
    # Loop over each chirp mass grid to get all max/min match values
    for chirp in data.keys():

        # Interpolate to standard e_val array
        max_interp = interp1d(data[chirp]['e_vals'], data[chirp][f'{key}_max'], bounds_error=False)
        min_interp = interp1d(data[chirp]['e_vals'], data[chirp][f'{key}_min'], bounds_error=False)
        max_vals = max_interp(common_e_vals)
        min_vals = min_interp(common_e_vals)
        non_nan_inds = np.array(1 - np.isnan(max_vals+min_vals), dtype='bool')
        
        # Normalise in both directions
        fid_e = data[chirp]['fid_params']['e']
        ecc_vals = common_e_vals/fid_e
        max_vals = max_vals/chirp
        min_vals = min_vals/chirp
        
        # Add non-nan vals to interpolation data points
        max_vals_arr += list(max_vals[non_nan_inds])
        min_vals_arr += list(min_vals[non_nan_inds])
        ecc_vals_arr += list(ecc_vals[non_nan_inds])
        fid_e_vals = [fid_e]*np.sum(non_nan_inds)
        fid_e_vals_arr += fid_e_vals
    
    # Create max/min interpolation objects
    max_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), max_vals_arr)
    min_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), min_vals_arr)

    return max_interp, min_interp

def find_ecc_range_samples(matches, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035]):
    """
    Find range of eccentricities corresponding to match values of samples. Assumes
    slope is increasing.

    Parameters:
        matches: Match values.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        ecc_arr: Minimum and maximum eccentricities for each sample.
    """

    # Ensure matches is numpy array
    matches = np.array(matches)

    # Handle either min/max lines or interpolating to chirp mass
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    if len(interps) == 1:
        fid_e = zero_ecc_chirp2fid_e(chirp, scaling_norms=scaling_norms)
        max_interp_arr = interps[0](fid_e, ecc_range/fid_e)*chirp
        min_interp_arr = interps[1](fid_e, ecc_range/fid_e)*chirp   
    else:
        max_interp, min_interp = interps
        max_interp_arr = max_interp(ecc_range)
        min_interp_arr = min_interp(ecc_range)

    # Create reverse interpolation object to get the eccentricity from match value
    max_nans = np.sum(np.isnan(max_interp_arr))
    min_nans = np.sum(np.isnan(min_interp_arr))
    if np.max([max_nans, min_nans]) > 0:
        max_interp_arr = max_interp_arr[:-np.max([max_nans, min_nans])]
        min_interp_arr = min_interp_arr[:-np.max([max_nans, min_nans])]
    max_interp = interp1d(max_interp_arr, ecc_range)
    min_interp = interp1d(min_interp_arr, ecc_range)

    # Check whether in range of each interp, deal with edge cases
    ecc_arr = np.array([np.full_like(matches, 5)]*2)
    ecc_arr[0][matches<np.min(max_interp_arr)] = 0
    ecc_arr[0][matches>np.max(max_interp_arr)] = ecc_range[np.argmax(max_interp_arr)]
    ecc_arr[1][matches<np.min(min_interp_arr)] = ecc_range[np.argmin(min_interp_arr)]
    ecc_arr[1][matches>np.max(min_interp_arr)] = 1
    
    # Find eccentricities corresponding to max and min lines
    ecc_arr[0][ecc_arr[0]==5] = max_interp(matches[ecc_arr[0]==5])
    ecc_arr[1][ecc_arr[1]==5] = min_interp(matches[ecc_arr[1]==5])

    return ecc_arr

def dist_CI(rv, x, CI=0.9):
    """
    Find 90% confidence bounds (in SNR^2 space) with x% cutoff from lower end 
    of distribution.

    Parameters:
        rv: Random variable distribution.
        x: Percentage cutoff from lower end of distribution.
        CI: Confidence interval.

    Returns:
        CI_bounds: Confidence interval bounds (in SNR**2 space).
    """
    q = np.array([x, x+CI])
    CI_bounds = rv.ppf(q)
    return CI_bounds

def dist_min_CI(rv, CI=0.9):
    """
    Find 90% confidence bounds (in SNR^2 space) with shortest possible distance (in SNR**2 space).

    Parameters:
        rv: Random variable distribution.
        CI: Confidence interval.

    Returns:
        CI_bounds: Confidence interval bounds.
    """
    min_result = minimize(lambda x: abs(np.diff(dist_CI(rv, x[0], CI=CI))[0]), 0.05, bounds=[(0,0.1)])
    min_x = min_result['x'][0]
    return np.sqrt(dist_CI(rv, min_x, CI=CI))

def find_ecc_CI(CI_bounds, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035]):
    """
    Maps confidence intervals in match space to eccentricity space.

    Parameters:
        CI_bounds: Confidence interval in match space.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        min_ecc, max_ecc: Confidence interval bounds on eccentricity.
    """

    # Find where CI matches cross min and max lines
    min_CI_eccs, max_CI_eccs = find_ecc_range_samples(CI_bounds, chirp, interps, max_ecc, scaling_norms=scaling_norms)

    # Find minimum and maximum eccentricity
    min_ecc = np.min([min_CI_eccs, max_CI_eccs])
    max_ecc = np.max([min_CI_eccs, max_CI_eccs])

    return min_ecc, max_ecc

def SNR_samples(obs_SNR, n):
    """
    Generates SNR samples.

    Parameters:
        obs_SNR: Observed SNR.
        n: Number of samples to generate.

    Returns:
        samples: SNR samples.
    """

    # Define distribution class
    class SNR_rv():
        def pdf(self, x):
            return ncx2.pdf(x**2, 2, obs_SNR**2)
        def cdf(self, x):
            return ncx2.cdf(x**2, 2, obs_SNR**2)

    # Generate samples
    rv = SNR_rv()
    sample_gen = sampling.NumericalInversePolynomial(rv, center=obs_SNR, domain=(0.000001, np.inf))
    samples = sample_gen.rvs(size=n)
    return samples

def SNR2ecc(matches, chirp, interps, max_ecc=0.4, scaling_norms=[10, 0.035], upper_lenience=0):
    """
    Maps SNR samples to eccentricity samples.

    Parameters:
        matches: SNR samples.
        chirp: Chirp mass at zero eccentricity.
        interps: Interpolation objects used to interpolate to the min/max lines of the desired chirp mass.
        max_ecc: Maximum value of eccentricity.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.
        upper_lenience: Allow upper bound of eccentricity samples to be higher than max_ecc.

    Returns:
        eccs: Eccentricity samples.
    """

    # Find upper and lower bounds on eccentricity for each sample
    ecc_arr = find_ecc_range_samples(matches, chirp, interps, max_ecc=max_ecc)

    # Put upper bound at max_ecc
    max_ecc *= 1+upper_lenience
    inds = ecc_arr>max_ecc
    ecc_arr[inds] = max_ecc

    # Uniformly draw random value between these bounds for each sample
    eccs = np.random.rand(len(matches))*(ecc_arr[1]-ecc_arr[0]) + ecc_arr[0]

    return eccs

def gen_zero_noise_data(zero_ecc_chirp, fid_e, ecc, f_low, f_match, MA_shift, total_SNR, ifos):
    """
    Generates zero noise data and psds.

    Parameters:
        zero_ecc_chirp: Chirp mass at zero eccentricity.
        fid_e: Fiducial eccentricity.
        ecc: Eccentricity of data.
        f_low: Waveform starting frequency.
        f_match: Low frequency cutoff to use.
        MA_shift: Anomaly.
        total_SNR: SNR of data.
        ifos: Detectors to use.

    Returns:
        data: Zero noise data.
        psds: PSDs.
        t_start: Data start time.
        t_end = Data end time.
        fid_chirp: Fiducial chirp mass.
    """
    
    # Get other required chirp masses along degeneracy line
    fid_chirp, chirp_mass = chirp_degeneracy_line(zero_ecc_chirp, np.array([fid_e, ecc]))
    
    # Calculate distance for specified SNR
    s_d_test = gen_wf(f_low, ecc, chirp2total(chirp_mass, 2), 2, 4096, 'TEOBResumS', distance=1)
    psd_d_test = gen_psd(s_d_test, f_low)
    s_d_test_sigma = sigma(s_d_test.real(), psd_d_test, low_frequency_cutoff=f_match, high_frequency_cutoff=psd_d_test.sample_frequencies[-1])
    distance = np.sqrt(len(ifos))*s_d_test_sigma/total_SNR
    
    # Calculate strain data (teobresums waveform) and psd
    s_f_2pi = f_low - shifted_f(f_low, ecc, chirp2total(chirp_mass, 2), 2)
    s_f = f_low - (MA_shift*s_f_2pi)
    s_e = shifted_e(s_f, f_low, ecc)
    s_teob = gen_wf(s_f, s_e, chirp2total(chirp_mass, 2), 2, 4096, 'TEOBResumS', distance=distance)
    fid_wf_len = gen_wf(f_low, fid_e, chirp2total(fid_chirp, 2), 2, 4096, 'TEOBResumS', distance=distance)
    _, s_teob = resize_wfs([fid_wf_len, s_teob])
    psd = gen_psd(s_teob, f_low)
    s_teob_f = s_teob.real().to_frequencyseries()
    
    # Creates objects used in SNR functions
    data = {'H1': s_teob_f, 'L1': s_teob_f}
    psds = {'H1': psd, 'L1': psd}
    t_start = s_teob.sample_times[0]
    t_end = s_teob.sample_times[-1]

    return data, psds, t_start, t_end, fid_chirp

def gen_ecc_samples(data, psds, t_start, t_end, fid_chirp, interps, max_ecc, n, zero_ecc_chirp, fid_e, f_low, f_match, match_key, ifos, verbose=False):
    """
    Generates samples on SNR and eccentricity.

    Parameters:
        data: Zero noise data.
        psds: PSDs.
        t_start: Data start time.
        t_end = Data end time.
        fid_chirp: Fiducial chirp mass.
        interps: Interpolation objects of min/max lines.
        max_ecc: Maximum eccentricity.
        n: Number of harmonics to use.
        zero_ecc_chirp: Chirp mass at zero eccentricity.
        fid_e: Fiducial eccentricity.
        f_low: Waveform starting frequency.
        f_match: Low frequency cutoff to use.
        match_key: Which harmonics to use in min/max line.
        ifos: Detectors to use.
        verbose: Whether to print out information.

    Returns:
        match_samples, ecc_samples: Samples on SNR and eccentricity.
        
    """

    # Generates fiducial waveforms in frequency domain
    start = time.time()
    all_wfs = list(get_h([1]*n, f_low, fid_e, chirp2total(fid_chirp, 2), 2, 4096))
    h0, h1, hn1, h2 = all_wfs[1:5]
    h0_f, h1_f, hn1_f, h2_f = [wf.real().to_frequencyseries() for wf in [h0, h1, hn1, h2]]
    h = {'h0': h0_f, 'h1': h1_f, 'h-1': hn1_f, 'h2': h2_f}
    
    # Loop over detectors
    z = {}
    for ifo in ifos:
    
        # Normalise waveform modes
        h_perp = {}
        for key in h.keys():
            h_perp[key] = h[key] / sigma(h[key], psds[ifo], low_frequency_cutoff=f_match, high_frequency_cutoff=psds[ifo].sample_frequencies[-1])
        
        # Calculate mode SNRs
        mode_SNRs, _ = calculate_mode_snr(data[ifo], psds[ifo], h_perp, t_start, t_end, f_match, h_perp.keys(), dominant_mode='h0')
        z[ifo] = mode_SNRs
    
    # Calculate network SNRs
    rss_snr, _ = network_mode_snr(z, ifos, z[ifos[0]].keys(), dominant_mode='h0')
    if verbose:
        for mode in rss_snr:
            print(f'rho_{mode[1:]} = ' + str(rss_snr[mode]))
    
    # Draw SNR samples and convert to eccentricity samples
    num_sqrd = 0
    for mode in rss_snr.keys():
        if mode != 'h0' and mode in match_key:
            num_sqrd += rss_snr[mode]**2
    match_samples = SNR_samples(np.sqrt(num_sqrd), n=10**6)/rss_snr['h0']
    ecc_samples = SNR2ecc(match_samples, zero_ecc_chirp, interps, max_ecc=max_ecc, scaling_norms=[fid_chirp, fid_e], upper_lenience=0.05)
    
    # Compute 90% confidence bounds on SNR
    rv = ncx2(2, num_sqrd)
    h1_CI_bounds = dist_min_CI(rv)
    h1_h0_CI_bounds = h1_CI_bounds/rss_snr['h0']
    
    # Compute 90% eccentric CI
    ecc_CI_bounds = find_ecc_CI(h1_h0_CI_bounds, zero_ecc_chirp, interps, max_ecc=max_ecc, scaling_norms=[fid_chirp, fid_e])
    
    # Output time taken
    end = time.time()
    if verbose:
        print(f'Eccentricity range of approximately {ecc_CI_bounds[0]:.3f} to {ecc_CI_bounds[1]:.3f} computed in {end-start:.3f} seconds.')

    return match_samples, ecc_samples