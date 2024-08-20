import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.optimize import curve_fit, minimize
from pycbc.filter import match, optimized_match
from calcwf import gen_psd

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
            prod = 1
            ms = np.zeros(j+2, dtype='int')
            ms[0] = i
            ms[-1] = n
            for k in range(1, j+1):
                ovlp_sum = 0
                for l in range(ms[k-1]+1, n+k-j):
                    ms[k] = l
                    if k == j:
                        ovlp_sum += est_coeffs[l]*ovlps[l][ms[k-1]]
                    else:
                        ovlp_sum += ovlps[l][ms[k-1]]
                prod *= ovlp_sum
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

def create_2D_interps(data, param_vals=np.linspace(0, 0.2, 101)):
    """
    Create interpolation objects which give the min and max match value at 
    arbitrary chirp mass and point in parameter space on line of degeneracy.

    Parameters:
        data: Dictionary containing matches.
        param_vals: Array of eccentricity values used to create data.

    Returns:
        interp_objs: Created interpolation objects.
    """

    interp_objs = []

    # Get param vals from the data if available
    first_chirp = list(data.keys())[0]
    if 'e_vals' in data[first_chirp].keys():
        param_vals = data[first_chirp]['e_vals']
    
    # Loop over h0, h1, h1/h0:
    for key in ['h1', 'h2', 'h2_h1']:

        max_vals = []
        min_vals = []
        
        # Loop over each chirp mass grid to get all max/min match values
        for chirp in data.keys():
            max_vals.append(data[chirp][f'{key}_max'])
            min_vals.append(data[chirp][f'{key}_min'])
        max_vals = np.array(max_vals).flatten()
        min_vals = np.array(min_vals).flatten()

        # Get flat chirp mass, param values arrays
        ecc_vals, chirp_vals = np.meshgrid(param_vals, list(data.keys()))
        ecc_vals = ecc_vals.flatten()
        chirp_vals = chirp_vals.flatten()
        
        # Create max/min interpolation objects
        max_interp = LinearNDInterpolator(list(zip(chirp_vals, ecc_vals)), max_vals)
        min_interp = LinearNDInterpolator(list(zip(chirp_vals, ecc_vals)), min_vals)
        interp_objs.append([max_interp, min_interp])

    return interp_objs

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
        fid_e = data[chirp]['fid_e']
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

def find_ecc_range_samples(matches, chirp, interps, max_ecc, scaling_norms=[10, 0.035], return_max_ecc=False):
    """
    Find range of eccentricities corresponding to match values of samples. Assumes
    slope is increasing.

    Parameters:
        matches: Match values.
        chirp: Chirp mass at zero_eccentricity.
        interps: Interpolation objects to use.
        max_ecc: Maximum value of eccentricity for this chirp mass.
        scaling_norms: Non-eccentric chirp mass and fiducial eccentricity used 
        to normalise relationship.

    Returns:
        ecc_arr: Minimum and maximum eccentricities for each sample.
    """

    # Ensure matches is numpy array
    matches = np.array(matches)

    # Get corresponding fiducial eccentricity value
    fid_e = zero_ecc_chirp2fid_e(chirp, scaling_norms=scaling_norms)

    # Create reverse interpolation object to get the eccentricity from match value
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    max_interp_arr = interps[0](fid_e, ecc_range/fid_e)*chirp
    min_interp_arr = interps[1](fid_e, ecc_range/fid_e)*chirp
    max_nans = np.sum(np.isnan(max_interp_arr))
    min_nans = np.sum(np.isnan(min_interp_arr))
    if np.max([max_nans, min_nans]) > 0:
        max_interp_arr = max_interp_arr[:-np.max([max_nans, min_nans])]
        min_interp_arr = min_interp_arr[:-np.max([max_nans, min_nans])]
        ecc_range = ecc_range[:-np.max([max_nans, min_nans])]
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

    if return_max_ecc:
        return ecc_arr, ecc_range[-1]
    else:
        return ecc_arr