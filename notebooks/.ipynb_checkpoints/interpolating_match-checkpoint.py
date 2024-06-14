import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.optimize import curve_fit, minimize

def find_min_max(data):
    """
    Finds minimum and maximum match of h0, h1, h1/h0 across varying mean anomaly.

    Parameters:
        data: Dictionary containing matches.

    Returns:
        data: Dictionary containing matches with min/max matches added.
    """

    # Loop over each chirp mass grid
    for chirp in data.keys():

        # Loop over h0, h1, h1/h0
        for key in list(data[chirp].keys()):
            
            # Find min/max vals and add to grid
            if key[0] == 'h' or key=='quad':
                data[chirp][f'{key}_max'] = np.max(np.array(data[chirp][key]), axis=1)
                data[chirp][f'{key}_min'] = np.min(np.array(data[chirp][key]), axis=1)

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

def fid_e_2_zero_ecc_chirp(fid_e, scaling_norms=[10, 0.03]):
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

def zero_ecc_chirp_2_fid_e(zero_ecc_chirp, scaling_norms=[10, 0.03]):
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
    
    # Loop over each chirp mass grid to get all max/min match values
    for chirp in data.keys():

        # Find normalisation in both directions
        fid_e = data[chirp]['fid_e']
        ecc_vals = data[chirp]['e_vals']/fid_e
        max_1d_interp = interpolate.interp1d(ecc_vals, data[chirp][f'{key}_max'])
        min_1d_interp = interpolate.interp1d(ecc_vals, data[chirp][f'{key}_max'])
        match_norm = (max_1d_interp(1)+min_1d_interp(1))/2
        max_vals = data[chirp][f'{key}_max']/match_norm
        min_vals = data[chirp][f'{key}_min']/match_norm
        
        # Add to interpolation data points
        max_vals_arr += max_vals
        min_vals_arr += min_vals
        ecc_vals_arr += ecc_vals
        fid_e_vals = [fid_e]*len(e_vals)
        fid_e_vals_arr += fid_e_vals
    
    # Create max/min interpolation objects
    max_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), max_vals_arr)
    min_interp = LinearNDInterpolator(list(zip(fid_e_vals_arr, ecc_vals_arr)), min_vals_arr)

    return max_interp, min_interp

def find_ecc_range(match, chirp, interps, slope='increasing', max_ecc=0.2):
    """
    Find range of eccentricities corresponding to match value.

    Parameters:
        match: Match value.
        chirp: Chirp mass at e_10=0.1.
        interps: Interpolation objects to use.
        slope: Slope direction of match against ecc along degeneracy line.
        max_ecc: Maximum value of eccentricity used to create interpolation objects.

    Returns:
        min_ecc: Minimum eccentricity.
        max_ecc: Maximum eccentricity.
    """

    # Check whether in range of each interp, deal with edge cases
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    line_eccs = [0,0]
    slope_dict = {'increasing': 0, 'decreasing': 1, None: 2}
    slope_ind = slope_dict[slope]
    for i in range(2):
        interp_arr = interps[i](chirp, ecc_range)
        edge_cases = [[[ecc_range[np.argmax(interp_arr)],0,None],[ecc_range[np.argmax(interp_arr)],1,None],[None, None, None]],
                      [[1,ecc_range[np.argmin(interp_arr)],None],[0,ecc_range[np.argmin(interp_arr)],None],[None, None, None]]]
        if match > np.max(interp_arr):
            range_ind = 0
        elif match < np.min(interp_arr):
            range_ind = 1
        else:
            range_ind = 2
        line_eccs[i] = edge_cases[i][slope_ind][range_ind]
        
    # Find eccentricities corresponding to max and min lines
    for i in range(2):
        if line_eccs[i] is None:
            max_result = minimize(lambda x: abs(interps[i](chirp, x) - match), max_ecc/2, bounds=[(0, max_ecc)], method='Powell')
            line_eccs[i] = max_result['x'][0]
    
    # Identify low and high of eccentricity range
    max_ecc = np.max(line_eccs)
    min_ecc = np.min(line_eccs)

    return min_ecc, max_ecc

def find_ecc_range_samples(matches, chirp, interps, max_ecc=0.2):
    """
    Find range of eccentricities corresponding to match values of samples. Assumes
    slope is increasing.

    Parameters:
        matches: Match values.
        chirp: Chirp mass at e_10=0.1.
        interps: Interpolation objects to use.
        max_ecc: Maximum value of eccentricity used to create interpolation objects.

    Returns:
        min_ecc: Minimum eccentricity.
        max_ecc: Maximum eccentricity.
    """

    # Ensure matches is numpy array
    matches = np.array(matches)

    # Create reverse interpolation object to get the eccentricity from match value
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    max_interp_arr = interps[0](chirp, ecc_range)
    min_interp_arr = interps[1](chirp, ecc_range)
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