import EOBRun_module
import numpy as np
import math
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx, sigma
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries
from scipy.optimize import minimize
import matplotlib.pyplot as plt

## Conversions

def f_kep2avg(f_kep, e):
    """
    Converts Keplerian frequency to the average frequency quantity used by TEOBResumS.

    Parameters:
        f_kep: Keplerian frequency to be converted.
        e: eccentricity of signal.

    Returns:
        Average frequency.
    """

    numerator = (1+e**2)
    denominator = (1-e**2)**(3/2)

    return f_kep*(numerator/denominator)

def f_avg2kep(f_avg, e):
    """
    Converts average frequency quantity used by TEOBResumS to Keplerian frequency.

    Parameters:
        f_kep: Average frequency to be converted.
        e: eccentricity of signal.

    Returns:
        Keplerian frequency.
    """

    numerator = (1-e**2)**(3/2)
    denominator = (1+e**2)

    return f_avg*(numerator/denominator)

def chirp2total(chirp, q):
    """
    Converts chirp mass to total mass.

    Parameters:
        chirp: Chirp mass.
        q: Mass ratio.

    Returns:
        Total mass.
    """
    
    q_factor = q/(1+q)**2
    total = q_factor**(-3/5) * chirp

    return total

def total2chirp(total, q):
    """
    Converts total mass to chirp mass.

    Parameters:
        total: Total mass.
        q: Mass ratio.

    Returns:
        Chirp mass.
    """
    
    q_factor = q/(1+q)**2
    chirp = q_factor**(3/5) * total

    return chirp

## Generating waveform

def gen_e_td_wf(f_low, e, M, q, sample_rate, phase):
    """
    Generates EccentricTD waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.

    Returns:
        Plus and cross polarisation of EccentricTD waveform.
    """
    
    m2 = M / (1+q)
    m1 = M - m2
    e_td_p, e_td_c = get_td_waveform(approximant='EccentricTD',
                                     mass1=m1,
                                     mass2=m2,
                                     eccentricity=e,
                                     coa_phase=phase,
                                     delta_t=1.0/sample_rate,
                                     f_lower=f_low)
    return e_td_p, e_td_c

def modes_to_k(modes):
    """
    Converts list of modes to use into the 'k' parameter accepted by TEOBResumS.

    Parameters:
        modes: List of modes to use.

    Returns:
        'k' parameter of TEOBResumS.
    """
    
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def gen_teob_wf(f_kep, e, M, q, sample_rate, phase):
    """
    Generates TEOBResumS waveform with chosen parameters.

    Parameters:
        f_kep: Starting (Keplerian) frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.

    Returns:
        Plus and cross polarisation of TEOBResumS waveform.
    """

    # Gets average frequency quantity used by TEOBResumS
    f_avg = f_kep2avg(f_kep, e)

    # Define parameters
    k = modes_to_k([[2,2]])
    pars = {
            'M'                  : M,
            'q'                  : q,
            'Lambda1'            : 0.,
            'Lambda2'            : 0.,     
            'chi1'               : 0.,
            'chi2'               : 0.,
            'domain'             : 0,            # TD
            'arg_out'            : 0,            # Output hlm/hflm. Default = 0
            'use_mode_lm'        : k,            # List of modes to use/output through EOBRunPy
            'srate_interp'       : sample_rate,  # srate at which to interpolate. Default = 4096.
            'use_geometric_units': 0,            # Output quantities in geometric units. Default = 1
            'initial_frequency'  : f_avg,        # in Hz if use_geometric_units = 0, else in geometric units
            'interp_uniform_grid': 1,            # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'distance'           : 1,
            'coalescence_angle'  : phase,
            'inclination'        : 0,
            'ecc'                : e,
            'output_hpc'         : 0,
            }

    # Calculate waveform and convert to pycbc TimeSeries object
    t, teob_p, teob_c = EOBRun_module.EOBRunPy(pars)
    teob = teob_p - 1j*teob_c
    tmrg = t[np.argmax(np.abs(teob))]
    t = t - tmrg
    teob_p = timeseries.TimeSeries(teob_p, 1/sample_rate, epoch=t[0])
    teob_c = timeseries.TimeSeries(teob_c, 1/sample_rate, epoch=t[0])
    
    return teob_p, teob_c

def gen_wf(f_low, e, M, q, sample_rate, approximant, phase=0):
    """
    Generates waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        approximant: Approximant to use to generate the waveform.
        phase: Phase of signal.

    Returns:
        Complex combination of plus and cross waveform polarisations.
    """

    # Chooses specified approximant
    if approximant=='EccentricTD':
        hp, hc = gen_e_td_wf(f_low, e, M, q, sample_rate, phase)
    elif approximant=='TEOBResumS':
        hp, hc = gen_teob_wf(f_low, e, M, q, sample_rate, phase)
    else:
        raise Exception('approximant not recognised')

    # Returns waveform as complex timeseries
    return hp - 1j*hc

## Varying true anomaly

def m1_m2_from_M_q(M, q):
    """
    Calculates component masses from total mass and mass ratio.

    Parameters:
        M: Total mass.
        q: Mass ratio.

    Returns:
        Masses of binary components.
    """
    
    m2 = M/(1+q)
    m1 = M - m2
    return m1, m2

def P_from_f(f):
    """
    Calculates orbital period from gravitational wave frequency.

    Parameters:
        f: Gravitational wave frequency.

    Returns:
        Orbital period.
    """
    
    f_orb = f/2
    return 1/f_orb

def a_from_P(P, M):
    """
    Calculates semi-major axis of orbit using Kepler's third law.

    Parameters:
        P: Orbital period.
        M: Total mass.

    Returns:
        Semi-major axis.
    """
    
    a_cubed = (const.G*M*P**2)/(4*np.pi**2)
    return a_cubed**(1/3)

def peri_advance_orbit(P, e, M):
    """
    Calculates periastron advance for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Periastron advance per orbit.
    """
    numerator = 6*np.pi*const.G*M
    a = a_from_P(P, M)
    denominator = const.c**2*a*(1-e**2)
    
    return numerator/denominator

def num_orbits(P, e, M):
    """
    Calculates number of orbits required for true anomaly to change by complete cycle of 2pi.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Number of orbits to shift true anomaly by 2pi.
    """
    
    delta_phi = peri_advance_orbit(P, e, M)
    n_orbit = (2*np.pi)/(2*np.pi - delta_phi)
    return n_orbit

def delta_freq_orbit(P, e, M, q):
    """
    Calculates shift in frequency for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Frequency shift per orbit.
    """
    
    m1, m2 = m1_m2_from_M_q(M, q)
    numerator = 2*192*np.pi*(2*np.pi*const.G)**(5/3)*m1*m2*(1+(73/24)*e**2+(37/96)*e**4)
    denominator = 5*const.c**5*P**(8/3)*(m1+m2)**(1/3)*(1-e**2)**(7/2)
    return numerator/denominator

def shifted_f(f, e, M, q):
    """
    Calculates how to shift frequency such that true anomaly changes by 2pi.

    Parameters:
        f: Original starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Shifted starting frequency.
    """
    
    M *= aconst.M_sun.value
    P = P_from_f(f)
    delta_f_orbit = delta_freq_orbit(P, e, M, q)
    n_orbit = num_orbits(P, e, M)
    return f - delta_f_orbit*n_orbit

def shifted_e(s_f, f, e):
    """
    Calculates how to shift eccentricity to match shifted frequency in such a way that the original frequency and eccentricity are recovered after one true anomaly cycle of 2pi.

    Parameters:
        s_f: Shifted starting frequency.
        f: Original starting frequency.
        e: Starting eccentricity.

    Returns:
        Shifted starting eccentricity.
    """
    
    s_e = e*(s_f/f)**(-19/18)
    return s_e

## Match waveforms

def ceiltwo(number):
    """
    Finds next highest power of two of a number.

    Parameters:
        number: Number to find next highest power of two for.

    Returns:
        Next highest power of two.
    """
    
    ceil = math.ceil(np.log2(number))
    return 2**ceil

def resize_wfs(wf_a, wf_b):
    """
    Resizes two input waveforms to both match the next highest power of two.

    Parameters:
        wf_a: First input waveform.
        wf_b: Second input waveform.

    Returns:
        Resized waveforms.
    """
    
    tlen = ceiltwo(max(len(wf_a), len(wf_b)))
    wf_a.resize(tlen)
    wf_b.resize(tlen)
    return wf_a, wf_b

def trim_wf(wf_trim, wf_ref):
    """
    Cuts the initial part of one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_trim: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """
    
    first_ind = np.argmin(np.abs(wf_ref.sample_times[0]-wf_trim.sample_times))
    wf_trim = wf_trim[first_ind:]
    wf_ref, wf_trim = resize_wfs(wf_ref, wf_trim)
    wf_trim.start_time = wf_ref.start_time
    assert np.array_equal(wf_ref.sample_times, wf_trim.sample_times)

    return wf_trim

def prepend_zeros(wf_pre, wf_ref):
    """
    Prepends zeros to one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_pre: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """

    num_zeros = len(np.where(wf_pre.sample_times[0] - wf_ref.sample_times > 0)[0])
    wf_pre.prepend_zeros(num_zeros)
    wf_pre, wf_ref = resize_wfs(wf_pre, wf_ref)
    wf_pre.start_time = wf_ref.start_time
    assert np.array_equal(wf_pre.sample_times, wf_ref.sample_times)

    return wf_pre

def match_wfs(wf1, wf2, f_low, subsample_interpolation, return_phase=False):
    """
    Calculates match (overlap maximised over time and phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Lower bound of frequency integral.
        subsample_interpolation: Whether to use subsample interpolation.
        return_phase: Whether to return phase of maximum match.
        
    Returns:
        Amplitude (and optionally phase) of match.
    """

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs(wf1, wf2)

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / wf1.duration
    flen = len(wf1)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # Perform match
    m = match(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low+3, subsample_interpolation=subsample_interpolation, return_phase=return_phase)

    # Additionally returns phase required to match waveforms up if requested
    if return_phase:
        return m[0], m[2]
    else:
        return m[0]

def overlap_cplx_wfs(wf1, wf2, f_low, normalized=True):
    """
    Calculates complex overlap (overlap maximised over phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Starting frequency of waveforms.
        normalized: Whether to normalise result between 0 and 1.
        
    Returns:
        Complex overlap.
    """

    # Trims earlier wf so same amount of data before merger
    if wf1.start_time < wf2.start_time:
        wf1 = trim_wf(wf1, wf2)
    elif wf1.start_time > wf2.start_time:
        wf2 = trim_wf(wf2, wf1)
    assert wf1.start_time == wf2.start_time

    # Ensures wfs are tapered
    if wf1[0] != 0:
        wf1 = taper_wf(wf1)
    if wf2[0] != 0:
        wf2 = taper_wf(wf2)
    
    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs(wf1, wf2)

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / wf1.duration
    flen = len(wf1)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # Perform complex overlap
    m = overlap_cplx(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=normalized)

    return m

def minimise_match(s_f, f_low, e, M, q, h_fid, sample_rate, approximant, subsample_interpolation):
    """
    Calculates match to fiducial waveform for a given shifted frequency.

    Parameters:
        s_f: Shifted frequency.
        f_low: Original starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        h_fid: Fiducial waveform.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        subsample_interpolation: Whether to use subsample interpolation.
        
    Returns:
        Match of waveforms.
    """

    # Calculate shifted eccentricity
    s_e = shifted_e(s_f[0], f_low, e)

    # Calculate trial waveform
    trial_wf = gen_wf(s_f[0], s_e, M, q, sample_rate, approximant)

    # Calculate match
    m = match_wfs(trial_wf, h_fid, s_f[0], subsample_interpolation)

    return m    

## Waveform components

def taper_wf(wf_taper):
    """
    Tapers start of input waveform using pycbc.waveform taper_timeseries() function.

    Parameters:
        wf_taper: Waveform to be tapered.
        
    Returns:
        Tapered waveform.
    """
    
    wf_taper_p = taper_timeseries(wf_taper.real(), tapermethod='start')
    wf_taper_c = taper_timeseries(-wf_taper.imag(), tapermethod='start')
    wf_taper = wf_taper_p - 1j*wf_taper_c

    return wf_taper

def norm_ap_peri(unnorm_h_ap, unnorm_h_peri, f_low):
    """
    Normalises h_ap and h_peri to ensure that (h1|h2) = 0.

    Parameters:
        unnorm_h_ap: Unnormalised h_ap waveform.
        unnorm_h_peri: Unnormalised h_peri waveform.
        f_low: Lower bound of frequency interval.
        
    Returns:
        Normalised h_ap and h_peri waveforms.
    """

    # Generate the aLIGO ZDHP PSD
    assert len(unnorm_h_ap) == len(unnorm_h_peri)
    delta_f = 1.0 / unnorm_h_ap.duration
    flen = len(unnorm_h_ap)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # Calculates normalisation factor using sigma function
    norm_ap = sigma(unnorm_h_ap.real(), psd=psd, low_frequency_cutoff=f_low+3)
    norm_peri = sigma(unnorm_h_peri.real(), psd=psd, low_frequency_cutoff=f_low+3)

    # Applies normalisation
    norm_h_ap = unnorm_h_ap
    norm_h_peri = unnorm_h_peri*norm_ap/norm_peri

    return norm_h_ap, norm_h_peri
    
def get_h_def(f_low, e, M, q, sample_rate, approximant, taper):
    """
    Generates waveform with chosen parameters at the default value of true anomaly.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        taper: Whether to taper start of waveform.
        
    Returns:
        Complex combination of plus and cross waveform polarisations.
    """
    
    h_def = gen_wf(f_low, e, M, q, sample_rate, approximant)

    # Tapers start of waveform if requested
    if taper:
        h_def = taper_wf(h_def)
        
    return h_def

def get_h_opp(f_low, e, M, q, h_def, sample_rate, approximant, opp_method, subsample_interpolation, taper):
    """
    Generates waveform with chosen parameters out of phase with the default value of true anomaly.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        h_def: Waveform with the default value of true anomaly.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        opp_method: Method to use to calculate the true anomaly shift of pi, either 'equation' or 'samples'.
        subsample_interpolation: Whether to use subsample interpolation.
        taper: Whether to taper start of waveform.
        
    Returns:
        Complex combination of plus and cross waveform polarisations.
    """

    # Gets amount to shift frequency by from equations in order to vary true anomaly by pi
    s_f_2pi = shifted_f(f_low, e, M, q)
    s_f_pi_len = 0.5*(f_low - s_f_2pi)

    # Calculates shifted frequency to vary anomaly by pi based on specified opp_method
    if opp_method == 'equation': # Simply takes result from approximate equations
        s_f_pi = f_low - s_f_pi_len
    elif opp_method == 'samples': # Refines result further by minimising match around local minimum
        args = (f_low, e, M, q, h_def, sample_rate, approximant, subsample_interpolation)
        bounds = [(f_low - 1.5*s_f_pi_len, f_low - 0.5*s_f_pi_len)]
        init_guess = f_low - (1.1 + 0.2*np.random.rand())*s_f_pi_len
        min_match_result = minimize(minimise_match, init_guess, args=args, bounds=bounds, method='Nelder-Mead')
        s_f_pi = min_match_result['x']
    else:
        raise Exception('opp_method not recognised')

    # Generates waveform
    s_e_pi = shifted_e(s_f_pi, f_low, e)
    h_opp = gen_wf(s_f_pi, s_e_pi, M, q, sample_rate, approximant)

    # Edits sample times of h_opp to match h_def
    h_opp = trim_wf(h_opp, h_def)

    # Calculate phase difference and generate in phase h_opp
    overlap = overlap_cplx_wfs(h_def, h_opp, f_low)
    phase_angle = -np.angle(overlap)/2
    h_opp = gen_wf(s_f_pi, s_e_pi, M, q, sample_rate, approximant, phase=phase_angle)
    h_opp = trim_wf(h_opp, h_def)

    # Tapers start of waveform if requested
    if taper:
        h_opp = taper_wf(h_opp)

    return h_opp 

## Overall waveform

def get_h_TD(coeffs, h_ap, h_peri):
    """
    Combines waveform components in time domain to form h1, h2 and h as follows:
    h1 = 0.5*(h_ap + h_peri)
    h2 = 0.5*(h_ap - h_peri)
    h = A*h1 + B*h2

    Parameters:
        coeffs: List containing A and B, the coefficients of h1 and h2.
        h_ap: Waveform starting at apastron.
        h_peri: Waveform starting at periastron.
        
    Returns:
        All waveform components and combinations: h, h1, h2, h_ap, h_peri
    """

    # Calculate h1, h2 components of waveform
    h1 = 0.5*(h_ap + h_peri)
    h2 = 0.5*(h_ap - h_peri)

    # Calculates overall waveform using complex coefficients A, B
    A, B = coeffs
    h = A*h1 + B*h2
    
    # Returns overall waveform and components for testing purposes
    return h, h1, h2, h_ap, h_peri

def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='TEOBResumS', opp_method='equation', subsample_interpolation=True, normalisation=True, taper=True):
    """
    Generates a total waveform and components defined by the following equations:
    h1 = 0.5*(h_ap + h_peri)
    h2 = 0.5*(h_ap - h_peri)
    h = A*h1 + B*h2

    Parameters:
        coeffs: List containing A and B, the coefficients of h1 and h2.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        opp_method: Method to use to calculate the true anomaly shift of pi, either 'equation' or 'samples'.
        subsample_interpolation: Whether to use subsample interpolation.
        normalisation: Whether to normalise h_ap and h_peri components to ensure (h1|h2) = 0.
        taper: Whether to taper start of waveform.
        
    Returns:
        All waveform components and combinations: h, h1, h2, h_ap, h_peri
    """

    # Gets h_def and h_opp components which make up overall waveform
    h_def = get_h_def(f_low, e, M, q, sample_rate, approximant, taper)
    h_opp = get_h_opp(f_low, e, M, q, h_def, sample_rate, approximant, opp_method, subsample_interpolation, taper)

    # Identify h_ap and h_peri based on waveform approximant used
    if approximant=='EccentricTD':
        h_ap, h_peri = h_opp, h_def
    elif approximant=='TEOBResumS':
        h_ap, h_peri = h_def, h_opp
    else:
        raise Exception('approximant not recognised')

    # Normalises h_ap and h_peri if required
    if normalisation:
        h_ap, h_peri = norm_ap_peri(h_ap, h_peri, f_low)

    # Calculate overall waveform and components in time domain
    wfs = get_h_TD(coeffs, h_ap, h_peri)
   
    return wfs