  import EOBRun_module
import numpy as np
import math
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx, sigma, sigmasq
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

## Conversions

def shifted_e_const(f, e):
    """
    Calculates constant of proportionality between gw frequency and function of eccentricity.

    Parameters:
        f: Gravitational wave frequency.
        e: Eccentricity.

    Returns:
        Proportionality constant.
    """

    constant = f*e**(18/19)*(1+(121/304)*e**2)**(1305/2299)*(1-e**2)**(-3/2)

    return constant

def total2f_ISCO(M):
    """
    Converts total mass of BBH to gravitational wave frequency at ISCO (approximated by assuming 
    circular, non-spinning).

    Parameters:
        M: Total mass.

    Returns:
        Gravitiational wave frequency at ISCO.
    """

    f_ISCO = const.c**3/(6*np.sqrt(6)*np.pi*const.G*M*aconst.M_sun.value)

    return f_ISCO

def const_eff_chirp_favata(given_e, given_chirp, e_vals, f_low=10, q=2, average_f=True, shift_e='approx', ISCO_upper=False):
    """
    Converts array of eccentricity values to chirp mass along a line of constant 
    effective chirp mass, as given by equation 1.1 in Favata et al. 
    https://arxiv.org/pdf/2108.05861.pdf.

    Parameters:
        given_e: Value of eccentricity for given point on line of constant effective chirp mass.
        given_chirp: Value of chirp mass for given point on line of constant effective chirp mass.
        e_vals: Frequency values to be converted.

    Returns:
        Converted chirp mass values.
    """

    # Find average value of f and evolve eccentricities if required
    if average_f:

        # Generate waveform at given point to use in sigmasq
        h = gen_wf(f_low, given_e, chirp2total(given_chirp, q), q, 4096, 'TEOBResumS')
        h.resize(ceiltwo(len(h)))
        
        # Generate the aLIGO ZDHP PSD
        delta_f = 1.0 / h.duration
        flen = len(h)//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

        # Sets upper bound of f_ISCO if requested
        if ISCO_upper:
            high_frequency_cutoff = total2f_ISCO(chirp2total(given_chirp, 2))
        else:
            high_frequency_cutoff = None

        # Calculate both integrals using sigmasq
        h = h.real().to_frequencyseries()
        ss = sigmasq(h, psd=psd, low_frequency_cutoff=f_low+3, high_frequency_cutoff=high_frequency_cutoff)
        ssf = sigmasq(h*np.sqrt(h.sample_frequencies), psd=psd, low_frequency_cutoff=f_low+3, 
                      high_frequency_cutoff=high_frequency_cutoff)

        # Use average frequency to evolve eccentricities
        avg_f = ssf/ss
        print('Average frequency: '+str(avg_f)+' Hz')
        if shift_e == 'approx':
            s_given_e = shifted_e(avg_f, f_low, given_e)
            print(f'Given_e shifted from {given_e} to {s_given_e}')
            s_e_vals = shifted_e(avg_f, f_low, e_vals)
        elif shift_e == 'exact':
            # For given_e
            constant = shifted_e_const(f_low, given_e)
            init_guess = shifted_e(avg_f, f_low, given_e)
            bounds = [(0, 0.999)]
            best_fit = minimize(lambda x: abs(shifted_e_const(avg_f, x)-constant), init_guess, bounds=bounds)
            s_given_e = np.array(best_fit['x'])
            print(f'Given_e shifted from {given_e} to {s_given_e}')
            # For e_vals
            constant = shifted_e_const(f_low, e_vals)
            init_guess = np.full(len(e_vals), shifted_e(avg_f, f_low, e_vals))
            bounds = [(0, 0.999)]
            best_fit = minimize(lambda x: np.sum(abs(shifted_e_const(avg_f, x)-constant)), init_guess, bounds=bounds)
            s_e_vals = np.array(best_fit['x'])
        else:
            raise Exception('shift_e not recognised')
    else:
        s_given_e = given_e
        s_e_vals = e_vals

    # Find effective chirp mass of given point
    eff_chirp = given_chirp/(1-(157/24)*s_given_e**2)**(3/5)

    # Convert to chirp mass values
    chirp_vals = eff_chirp*(1-(157/24)*s_e_vals**2)**(3/5)

    return chirp_vals

def eff_chirp_bose(chirp, e):
    """
    Calculates the effective chirp mass parameter defined by Bose and Pai (2021).

    Parameters:
        chirp: chirp mass in solar masses.
        e: eccentricity at 10Hz.

    Returns:
        Effective chirp mass parameter in solar masses.
    """

    # Define polynomial constants
    epsilon = 0.06110974175360381
    delta = -0.4193723077257345
    Theta_beta = 0.00801015132110059
    Delta_beta = -2.14807199936756*10**-5
    kappa_beta = 1.12702400406416*10**-8
    zeta_beta = -1.9753003183066*10**-12
    Theta_gamma = 0.024204222771565382
    Delta_gamma = -6.261945897154536*10**-6
    kappa_gamma = 1.1175104924576945*10**-8
    zeta_gamma = -3.681726165703978*10**-12

    # Coefficient of e_10^2
    alpha = epsilon*chirp + delta

    # Coefficient of e_10^4
    beta = Theta_beta*chirp**2 + Delta_beta*chirp**2 + kappa_beta*chirp**6 + zeta_beta*chirp**8

    # Coefficient of e_10^6
    gamma = Theta_gamma*chirp**2 + Delta_gamma*chirp**2 + kappa_gamma*chirp**6 + zeta_gamma*chirp**8

    # Effective chirp mass
    eff_chirp = chirp*(1 + alpha*e**2 + beta*e**4 + gamma*e**6)

    return eff_chirp
    
def const_eff_chirp_bose(given_e, given_chirp, e_vals):
    """
    Converts array of eccentricity values to chirp mass along a line of constant 
    effective chirp mass, as given by equation 7 in Bose and Pai (2021) 
    https://arxiv.org/pdf/2107.14736.pdf.

    Parameters:
        given_e: Value of eccentricity for given point on line of constant effective chirp mass.
        given_chirp: Value of chirp mass for given point on line of constant effective chirp mass.
        e_vals: Frequency values to be converted.

    Returns:
        Converted chirp mass values.
    """

    # Find effective chirp mass of given point
    eff_chirp = eff_chirp_bose(given_chirp, given_e)

    # Optimise to find chirp mass values corresponding to eccentricity values
    init_guess = np.full(len(e_vals), given_chirp)
    best_fit = minimize(lambda x: np.sum(abs(eff_chirp_bose(x, e_vals)-eff_chirp)), init_guess)
    chirp_vals = np.array(best_fit['x'])
    
    return chirp_vals

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

    wf_trim_interpolate = interp1d(wf_trim.sample_times, wf_trim, bounds_error=False, fill_value=0)
    wf_trim_strain = wf_trim_interpolate(wf_ref.sample_times)
    wf_trim = timeseries.TimeSeries(wf_trim_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
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

    wf_pre_interpolate = interp1d(wf_pre.sample_times, wf_pre, bounds_error=False, fill_value=0)
    wf_pre_strain = wf_pre_interpolate(wf_ref.sample_times)
    wf_pre = timeseries.TimeSeries(wf_pre_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
    assert np.array_equal(wf_ref.sample_times, wf_pre.sample_times)

    return wf_pre

def match_h1_h2(wf_h1, wf_h2, wf_s, f_low, return_index=False):
    """
    Calculates match between fiducial h1 waveform and a trial waveform, and uses the time shift 
    in this match to compute the complex overlap between the time-shifted fiducial h2 waveform 
    and a trial waveform. This ensures the 'match' is calculated for both h1 and h2 at the same 
    time.

    Parameters:
        wf_h1: Fiducial h1 waveform.
        wf_h2: Fiducial h2 waveform.
        wf_s: Trial waveform
        f_low: Starting frequency of waveforms.
        return_index: Whether to return index shift of h1 match.
        
    Returns:
        Complex matches of trial waveform to h1 and h2 respectively.
    """

    # Creates new versions of waveforms to avoid editing originals
    wf_h1 = timeseries.TimeSeries(wf_h1.copy(), wf_h1.delta_t, epoch=wf_h1.start_time)
    wf_h2 = timeseries.TimeSeries(wf_h2.copy(), wf_h2.delta_t, epoch=wf_h2.start_time)
    wf_s = timeseries.TimeSeries(wf_s.copy(), wf_s.delta_t, epoch=wf_s.start_time)

    # Resize waveforms to the same length
    assert len(wf_h1) == len(wf_h2)
    wf_h1, wf_s = resize_wfs(wf_h1, wf_s)
    wf_h1, wf_h2 = resize_wfs(wf_h1, wf_h2)

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / wf_h1.duration
    flen = len(wf_h1)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # Perform match on h1
    m_h1_amp, m_index, m_h1_phase = match(wf_h1.real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_low+3, subsample_interpolation=True, return_phase=True)
    m_h1 = m_h1_amp*np.e**(1j*m_h1_phase)

    # Shift fiducial h2
    if m_index <= len(wf_h1)/2:
        # If fiducial h2 needs to be shifted forward, prepend zeros to it
        wf_h2.prepend_zeros(int(m_index))
    else:
        # If fiducial h2 needs to be shifted backward, prepend zeros to trial waveform instead
        wf_s.prepend_zeros(int(len(wf_h1) - m_index))

    # As subsample_interpolation=True, require interpolation of h2 to account for non-integer index shift
    delta_t = wf_h1.delta_t
    if m_index <= len(wf_h1)/2:
        # If fiducial h2 needs to be shifted forward, interpolate h2 waveform forward
        inter_index = m_index - int(m_index)
        wf_h2_interpolate = interp1d(wf_h2.sample_times, wf_h2, bounds_error=False, fill_value=0)
        wf_h2_strain = wf_h2_interpolate(wf_h2.sample_times-(inter_index*delta_t))
        wf_h2 = timeseries.TimeSeries(wf_h2_strain, wf_h2.delta_t, epoch=wf_h2.start_time-(inter_index*delta_t))
    else:
        # If fiducial h2 needs to be shifted backward, interpolate h2 waveform backward
        inter_index = (len(wf_h1) - m_index) - int(len(wf_h1) - m_index)
        wf_h2_interpolate = interp1d(wf_h2.sample_times, wf_h2, bounds_error=False, fill_value=0)
        wf_h2_strain = wf_h2_interpolate(wf_h2.sample_times+(inter_index*delta_t))
        wf_h2 = timeseries.TimeSeries(wf_h2_strain, wf_h2.delta_t, epoch=wf_h2.start_time+(inter_index*delta_t))

    # Resize waveforms to the same length
    wf_h2, wf_s = resize_wfs(wf_h2, wf_s)

    # Generate the aLIGO ZDHP PSD again as waveform length doubled
    delta_f = 1.0 / wf_h2.duration
    flen = len(wf_h2)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # Perform complex overlap on h2
    m_h2 = overlap_cplx(wf_h2.real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_low+3)
    
    # Returns index shift if requested
    if return_index:
        return m_h1, m_h2, m_index
    else:
        return m_h1, m_h2
    

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

    # Prepends earlier wf with zeroes so same amount of data before merger (required for overlap_cplx)
    if wf1.start_time > wf2.start_time:
        wf1 = prepend_zeros(wf1, wf2)
    elif wf1.start_time < wf2.start_time:
        wf2 = prepend_zeros(wf2, wf1)
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

## Maximising over shifted frequency

def sine_model_coeffs(m_0, m_1, m_2):
    """
    Calculates coefficients A, B, C in equation m(x) = A*sin(x+B)+C given the value of 
    m(0), m(-pi/2) and m(-pi).

    Parameters:
        m_0: Value of m(0).
        m_1: Value of m(-pi/2).
        m_2: Value of m(-pi).
    
    Returns:
        Coefficients A, B, C.
    """

    # Ensure amplitude of match is given
    m_0, m_1, m_2 = abs(m_0), abs(m_1), abs(m_2)

    # Calculate C
    C = (m_0 + m_2)/2
    
    # Calculate A
    A = np.sqrt((m_0 - C)**2 + (m_1 - C)**2)

    # Calculate B
    B = np.arctan2(m_0 - C, -(m_1 - C))

    return A, B, C

def sine_model(x, A, B, C):
    """
    Calculates sinusoid modelled as m(x) = A*sin(x+B)+C at a given value of x.

    Parameters:
        x: Value at which to evaluate m(x).
        A_1: Amplitude of sinusoid.
        B_1: Phase offset of sinusoid.
        C_1: Offset of sinusoid.
        
    Returns:
        Value of m(x) at given value of x.
    """
    
    m = A*np.sin(x+B)+C

    return m

def quad_sine_model(x, A_1, B_1, C_1, A_2, B_2, C_2, sign=1):
    """
    Calculates quadrature sum of two sinusoids modelled as m_quad(x) = sqrt(m_1^2(x) + m_2^2(x)) 
    where m_n(x) = A_n*sin(x+B_n)+C_n for n=1,2 at a given value of x.

    Parameters:
        x: Value at which to evaluate m_n(x).
        A_1: Amplitude of first sinusoid.
        B_1: Phase offset of first sinusoid.
        C_1: Offset of first sinusoid.
        A_2: Amplitude of second sinusoid.
        B_2: Phase offset of second sinusoid.
        C_2: Offset of second sinusoid.
        
    Returns:
        Value of m_quad(x) at given value of x.
    """
    
    # Calculates m_n functions for this value of x
    m_1 = sine_model(x, A_1, B_1, C_1)
    m_2 = sine_model(x, A_2, B_2, C_2)

    # Calculates quadrature sum for this value of x
    m_quad = np.sqrt(m_1**2 + m_2**2)

    return m_quad

def maximise_quad_sine(A_1, B_1, C_1, A_2, B_2, C_2):
    """
    Maximises quadrature sum of two sinusoids modelled as m_quad(x) = sqrt(m_1^2(x) + m_2^2(x)) 
    where m_n(x) = A_n*sin(x+B_n)+C_n for n=1,2.

    Parameters:
        A_1: Amplitude of first sinusoid.
        B_1: Phase offset of first sinusoid.
        C_1: Offset of first sinusoid.
        A_2: Amplitude of second sinusoid.
        B_2: Phase offset of second sinusoid.
        C_2: Offset of second sinusoid.
        
    Returns:
        Value of x which maximises m_quad(x).
    """

    # Use location of peak of first sinusoid for initial guess
    init_guess = np.pi/2 - B_1
    if init_guess > 0:
        init_guess -= 2*np.pi

    # Set bounds and arguments of function
    args = (A_1, B_1, C_1, A_2, B_2, C_2)
    bounds = [(-2*np.pi, 0)]

    # Perform maximisation
    max_result = minimize(lambda x: -quad_sine_model(x, *args), init_guess, bounds=bounds)
    max_location = max_result['x']

    return max_location

def s_f_max_sine_approx(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant, return_coeffs=False):
    """
    Calculates match between fiducial h1, h2 waveforms and a trial waveform, maximised 
    over true anomaly/shifted frequency by approximating the matches of h1/h2 against 
    as sinusoidal curves.

    Parameters:
        wf_h1: Fiducial h1 waveform.
        wf_h2: Fiducial h2 waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        return_coeffs: whether to return calculated coefficients of sine models.
        
    Returns:
        Complex matches to h1,h2 maximised to quad match peak.
    """
    
    # Converts necessary phase shifts to shifted frequency and eccentricity
    phase_shifts = np.array([0, -np.pi/2, -np.pi])
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f_vals = f_low + (phase_shifts/(2*np.pi))*s_f_range
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    # Calculates matches to h1, h2 at each phase shift
    m1_vals, m2_vals = np.empty(3, dtype=np.complex128), np.empty(3, dtype=np.complex128)
    for i, (s_f, s_e) in enumerate(zip(s_f_vals, s_e_vals)):
        wf_s = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
        m1_vals[i], m2_vals[i] = match_h1_h2(wf_h1, wf_h2, wf_s, f_low)
    
    # Calculates both sets of sine model coefficients
    coeffs_h1 = sine_model_coeffs(*m1_vals)
    coeffs_h2 = sine_model_coeffs(*m2_vals)

    # Find location of quad match peak in terms of required phase shift
    phase_shift_quad_max = maximise_quad_sine(*coeffs_h1, *coeffs_h2)

    # Perform final match to h1, h2 at this phase shift
    s_f_quad_max = f_low + (phase_shift_quad_max/(2*np.pi))*s_f_range
    s_e_quad_max = shifted_e(s_f_quad_max, f_low, e)
    wf_quad_max = gen_wf(s_f_quad_max, s_e_quad_max, M, q, sample_rate, approximant)
    matches = match_h1_h2(wf_h1, wf_h2, wf_quad_max, f_low)

    # Additionall returns coefficients if requested
    if return_coeffs:
        return matches, list(coeffs_h1) + list(coeffs_h2)
    else:
        return matches

def s_f_max_phase_diff(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant):
    """
    Calculates match between fiducial h1, h2 waveforms and a trial waveform, maximised 
    over true anomaly/shifted frequency using the difference between the phase of matches 
    to the h1,h2 waveforms when the trial waveform starts at f=f_low.

    Parameters:
        wf_h1: Fiducial h1 waveform.
        wf_h2: Fiducial h2 waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        
    Returns:
        Complex matches to h1,h2 maximised to quad match peak.
    """

    # Calculates matches to h1, h2 at f_low
    wf_f_low = gen_wf(f_low, e, M, q, sample_rate, approximant)
    m1_f_low, m2_f_low = match_h1_h2(wf_h1, wf_h2, wf_f_low, f_low)

    # Gets phase difference
    phase_diff = np.angle(m2_f_low) - np.angle(m1_f_low)
    if phase_diff > 0:
        phase_diff -= 2*np.pi

    # Converts phase difference to shifted frequency and eccentricity
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f = f_low + (phase_diff/(2*np.pi))*s_f_range
    s_e = shifted_e(s_f, f_low, e)

    # Calculates matches to h1, h2 at shifted frequency
    wf_s_f = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
    matches =  match_h1_h2(wf_h1, wf_h2, wf_s_f, f_low)

    return matches
    

def match_s_f_max(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant, max_method):
    """
    Calculates match between fiducial h1, h2 waveforms and a trial waveform, maximised 
    over true anomaly/shifted frequency using the specified method.

    Parameters:
        wf_h1: Fiducial h1 w_anomalyaveform.
        wf_h2: Fiducial h2 waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        max_method: Which method to use to maximise over shifted frequency, either 'sine_approx' or 'phase_diff'.
        
    Returns:
        Complex matches to h1,h2 maximised to quad match peak.
    """

    # Calculates matches maximised over shifted frequency using specified method
    if max_method == 'sine_approx':
        matches = s_f_max_sine_approx(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant)
    elif max_method == 'phase_diff':
        matches = s_f_max_phase_diff(wf_h1, wf_h2, f_low, e, M, q, sample_rate, approximant)
    else:
        raise Exception('max_method not recognised')

    # Returns matches
    return matches

def match_true_anomaly(wf_h, f_low, e, M, q, sample_rate, approximant, final_match):
    """
    Calculates match between two waveforms, maximised over shifted frequency 
    by calculating the true anomaly using matches to h1, h2 waveforms.

    Parameters:
        wf_h: Fiducial waveform.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        final_match: Whether to perform final match to TEOBResumS waveform or h1, h2 quad match.
        
    Returns:
        Complex match between waveforms maximised over shifted frequency/true anomaly.
    """

    # Calculates matches to h1, h2 at f_low
    _, wf_h1, wf_h2, _, _ = get_h([1,1], f_low, e, M, q, sample_rate, approximant=approximant)
    m1, m2 = match_h1_h2(wf_h1, wf_h2, wf_h, f_low)

    # Gets phase difference
    phase_diff = np.angle(m1) - np.angle(m2)
    if phase_diff > 0:
        phase_diff -= 2*np.pi

    # Converts phase difference to shifted frequency and eccentricity
    s_f_range = f_low - shifted_f(f_low, e, M, q)
    s_f = f_low + (phase_diff/(2*np.pi))*s_f_range
    s_e = shifted_e(s_f, f_low, e)

    # Calculates match(es) to final_match at shifted frequency
    if final_match == 'TEOB':
        wf_s_f = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
        m_amp, m_phase =  match_wfs(wf_s_f, wf_h, f_low, True, return_phase=True)
        match = m_amp*np.e**(1j*m_phase)
    elif final_match == 'quad':
        _, wf_s_f_h1, wf_s_f_h2, _, _ = get_h([1,1], s_f, s_e, M, q, sample_rate, approximant=approximant)
        match = match_h1_h2(wf_s_f_h1, wf_s_f_h2, wf_h, f_low, return_index=False)
    else:
        raise Exception('final_match not recognised')

    # Returns match(es)
    return match

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