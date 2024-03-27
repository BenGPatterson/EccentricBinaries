import EOBRun_module
import numpy as np
import math
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx, sigma, sigmasq
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries, frequencyseries
from scipy.optimize import minimize
from scipy.interpolate import interp1d
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
            'output_hpc'         : 0
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

def gen_psd(h_psd, f_low, kind='f'):
    """
    Generates psd required for a real or complex time series.

    Parameters:
        h_psd: Time series to generate psd for.
        f_low: Starting frequency of waveform.
        kind: Whether a psd should be made for a float or complex time series.

    Returns:
        Psd and high frequency cutoff to use.
    """

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / h_psd.duration
    flen = len(h_psd)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low+3)

    # If time series is not complex, we are done
    if kind == 'f':
        return psd, None
    assert kind == 'c'

    # Complex PSD
    cplx_psd = np.zeros(len(h_psd))
    cplx_psd[:flen] = psd
    if len(h_psd)%2 == 0:
        cplx_psd[flen:] = np.flip(psd[1:-1])
    else:
        cplx_psd[flen:] = np.flip(psd[1:])
    cplx_psd[int(len(cplx_psd)/2):int(len(cplx_psd)/2)+1+(len(h_psd)%2)] = cplx_psd[int(len(cplx_psd)/2)-1]
    cplx_psd = frequencyseries.FrequencySeries(cplx_psd, delta_f=delta_f)

    # High frequency cutoff
    low_cutoff_ind = int((f_low+3)/cplx_psd.delta_f)
    high_cutoff_ind = len(h_psd) - (low_cutoff_ind - 1)
    high_cutoff_freq = high_cutoff_ind*cplx_psd.delta_f

    return cplx_psd, high_cutoff_freq

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

def resize_wfs(wfs):
    """
    Resizes two or more input waveforms to all match the next highest power of two.

    Parameters:
        wfs: List of input waveforms.
        wf_b: Second input waveform.

    Returns:
        Resized waveforms.
    """
    
    lengths = [len(i) for i in wfs]
    tlen = ceiltwo(max(lengths))
    for wf in wfs:
        wf.resize(tlen)
    return wfs

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

def match_hn(wf_hjs_, wf_s, f_low, return_index=False):
    """
    Calculates match between fiducial h1 waveform and a trial waveform, and uses the time shift 
    in this match to compute the complex overlaps between the time-shifted fiducial h2,...,hn waveforms
    and a trial waveform. This ensures the 'match' is calculated for h1 and h2,...,hn at the same 
    time.

    Parameters:
        wf_hjs_: List of fiducial h1,...,hn waveforms.
        wf_s: Trial waveform.
        f_low: Starting frequency of waveforms.
        return_index: Whether to return index shift of h1 match.
        
    Returns:
        Complex matches of trial waveform to h1,...,hn.
    """

    # Creates new versions of waveforms to avoid editing originals
    wf_hjs = []
    for i in range(len(wf_hjs_)):
        wf_new = timeseries.TimeSeries(wf_hjs_[i].copy(), wf_hjs_[i].delta_t, epoch=wf_hjs_[i].start_time)
        wf_hjs.append(wf_new)
    wf_s = timeseries.TimeSeries(wf_s.copy(), wf_s.delta_t, epoch=wf_s.start_time)

    # Resize waveforms to the same length
    all_wfs = resize_wfs([*wf_hjs, wf_s])
    wf_hjs = all_wfs[:-1]
    wf_s = all_wfs[-1]

    # Generate the aLIGO ZDHP PSD
    psd, _ = gen_psd(wf_hjs[0], f_low)

    # Perform match on h1
    m_h1_amp, m_index, m_h1_phase = match(wf_hjs[0].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_low+3, subsample_interpolation=True, return_phase=True)
    m_h1 = m_h1_amp*np.e**(1j*m_h1_phase)

    # Shift fiducial h2,...,hn
    if m_index <= len(wf_hjs[0])/2:
        # If fiducial h2,...,hn needs to be shifted forward, prepend zeros to it
        for i in range(1,len(wf_hjs)):
            wf_hjs[i].prepend_zeros(int(m_index))
    else:
        # If fiducial h2,...,hn needs to be shifted backward, prepend zeros to trial waveform instead
        wf_s.prepend_zeros(int(len(wf_hjs[0]) - m_index))

    # As subsample_interpolation=True, require interpolation of h2,...,hn to account for non-integer index shift
    delta_t = wf_hjs[0].delta_t
    if m_index <= len(wf_hjs[0])/2:
        # If fiducial h2 needs to be shifted forward, interpolate h2,...,hn waveform forward
        inter_index = m_index - int(m_index)
        for i in range(1,len(wf_hjs)):
            wf_hj_interpolate = interp1d(wf_hjs[i].sample_times, wf_hjs[i], bounds_error=False, fill_value=0)
            wf_hj_strain = wf_hj_interpolate(wf_hjs[i].sample_times-(inter_index*delta_t))
            wf_hjs[i] = timeseries.TimeSeries(wf_hj_strain, wf_hjs[i].delta_t, epoch=wf_hjs[i].start_time-(inter_index*delta_t))
    else:
        # If fiducial h2 needs to be shifted backward, interpolate h2,...,hn waveform backward
        inter_index = (len(wf_hjs[0]) - m_index) - int(len(wf_hjs[0]) - m_index)
        for i in range(1,len(wf_hjs)):
            wf_hj_interpolate = interp1d(wf_hjs[i].sample_times, wf_hjs[i], bounds_error=False, fill_value=0)
            wf_hj_strain = wf_hj_interpolate(wf_hjs[i].sample_times+(inter_index*delta_t))
            wf_hjs[i] = timeseries.TimeSeries(wf_hj_strain, wf_hjs[i].delta_t, epoch=wf_hjs[i].start_time+(inter_index*delta_t))

    # Resize waveforms to the same length
    all_wfs = resize_wfs([*wf_hjs, wf_s])
    wf_hjs = all_wfs[:-1]
    wf_s = all_wfs[-1]

    # Generate the aLIGO ZDHP PSD again as waveform length may have doubled
    psd, _ = gen_psd(wf_hjs[1], f_low)

    # Perform complex overlap on h2,...,hn
    matches = [m_h1]
    for i in range(1,len(wf_hjs)):
        m = overlap_cplx(wf_hjs[i].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_low+3)
        matches.append(m)
    
    # Returns index shift if requested
    if return_index:
        return *matches, m_index
    else:
        return matches

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

    return match_hn([wf_h1, wf_h2], wf_s, f_low, return_index=return_index)
    

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
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd, _ = gen_psd(wf1, f_low)

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
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd, _ = gen_psd(wf1, f_low)

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

def quad_sine_model(x, A_1, B_1, C_1, A_2, B_2, C_2):
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

    # Additionally returns coefficients if requested
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
        wf_h1: Fiducial h1 waveform.
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

def match_true_anomaly(wf_h, n, f_low, e, M, q, sample_rate, approximant, final_match):
    """
    Calculates match between two waveforms, maximised over shifted frequency 
    by calculating the true anomaly using matches to h1,...,hn waveforms.

    Parameters:
        wf_h: Fiducial waveform.
        n: Number of waveform components to use.
        f_low: Starting frequency of waveforms.
        e: Eccentricity of trial waveform.
        M: Total mass of trial waveform.
        q: Mass ratio of trial waveform.
        sample_rate: Sample rate of trial waveform.
        approximant: Approximant of trial waveform.
        final_match: Whether to perform final match to TEOBResumS waveform or h1,...,hn quad match.
        
    Returns:
        Complex match between waveforms maximised over shifted frequency/true anomaly.
    """

    # Calculates matches to h1,...,hn at f_low
    all_wfs = list(get_h([1]*n, f_low, e, M, q, sample_rate, approximant=approximant))
    matches = match_hn(all_wfs[1:n+1], wf_h, f_low)

    # Gets phase difference
    phase_diff = np.angle(matches[0]) - np.angle(matches[1])
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
        all_s_f_wfs = list(get_h([1]*n, s_f, s_e, M, q, sample_rate, approximant=approximant))
        match = match_hn(all_s_f_wfs[1:n+1], wf_h, f_low)
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

def get_comp_shifts(h, f_low, e, M, q, n, sample_rate, approximant, shift_corr):
    '''
    Calculates shifted frequency and eccentricity required to create each component
    waveform (beyond first).

    Parameters:
        h: First (unshifted) component waveform.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        shift_corr: Whether to use trial waveforms to correct shifting of component waveforms.

    Returns:
        Shifted frequency and eccentricity for all components beyond first.
    '''

    # Finds shifted frequency and eccentricity without correction
    max_s_f = shifted_f(f_low, e, M, q)
    s_f_vals = np.linspace(f_low, max_s_f, n, endpoint=False)[1:]
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    # Corrects this by generating trial component waveforms if requested
    if shift_corr:

        # Generates trial waveforms and find time of each peak
        wfs = [h]
        for s_f, s_e in zip(s_f_vals, s_e_vals):
            wf = gen_wf(s_f, s_e, M, q, sample_rate, approximant)
            wfs.append(wf)

        # Find peak and min times
        est_2pi_ind = (wfs[0].sample_times[0] - wfs[-1].sample_times[0])*sample_rate*n/(n-1)
        wf_peak_times = []
        wf_min_times = []
        for i, wf in enumerate(wfs):

            # Peak times
            peak_ind = np.argmax(abs(wf[int(est_2pi_ind):int(2*est_2pi_ind)]))+int(est_2pi_ind)
            peak_time = wf.sample_times[peak_ind]
            wf_peak_times.append(peak_time)

            # Min times
            min_ind = np.argmin(abs(wf[int(3*est_2pi_ind/2):int(5*est_2pi_ind/2)]))+int(3*est_2pi_ind/2)
            min_time = wf.sample_times[min_ind]
            wf_min_times.append(min_time)

        # Add first and third peak/min of first component waveform
        peak_ind1 = np.argmax(abs(wfs[0][0:int(est_2pi_ind)]))
        wf0_peak_time1 = wfs[0].sample_times[peak_ind1]
        peak_ind3 = np.argmax(abs(wfs[0][int(2*est_2pi_ind):int(3*est_2pi_ind)]))+int(2*est_2pi_ind)
        wf0_peak_time3 = wfs[0].sample_times[peak_ind3]
        min_ind1 = np.argmin(abs(wfs[0][int(est_2pi_ind/2):int(3*est_2pi_ind/2)]))+int(est_2pi_ind/2)
        wf0_min_time1 = wfs[0].sample_times[min_ind1]
        min_ind3 = np.argmin(abs(wfs[0][5*int(est_2pi_ind/2):int(7*est_2pi_ind/2)]))+int(5*est_2pi_ind/2)
        wf0_min_time3 = wfs[0].sample_times[min_ind3]

        # Calculate time ranges and orbital evolution factors
        peak_range_time = wf_peak_times[0] - wf0_peak_time1
        min_range_time = wf_min_times[0] - wf0_min_time1
        peak_orb_ev_factor = (2*wf_peak_times[0] - wf0_peak_time1 - wf0_peak_time3)/n
        min_orb_ev_factor = (2*wf_min_times[0] - wf0_min_time1 - wf0_min_time3)/n

        # Calculates correction factor for each component
        corr_factors = []
        for i in range(1, n):

            # Calculate target time shift taking orbital evolution into account
            peak_orb_ev_corr = peak_orb_ev_factor*(n-i)*i/(2*n)
            peak_target_time = peak_range_time*i/n - peak_orb_ev_corr
            min_orb_ev_corr = min_orb_ev_factor*(n-i)*i/(2*n)
            min_target_time = min_range_time*i/n - min_orb_ev_corr

            # Calculate corresponding correction factors
            peak_corr_factor = peak_target_time/(wf_peak_times[0] - wf_peak_times[i])
            min_corr_factor = min_target_time/(wf_min_times[0] - wf_min_times[i])
            corr_factors.append(np.mean([peak_corr_factor, min_corr_factor]))
    
        # Calculates corrected shifted frequencies and eccentricities
        shift_f_dist = f_low - s_f_vals
        corr_shift_f_dist = shift_f_dist*np.array(corr_factors)
        s_f_vals = f_low - corr_shift_f_dist
        s_e_vals = shifted_e(s_f_vals, f_low, e)

    return s_f_vals, s_e_vals

def gen_component_wfs(f_low, e, M, q, n, sample_rate, approximant, normalisation, phase, shift_corr, taper):
    '''
    Creates n component waveforms used to make h_1,...,h_n, all equally spaced in
    true anomaly.
    
    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        normalisation: Whether to normalise s_1,...,s_n components to ensure (sj|sj) is constant.
        phase: Initial phase of s_1,...,s_n components.
        shift_corr: Whether to use trial waveforms to correct shifting of component waveforms.
        taper: Whether to taper start of waveform.
        
    Returns:
        Component waveforms.
    '''

    # Generates first (unshifted) component waveform and shifts required for others
    h = gen_wf(f_low, e, M, q, sample_rate, approximant, phase=phase)
    s_f_vals, s_e_vals = get_comp_shifts(h, f_low, e, M, q, n, sample_rate, approximant, shift_corr)

    # Tapers first waveform if requested
    if taper:
        h = taper_wf(h)

    # Calculates normalisation factor using sigma function
    if normalisation:
        # Generate the aLIGO ZDHP PSD
        h.resize(ceiltwo(len(h))) 
        psd, _ = gen_psd(h, f_low)
        sigma_0 = sigma(h.real(), psd=psd, low_frequency_cutoff=f_low+3)

    comp_wfs = [h]
    
    # Generate all component waveforms
    for i in range(n-1):

        # Create waveform
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase)

        # Trim waveform to same size as first (shortest), and corrects phase
        h = trim_wf(h, comp_wfs[0])
        overlap = overlap_cplx_wfs(h, comp_wfs[0], f_low)
        phase_angle = np.angle(overlap)/2
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase+phase_angle)
        h = trim_wf(h, comp_wfs[0])

        # Tapers if requested
        if taper:
            h = taper_wf(h)
        
        # Normalises waveform if requested
        if normalisation:
            sigma_h = sigma(h.real(), psd=psd, low_frequency_cutoff=f_low+3)
            h *= sigma_0/sigma_h

        comp_wfs.append(h)

    return comp_wfs

def get_dominance_order(n):
    '''
    Creates indexing array to order h1, ..., hn waveforms from their natural roots of unity order 
    to their order of dominance.
    
    Parameters:
        n: Number of waveform components.
        
    Returns:
        Indexing array.
    '''

    # Always start with j=0
    j_order = [0]

    # Add increasing pairs of j and n-j
    for i in range(1, int((n+1)/2)):
        j_order.append(i)
        j_order.append(n-i)

    # Add n/2 if n is even
    if n%2 == 0:
        j_order.append(int(n/2))

    return j_order

def GS_proj(u, v, f_low, psd):
    '''
    Performs projection used in Grant-Schmidt orthogonalisation, defined as 
    u*(v|u)/(u|u).
    
    Parameters:
        u: Waveform u defined above.
        v: Waveform v defined above.
        f_low: Starting frequency.
        psd: Psd to use to weight complex overlap.
        
    Returns:
        Grant-Schmidt orthogonalised h1,...,hn.
    '''

    numerator = overlap_cplx(v.real(), u.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=False)
    denominator = overlap_cplx(u.real(), u.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=False)

    return u*numerator/denominator

def GS_orthogonalise(f_low, wfs):
    '''
    Performs Grant-Schmidt orthogonalisation on waveforms h1,...,hn to ensure 
    (hj|hm) = 0 for j!=m.
    
    Parameters:
        f_low: Starting frequency.
        wfs: Waveforms h1,...,hn.
        
    Returns:
        Grant-Schmidt orthogonalised h1,...,hn.
    '''

    # Generates psd for use in orthogonalisation
    psd, _ = gen_psd(wfs[0], f_low)

    # Orthogonalises each waveform in turn
    for i in range(1,len(wfs)):
        for j in range(i):
            wfs[i] = wfs[i] - GS_proj(wfs[j], wfs[i], f_low, psd)

    return wfs

def get_h_TD(f_low, coeffs, comp_wfs, GS_normalisation):
    """
    Combines waveform components in time domain to form h1, ..., hn and h as follows:

    Parameters:
        f_low: Starting frequency.
        coeffs: List containing coefficients of h_1, ..., h_n.
        comp_wfs: Waveform components s_1, ..., s_n.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        
    Returns:
        All waveform components and combinations: h, h1, ..., h_n, s_1, ..., s_n
    """

    # Find first primitive root of unity
    prim_root = np.e**(2j*np.pi/len(coeffs))
    
    # Build h1, ..., hn
    hs = []
    for i in range(len(coeffs)):
        hs.append((1/len(coeffs))*comp_wfs[0])
        for j in range(len(coeffs)-1):
            hs[-1] += (1/len(coeffs))*comp_wfs[j+1]*prim_root**(i*(j+1))

    # Re-order by dominance rather than natural roots of unity order
    j_order = get_dominance_order(len(coeffs))
    hs = [hs[i] for i in j_order]

    # Perform Grant-Schmidt orthogonalisation if requested
    if GS_normalisation:
        hs = GS_orthogonalise(f_low, hs)

    # Calculates overall waveform using complex coefficients A, B, C, ...
    h = coeffs[0]*hs[0]
    for i in range(len(coeffs)-1):
        h += coeffs[i+1]*hs[i+1]
    
    # Returns overall waveform and components for testing purposes
    return h, *hs, *comp_wfs

def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='TEOBResumS', subsample_interpolation=True, GS_normalisation=True, comp_normalisation=False, comp_phase=0, comp_shift_corr=False, taper=True):
    """
    Generates a overall h waveform, h_1,...h_n, and s_1,...,s_n.

    Parameters:
        coeffs: List containing coefficients of h_1,...,h_n.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        subsample_interpolation: Whether to use subsample interpolation.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        comp_normalisation: Whether to normalise s_1,...,s_n components to ensure (sj|sj) is constant.
        comp_phase: Initial phase of s_1,...,s_n components.
        comp_shift_corr: Whether to use trial waveforms to correct shifting of component waveforms.
        taper: Whether to taper start of waveform.
        
    Returns:
        All waveform components and combinations: h, h1, ..., h_n, s_1, ..., s_n
    """

    # Other approximants are deprecated
    assert approximant == 'TEOBResumS'

    # Gets (normalised) components which make up overall waveform
    component_wfs = gen_component_wfs(f_low, e, M, q, len(coeffs), sample_rate, approximant, comp_normalisation, comp_phase, comp_shift_corr, taper)

    # Calculate overall waveform and components in time domain
    wfs = get_h_TD(f_low, coeffs, component_wfs, GS_normalisation)
   
    return wfs