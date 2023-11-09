import EOBRun_module
import numpy as np
import math
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries
from scipy.optimize import minimize

## Frequency definition conversions

# Converts Keplerian frequency to average frequency used by TEOBResumS
def f_kep2avg(f_kep, e):

    numerator = (1+e**2) * np.sqrt(1-e**2)
    denominator = (1+e)**2 * (1-e)**2

    return f_kep*(numerator/denominator)

# Converts average frequency used by TEOBResumS to Keplerian frequency
def f_avg2kep(f_avg, e):

    numerator = (1+e)**2 * (1-e)**2
    denominator = (1+e**2) * np.sqrt(1-e**2)

    return f_avg*(numerator/denominator)

## Generating waveform

# Generates EccentricTD waveform with given parameters
def gen_e_td_wf(f_low, e, M, q, sample_rate, phase):
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

# Converts modes to use into language TEOBResumS understands
def modes_to_k(modes):
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

# Generates TEOBResumS waveform with given parameters
def gen_teob_wf(f_kep, e, M, q, sample_rate, phase):

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

# Generates waveform with given parameters and approximant
def gen_wf(f_low, e, M, q, sample_rate, approximant, phase=0):

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

# Calculates component masses from total mass and mass ratio
def m1_m2_from_M_q(M, q):
    m2 = M/(1+q)
    m1 = M - m2
    return m1, m2

# Calculates orbital period from gw frequency
def P_from_f(f):
    f_orb = f/2
    return 1/f_orb

# Uses Kepler's 3rd to get semi-major axis from period of orbit
def a_from_P(P, M):
    a_cubed = (const.G*M*P**2)/(4*np.pi**2)
    return a_cubed**(1/3)

# Calculates periastron advance per orbital revolution
def peri_advance_orbit(P, e, M):
    numerator = 6*np.pi*const.G*M
    a = a_from_P(P, M)
    denominator = const.c**2*a*(1-e**2)
    
    return numerator/denominator

# Calculates number of orbits for true anomaly to shift by 2pi
def num_orbits(P, e, M):
    delta_phi = peri_advance_orbit(P, e, M)
    n_orbit = (2*np.pi)/(2*np.pi - delta_phi)
    return n_orbit

# How much frequency is shifted by per orbit
def delta_freq_orbit(P, e, M, q):
    m1, m2 = m1_m2_from_M_q(M, q)
    numerator = 2*192*np.pi*(2*np.pi*const.G)**(5/3)*m1*m2*(1+(73/24)*e**2+(37/96)*e**4)
    denominator = 5*const.c**5*P**(8/3)*(m1+m2)**(1/3)*(1-e**2)**(7/2)
    return numerator/denominator

# Calculates what new shifted frequency should be such that true anomaly changes by 2pi
def shifted_f(f, e, M, q):
    M *= aconst.M_sun.value
    P = P_from_f(f)
    delta_f_orbit = delta_freq_orbit(P, e, M, q)
    n_orbit = num_orbits(P, e, M)
    return f - delta_f_orbit*n_orbit

# Calculates what new shifted frequency and eccentricity should be to shift such that true anomaly changes by 2pi
def shifted_e(s_f, f, e):
    s_e = e*(s_f/f)**(-19/18)
    return s_e

## Match waveforms

# Finds next highest power of two
def ceiltwo(number):
    ceil = math.ceil(np.log2(number))
    return 2**ceil

# Resize both waveform lengths to next highest power of two
def resize_wfs(wf1, wf2):
    tlen = ceiltwo(max(len(wf1), len(wf2)))
    wf1.resize(tlen)
    wf2.resize(tlen)
    return wf1, wf2

# Calculates match between two waveforms
def match_wfs(wf1, wf2, f_low, subsample_interpolation, return_phase=False):

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs(wf1, wf2)

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / wf1.duration
    flen = len(wf1)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    # Perform match
    m = match(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low, subsample_interpolation=subsample_interpolation, return_phase=return_phase)

    # Additionally returns phase required to match waveforms up if requested
    if return_phase:
        return m[0], m[2]
    else:
        return m[0]

# Calculates complex overlap between two waveforms
def overlap_cplx_wfs(wf1, wf2, f_low):

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs(wf1, wf2)

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / wf1.duration
    flen = len(wf1)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    # Perform match
    m = overlap_cplx(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low)

    # Additionally returns phase required to match waveforms up if requested
    return m

# Function to vary s_f to minimise match and find out of phase waveform
def minimise_match(s_f, f_low, e, M, q, h_fid, sample_rate, approximant, subsample_interpolation):

    # Calculate shifted eccentricity
    s_e = shifted_e(s_f[0], f_low, e)

    # Calculate trial waveform
    trial_wf = gen_wf(s_f[0], s_e, M, q, sample_rate, approximant)

    # Calculate match
    m = match_wfs(trial_wf, h_fid, s_f[0], subsample_interpolation)

    return m    

## Waveform components

# Ensures a waveform starts at the same index/time as a reference waveform
def trim_wf(wf2, wf_ref):
    
    first_ind = np.argmin(np.abs(wf_ref.sample_times[0]-wf2.sample_times))
    wf2 = wf2[first_ind:]
    wf_ref, wf2 = resize_wfs(wf_ref, wf2)
    wf2.start_time = wf_ref.start_time
    assert np.array_equal(wf_ref.sample_times, wf2.sample_times)

    return wf2
    
# Gets waveform with given parameters at default true anomaly
def get_h_def(f_low, e, M, q, sample_rate, approximant):
    h_def = gen_wf(f_low, e, M, q, sample_rate, approximant)
    return h_def

# Gets waveform with given parameters out of phase with default true anomaly
def get_h_opp(f_low, e, M, q, h_def, sample_rate, approximant, opp_method, subsample_interpolation):

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
        s_f_pi = f_low - 0.12/2
        #raise Exception('opp_method not recognised')

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

    return h_opp 

## Overall waveform

# Combines waveform components in time domain
def get_h_TD(coeffs, h_ap, h_peri):

    # Calculate h1, h2 components of waveform
    h1 = 0.5*(h_ap + h_peri)
    h2 = 0.5*(h_ap - h_peri)

    # Calculates overall waveform using complex coefficients A, B
    A, B = coeffs
    h = A*h1 + B*h2
    
    # Returns overall waveform and components for testing purposes
    return h, h1, h2, h_ap, h_peri

# Combines waveform components in frequency domain - BROKEN DO NOT USE
def get_h_FD(coeffs, h_ap, h_peri):

    # Calculate h1, h2 components of waveform
    h1p_tilde = 0.5*(h_ap.real().to_frequencyseries() + h_peri.real().to_frequencyseries())
    h1c_tilde = -0.5*(h_ap.imag().to_frequencyseries() + h_peri.imag().to_frequencyseries())
    h2p_tilde = 0.5*(h_ap.real().to_frequencyseries() - h_peri.real().to_frequencyseries())
    h2c_tilde = -0.5*(h_ap.imag().to_frequencyseries() - h_peri.imag().to_frequencyseries())
    h1_tilde = h1p_tilde - 1j*h1c_tilde
    h2_tilde = h2p_tilde - 1j*h2c_tilde

    #plt.plot(h1p_tilde.to_timeseries().sample_times, h1p_tilde.to_timeseries().real())
    plt.plot(h1_tilde.sample_frequencies, h1_tilde.real())
    plt.plot(h1c_tilde.sample_frequencies, h1c_tilde.imag())
    plt.show()

    # Calculates overall waveform using complex coefficients A, B
    A, B = coeffs
    hp_tilde = A*h1p_tilde + B*h2p_tilde
    hc_tilde = A*h1c_tilde + B*h2c_tilde

    #h_tilde = hp_tilde - 1j*hc_tilde

    # Returns overall waveform and components for testing purposes
    return hp_tilde.to_timeseries(), h1_tilde.to_timeseries(), h2_tilde.to_timeseries(), h_ap, h_peri

# Gets overall waveform h = A*h1 + B*h2
def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='EccentricTD', opp_method='equation', subsample_interpolation=True, domain='TD'):

    # Gets h_def and h_opp components which make up overall waveform
    h_def = get_h_def(f_low, e, M, q, sample_rate, approximant)
    h_opp = get_h_opp(f_low, e, M, q, h_def, sample_rate, approximant, opp_method, subsample_interpolation)

    # Edits sample times of h_opp to match h_def
    first_ind = np.argmin(np.abs(h_def.sample_times[0]-h_opp.sample_times))
    h_opp = h_opp[first_ind:]
    h_def, h_opp = resize_wfs(h_def, h_opp)
    h_opp.start_time = h_def.start_time
    assert np.array_equal(h_def.sample_times, h_opp.sample_times)

    # Rotate h_opp by necessary amount to bring into phase with h_def
    _, phase_diff = match_wfs(h_def, h_opp, f_low, subsample_interpolation, return_phase=True)
    h_opp *= np.e**(-1j*phase_diff)

    # Identify h_ap and h_peri based on waveform approximant used
    if approximant=='EccentricTD':
        h_ap, h_peri = h_opp, h_def
    elif approximant=='TEOBResumS':
        h_ap, h_peri = h_def, h_opp
    else:
        raise Exception('approximant not recognised')

    # Calculate overall waveform and components in specified domain
    if domain == 'TD':
        wfs = get_h_TD(coeffs, h_ap, h_peri)
    elif domain =='FD':
        wfs = get_h_FD(coeffs, h_ap, h_peri)
    else:
        raise Exception('domain not recognised')

    return wfs