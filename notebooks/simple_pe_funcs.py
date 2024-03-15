import numpy as np
from pycbc.filter.matchedfilter import sigma, overlap_cplx, matched_filter

def orthonormalize_modes(h, ifo_psd, f_low, modes, dominant_mode='22'):
    """
    Orthonormalize a set of waveforms for a given PSD
    Return normalized waveforms orthogonal to the dominant mode,
    sigmas and (complex) overlaps of original waveforms

    Parameters
    ----------
    h: dict
        dictionary of waveform modes
    ifo_psd: pycbc.Frequency_Series
        PSD to use for orthonormalization
    f_low: float
        low frequency cutoff
    modes: list
        modes to consider
    dominant_mode: str
        mode to use for orthonormalization

    Returns
    -------
    h_perp: dict
    orthonormalized waveforms
    sigmas: dict
        waveform normalizations, pre-orthogonalization
    zetas: dict
        complex overlap with dominant mode
    """
    sigmas = {}
    for mode in modes:

        try:
            sigmas[mode] = sigma(h[mode], ifo_psd, low_frequency_cutoff=f_low,
                                 high_frequency_cutoff=
                                 ifo_psd.sample_frequencies[-1])
            h[mode] /= sigmas[mode]
        except:
            print("No power in mode %s" % mode)
            h.pop(mode)

    zetas = {}
    h_perp = {}
    for mode in modes:
        zetas[mode] = overlap_cplx(h[dominant_mode], h[mode], psd=ifo_psd,
                                   low_frequency_cutoff=f_low,
                                   high_frequency_cutoff=
                                   ifo_psd.sample_frequencies[-1],
                                   normalized=True)

        # generate the orthogonal waveform
        if mode == dominant_mode:
            h_perp[mode] = h[mode]
        else:
            h_perp[mode] = (h[mode] - zetas[mode] * h[dominant_mode]) / \
                           (np.sqrt(1 - np.abs(zetas[mode]) ** 2))

    return h_perp, sigmas, zetas

def calculate_mode_snr(strain_data, ifo_psd, waveform_modes, t_start, t_end,
                       f_low, modes, dominant_mode='22'):
    """
    Calculate the SNR in each of the modes.  This is done by finding time of
    the peak SNR for the dominant mode, and then calculating the SNR of other
    modes at that time.

    Parameters
    ----------
    strain_data: pycbc.Time_Series
        the ifo data
    ifo_psd: pycbc.Frequency_Series
        PSD for ifo
    waveform_modes: dict
        dictionary of waveform modes (time/frequency series)
    t_start: float
        beginning of time window to look for SNR peak
    t_end: float
        end of time window to look for SNR peak
    f_low: float
        low frequency cutoff
    modes: list
        the modes to calculate SNR for
    dominant_mode: str
        mode that is used to define the peak time

    Returns
    -------
    z: dict
        dictionary of complex SNRs for each mode
    t: float
        the time of the max SNR
    """

    if dominant_mode not in waveform_modes.keys():
        print("Please give the waveform for the dominant mode")
        return

    s = matched_filter(waveform_modes[dominant_mode], strain_data, ifo_psd,
                       low_frequency_cutoff=f_low)
    snr = s.crop(t_start - s.start_time, s.end_time - t_end)

    # find the peak and use this for the other modes later
    i_max = snr.abs_arg_max()
    t_max = snr.sample_times[i_max]

    z = {}
    for mode in modes:
        s = matched_filter(waveform_modes[mode], strain_data, psd=ifo_psd,
                           low_frequency_cutoff=f_low,
                           high_frequency_cutoff=ifo_psd.sample_frequencies[-1],
                           sigmasq=None)
        snr_ts = s.crop(t_start - s.start_time, s.end_time - t_end)
        z[mode] = snr_ts[i_max]

    return z, t_max

def network_mode_snr(z, ifos, modes, dominant_mode='22'):
    """
    Calculate the Network SNR in each of the specified modes.  For the
    dominant mode, this is simply the root sum square of the snrs in each
    ifo.  For the other modes, we calculate both the rss SNR and the network
    SNR which requires the relative phase between ifos is consistent with
    the dominant.

    Parameters
    ----------
    z: dict
        dictionary of dictionaries of SNRs in each mode (in each ifo)
    ifos: list
        A list of ifos to use
    modes: list
        A list of modes to use
    dominant_mode: str
        the mode with most power (for orthogonalization)

    Returns
    -------
    rss_snr: dict
        the root sum squared SNR in each mode
    net_snr: dict
        the SNR in each mode that is consistent (in amplitude and
        phase) with the dominant mode SNR
    """

    z_array = {}

    rss_snr = {}

    for mode in modes:
        z_array[mode] = np.array([z[ifo][mode] for ifo in ifos])
        rss_snr[mode] = np.linalg.norm(z_array[mode])

    net_snr = {}

    for mode in modes:
        net_snr[mode] = np.abs(np.inner(z_array[dominant_mode],
                                        z_array[mode].conjugate())) / \
                        rss_snr[dominant_mode]

    return rss_snr, net_snr


def matched_filter_network(
    ifos, data, psds, t_start, t_end, h, f_low, dominant_mode=0
):
    """
    Find the maximum SNR in the network for a waveform h within a given time
    range

    :param ifos: list of ifos
    :param data: a dictionary containing data from the ifos
    :param psds: a dictionary containing psds from the given ifos
    :param t_start: start time to consider SNR peak
    :param t_end: end time to consider SNR peak
    :param h: waveform (either a time series or dictionary of time series)
    :param f_low: low frequency cutoff
    :param dominant_mode: the dominant waveform mode
    (if a dictionary was passed)
    :return snr: the network snr
    :return smax: the max snr in each ifo
    :return tmax: return the time of max snr in each ifo
    """
    if not isinstance(h, dict):
        h = {0: h}
    modes = list(h.keys())
    snrsq = 0
    smax = {}
    tmax = {}
    for ifo in ifos:
        h_perp, _, _ = orthonormalize_modes(
            h, psds[ifo], f_low, modes, dominant_mode
        )
        z_dict, tmax[ifo] = calculate_mode_snr(
            data[ifo], psds[ifo], h_perp, t_start, t_end, f_low,
            h.keys(), dominant_mode
        )
        if len(modes) > 1:
            # return the RSS SNR
            smax[ifo] = np.linalg.norm(np.array(list(z_dict.values())))
        else:
            # return the complex SNR for the 1 mode
            smax[ifo] = z_dict[modes[0]]

        snrsq += np.abs(smax[ifo]) ** 2
    return np.sqrt(snrsq), smax, tmax
