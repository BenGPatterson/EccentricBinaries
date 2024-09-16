import os
import pickle
import numpy as np
from pycbc.filter import sigma
from pycbc.noise import frequency_noise_from_psd
from simple_pe.waveforms import calculate_mode_snr, network_mode_snr
import p_tqdm
from calcwf import *
from interpolating_match import *

# Waveform parameters
n = 6
zero_ecc_chirp = 24.16193848
fid_e = zero_ecc_chirp2fid_e(0.0730036413528654, scaling_norms=[10, 0.035])
fid_chirp = chirp_degeneracy_line(zero_ecc_chirp, fid_e)
f_match = 20
ifos=['H1']

# Generate harmonics
all_wfs = list(get_h([1]*n, 10, fid_e, chirp2total(fid_chirp, 2), 2, 4096))
wfs = {'h0': all_wfs[1], 'h1': all_wfs[2], 'h-1': all_wfs[3], 'h2': all_wfs[4]}
for key in wfs:
    wfs[key].start_time = 1
    wfs[key].prepend_zeros(4096)
    wfs[key].resize(16*4096)
wfs_f = {}
for key in wfs:
    wfs_f[key] = wfs[key].real().to_frequencyseries()
psd = gen_psd(wfs['h0'], 10)

# Normalise waveform modes
h_perp = {}
h_perp_f = {}
for key in wfs.keys():
    norm = sigma(wfs_f[key], psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])
    h_perp[key] = wfs[key] / norm
    h_perp_f[key] = wfs_f[key] / norm

# Create mode part of data
mode_data = 5*h_perp['h0'] + 2*h_perp['h1'] + 1*h_perp['h-1'] + 1*h_perp['h2']
mode_data_f = mode_data.real().to_frequencyseries()

# Get single sample of harmonic SNRs
def single_sample(counter):

    # Create noise and add modes to boost SNR
    gaussian_data = mode_data_f + frequency_noise_from_psd(psd)

    # Calculate mode SNRs
    mode_SNRs, _ = calculate_mode_snr(gaussian_data, psd, h_perp_f, 16-1/8192, 16, f_match, h_perp.keys(), dominant_mode='h0')
    z = {'H1': mode_SNRs}

    # Calculate network SNRs
    rss_snr, _ = network_mode_snr(z, ['H1'], z[ifos[0]].keys(), dominant_mode='h0')

    # Calculate phase consistent combinations
    cplx_SNRs = [mode_SNRs['h0'], mode_SNRs['h1'], mode_SNRs['h-1'], mode_SNRs['h2']]
    frac, denom = comb_harm_consistent(np.abs(cplx_SNRs[:-1]), np.angle(cplx_SNRs[:-1]), harms=[0,1,-1], return_denom=True)
    h1_hn1_pc_SNR = frac*denom
    frac, denom = comb_harm_consistent(np.abs(cplx_SNRs), np.angle(cplx_SNRs), harms=[0,1,-1,2], return_denom=True)
    h1_hn1_h2_pc_SNR = frac*denom

    return *rss_snr.values(), h1_hn1_pc_SNR, h1_hn1_h2_pc_SNR

# Generate samples
SNR_arr = np.array(p_tqdm.p_map(single_sample, np.arange(10**6))).T

# Unpack SNRs
SNRs = {'gaussian': {}, 'modes_only': {}}
for i, key in enumerate(['h0', 'h1', 'h-1', 'h2', 'h1_h-1_pc', 'h1_h-1_h2_pc']):
    SNRs['gaussian'][key] = SNR_arr[i]

# Calculate SNRs with modes only
mode_SNRs, _ = calculate_mode_snr(mode_data_f, psd, h_perp_f, 16-1/8192, 16, f_match, h_perp.keys(), dominant_mode='h0')
z = {'H1': mode_SNRs}
SNRs['modes_only'], _ = network_mode_snr(z, ['H1'], z[ifos[0]].keys(), dominant_mode='h0')

# Save SNRs dictionary
with open('SNR_samples', 'wb') as fp:
        pickle.dump(SNRs, fp)
