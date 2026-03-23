import numpy as np
import pickle
from tqdm import tqdm
from pycbc.filter import sigma
from pycbc.psd import aLIGOZeroDetHighPower
from simple_pe.waveforms import make_waveform, calculate_mode_snr, gen_component_wfs
from simple_pe.param_est import SimplePESamples
import logging
_logger = logging.getLogger('PESummary')
_logger.setLevel(logging.CRITICAL + 10)

def generate_snrs(vary_key, fixed_pars, vary_array, n_MAs, n_ecc_harms, n_ecc_gen, df, f_low, flen, psd):

    print(f'Varying {vary_key} over {vary_array[0]} to {vary_array[-1]}')
    matches = []
    for vary_par in tqdm(vary_array):
        matches.append([])
        pars = fixed_pars.copy()
        pars[vary_key] = vary_par
        harms = make_waveform(pars, df, f_low, flen, approximant='TEOBResumS-Dali-Harms',
                              n_ecc_gen=n_ecc_gen, n_ecc_harms=n_ecc_harms, ecc_harm_psd=psd)
        pars = SimplePESamples(pars)
        if 'phase' not in pars.keys():
            pars['phase'] = np.zeros_like(list(pars.values())[0])
        if 'f_ref' not in pars.keys():
            pars['f_ref'] = f_low * np.ones_like(list(pars.values())[0])
        if ('spin_1z' not in pars.keys()) or ('spin_2z' not in pars.keys()):
            pars.generate_spin_z()
        if "inc" not in pars.keys():
            pars["inc"] = 0
        pars.generate_all_posterior_samples(f_low=f_low, f_ref=pars["f_ref"][0],
                                            delta_f=df, disable_remnant=True)
        s_rate = 2 * int(flen*df)
        tlen = int(1/df)
        f_gen = 10
        wfs = gen_component_wfs(pars['total_mass'][0], pars['inverted_mass_ratio'][0], pars['ecc10sqrd'][0]**0.5, pars['spin_1z'][0], pars['spin_2z'][0],
                                f_gen, s_rate, pars['phase'][0], pars['inc'][0], pars['distance'][0], tlen, n_MAs, align_merger=True)
        for wf in wfs['22']:
            z, _ = calculate_mode_snr(wf.real(), psd, harms, wf.start_time, wf.end_time,
                                      f_low, harms.keys(), dominant_mode=0,
                                      subsample_interpolation=True)
            matches[-1].append(z)
            matches[-1][-1]['total'] = sigma(wf.real(), psd, low_frequency_cutoff=f_low, high_frequency_cutoff=s_rate/2)

    return matches


def main():

    # Fixed params
    fixed_pars = {
        'chirp_mass': 24,
        'mass_ratio': 0.5,
        'chi_align': 0,
        'ecc10sqrd': 0.2**2,
        'distance': 1680
    }

    # Varying params
    len_params = 101
    n_MAs = 32
    n_ecc_harms = 6
    n_ecc_gen = 12
    vary_params = {
        'chirp_mass': np.linspace(10, 40, len_params),
        'mass_ratio': np.linspace(1, 0.2, len_params),
        'chi_align': np.linspace(-0.5, 0.5, len_params),
        'ecc10sqrd': np.linspace(0, 0.5, len_params)**2
    }

    # Generate psd
    tlen = 32
    s_rate = 4096
    f_low = 20
    df = 1/tlen
    flen = int(s_rate * tlen)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, df, f_low)

    # Generate snrs
    matches = {}
    for vary_key in vary_params:
        matches[vary_key] = generate_snrs(vary_key, fixed_pars, vary_params[vary_key], n_MAs, n_ecc_harms, n_ecc_gen, df, f_low, flen, psd)

    # Save results
    matches['metadata'] = {}
    matches['metadata']['fixed_pars'] = fixed_pars
    matches['metadata']['vary_pars'] = vary_params
    matches['metadata']['n_ecc_gen'] = n_ecc_gen
    with open('all_matches.pickle', 'wb') as handle:
        pickle.dump(matches, handle)

if __name__ == "__main__":
    main()
