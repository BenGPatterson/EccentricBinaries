#!/usr/bin/env bash

# datafind
# PARENTS 
# CHILDREN filter
/home/ben.patterson/.conda/envs/igwn_eccentric/bin/simple_pe_datafind --outdir outdir --injection injection_params.json --delta_f 0.0625 --f_high 2048.0 --f_low 20.0 --seed 123456789 --multipoles 22 33 32 21 44 --channels H1:INJ L1:INJ V1:INJ --asd H1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt L1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt V1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/avirgo_O4high_NEW.txt

# filter
# PARENTS datafind
# CHILDREN analysis
/home/ben.patterson/.conda/envs/igwn_eccentric/bin/simple_pe_filter --trigger_parameters trigger_parameters.json --approximant IMRPhenomXPHM --f_low 20.0 --f_high 2048.0 --minimum_data_length 16 --seed 123456789 --peak_finder scipy --strain outdir/output/strain_cache.json --metric_directions chirp_mass symmetric_mass_ratio chi_align chi_p --asd H1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt L1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt V1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/avirgo_O4high_NEW.txt --outdir outdir/output

# analysis
# PARENTS filter
# CHILDREN corner postprocessing
/home/ben.patterson/.conda/envs/igwn_eccentric/bin/simple_pe_analysis --approximant IMRPhenomXPHM --f_low 20.0 --delta_f 0.0625 --f_high 2048.0 --minimum_data_length 16 --seed 123456789 --snr_threshold 4 --localization_method fullsky --metric_directions chirp_mass symmetric_mass_ratio chi_align chi_p --precession_directions symmetric_mass_ratio chi_align chi_p --asd H1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt L1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/aligo_O4high.txt V1:/home/ben.patterson/projects/simple-pe/examples/zero-noise/avirgo_O4high_NEW.txt --outdir outdir/output --peak_parameters outdir/output/peak_parameters.json --peak_snrs outdir/output/peak_snrs.json

# postprocessing
# PARENTS analysis
# CHILDREN 
/home/ben.patterson/.conda/envs/igwn_eccentric/bin/summarypages --approximant IMRPhenomXPHM --f_low 20.0 --webdir outdir/webpage --gw --no_ligo_skymap --disable_interactive --label simple_pe --add_to_corner theta_jn network_precessing_snr network_33_multipole_snr --samples outdir/output/posterior_samples.dat --config config.ini

# corner
# PARENTS analysis
# CHILDREN 
/home/ben.patterson/.conda/envs/igwn_eccentric/bin/simple_pe_corner --truth injection_params.json --outdir outdir/output --posterior outdir/output/posterior_samples.dat --parameters chirp_mass symmetric_mass_ratio chi_align theta_jn luminosity_distance chi_p

