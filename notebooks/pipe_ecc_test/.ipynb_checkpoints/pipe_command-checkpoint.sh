#!/usr/bin/bash

source "/cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh"
conda activate igwn_eccentric_new
simple_pe_pipe config.ini
