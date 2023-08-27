"""
This script extracts z_avg, x & y data from hdf5 files created during our experiments
into a smaller hdf5 file that is easier to deal with.

Example usage: python3 hdf5_extraction.py /Users/kyle/Documents/fast-flux-line/090729_somerset_qubit_spc.h5 /Users/kyle/Documents/fast-flux-line/090729_cut.h5 
"""

import sys

import h5py

# input_file = "C:\\Users\\qcrew\\Desktop\\qcrew\\data\\somerset\\20230703\\150526_somerset_pi_pulse_scope.h5"
input_file = "C:\\Users\\qcrew\\Desktop\\qcrew\\data\\somerset\\20230719\\121741_somerset_pi_pulse_scope.h5"
output_file = "20230719_121741_somerset_pi_pulse_scope_cut.h5"

infile = h5py.File(input_file, "r")
outfile = h5py.File(output_file, "w")

# outfile.create_dataset("data", data=infile["data"]["state"])
outfile.create_dataset("data", data=infile["data"]["I_AVG"])
outfile.create_dataset("x", data=infile["data"]["x"])
outfile.create_dataset("y", data=infile["data"]["y"])

infile.close()
outfile.close()
