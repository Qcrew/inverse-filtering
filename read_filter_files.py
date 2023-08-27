import pickle
import numpy as np
import matplotlib.pyplot as plt

FILTER_PATHS = ["filters/pi_scope_filter_20230811_IIR.pickle"]

# SAVE_PATH = "pi_scope_filter_20230725_IIR_corrected_3.pickle"

iir_filters = []
fir_filters = []
with open(FILTER_PATHS[0], "rb") as filter_file:
    curr_filters = pickle.load(filter_file)
    iir_filters.extend(curr_filters["iir_filters"])
    fir_filters.extend(curr_filters["fir_filters"])
    print(len(iir_filters))
    print(len(fir_filters))

# filters = {"iir_filters": iir_filters[:-1], "fir_filters": []}
# with open(SAVE_PATH, "wb") as output_file:
#     pickle.dump(filters, output_file, protocol=pickle.HIGHEST_PROTOCOL)
