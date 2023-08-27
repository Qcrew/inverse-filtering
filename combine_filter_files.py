import pickle

FILTER_PATHS = ["filters/filter_ApBpC_20230816.pickle", "filters/filter_D_20230822.pickle"]

SAVE_PATH = "filter_ApBpCpD_20230822.pickle"

iir_filters = []
fir_filters = []
for i in range(len(FILTER_PATHS)):
    with open(FILTER_PATHS[i], "rb") as filter_file:
        curr_filters = pickle.load(filter_file)
        iir_filters.extend(curr_filters["iir_filters"])
        fir_filters.extend(curr_filters["fir_filters"])
print(len(fir_filters))
print(len(iir_filters))

filters = {"iir_filters": iir_filters, "fir_filters": fir_filters}
with open(SAVE_PATH, "wb") as output_file:
    pickle.dump(filters, output_file, protocol=pickle.HIGHEST_PROTOCOL)
