import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.optimize import curve_fit

from plot_flux_tunability_curve import map_freq_to_current


data_file_path = "/Users/kyle/Documents/qcrew/inverse-filtering/examples/sample_somerset_pi_pulse_scope_cut.h5"
output_filename = "/Users/kyle/Documents/qcrew/inverse-filtering/examples/line_response/sample_line_response_with_freq_mapping.npz"

CLOCK_PERIOD = 4e-9  # It is unlikely that this should ever be changed

PLOT_INDIVIDUAL_FITS = False
PLOT_FINAL_GRAPH = True

# Set to FALSE if we're to the right max freq point (freq goes down as current increases)
# since np.arccos gives results from 0-pi, we don't need to flip the point about the
# middle.
# Set to TRUE if we're to the left of the max freq point (freq goes up as current increases)
# we need to flip our result about the max freq point.
# This should be set to FALSE unless you really want to get the exact number on the other half
# of the curve
FLIP_POINT = True

lo_freq = 6.46289425e9 - 120e6 - 200e6 - 230e6
current = np.array(
    [15.4, 8.08, 8.05, 8, 7.8, 7.5, 7.3, 6.8, 6.5, 6.3, 5.9, 5.5, 5.3, 4.9, 2.7]
)
freq = np.array(
    [
        6.413,
        5.68279,
        5.676,
        5.6662,
        5.6352,
        5.5465,
        5.5076,
        5.3889,
        5.3122,
        5.2594,
        5.1493,
        5.0322,
        4.9715,
        4.8453,
        4.1345,
    ]
)


# Gaussian function for curve fitting
def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c


def gaussian_fit(y, x):
    mean_arg = np.argmax(y)  # np.argmin(y)
    mean = x[mean_arg]
    sigma = 5e6  # np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    fit_range = int(0.2 * len(x))
    x_sample = x[max([mean_arg - fit_range, 0]) : min([mean_arg + fit_range, len(x)])]
    y_sample = y[max([mean_arg - fit_range, 0]) : min([mean_arg + fit_range, len(x)])]

    # popt, pcov = curve_fit(
    #     gaussian,
    #     x_sample,
    #     y_sample,
    #     bounds=(
    #         (-np.inf, min(x_sample), -np.inf, -np.inf),
    #         (0, max(x_sample), np.inf, np.inf),
    #     ),
    #     p0=[min(y_sample) - max(y_sample), mean, sigma, max(y_sample)],
    # )
    popt, pcov = curve_fit(
        gaussian,
        x_sample,
        y_sample,
        bounds=(
            (0, min(x_sample), -np.inf, -np.inf),
            (np.inf, max(x_sample), np.inf, np.inf),
        ),
        p0=[max(y_sample) - min(y_sample), mean, sigma, min(y_sample)],
    )
    return popt


def convert_flux_pi_plot(
    z_avg,
    qubit_freq,
    time_delay,
    plot_final: bool = False,
    plot_individual_fits: bool = False,
) -> np.ndarray:
    """
    To convert our pi pulse from clock cycles to ns, we use interpolation
    to fill in the missing points. While this creates additional data, it
    will not affect our results as interpolation will be done in dlsim if
    it is not already done here. Hence, we'll do the interpolation here to
    simplify things.
    """
    z_avg_t = np.transpose(z_avg)
    qubit_freq_t = np.transpose(qubit_freq)
    data_points = len(z_avg_t)
    fit_data = [0] * data_points
    print("---------- PERFORMING GAUSSIAN FIT ----------")
    for i in range(data_points):
        curr_opt = gaussian_fit(z_avg_t[i], qubit_freq_t[i])
        fit_data[i] = curr_opt[1]
        print(f"Fitting for point {i+1} out of {data_points}")
        print(curr_opt)
        if plot_individual_fits:
            plt.scatter(qubit_freq_t[i], z_avg_t[i], label="fit")
            plt.plot(
                qubit_freq_t[i], gaussian(qubit_freq_t[i], *curr_opt), label="data"
            )
            plt.show()
    print("---------------------- DONE ----------------------")

    freq_response = fit_data
    fit_data = map_freq_to_current(fit_data, current, freq, True, lo_freq, FLIP_POINT)

    print("---------- CONVERTING TIMESCALE FROM CLOCK CYCLES TO NS ----------")
    time_delay_ns = time_delay * CLOCK_PERIOD
    num_points = 4 * max(time_delay)
    full_timesteps = np.linspace(time_delay_ns[0], max(time_delay_ns), num_points)
    converted_data = np.interp(full_timesteps, time_delay_ns, fit_data)
    freq_response = np.interp(full_timesteps, time_delay_ns, freq_response)
    print("------------------------------ DONE ------------------------------")

    if plot_final:
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("current(mA))")
        ax1.set_ylabel("flux", color=color)
        p1 = ax1.plot(full_timesteps, converted_data, label="flux", color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(
            "freq responmse", color=color
        )  # we already handled the x-label with ax1
        p2 = ax2.plot(full_timesteps, freq_response, label="freq response", color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        plots = p1 + p2
        labels = [l.get_label() for l in plots]
        ax1.legend(plots, labels, loc=0)
        plt.show()
    return converted_data


def run_convert_flux_pi_plot(
    file_path: str, plot_final: bool = False, plot_individual_fits: bool = False
) -> np.ndarray:
    data_file = h5py.File(file_path)
    z_avg = data_file["data"]
    qubit_freq = data_file["x"]
    time_delay = data_file["y"][0]

    converted_data = convert_flux_pi_plot(
        z_avg,
        qubit_freq,
        time_delay,
        plot_final=plot_final,
        plot_individual_fits=plot_individual_fits,
    )
    normalized_data = converted_data / converted_data[np.argmax(np.abs(converted_data))]
    return normalized_data


if __name__ == "__main__":
    converted_data = run_convert_flux_pi_plot(
        file_path=data_file_path,
        plot_final=PLOT_FINAL_GRAPH,
        plot_individual_fits=PLOT_INDIVIDUAL_FITS,
    )
    np.savez(output_filename, data=converted_data)
    print(f"Converted waveform saved to {os.getcwd()}/{output_filename}")
