import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

current = np.array(
    [15.4, 8.08, 8.05, 8, 7.8, 7.5, 7.3, 6.8, 6.5, 6.3, 5.9, 5.5, 5.3, 4.9, 2.7]
)  # in mA
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
)  # in GHz


def cosine_fn(current, a, omega, psi):
    return a * np.sqrt(np.abs(np.cos(omega * current + psi)))


def arccosine_fn(freq, a, omega, psi):
    return (np.arccos(np.abs(freq**2 / a**2)) - psi) / omega


def get_max_point(a, omega, psi):
    return (np.arccos(1.0) - psi) / omega


def cosine_fit(current, freq):
    a_guess = max(freq)
    psi_guess = 0
    omega_guess = (2 * np.pi) / 66

    popt, pcov = curve_fit(
        cosine_fn,
        current,
        freq,
        bounds=(
            (0, 0, -2 * np.pi),
            (np.inf, np.inf, 2 * np.pi),
        ),
        p0=[a_guess, omega_guess, psi_guess],
        maxfev=8000,
    )
    return popt


def reflect_about(x, axis):
    if x > axis:
        return axis - (x - axis)
    else:
        return axis + (axis - x)


def map_freq_to_current(pulse, current, freq, plot_fit, lo_freq, flip_point):
    opt_params = cosine_fit(current=current, freq=freq)
    print(f"cos curve params are: {opt_params}")
    if plot_fit:
        current_sweep = np.linspace(current[0], current[-1], endpoint=True)
        plt.title("Flux tunbaility curve fit")
        plt.plot(
            current_sweep,
            [cosine_fn(i, *opt_params) for i in current_sweep],
            label="fit",
        )
        plt.scatter(current, freq, label="data")
        plt.show()

    # Get the current where freq is max based on fitting
    max_freq_point_current = get_max_point(*opt_params)

    current_points = np.zeros(len(pulse))
    for i in range(len(pulse)):
        point = (pulse[i] + lo_freq) * 1e-9
        curr_point = arccosine_fn(point, *opt_params)
        if flip_point:
            curr_point = reflect_about(curr_point, max_freq_point_current)
        current_points[i] = curr_point
    return current_points


if __name__ == "__main__":
    curr_opt = cosine_fit(current=current, freq=freq)
    print(curr_opt)
    current_sweep = np.arange(0, 20, 0.1)
    plt.plot(
        current_sweep,
        [cosine_fn(i, *curr_opt) for i in current_sweep],
        label="data",
    )
    plt.scatter(current, freq)
    plt.show()

    max_point = get_max_point(*curr_opt)
    print(max_point)
    # print(cosine_fn(y, *curr_opt))
    # offset = current[0] - y

    z = arccosine_fn(freq[6], *curr_opt)
    neg_z = reflect_about(z, max_point)
    print(z)
    print(neg_z)
    print(freq[6])
    print(cosine_fn(z, *curr_opt))
    print(cosine_fn(neg_z, *curr_opt))
