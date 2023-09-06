import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

current = np.array(
    [15.4, 8.08, 8.05, 8, 7.8, 7.5, 7.3, 6.8, 6.5, 6.3, 5.9, 5.5, 5.3, 4.9]
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
    ]
)


def cosine_fn(current, a, omega, psi, b):
    return (a * np.cos((omega * current) + psi)) + b


def simple_cosine_fn(current, a, omega, psi):
    return a * np.cos((omega * current) + psi)


def simple_arccosine_fn(freq, a, omega, b):
    return (np.arccos((freq - b) / a)) / omega


def arccosine_fn(freq, a, omega, psi, b):
    # print(np.arccos((freq - b) / a))
    return (np.arccos((freq - b) / a) - psi) / omega


def cosine_simple_fit(current, freq):
    a_guess = max(freq) - min(freq)
    psi_guess = -1 * (15.4 % (2 * np.pi))
    omega_guess = 0.1

    popt, pcov = curve_fit(
        simple_cosine_fn,
        current,
        freq,
        bounds=(
            (0, 0, -np.inf),
            (np.inf, 1, np.inf),
        ),
        p0=[a_guess, omega_guess, psi_guess],
        maxfev=5000,
    )

    return popt


def cosine_fit(current, freq):
    a_guess = max(freq) - min(freq)
    b_guess = min(freq)
    psi_guess = -1 * (15.4 % (2 * np.pi))
    omega_guess = 1

    popt, pcov = curve_fit(
        cosine_fn,
        current,
        freq,
        bounds=(
            (0, 0, -np.inf, 0),
            (np.inf, np.inf, np.inf, min(freq)),
        ),
        p0=[a_guess, omega_guess, psi_guess, b_guess],
        maxfev=5000,
    )

    # small changes in b don't make the fit anymore accurate but can
    # cause significant errors when you do arccos.
    if popt[3] < 1e-3:
        popt[3] = 0
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

    starting_level = pulse[0]
    end_level = pulse[-1]
    # Get the current where freq is max based on fitting
    max_freq_point = opt_params[0] - opt_params[3]
    max_freq_point_current = arccosine_fn(max_freq_point, *opt_params)

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
    current_sweep = np.arange(4, 20, 0.1)
    plt.plot(
        current_sweep, [cosine_fn(i, *curr_opt) for i in current_sweep], label="data"
    )
    plt.scatter(current, freq)
    plt.show()

    y = arccosine_fn(curr_opt[0], *curr_opt)
    print(y)
    # print(cosine_fn(y, *curr_opt))
    # offset = current[0] - y

    z = arccosine_fn(freq[6] - min(freq), *curr_opt)
    neg_z = reflect_about(z, y)
    print(z)
    print(neg_z)
    print(freq[6] - min(freq))
    print(cosine_fn(z, *curr_opt))
    print(cosine_fn(neg_z, *curr_opt))
