import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm, lstsq
from scipy.signal import dlti, dlsim
from scipy.optimize import curve_fit, least_squares, leastsq
from typing import Tuple


def fitting_func(t, a, b, tau):
    return a + (b * np.exp(-t / tau))


def generate_initial_guess(
    data: np.ndarray, timesteps: np.ndarray
) -> Tuple[float, float, float]:
    avg = np.average(data)
    min_v = np.min(data)
    max_v = np.max(data)

    start = data[0]
    end = data[-1]

    a_guess = 0  # assume the exponential is decaying, so offset is the last point

    # estimate y-intercept
    # quarter_point = int(len(data) / 4)
    # est_grad = (data[quarter_point] - start) / (timesteps[quarter_point] - timesteps[0])
    # est_yinter = start - (est_grad * timesteps[0])

    b_guess = start - end  # b is the amplitude of the decay
    # print(timesteps)
    tau_guess = (timesteps[-1] - timesteps[0]) / 5  # assume the fit length is 5x tau

    print(f"min, max, avg is {min_v} {max_v} {avg}")
    print(f"Guess is {a_guess} {b_guess} {tau_guess}")
    return (a_guess, b_guess, tau_guess)


def generate_param_bounds(
    data: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    max_a = 1e-5
    min_a = 0

    max_b = np.inf
    min_b = -np.inf

    max_tau = np.inf
    min_tau = 0

    return ((min_a, min_b, min_tau), (max_a, max_b, max_tau))


def iir_fitting_pass(data_points, timesteps, param_bounds, initial_guess, dt=1e-9):
    if param_bounds is None:
        param_bounds = generate_param_bounds(data_points)
    if initial_guess is None:
        initial_guess = generate_initial_guess(data_points, timesteps)

    opt_params = curve_fit(
        fitting_func,
        timesteps - timesteps[0],
        data_points,
        p0=initial_guess,
        bounds=param_bounds,
        maxfev=100000,
    )[0]
    a, b, tau = tuple(opt_params)
    # Normalize filter amplitude
    a_orig, b_orig = a, b
    # b /= a
    # a /= a

    tau_dt = tau  # * dt
    lamb = (2 * a * tau_dt) + (2 * b * tau_dt) + (a * dt)
    a1 = ((2 * a * tau_dt) + (2 * b * tau_dt) - (a * dt)) / lamb
    b0 = (2 * tau_dt + dt) / lamb
    b1 = (-2 * tau_dt + dt) / lamb

    H_correction = dlti([b0, b1], [1, -a1], dt=dt)
    print("IIR filter: ", H_correction)
    fitting_params = {"a": a_orig, "b": b_orig, "tau": tau}
    filter_params = {"a1": a1, "b0": b0, "b1": b1}

    return {
        "filter_params": filter_params,
        "fitting_params": fitting_params,
        "correction": H_correction,
    }


def iterative_iir_fitting(
    data_points,
    sample_points,
    param_bounds,
    initial_guess,
    timesteps,
    dt=1e-9,
    plot_fits=False,
):
    num_iters = len(sample_points)
    results = [None] * num_iters
    orig_points = data_points
    orig_timesteps = timesteps
    for i in range(num_iters):
        print(f"Iteration #{i}")
        curr_sample = data_points[sample_points[i][0] : sample_points[i][1]]
        curr_timesteps = timesteps[sample_points[i][0] : sample_points[i][1]]

        results[i] = iir_fitting_pass(
            data_points=curr_sample,
            timesteps=curr_timesteps,
            param_bounds=param_bounds[i],
            initial_guess=initial_guess[i],
            dt=dt,
        )

        # apply filter to samples
        prev_points = data_points
        prev_timesteps = timesteps
        timesteps, data_points = dlsim(
            results[i]["correction"], data_points, t=timesteps
        )

        if plot_fits:
            fig, ax = plt.subplots(2, 2, figsize=(10, 7))

            fit_params = results[i]["fitting_params"]
            a, b, tau = fit_params["a"], fit_params["b"], fit_params["tau"]
            best_fit_points = [0] * len(curr_sample)
            for k in range(len(curr_sample)):
                best_fit_points[k] = fitting_func(
                    curr_timesteps[k] - curr_timesteps[0], a, b, tau
                )

            ax[0, 0].scatter(curr_timesteps, curr_sample, c="C0", label="data")
            ax[0, 0].plot(curr_timesteps, best_fit_points, c="C1", label="fit")
            ax[0, 0].set_title(f"Fitting #{i + 1}")
            ax[0, 0].legend()
            ax[0, 1].plot(
                prev_timesteps,
                prev_points / max(prev_points),
                color="C0",
                label="Previous correction",
            )
            ax[0, 1].plot(
                timesteps,
                data_points / max(data_points),
                color="C1",
                label="Current correction",
            )
            ax[0, 1].plot(
                orig_timesteps,
                orig_points,
                color="k",
                linestyle="--",
                label="No correction",
            )
            ax[0, 1].set_title(f"Correction #{i + 1}")
            ax[0, 1].legend()

            w = np.logspace(2, 9, num=5000) * 2 * np.pi * 1e-9
            sys = results[i]["correction"]
            w, mag, phase = sys.bode(w=w)
            ax[1, 0].semilogx(w, mag)  # Bode magnitude plot
            ax[1, 0].set_title("Correction magnitude plot")
            # ax[1,0].set_yscale('log')
            ax[1, 0].grid(True, which="major", axis="both", ls="-")
            ax[1, 1].semilogx(w, phase)  # Bode phase plot
            ax[1, 1].set_title("Correction phase plot")
            ax[1, 1].grid(True, which="major", axis="x", ls="-")
            fig.tight_layout()
            plt.legend()
            plt.show()

        data_points = np.squeeze(data_points)
        print("Filter params are:")
        print(results[i]["fitting_params"])

    data_points = data_points / max(data_points)
    return results, data_points


def build_regularizer_mat(sample_size: int):
    """
    Build D, a N-1 * N matrix with the form
    [1  -1  0  0 ...  0]
    [0   1 -1  0 ...  0]
    [0   0  1 -1 ...  0]
    [        ...       ]
    [0   0  0 ... 1  -1]
    """
    d_matrix = np.zeros((sample_size - 1, sample_size))
    np.fill_diagonal(d_matrix, 1)

    r_ids, c_ids = np.indices((sample_size - 1, sample_size))
    off_diag_row_ids = np.diag(r_ids, k=1)
    off_diag_col_ids = np.diag(c_ids, k=1)
    d_matrix[off_diag_row_ids, off_diag_col_ids] = -1
    return d_matrix


def build_conv_mat(h: np.ndarray):
    sample_size = h.size
    conv_mat = np.zeros((sample_size, sample_size))
    for r in range(sample_size):
        for c in range(sample_size):
            if r >= c:
                conv_mat[r][c] = h[r - c]
            else:
                conv_mat[r][c] = 0
    return conv_mat


def reg_function_builder(h: np.ndarray, y: np.ndarray, alpha: float):
    reg_mat = build_regularizer_mat(h.size)
    conv_mat = build_conv_mat(h)

    def reg_function(x: np.ndarray):
        reg_term = alpha * ((norm(reg_mat.dot(x))) ** 2)
        conv_term = (norm(conv_mat.dot(x) - y)) ** 2
        return conv_term + reg_term

    return reg_function


def linear_least_square(h: np.ndarray, alpha: float, y=np.ndarray):
    reg_mat = build_regularizer_mat(h.size)
    conv_mat = build_conv_mat(h)
    a = np.concatenate([conv_mat, np.sqrt(alpha) * reg_mat])
    b = np.concatenate([y, np.zeros(len(reg_mat))])
    res = lstsq(a, b)
    return res


def fir_fitting(
    step_response: np.ndarray,
    expxected_response: np.ndarray,
    alpha: float,
    use_old_method: bool,
):
    impulse_response = np.gradient(step_response)
    if use_old_method:
        reg_function = reg_function_builder(
            h=impulse_response, y=expxected_response, alpha=alpha
        )
        initial_guess = expxected_response
        results = least_squares(reg_function, initial_guess)
        h_inv = results.x
    else:
        results = linear_least_square(
            h=impulse_response, alpha=alpha, y=expxected_response
        )
        h_inv = results[0]
    return h_inv
