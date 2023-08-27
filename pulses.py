import numpy as np

from typing import List, Tuple


def square_pulse(
    on_time: int, lpad: int, rpad: int, amp: int = 1, dt: float = 1e-9
) -> Tuple[np.ndarray, List[float]]:
    total_len = lpad + rpad + on_time
    pulse = ([0] * lpad) + ([amp] * on_time) + ([0] * rpad)
    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


def gaussian_pulse(sigma, chop, lpad, rpad, amp=1, dt=1e-9) -> tuple[np.ndarray]:
    """ """
    length = int(sigma * chop)
    total_len = lpad + length + rpad

    start, stop = -chop / 2 * sigma, chop / 2 * sigma
    ts = np.arange(start, stop, 1)
    exponential = np.exp(-(ts**2) / (2.0 * sigma**2))
    pulse = amp * exponential
    pulse = np.concatenate(([0] * lpad, pulse, [0] * rpad))

    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


# generates a generic square pulse that lets users determine the starting, hold and end amps
def square_generic(
    on_time: int,
    lpad: int,
    rpad: int,
    amp: Tuple[float, float, float],
    dt: float = 1e-9,
) -> Tuple[np.ndarray, List[float]]:
    total_len = lpad + rpad + on_time
    pulse = ([amp[0]] * lpad) + ([amp[1]] * on_time) + ([amp[2]] * rpad)
    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


def constant_cosine(length_constant, length_ring, lpad=0, rpad=0, amp=1, dt=1e-9):
    def ring_up_wave(length_ring, reverse=False, shape="cos"):
        if shape == "cos":
            i_wave = ring_up_cos(length_ring)
        elif shape == "tanh":
            i_wave = ring_up_tanh(length_ring)
        else:
            raise ValueError("Type must be 'cos' or 'tanh', not %s" % shape)
        if reverse:
            i_wave = i_wave[::-1]
        return i_wave

    def ring_up_cos(length_ring):
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length_ring))) * amp

    def ring_up_tanh(length_ring):
        ts = np.linspace(-2, 2, length_ring)
        return (1 + np.tanh(ts)) / 2 * amp

    ring_up = ring_up_wave(length_ring)
    ring_down = ring_up_wave(length_ring, reverse=True)
    constant = np.full(length_constant, amp)
    pulse = np.concatenate(([0] * lpad, ring_up, constant, ring_down, [0] * rpad))
    # pulse = [0] * lpad + list(pulse)
    total_length = lpad + length_constant + 2 * length_ring + rpad

    timesteps = np.array([dt * i for i in range(total_length)])
    return timesteps, pulse


def constant_cosine_reset(length_constant, length_ring, lpad=0, rpad=0, amp=1, dt=1e-9):
    def ring_up_wave(length_ring, reverse=False, shape="cos"):
        if shape == "cos":
            i_wave = ring_up_cos(length_ring)
        elif shape == "tanh":
            i_wave = ring_up_tanh(length_ring)
        else:
            raise ValueError("Type must be 'cos' or 'tanh', not %s" % shape)
        if reverse:
            i_wave = i_wave[::-1]
        return i_wave

    def ring_up_cos(length_ring):
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length_ring))) * amp

    def ring_up_tanh(length_ring):
        ts = np.linspace(-2, 2, length_ring)
        return (1 + np.tanh(ts)) / 2 * amp

    ring_up = ring_up_wave(length_ring)
    ring_middle = 2 * ring_up_wave(2 * length_ring, reverse=True) - 1
    ring_down = ring_up_wave(length_ring, reverse=False) - 1
    constant = np.full(length_constant, amp)
    constant_reverse = np.full(length_constant, -amp)
    pulse = np.concatenate(
        (
            [0] * lpad,
            ring_up,
            constant,
            ring_middle,
            constant_reverse,
            ring_down,
            [0] * rpad,
        )
    )
    # pulse = [0] * lpad + list(pulse)
    total_length = lpad + length_constant * 2 + 4 * length_ring + rpad

    timesteps = np.array([dt * i for i in range(total_length)])
    return timesteps, pulse


# import matplotlib.pyplot as plt

# ts, pulse = gaussian_pulse(
#     sigma=500,
#     chop=8,
#     lpad=200,
#     rpad=200,
#     dt=1e-9,
#     amp=1,
# )
# plt.scatter(ts, pulse)
# plt.show()
