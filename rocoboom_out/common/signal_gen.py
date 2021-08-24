from dataclasses import dataclass
import numpy as np
import numpy.random as npr

from SIPPY.sippy import functionset as fset


@dataclass
class SigParam:
    method: str
    mean: float
    scale: float
    ma_length: int


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def make_sig(T, m=1, params=None):
    # Function that creates signal e.g. for exploratory control actions
    if params is None:
        params = [SigParam(method='gbn', mean=0.0, scale=1.0, ma_length=None)]

    u_hist = np.zeros([T, m])
    for i in range(m):
        for param in params:
            method = param.method
            ma_length = param.ma_length
            mean = param.mean
            scale = param.scale

            # Increase time length if using moving average
            if ma_length is not None:
                Ti = T + ma_length - 1
            else:
                Ti = T

            # Create the signal
            if method == 'gbn':
                sig = fset.GBN_seq(Ti, p_swd=0.01)[0]  # Binary switching signal
            elif method == 'wgn':
                u_explore_var = 1.0
                sig = np.sqrt(u_explore_var)*npr.randn(Ti)  # White Gaussian noise
            elif method == 'rgw':
                sig = fset.RW_seq(Ti, 0, sigma=1.0)  # Random Gaussian walk
            elif method == 'zeros':
                sig = np.zeros(Ti)
            elif method == 'ones':
                sig = np.ones(Ti)
            else:
                raise ValueError('Invalid signal generation method!')

            # Moving average
            if ma_length is not None:
                sig = moving_average(sig, ma_length)

            # Centering
            if mean is not None:
                sig += mean - np.mean(sig)
            u_hist[:, i] += scale*sig
    return u_hist


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    npr.seed(1)

    params = [SigParam(method='gbn', mean=0.0, scale=1.0, ma_length=4),
              SigParam(method='gbn', mean=0.0, scale=1.0, ma_length=8),
              SigParam(method='rgw', mean=0.0, scale=0.2, ma_length=20)]
    sig = make_sig(T=100, params=params)
    plt.plot(sig)
