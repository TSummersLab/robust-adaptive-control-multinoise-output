from dataclasses import dataclass, field
import numpy as np
import numpy.random as npr

from SIPPY.sippy import functionset as fset


@dataclass
class SigParam:
    method: str
    mean: float = None
    scale: float = 1.0
    ma_length: int = None
    options: dict = field(default_factory=dict)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def make_sig(T, m=1, params=None):
    # Function that creates signal e.g. for exploratory control actions
    if params is None:
        params = [SigParam(method='gbn')]

    u_hist = np.zeros([T, m])
    for i in range(m):
        for param in params:
            method = param.method
            ma_length = param.ma_length
            mean = param.mean
            scale = param.scale

            # Increase time length if using moving average
            if ma_length == 0 or ma_length is None:
                Ti = T
            else:
                Ti = T + ma_length - 1

            # Create the signal
            if method == 'gbn':
                # Set the switching probability
                if 'p_swd' not in param.options:
                    p_swd = 0.1
                else:
                    p_swd = param.options['p_swd']
                sig = fset.GBN_seq(Ti, p_swd=p_swd)[0]  # Binary switching signal
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
            if ma_length == 0 or ma_length is None:
                pass
            else:
                sig = moving_average(sig, ma_length)

            # Centering
            if mean is not None:
                sig += mean - np.mean(sig)

            # Scaling
            sig *= scale

            # Add signal component to the u_hist
            u_hist[:, i] += sig
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
