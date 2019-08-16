import numpy as np
from scipy import stats


def poisson_gaussian_noise(arr, sigma, alpha, mu=0):
    p = stats.poisson.rvs(arr)
    n = stats.norm.rvs(scale=sigma, loc=mu, size=arr.shape)
    return alpha * p + n


def gaussian_noise(arr, sigma):
    n = stats.norm.rvs(scale=sigma, size=arr.shape)
    return arr + n


def std_mad(a, loc=None, axis=None):
    if loc is None:
        loc = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - loc), axis=axis) / 0.6744897501960817


def std_sn(a):
    diff = np.abs(a[:, np.newaxis] - a[np.newaxis, :])
    return 1.1926 * np.median(np.median(diff, axis=0))


def half_sample_mode(x, sort=True, axis=None):
    """
    Algorithm from [1]. Based on the implementation in [2],
    with modifications to handle len(x) == 3
    [1] http://arxiv.org/abs/math.ST/0505419
    [2] https://stats.stackexchange.com/questions/278237/half-sample-mode-estimate-of-sample-of-weighted-data
    """
    if axis is not None:
        return np.apply_along_axis(half_sample_mode, axis, x)

    if len(x) <= 2:
        return np.mean(x, axis=axis)
    if len(x) == 3:
        if np.mean(x[1:]) < np.mean(x[:-1]):
            return np.mean(x[1:], axis=axis)
        if np.mean(x[1:]) > np.mean(x[:-1]):
            return np.mean(x[:-1], axis=axis)
        # i.e., np.mean(x[1:]) == np.mean(x[:-1]):
        return x[1]

    if sort:
        sorted_x = np.sort(x, axis=axis)
    else:
        sorted_x = x
    # Round up to include the middle value, in the case of an odd-length array
    half_idx = (len(x) + 1) // 2

    # Calculate all interesting ranges that span half of all data points
    ranges = sorted_x[-half_idx:] - sorted_x[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    # Now repeat the procedure on the half that spans the smallest range
    x_subset = sorted_x[smallest_range_idx:(smallest_range_idx + half_idx)]
    return half_sample_mode(x_subset, sort=False)


def half_range_mode(x, sort=True):
    """
    Algorithm from [1]. Based on the implementation in [2],
    with modifications to handle len(x) == 3
    [1] http://arxiv.org/abs/math.ST/0505419
    [2] https://stats.stackexchange.com/questions/278237/half-sample-mode-estimate-of-sample-of-weighted-data
    """
    if len(x) <= 2:
        return np.mean(x)
    if len(x) == 3:
        if np.mean(x[1:]) < np.mean(x[:-1]):
            return np.mean(x[1:])
        if np.mean(x[1:]) > np.mean(x[:-1]):
            return np.mean(x[:-1])
        # np.mean(x[1:]) == np.mean(x[:-1]):
        return x[1]

    if sort:
        sorted_x = np.sort(x)
    else:
        sorted_x = x
    # Round up to include the middle value, in the case of an odd-length array
    half_idx = (len(x) + 1) // 2

    # Calculate all interesting ranges that span half of all data points
    ranges = sorted_x[-half_idx:] - sorted_x[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    # Now repeat the procedure on the half that spans the smallest range
    x_subset = sorted_x[smallest_range_idx:(smallest_range_idx + half_idx)]
    return half_sample_mode(x_subset, sort=False)
