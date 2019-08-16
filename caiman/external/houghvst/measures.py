import numpy as np
from houghvst.estimation import gat


def compare_variance_stabilization(img, img_noisy, sigma_gt, alpha_gt, sigma_est, alpha_est):
    assess_variance_stabilization(img, img_noisy, sigma_gt, alpha_gt, heading='Ground truth')
    assess_variance_stabilization(img, img_noisy, sigma_est, alpha_est)


def assess_variance_stabilization(img,
                                  img_noisy,
                                  sigma,
                                  alpha,
                                  correct_noiseless=True,
                                  verbose=True,
                                  heading='Estimated'):
    if correct_noiseless:
        img = alpha * img

    img_gat = gat.compute_gat(img, sigma, alpha=alpha)
    img_noisy_gat = gat.compute_gat(img_noisy, sigma, alpha=alpha)
    diff = img_gat - img_noisy_gat

    variance = np.var(diff, ddof=1)
    if verbose:
        print('--->', heading, 'variance', variance)
        # print(np.var(diff, ddof=1, axis=1).min(),
        #       np.var(diff, ddof=1, axis=1).max())
    return variance


def compute_temporal_mean_var(movie):
    means = np.mean(movie, axis=0)
    variances = np.var(movie, axis=0, ddof=1)
    return means, variances
