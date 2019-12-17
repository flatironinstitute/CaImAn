import numpy as np


def compute_gat(arr, sigma_sq, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation

    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)


    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha**2), 0)
    f = 2. * np.sqrt(v)
    return f


def compute_inverse_gat(arr, sigma_sq, m=0, alpha=1, method='asym'):
    """
    Inverse of the Generalized Anscombe variance-stabilizing
    transformation

    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    
    
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param m: mean of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :param method: 'closed_form' applies the closed-form approximation
    of the exact unbiased inverse. 'asym' applies the asymptotic
    approximation of the exact unbiased inverse.
    :return: inverse variance-stabilized array
    """
    sigma_sq /= alpha**2

    if method == 'closed-form':
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = ((arr_trunc / 2.)**2 + 0.25 * np.sqrt(1.5) * arr_trunc**-1 - (11. / 8.) * arr_trunc**-2 +
                   (5. / 8.) * np.sqrt(1.5) * arr_trunc**-3 - (1. / 8.) - sigma_sq)
    elif method == 'asym':
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.)**2 - 1. / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError('Only supports the closed-form')

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse
