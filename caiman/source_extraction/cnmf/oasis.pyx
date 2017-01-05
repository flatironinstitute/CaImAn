"""Extract neural activity from a fluorescence trace using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Mon Apr 4 18:21:13 2016
@author: Johannes Friedrich
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp
from scipy.optimize import fminbound, minimize
from cpython cimport bool

ctypedef np.float_t DOUBLE


def oasisAR1(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, DOUBLE lam=0, DOUBLE s_min=0):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    min 1/2|c-y|^2 + lam |s|_1 subject to s_t = c_t-g c_{t-1} >=s_min or =0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    s_min : float, optional, default 0
        Minimal non-zero activity within each bin (minimal 'spike size').

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, f, l
        unsigned int len_active_set
        DOUBLE v, w
        np.ndarray[DOUBLE, ndim = 1] solution, h

    T = len(y)
    solution = np.empty(T)
    # [value, weight, start time, length] of pool
    active_set = [[y[i] - lam * (1 - g), 1, i, 1] for i in [0, 1]]
    c = 0
    i = 1
    while i < T:
        while i < T and \
            (active_set[c][0] / active_set[c][1] * g**active_set[c][3] + s_min <=
             active_set[c + 1][0] / active_set[c + 1][1]):
            c += 1
            i = active_set[c][2] + active_set[c][3]
            if i < T:
                active_set.append([y[i] - lam * (1 if i == T - 1 else (1 - g)), 1, i, 1])
        if i == T:
            break
        # merge two pools
        active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
        active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
        active_set[c][3] += active_set[c + 1][3]
        active_set.pop(c + 1)
        while (c > 0 and  # backtrack until violations fixed
               (active_set[c - 1][0] / active_set[c - 1][1] * g**active_set[c - 1][3] + s_min >
                active_set[c][0] / active_set[c][1])):
            c -= 1
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
            active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
            active_set[c][3] += active_set[c + 1][3]
            active_set.pop(c + 1)
        i = active_set[c][2] + active_set[c][3]
        if i < T:
            active_set.append([y[i] - lam * (1 if i == T - 1 else (1 - g)), 1, i, 1])
    # construct solution
    # calc explicit kernel h up to required length just once
    h = np.exp(log(g) * np.arange(max([a[-1] for a in active_set])))
    for v, w, f, l in active_set:
        solution[f:f + l] = max(v, 0) / w * h[:l]
    return solution, np.append(0, solution[1:] - g * solution[:-1])


def constrained_oasisAR1(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, DOUBLE sn,
                         bool optimize_b=False, bool b_nonneg=True, int optimize_g=0,
                         int decimate=1, int max_iter=5, int penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted, if known, see optimize_b) with one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default False
        Optimize baseline if True else it is set to 0, see y.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 1
        Decimation factor for estimating hyper-parameters faster on decimated data.
    max_iter : int, optional, default 5
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, f, l
        unsigned int len_active_set, ma, count, T
        DOUBLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi
        bool g_converged
        np.ndarray[DOUBLE, ndim = 1] solution, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll

    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        g = g**decimate
        thresh = thresh / decimate / decimate
        T = len(y)
    h = np.exp(log(g) * np.arange(T))  # explicit kernel, useful for constructing solution
    solution = np.empty(T)
    # [value, weight, start time, length] of pool
    lam = 0  # sn/sqrt(1-g*g)
    if T < 5000:  # for 5000 or more frames grow set of pools in first run: faster & memory
        active_set = [[y[i], 1, i, 1] for i in range(T)]
    else:
        def oasis1strun(y, g, h, solution):
            T = len(y)
            # [value, weight, start time, length] of pool
            active_set = [[y[i], 1, i, 1] for i in [0, 1]]
            c = 0
            i = 1
            while i < T:
                while i < T and \
                    (active_set[c][0] * active_set[c + 1][1] * g**active_set[c][3] <=
                     active_set[c][1] * active_set[c + 1][0]):
                    c += 1
                    i = active_set[c][2] + active_set[c][3]
                    if i < T:
                        active_set.append([y[i], 1, i, 1])
                if i == T:
                    break
                # merge two pools
                active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
                active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
                active_set[c][3] += active_set[c + 1][3]
                active_set.pop(c + 1)
                while (c > 0 and  # backtrack until violations fixed
                       (active_set[c - 1][0] * active_set[c][1] * g**active_set[c - 1][3] >
                        active_set[c - 1][1] * active_set[c][0])):
                    c -= 1
                    # merge two pools
                    active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
                    active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
                    active_set[c][3] += active_set[c + 1][3]
                    active_set.pop(c + 1)
                i = active_set[c][2] + active_set[c][3]
                if i < T:
                    active_set.append([y[i], 1, i, 1])
            # construct solution
            for v, w, f, l in active_set:
                solution[f:f + l] = v / w * h[:l]
            solution[solution < 0] = 0
            return solution, active_set

    def oasis(active_set, g, h, solution):
        solution = np.empty(active_set[-1][2] + active_set[-1][3])
        len_active_set = len(active_set)
        c = 0
        while c < len_active_set - 1:
            while c < len_active_set - 1 and \
                (active_set[c][0] * active_set[c + 1][1] * g**active_set[c][3] <=
                 active_set[c][1] * active_set[c + 1][0]):
                c += 1
            if c == len_active_set - 1:
                break
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
            active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
            active_set[c][3] += active_set[c + 1][3]
            active_set.pop(c + 1)
            len_active_set -= 1
            while (c > 0 and  # backtrack until violations fixed
                   (active_set[c - 1][0] * active_set[c][1] * g**active_set[c - 1][3] >
                    active_set[c - 1][1] * active_set[c][0])):
                c -= 1
                # merge two pools
                active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
                active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
                active_set[c][3] += active_set[c + 1][3]
                active_set.pop(c + 1)
                len_active_set -= 1
        # construct solution
        for v, w, f, l in active_set:
            solution[f:f + l] = v / w * h[:l]
        solution[solution < 0] = 0
        return solution, active_set

    if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
        if T < 5000:
            solution, active_set = oasis(active_set, g, h, solution)
        else:
            solution, active_set = oasis1strun(y, g, h, solution)
        tmp = np.empty(len(solution))
        res = y - solution
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and sum(solution) > 1e-9:
            # calc RSS
            res = y - solution
            RSS = res.dot(res)
            # update lam
            for i, (v, w, f, l) in enumerate(active_set):
                if i == len(active_set) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for a in active_set:     # perform shift
                a[0] -= dlam * (1 - g**a[3])
            solution, active_set = oasis(active_set, g, h, solution)

    else:  # optimize b and dependend on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = max(b, 0)
        if T < 5000:
            for a in active_set:   # subtract baseline
                a[0] -= b
            solution, active_set = oasis(active_set, g, h, solution)
        else:
            solution, active_set = oasis1strun(y - b, g, h, solution)
        # update b and lam
        db = max(np.mean(y - solution), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
        v, w, f, l = active_set[-1]
        solution[f:f + l] = max(0, v) / w * h[:l]
        # calc RSS
        res = y - b - solution
        RSS = res.dot(res)
        tmp = np.empty(len(solution))
        g_converged = False
        count = 0
        # until noise constraint is tight or spike train is empty or max_iter reached
        while (RSS < thresh * (1 - 1e-4) or RSS > thresh * (1 + 1e-4)) \
                and sum(solution) > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i, (v, w, f, l) in enumerate(active_set):
                if i == len(active_set) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            tmp -= 1. / T / (1 - g) * np.sum([(1 - g**l)**2 / w for (_, w, _, l) in active_set])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = max(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for a in active_set:     # perform shift
                a[0] -= dphi * (1 - g**a[3])
            solution, active_set = oasis(active_set, g, h, solution)
            # update b and lam
            db = max(np.mean(y - solution), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            active_set[-1][0] -= dlam * g**active_set[-1][3]  # |s|_1 instead |c|_1
            v, w, f, l = active_set[-1]
            solution[f:f + l] = max(0, v) / w * h[:l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([a[3] for a in active_set])
                idx = np.argsort([a[0] for a in active_set])

                def bar(y, opt, a_s):
                    b, g = opt
                    h = np.exp(log(g) * np.arange(ma))

                    def foo(y, t_hat, len_set, q, b, g, lam=lam):
                        yy = y[t_hat:t_hat + len_set] - b
                        if t_hat + len_set == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**len_set)) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, a_s[i][2], a_s[i][3], h[:a_s[i][3]], b, g)
                                for i in idx[-optimize_g:]])

                def baz(y, active_set):
                    return minimize(lambda x: bar(y, x, active_set), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, active_set)
                if abs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                # explicit kernel, useful for constructing solution
                h = np.exp(log(g) * np.arange(T))
                for a in active_set:
                    q = h[:a[3]]
                    a[0] = q.dot(y[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
                    a[1] = q.dot(q)
                active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
                solution, active_set = oasis(active_set, g, h, solution)
                # update b and lam
                db = max(np.mean(y - solution), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                active_set[-1][0] -= dlam * g**active_set[-1][3]  # |s|_1 instead |c|_1
                v, w, f, l = active_set[-1]
                solution[f:f + l] = max(0, v) / w * h[:l]

            # calc RSS
            res = y - solution - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam = lam * (1 - g)
        g = g**(1. / decimate)
        lam = lam / (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.hstack([a[2] * decimate + np.arange(-decimate, 3 * decimate / 2)
                        for a in active_set])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        active_set = map(list, zip([0.] * len(ll), [0.] * len(ll), list(ff), list(ll)))
        ma = max([a[3] for a in active_set])
        h = np.exp(log(g) * np.arange(T))
        for a in active_set:
            q = h[:a[3]]
            a[0] = q.dot(fluor[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
            a[1] = q.dot(q)
        active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
        solution = np.empty(T)

        solution, active_set = oasis(active_set, g, h, solution)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(active_set[i + 1][0] / active_set[i + 1][1] -
                active_set[i][0] / active_set[i][1] * g**active_set[i][3])
               for i in range(len(active_set) - 1)]
        pos = [active_set[i + 1][2] for i in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        solution = np.zeros_like(y)
        a_s = [[0, 1, 0, len(y)]]
        for p in pos:
            c = 0
            while a_s[c][2] + a_s[c][3] <= p:
                c += 1
            # split current pool at pos
            v, w, f, l = a_s[c]
            q = h[:f - p + l]
            a_s.insert(c + 1, [q.dot(y[p:f + l]), q.dot(q), p, f - p + l])
            q = h[:p - f]
            a_s[c] = [q.dot(y[f:p]), q.dot(q), f, p - f]
            for i in [c, c + 1]:
                v, w, f, l = a_s[i]
                solution[f:f + l] = max(0, v) / w * h[:l]
            # calc RSS
            RSS -= res[a_s[c][2]:f + l].dot(res[a_s[c][2]:f + l])
            res[a_s[c][2]:f + l] = solution[a_s[c][2]:f + l] - y[a_s[c][2]:f + l]
            RSS += res[a_s[c][2]:f + l].dot(res[a_s[c][2]:f + l])
            if RSS < thresh:
                break

    return solution, np.append(0, solution[1:] - g * solution[:-1]), b, g, lam
