"""Extract neural activity from a fluorescence trace using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Mon Apr 4 18:21:13 2016
@author: Johannes Friedrich
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp, fmax, fabs
from scipy.optimize import fminbound, minimize
from cpython cimport bool
from libcpp.vector cimport vector

ctypedef np.float32_t SINGLE


cdef struct Pool:
    SINGLE v
    SINGLE w
    Py_ssize_t t
    Py_ssize_t l


cdef min1000(a):
    return a if a < 1000 else 1000


cdef class OASIS:
    """
    Deconvolution class implementing OASIS
    Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    min 1/2|c-y|^2 + lam |s|_1 subject to s_t = c_t-g c_{t-1} >=s_min or =0

    Parameters
    ----------
    g : float
        Parameter of the AR(1) or 1st parameter of the AR(2) process that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    s_min : float, optional, default 0
        Minimal non-zero activity within each bin (minimal 'spike size').
    b : float, optional, default 0
        Baseline that is substracted.
    num_empty_samples : int
        Number of elapsed frames until neuron is added and OASIS initialized
    g2 : float
        0 for AR(1) or 2nd parameter of the AR(2) process that models the fluorescence impulse response.


    Attributes
    ----------
    g, lam, smin, b: see Parameters above
    p : order of the AR process
    P : vector of struct of (float, float, int, int)
        Pools of the active set method, i.e. a sufficient statistics.
    t : int
        Number of processed time steps.
    h : array of float
        Explicit calcium kernel to avoid duplicated recalculations.
    d : float
        Decay factor. Only for AR(2).
    r : float
        Rise factor. Only for AR(2).
    g12, g11g11, g11g12 : arrays of float
        Precomputed quantitites related to the calcium kernel. Only for AR(2).

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """
    cdef:
        Py_ssize_t i
        SINGLE v, w, g, lam, s_min, b, g2, d, r, yt
        vector[Pool] P
        unsigned int t
        SINGLE[1000] h, g12, g11g11, g11g12  # assume kernel length <= 1000
        vector[SINGLE] _y

    def __init__(self, g, lam=0, s_min=0, b=0, num_empty_samples=0, g2=0):
        # save the parameters as attributes
        # self.p = len(np.ravel(g))
        cdef SINGLE lg, ld, lr
        cdef Py_ssize_t k
        cdef Pool newpool
        self.g = g
        self.g2 = g2
        self.lam = lam
        self.s_min = s_min
        self.b = b
        self.P = []
        # precompute
        if g2 == 0:  # AR(1)
            # calc explicit kernel h just once; length should be >=max ISI
            lg = log(g)
            for k in range(1000):
                self.h[k] = exp(lg * k)
            if num_empty_samples > 0:
                newpool.w = (1 - g**(2 * num_empty_samples)) / (1 - g * g)
                newpool.v, newpool.t, newpool.l = 0, 0, num_empty_samples
                self.P.push_back(newpool)
                self.t = num_empty_samples
                self.i = 0  # index of last pool
            else:
                self.t = 0
                self.i = -1
        else:  # AR(2)
            self.d = (g + sqrt(g * g + 4 * g2)) / 2
            self.r = (g - sqrt(g * g + 4 * g2)) / 2
            ld = log(self.d)
            if self.d == self.r:
                for k in range(1000):
                    self.h[k] = exp(ld * k) * (k + 1)
            else:
                lr = log(self.r)
                lg = self.d - self.r
                for k in range(1, 1001):
                    self.h[k - 1] = (exp(ld * k) - exp(lr * k)) / lg
            self.g12[0] = 0
            for k in range(1, 1000):
                self.g12[k] = g2 * self.h[k - 1]
            self.g11g11[0] = 1  # h[0] * h[0]
            self.g11g12[0] = 0  # h[0] * g12[0]
            for k in range(1, 1000):
                self.g11g11[k] = self.g11g11[k - 1] + self.h[k] * self.h[k]
                self.g11g12[k] = self.g11g12[k - 1] + self.h[k] * self.g12[k]
            if num_empty_samples > 0:
                newpool.v, newpool.w, newpool.t, newpool.l = 0, 0, 0, num_empty_samples
                self.P.push_back(newpool)
                self.t = num_empty_samples
                self.i = 0  # index of last pool
            else:
                self.t = 0
                self.i = -1
            self._y = [0] * num_empty_samples

    def fit_next(self, yt):
        """
        fit next time step t
        """
        cdef Pool newpool
        cdef Py_ssize_t j, k
        cdef np.ndarray[SINGLE, ndim = 1] h, g12, g11g11, g11g12
        cdef SINGLE tmp
        if self.g2 == 0:  # AR(1)
            newpool.v = yt - self.b - self.lam * (1 - self.g)
            newpool.w, newpool.t, newpool.l = 1, self.t, 1
            self.P.push_back(newpool)
            self.t += 1
            self.i += 1
            while (self.i > 0 and  # backtrack until violations fixed
                   (self.P[self.i - 1].v / self.P[self.i - 1].w * self.g**self.P[self.i - 1].l +
                    self.s_min > self.P[self.i].v / self.P[self.i].w)):
                self.i -= 1
                # merge two pools
                self.P[self.i].v += self.P[self.i + 1].v * self.g**self.P[self.i].l
                self.P[self.i].w += self.P[self.i + 1].w * self.g**(2 * self.P[self.i].l)
                self.P[self.i].l += self.P[self.i + 1].l
                self.P.pop_back()
        else:  # AR(2)
            self._y.push_back(yt - self.b - self.lam * (1 - self.g - self.g2))
            newpool.v = fmax(0, self._y[self.t])
            newpool.w, newpool.t, newpool.l = newpool.v, self.t, 1
            self.P.push_back(newpool)
            self.t += 1
            self.i += 1
            while (self.i > 0 and  # backtrack until violations fixed
                   (((((self.P[self.i - 1].v * self.d**(self.P[self.i - 1].l + 1) /
                        (self.d - self.r))
                       if self.d != self.r else
                       (self.P[self.i - 1].v * self.d**self.P[self.i - 1].l * \
                        (self.P[self.i - 1].l + 1) -
                        self.P[self.i - 2].w * self.d**(self.P[self.i - 1].l + 2) * \
                        (self.P[self.i - 1].l + 1)))
                       if self.P[self.i - 1].l >= 1000 else
                       (self.h[self.P[self.i - 1].l] * self.P[self.i - 1].v +
                        self.g12[self.P[self.i - 1].l] * self.P[self.i - 2].w)) >
                       self.P[self.i].v - self.s_min)
                    if self.i > 1 else
                    (self.P[self.i - 1].w * self.d > self.P[self.i].v - self.s_min))):
                self.i -= 1
                # merge two pools
                self.P[self.i].l += self.P[self.i + 1].l
                k = self.P[self.i].l - 1
                if self.i > 0:
                    if k >= 1000:
                        k = 999  # precomputed kernel shorter than ISI -> simply truncate
                    tmp = 0
                    for j in range(min1000(self.P[self.i].l)):
                        tmp += self.h[j] * self._y[self.P[self.i].t + j]
                    self.P[self.i].v = ((tmp - self.g11g12[k] * self.P[self.i - 1].w) /
                                        self.g11g11[k])
                    self.P[self.i].w = (self.h[k] * self.P[self.i].v +
                                        self.g12[k] * self.P[self.i - 1].w)
                else:  # update first pool
                    tmp = 0
                    for j in range(self.P[self.i].l):
                        tmp += self.d**j * self._y[j]
                    self.P[self.i].v = fmax(0, tmp * (1 - self.d * self.d) /
                                            (1 - self.d**(2 * self.P[self.i].l)))
                    self.P[self.i].w = self.d**k * self.P[self.i].v
                self.P.pop_back()

    def fit_next_tmp(self, yt, num):
        """
        fit next time step t temporarily and return denoised calcium for last num time steps
        """
        cdef Pool newpool
        cdef np.ndarray[SINGLE, ndim = 1] c
        cdef Py_ssize_t t, j, k, tmp2
        cdef SINGLE tmp, _yt
        if self.g2 == 0:  # AR(1)
            newpool.v = yt - self.b - self.lam * (1 - self.g)
            newpool.w, newpool.t, newpool.l = 1, self.t, 1
            j = self.i
            while (j >= 0 and  # backtrack until violations fixed
                   (self.P[j].v / self.P[j].w * self.g**self.P[j].l + self.s_min >
                    newpool.v / newpool.w)):
                # merge two pools
                newpool.v = self.P[j].v + newpool.v * self.g**self.P[j].l
                newpool.w = self.P[j].w + newpool.w * self.g**(2 * self.P[j].l)
                newpool.t = self.P[j].t
                newpool.l = self.P[j].l + newpool.l
                j -= 1
            # return deconvolved activity for last num time steps
            c = np.zeros(num, dtype='float32')
            t = num
            tmp = fmax(newpool.v, 0) / newpool.w
            if newpool.l <= t:  # safely assume t<=num=buffer<1000
                for k in range(newpool.l):
                    c[k + t - newpool.l] = tmp * self.h[k]
            else:
                tmp2 = 1000 - newpool.l + t
                for k in range(t if t < tmp2 else tmp2):
                    c[k] = tmp * self.h[k + newpool.l - t]
                # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                # if tmp2 <= 0:
                #     c[0] = tmp * self.g**(newpool.l - t)
                #     for k in range(1, t):
                #         c[k] = c[k - 1] * self.g
                # else:
                #     for k in range(tmp2, t):
                #         c[k] = c[k - 1] * self.g
            t -= newpool.l
            while t > 0:
                tmp = fmax(self.P[j].v, 0) / self.P[j].w
                if self.P[j].l <= t:  # safely assume t<=num=buffer<1000
                    for k in range(self.P[j].l):
                        c[k + t - self.P[j].l] = tmp * self.h[k]
                else:
                    tmp2 = 1000 - self.P[j].l + t
                    for k in range(t if t < tmp2 else tmp2):
                        c[k] = tmp * self.h[k + self.P[j].l - t]
                    # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                    # if tmp2 <= 0:
                    #     c[0] = tmp * self.g**(self.P[j].l - t)
                    #     for k in range(1, t):
                    #         c[k] = c[k - 1] * self.g
                    # else:
                    #     for k in range(tmp2, t):
                    #         c[k] = c[k - 1] * self.g
                t -= self.P[j].l
                j -= 1
        else:
            _yt = yt - self.b - self.lam * (1 - self.g - self.g2)
            newpool.v = fmax(0, _yt)
            newpool.w, newpool.t, newpool.l = newpool.v, self.t, 1
            j = self.i
            while (j >= 0 and  # backtrack until violations fixed
                   ((((self.P[j].v * self.d**(self.P[j].l + 1) / (self.d - self.r)
                       if self.d != self.r else
                       (self.P[j].v * self.d**self.P[j].l * (self.P[j].l + 1)) -
                       self.P[j - 1].w * self.d**(self.P[j].l + 2) * (self.P[j].l + 1))
                       if self.P[j].l >= 1000 else
                       (self.h[self.P[j].l] * self.P[j].v +
                        self.g12[self.P[j].l] * self.P[j - 1].w)) >
                       newpool.v - self.s_min)
                    if j > 0 else
                    (self.P[j].w * self.d > newpool.v - self.s_min))):
                # merge two pools
                newpool.l += self.P[j].l
                newpool.t = self.P[j].t
                k = newpool.l - 1
                if j > 0:
                    if k >= 1000:
                        k = 999  # precomputed kernel shorter than ISI -> simply truncate
                    tmp = 0
                    for tmp2 in range(k):
                        tmp += self.h[tmp2] * self._y[newpool.t + tmp2]
                    # tmp += self.h[k] * (self._y[newpool.t + k] if newpool.l > 1000 else _yt)
                    tmp += self.h[k] * (self._y[newpool.t + k] if newpool.t +
                                        k < self._y.size() else _yt)
                    newpool.v = (tmp - self.g11g12[k] * self.P[j-1].w) / self.g11g11[k]
                    newpool.w = self.h[k] * newpool.v + self.g12[k] * self.P[j-1].w
                else:  # update first pool
                    tmp = 0
                    for tmp2 in range(k):
                        tmp += self.d**tmp2 * self._y[tmp2]
                    tmp += self.d**k * self._yt
                    newpool.v = fmax(0, tmp * (1 - self.d * self.d) /
                                     (1 - self.d**(2 * newpool.l)))
                    newpool.w = self.d**k * newpool.v
                j -= 1
            # return deconvolved activity for last num time steps
            c = np.zeros(num, dtype='float32')
            t = num

            if j == -1:  # newpool is first and only pool
                c[0] = newpool.v * self.d**(newpool.l - t)
                for k in range(1, t):
                    c[k] = c[k - 1] * self.d
            elif newpool.l <= t:  # safely assume t<=num=buffer<1000
                for k in range(newpool.l):
                    c[k + t - newpool.l] = self.h[k] * newpool.v + \
                        self.g12[k] * self.P[j].w
            else:
                tmp2 = 1000 - newpool.l + t
                for k in range(t if t < tmp2 else tmp2):
                    c[k] = self.h[k + newpool.l - t] * newpool.v + \
                        self.g12[k + newpool.l - t] * self.P[j].w
                # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                # if tmp2 <= 0:
                #     c[0] = newpool.v * self.d**(newpool.l - t + 1) / (self.d - self.r)
                #     for k in range(1, t):
                #         c[k] = c[k - 1] * self.d
                # else:
                #     for k in range(tmp2, t):
                #         c[k] = c[k - 1] * self.d
            t -= newpool.l
            while t > 0:
                if j == 0:  # first pool
                    c[0] = self.P[0].v * self.d**(self.P[0].l - t)
                    for k in range(1, t):
                        c[k] = c[k - 1] * self.d
                elif self.P[j].l <= t:  # safely assume t<=num=buffer<1000
                    for k in range(self.P[j].l):
                        c[k + t - self.P[j].l] = self.h[k] * self.P[j].v + \
                            self.g12[k] * self.P[j - 1].w
                else:
                    tmp2 = 1000 - self.P[j].l + t
                    for k in range(t if t < tmp2 else tmp2):
                        c[k] = self.h[k + self.P[j].l - t] * self.P[j].v + \
                            self.g12[k + self.P[j].l - t] * self.P[j - 1].w
                    # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                    # if tmp2 <= 0:
                    #     c[0] = self.P[j].v * self.d**(self.P[j].l - t + 1) / (self.d - self.r)
                    #     for k in range(1, t):
                    #         c[k] = c[k - 1] * self.d
                    # else:
                    #     for k in range(tmp2, t):
                    #         c[k] = c[k - 1] * self.d
                t -= self.P[j].l
                j -= 1

        return c

    def fit(self, y):
        """
        fit all time steps
        """
        for yt in y:
            self.fit_next(yt)
        return self

    def get_c(self, num):
        """
        return denoised calcium for last num time steps
        """
        cdef np.ndarray[SINGLE, ndim = 1] c
        cdef Py_ssize_t t, j, k, tmp2
        cdef SINGLE tmp
        t = num
        c = np.zeros(t, dtype='float32')
        j = self.i
        if self.g2 == 0:  # AR(1)
            while t > 0:
                tmp = fmax(self.P[j].v / self.P[j].w, 0)
                if self.P[j].l <= t:
                    for k in range(min1000(self.P[j].l)):
                        c[k + t - self.P[j].l] = tmp * self.h[k]
                    # # more accurate exponential decay, but maybe simple truncation suffices
                    # for k in range(1000 + t - self.P[j].l, t):
                    #     c[k] = c[k - 1] * self.g
                else:
                    tmp2 = 1000 - self.P[j].l + t
                    for k in range(t if t < tmp2 else tmp2):
                        c[k] = tmp * self.h[k + self.P[j].l - t]
                    # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                    # if tmp2 <= 0:
                    #     c[0] = tmp * self.g**(self.P[j].l - t)
                    #     for k in range(1, t):
                    #         c[k] = c[k - 1] * self.g
                    # else:
                    #     for k in range(tmp2, t):
                    #         c[k] = c[k - 1] * self.g
                t -= self.P[j].l
                j -= 1
        else:  # AR(2)
            while t > 0:
                if j == 0:  # first pool
                    for k in range(t):
                        c[k] = self.P[0].v * self.d**(k + self.P[0].l - t)
                elif self.P[j].l <= t:
                    for k in range(min1000(self.P[j].l)):
                        c[k + t - self.P[j].l] = self.h[k] * self.P[j].v + \
                            self.g12[k] * self.P[j - 1].w
                    # # more accurate exponential decay, but maybe simple truncation suffices
                    # for k in range(1000, self.P[j].l):
                    #     c[k] = c[k - 1] * self.d
                else:
                    tmp2 = 1000 - self.P[j].l + t
                    for k in range(t if t < tmp2 else tmp2):
                        c[k] = self.h[k + self.P[j].l - t] * self.P[j].v + \
                            self.g12[k + self.P[j].l - t] * self.P[j - 1].w
                    # # more accurate exponential decay, but maybe simple truncation c[k]=0 suffices
                    # if tmp2 <= 0:
                    #     c[0] = self.P[j].v * self.d**(self.P[j].l - t + 1) / (self.d - self.r)
                    #     for k in range(1, t):
                    #         c[k] = c[k - 1] * self.d
                    # else:
                    #     for k in range(tmp2, t):
                    #         c[k] = c[k - 1] * self.d
                t -= self.P[j].l
                j -= 1
        return c

    def get_s(self, num):
        """
        return deconvolved activity for last num time steps
        """
        cdef np.ndarray[SINGLE, ndim = 1] s
        cdef Py_ssize_t t, j
        j = self.i
        t = num - self.P[j].l
        s = np.zeros(num, dtype='float32')
        if self.g2 == 0:  # AR(1)
            while t >= (1 if num == self.t else 0):
                s[t] = self.P[j].v / self.P[j].w - \
                    self.P[j - 1].v / self.P[j - 1].w * self.g**self.P[j - 1].l
                j -= 1
                t -= self.P[j].l
        else:  # AR(2)
            while t > 0:
                if j > 1:
                    if self.P[j - 1].l >= 1000:
                        s[t] = self.P[j].v
                        # # more accurate exponential decay
                        # s[t] = self.P[j].v - (self.d**(self.P[j - 1].l+1) /
                        #                       (self.d - self.r) * self.P[j - 1].v)
                    else:
                        s[t] = self.P[j].v - (self.h[self.P[j - 1].l] * self.P[j - 1].v +
                                              self.g12[self.P[j - 1].l] * self.P[j - 2].w)
                else:  # j==1; spike at the beginning of 2nd pool
                    s[t] = (self.P[1].v - self.P[0].w * self.d)
                    return s
                j -= 1
                t -= self.P[j].l
        return s

    def get_l_of_last_pool(self):
        """
        return length of last pool
        """
        return self.P[self.i].l

    def get_c_of_last_pool(self):
        """
        return denoised calcium of last pool, i.e. the part of c that actually changed
        """
        cdef np.ndarray[SINGLE, ndim = 1] c
        cdef Py_ssize_t k
        cdef SINGLE tmp
        c = np.zeros(self.P[self.i].l, dtype='float32')
        if self.g2 == 0:  # AR(1)
            tmp = self.P[self.i].v / self.P[self.i].w
            for k in range(min1000(self.P[self.i].l)):
                c[k] = tmp * self.h[k]
            # # more accurate exponential decay instead truncation
            # for k in range(1000, self.P[self.i].l):
            #     c[k] = c[k - 1] * self.g
        else:  # AR(2)
            if self.i == 0:  # first pool
                c[0] = self.P[0].v
                for k in range(1, self.P[0].l):
                    c[k] = c[k - 1] * self.d
            else:
                for k in range(min1000(self.P[self.i].l)):
                    c[k] = self.h[k] * self.P[self.i].v + self.g12[k] * self.P[self.i - 1].w
                # # more accurate exponential decay instead truncation
                # for k in range(1000, self.P[self.i].l):
                #     c[k] = c[k - 1] * self.d
        return c

    def remove_last_pool(self):
        cdef Py_ssize_t k
        self.t -= self.P[self.i].l
        if self.g2 != 0:
            for k in range(self.P[self.i].l):
                self._y.pop_back()
        self.i -= 1
        self.P.pop_back()

    def get_l_of_pool(self, idx_from_end=0):
        return self.P[self.i - idx_from_end].l

    def set_poolvalue(self, val, idx_from_end=0):
        self.P[self.i - idx_from_end].v = val

    @property
    def P(self):
        cdef Py_ssize_t j
        tmp = []
        for j in range(self.i+1):
            tmp.append([self.P[j].v, self.P[j].w, self.P[j].t, self.P[j].l])
        return tmp

    @property
    def g(self):
        return self.g

    @property
    def g2(self):
        return self.g2

    @property
    def lam(self):
        return self.lam

    @property
    def s_min(self):
        return self.s_min

    @property
    def b(self):
        return self.b

    @property
    def t(self):
        return self.t

    @property
    def c(self):
        """
        construct and return full calcium trace
        """
        cdef np.ndarray[SINGLE, ndim = 1] c
        cdef Py_ssize_t j, k
        cdef SINGLE tmp
        if self.g2 == 0:  # AR(1)
            c = np.zeros(self.P[self.i].t + self.P[self.i].l, dtype='float32')
            for j in range(self.i + 1):
                tmp = fmax(self.P[j].v, 0) / self.P[j].w
                for k in range(min1000(self.P[j].l)):
                    c[k + self.P[j].t] = tmp * self.h[k]
                # # more accurate exponential decay instead truncation
                # for k in range(1000, self.P[j].l):
                #     c[k + self.P[j].t] = tmp * self.g**k
            return c
        else:  # AR(2)
            c = np.zeros(self.P[self.i].t + self.P[self.i].l, dtype='float32')
            # first pool
            c[0] = self.P[0].v
            for k in range(1, self.P[0].l):
                c[k] = c[k - 1] * self.d
            # remaining pools
            for j in range(1, self.i + 1):
                for k in range(min1000(self.P[j].l)):
                    c[k + self.P[j].t] = self.h[k] * self.P[j].v + self.g12[k] * self.P[j - 1].w
                # # more accurate exponential decay instead truncation
                # for k in range(1000, self.P[j].l):
                #     c[k + self.P[j].t] = c[k + self.P[j].t - 1] * self.d
            return c

    @property
    def s(self):
        """
        construct and return full deconvolved activity, 'spike rates'
        """
        cdef np.ndarray[SINGLE, ndim = 1] s
        cdef Py_ssize_t j
        if self.g2 == 0:  # AR(1)
            s = np.zeros(self.P[self.i].t + self.P[self.i].l, dtype='float32')
            for j in range(self.i):
                s[self.P[j + 1].t] = self.P[j + 1].v / self.P[j + 1].w - \
                    self.P[j].v / self.P[j].w * self.g**self.P[j].l
            return s
        else:  # AR(2)
            return self.get_s(self.P[self.i].t + self.P[self.i].l)


@cython.cdivision(True)
def oasisAR1(np.ndarray[SINGLE, ndim=1] y, SINGLE g, SINGLE lam=0, SINGLE s_min=0):
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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, j, k, t, T
        SINGLE tmp
        np.ndarray[SINGLE, ndim= 1] c, s
        vector[Pool] P
        Pool newpool

    T = len(y)
    # [value, weight, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = y[0] - lam * (1 - g), 1, 0, 1
    P.push_back(newpool)
    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v = y[t] - lam * (1 if t == T - 1 else (1 - g))
        newpool.w, newpool.t, newpool.l = 1, t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
               (P[i - 1].v / P[i - 1].w * g**P[i - 1].l + s_min > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i + 1].v * g**P[i].l
            P[i].w += P[i + 1].w * g**(2 * P[i].l)
            P[i].l += P[i + 1].l
            P.pop_back()
    # construct c
    c = np.empty(T, dtype=np.float32)
    for j in range(i + 1):
        tmp = fmax(P[j].v, 0) / P[j].w
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s


@cython.cdivision(True)
def constrained_oasisAR1(np.ndarray[SINGLE, ndim=1] y, SINGLE g, SINGLE sn,
                         bool optimize_b=False, bool b_nonneg=True, int optimize_g=0,
                         int decimate=1, int max_iter=5, int penalty=1, SINGLE s_min=-3):
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
    s_min : float, optional, default -3
        Minimal non-zero activity within each bin (minimal 'spike size').
        For negative values the threshold is |s_min| * sn * sqrt(1-g)
        If 0 the threshold is determined automatically such that RSS <= sn^2 T

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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, j, k, t, l
        unsigned int ma, count, T
        SINGLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi
        bool g_converged
        np.ndarray[SINGLE, ndim = 1] c, s, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll
        vector[Pool] P
        Pool newpool

    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        g = g**decimate
        thresh = thresh / decimate / decimate
        T = len(y)
    # explicit kernel, useful for constructing solution
    h = g ** np.arange(T, dtype=np.float32)
    c = np.empty(T, dtype=np.float32)
    lam = 0  # sn/sqrt(1-g*g)

    def oasis1strun(np.ndarray[SINGLE, ndim=1] y, SINGLE g, np.ndarray[SINGLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k, t, T
            SINGLE tmp
            vector[Pool] P
            Pool newpool

        T = len(y)
        # [value, weight, start time, length] of pool
        # newpool.v, newpool.w, newpool.t, newpool.l = y[0] - lam * (1 - g), 1, 0, 1
        newpool.v, newpool.w, newpool.t, newpool.l = y[0], 1, 0, 1
        P.push_back(newpool)
        i = 0  # index of last pool
        t = 1  # number of time points added = index of next data point
        while t < T:
            # add next data point as pool
            newpool.v = y[t]  # - lam * (1 if t == T - 1 else (1 - g))
            newpool.w, newpool.t, newpool.l = 1, t, 1
            P.push_back(newpool)
            t += 1
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i - 1].v / P[i - 1].w * g**P[i - 1].l > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i + 1].v * g**P[i].l
                P[i].w += P[i + 1].w * g**(2 * P[i].l)
                P[i].l += P[i + 1].l
                P.pop_back()
        # construct c
        c = np.empty(T, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    def oasis(vector[Pool] P, SINGLE g, np.ndarray[SINGLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k
            SINGLE tmp

        i = 0
        while i < P.size() - 1:
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i - 1].v / P[i - 1].w * g**P[i - 1].l > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i + 1].v * g**P[i].l
                P[i].w += P[i + 1].w * g**(2 * P[i].l)
                P[i].l += P[i + 1].l
                P.erase(P.begin() + i + 1)
        # construct c
        c = np.empty(P[P.size() - 1].t + P[P.size() - 1].l, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    g_converged = False
    count = 0
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g>0
        c, P = oasis1strun(y, g, c)
        tmp = np.empty(T, dtype=np.float32)
        res = y - c
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and c.sum() > 1e-9:
            # update lam
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    # faster than tmp[P[i].t:P[i].t + P[i].l] = 1 / P[i].w * h[:P[i].l]
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - g**P[i].l) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for i in range(P.size() - 1):  # perform shift
                P[i].v -= dlam * (1 - g**P[i].l)
            P[P.size() - 1].v -= dlam  # correct last pool; |s|_1 instead |c|_1
            c, P = oasis(P, g, c)

            # update g
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, g, Pt, Pl):
                    h = np.exp(log(g) * np.arange(ma, dtype=np.float32))

                    def foo(y, t, l, q, g, lam=lam):
                        yy = y[t:t + l]
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - g**(2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**l)) * (1 - g * g) /
                                   (1 - g**(2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], g)
                                for i in range(optimize_g)])

                def baz(y, Pt, Pl):
                    # minimizes residual
                    return fminbound(lambda x: bar(y, x, Pt, Pl), 0, 1, xtol=1e-4, maxfun=50)
                aa = baz(y, Pt, Pl)
                if abs(aa - g) < 1e-4:
                    g_converged = True
                g = aa
                # explicit kernel, useful for constructing c
                h = np.exp(log(g) * np.arange(T, dtype=np.float32))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - lam * (1 - g**P[i].l)
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * g**P[P.size() - 1].l  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:  # optimize b and dependent on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = fmax(b, 0)
        c, P = oasis1strun(y - b, g, c)
        # update b and lam
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        i = P.size() - 1
        P[i].v -= lam * g**P[i].l  # |s|_1 instead |c|_1
        c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
        # calc RSS
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(T, dtype=np.float32)
        # until noise constraint is tight or spike train is empty or max_iter reached
        while fabs(RSS - thresh) > thresh * 1e-4 and c.sum() > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - g**P[i].l) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            tmp -= 1. / T / (1 - g) * np.sum([(1 - g**P[i].l) ** 2 / P[i].w
                                              for i in range(P.size())])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = fmax(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for i in range(P.size()):  # perform shift
                P[i].v -= dphi * (1 - g**P[i].l)
            c, P = oasis(P, g, c)
            # update b and lam
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            i = P.size() - 1
            P[i].v -= dlam * g**P[i].l  # |s|_1 instead |c|_1
            c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, opt, Pt, Pl):
                    b, g = opt
                    h = np.exp(log(g) * np.arange(ma, dtype=np.float32))

                    def foo(y, t, l, q, b, g, lam=lam):
                        yy = y[t:t + l] - b
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - g**(2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**l)) * (1 - g * g) /
                                   (1 - g**(2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], b, g)
                                for i in range(P.size() if P.size() < optimize_g else optimize_g)])

                def baz(y, Pt, Pl):
                    return minimize(lambda x: bar(y, x, Pt, Pl), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, Pt, Pl)
                if fabs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                # explicit kernel, useful for constructing c
                h = np.exp(log(g) * np.arange(T, dtype=np.float32))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - \
                        (b / (1 - g) + lam) * (1 - g**P[i].l)
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * g**P[P.size() - 1].l  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
                # update b and lam
                db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                i = P.size() - 1
                P[i].v -= dlam * g**P[i].l  # |s|_1 instead |c|_1
                c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # calc RSS
            res = y - c - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam *= (1 - g)
        g = g**(1. / decimate)
        lam /= (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.ravel([P[i].t * decimate + np.arange(-decimate, 3 * decimate / 2)
                       for i in range(P.size())])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        h = np.exp(log(g) * np.arange(T, dtype=np.float32))
        P.resize(0)
        for i in range(len(ff)):
            q = h[:ll[i]]
            newpool.v = q.dot(fluor[ff[i]:ff[i] + ll[i]]) - (b / (1 - g) + lam) * (1 - g**ll[i])
            newpool.w = q.dot(q)
            newpool.t = ff[i]
            newpool.l = ll[i]
            P.push_back(newpool)
        P[P.size() - 1].v -= lam * g**P[P.size() - 1].l  # |s|_1 instead |c|_1
        c = np.empty(T, dtype=np.float32)

        c, P = oasis(P, g, c)

    if penalty == 0:  # get (locally optimal) L0 solution
        y = y - b
        if s_min == 0:
            lls = [(P[i + 1].v / P[i + 1].w - P[i].v / P[i].w * g**P[i].l)
                   for i in range(P.size() - 1)]
            pos = [P[i + 1].t for i in np.argsort(lls)[::-1]]
            res = -y
            RSS = y.dot(y)
            c = np.zeros_like(y)
            P.resize(0)
            newpool.v, newpool.w, newpool.t, newpool.l = 0, 1, 0, len(y)
            P.push_back(newpool)
            for p in pos:
                i = 0
                while P[i].t + P[i].l <= p:
                    i += 1
                # split current pool at pos
                j, k = P[i].t, P[i].l
                q = h[:j - p + k]
                newpool.v = q.dot(y[p:j + k])
                newpool.w, newpool.t, newpool.l = q.dot(q), p, j - p + k
                P.insert(P.begin() + i + 1, newpool)
                q = h[:p - j]
                P[i].v, P[i].w, P[i].t, P[i].l = q.dot(y[j:p]), q.dot(q), j, p - j
                for t in [i, i + 1]:
                    c[P[t].t:P[t].t + P[t].l] = fmax(0, P[t].v) / P[t].w * h[:P[t].l]
                # calc RSS
                RSS -= res[j:j + k].dot(res[j:j + k])
                res[P[i].t:j + k] = c[P[i].t:j + k] - y[P[i].t:j + k]
                RSS += res[P[i].t:j + k].dot(res[P[i].t:j + k])
                if RSS < thresh:
                    break
        else:
            c, s = oasisAR1(y, g, s_min=s_min if s_min > 0 else -s_min * sn * sqrt(1 - g))

    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s, b, g, lam
