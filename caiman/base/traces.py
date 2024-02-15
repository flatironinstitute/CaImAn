#!/usr/bin/env python

import cv2
import logging
import numpy as np
import pylab as pl
pl.ion()

import caiman.base.timeseries

try:
    cv2.setNumThreads(0)
except:
    pass

################
# This holds the trace class, which is a specialised Caiman timeseries class.


class trace(caiman.base.timeseries.timeseries):
    """
    Class representing a trace.

    Example of usage

    TODO

    Args:
        input_trace: np.ndarray (time x ncells)
        start_time: time beginning trace
        fr: frame rate
        meta_data: dictionary including any custom meta data
    """

    def __new__(cls, input_arr, **kwargs):
        return super(trace, cls).__new__(cls, input_arr, **kwargs)

    @staticmethod
    def load(file_name):
        """
        load movie from file
        """
        return trace(**np.load(file_name))

    def computeDFF(self, window_sec=5, minQuantile=20):
        """
        compute the DFF of the movie

        In order to compute the baseline frames are binned according to the window length parameter
        and then the intermediate values are interpolated.

        Args:
            secsWindow: length of the windows used to compute the quantile
            quantilMin : value of the quantile

        Raises:
            ValueError "All traces must be positive"
            ValueError "The window must be shorter than the total length"
        """
        if np.min(self) <= 0:
            raise ValueError("All traces must be positive")

        T, _ = self.shape
        window = int(window_sec * self.fr)
        logging.debug(window)
        if window >= T:
            raise ValueError("The window must be shorter than the total length")

        tracesDFF = []
        for tr in self.T:
            logging.debug(f"TR Shape is {tr.shape}")
            traceBL = [np.percentile(tr[i:i + window], minQuantile) for i in range(1, len(tr) - window)]
            missing = np.percentile(tr[-window:], minQuantile)
            missing = np.repeat(missing, window + 1)
            traceBL = np.concatenate((traceBL, missing))
            tracesDFF.append((tr - traceBL) / traceBL)

        return self.__class__(np.asarray(tracesDFF).T, **self.__dict__)

    def resample(self, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        raise Exception('Not Implemented. Look at movie resize')

    def plot(self, stacked=True, subtract_minimum=False, cmap=pl.cm.jet, **kwargs):
        """Plot the data

        author: ben deverett

        Args:
            stacked : bool
                for multiple columns of data, stack instead of overlaying
            subtract_minimum : bool
                subtract minimum from each individual trace
            cmap : matplotlib.LinearSegmentedColormap
                color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
            kwargs : dict
                any arguments accepted by matplotlib.plot

        Returns:
            The matplotlib axes object corresponding to the data plot
        """

        d = self.copy()
        n = 1  # number of traces
        if len(d.shape) > 1:
            n = d.shape[1]

        ax = pl.gca()

        colors = cmap(np.linspace(0, 1, n))
        ax.set_color_cycle(colors)

        if subtract_minimum:
            d -= d.min(axis=0)
        if stacked and n > 1:
            d += np.append(0, np.cumsum(d.max(axis=0))[:-1])
        ax.plot(self.time, d, **kwargs)

        # display trace labels along right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.atleast_1d(d.mean(axis=0)))
        ax2.set_yticklabels([str(i) for i in range(n)], weight='bold')
        [l.set_color(c) for l, c in zip(ax2.get_yticklabels(), colors)]

        pl.gcf().canvas.draw()
        return ax

    def extract_epochs(self, trigs=None, tb=1, ta=1):
        raise Exception('Not Implemented. Look at movie resize')

