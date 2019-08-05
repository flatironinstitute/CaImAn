import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
import math

#%% INITIAL RESULTS SCREENING

def check_eval_results(cnm, idx):
    """Checks results of component evaluation and determines why the component got rejected or accepted

    Args:
        cnm:                caiman CNMF object containing estimates and evaluate_components() results

        idx:                int or iterable (array, list...)
                            index or list of indices of components to be checked

    Returns:
        printout of evaluation results
    """
    try:
        iter(idx)
        idx = list(idx)
    except:
        idx = [idx]

    snr_min = cnm.params.quality['SNR_lowest']
    snr_max = cnm.params.quality['min_SNR']
    r_min = cnm.params.quality['rval_lowest']
    r_max = cnm.params.quality['rval_thr']
    cnn_min = cnm.params.quality['cnn_lowest']
    cnn_max = cnm.params.quality['min_cnn_thr']

    for i in range(len(idx)):
        snr = cnm.estimates.SNR_comp[idx[i]]
        r = cnm.estimates.r_values[idx[i]]
        cnn = cnm.estimates.cnn_preds[idx[i]]
        cnn_round = str(round(cnn, 2))

        red_start = '\033[1;31;49m'
        red_end = '\033[0;39;49m'

        green_start = '\033[1;32;49m'
        green_end = '\033[0;39;49m'

        upper_thresh_failed = 0
        lower_thresh_failed = False

        print(f'Checking component {idx[i]+1}...')
        if idx[i] in cnm.estimates.idx_components:
            print(green_start+f'\nComponent {idx[i]+1} got accepted, all lower threshold were passed!'+green_end+'\n\n\tUpper thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds threshold of {snr_max}\n'+green_end)
            else:
                print(f'\tSNR of {round(snr,2)} does not exceed threshold of {snr_max}\n')

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds threshold of {r_max}\n'+green_end)
            else:
                print(f'\tR-value of {round(r,2)} does not exceed threshold of {r_max}\n')

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            else:
                print('\tCNN-value of '+cnn_round+f' does not exceed threshold of {cnn_max}\n')
            print(f'\n')

        else:
            print(f'\nComponent {idx[i] + 1} did not get accepted. \n\n\tChecking thresholds:\n')

            if snr >= snr_max:
                print(green_start+f'\tSNR of {round(snr,2)} exceeds upper threshold of {snr_max}\n'+green_end)
            elif snr >= snr_min and snr < snr_max:
                print(f'\tSNR of {round(snr,2)} exceeds lower threshold of {snr_min}, but not upper threshold of {snr_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+f'\tSNR of {round(snr,2)} does not pass lower threshold of {snr_min}\n'+red_end)
                lower_thresh_failed = True

            if r >= r_max:
                print(green_start+f'\tR-value of {round(r,2)} exceeds upper threshold of {r_max}\n'+green_end)
            elif r >= r_min and r < r_max:
                print(f'\tR-value of {round(r,2)} exceeds lower threshold of {r_min}, but not upper threshold of {r_max}\n')
                upper_thresh_failed += 1
            else:
                print(f'\tR-value of {round(r,2)} does not pass lower threshold of {r_min}\n')
                lower_thresh_failed = True

            if cnn >= cnn_max:
                print(green_start+'\tCNN-value of '+cnn_round+f' exceeds threshold of {cnn_max}\n'+green_end)
            elif cnn >= cnn_min and cnn < cnn_max:
                print('\tCNN-value of '+cnn_round+f' exceeds lower threshold of {cnn_min}, but not upper threshold of {cnn_max}\n')
                upper_thresh_failed += 1
            else:
                print(red_start+'\tCNN-value of '+cnn_round+f' does not pass lower threshold of {cnn_min}\n'+red_end)
                lower_thresh_failed = True

            if lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it failed at least one lower threshold!\n\n'+red_end)
            elif upper_thresh_failed == 3 and not lower_thresh_failed:
                print(red_start+f'Result: Component {idx[i]+1} got rejected because it met all lower, but no upper thresholds!\n\n'+red_end)
            else:
                print('This should not appear, check code logic!\n\n')


def plot_component_traces(cnm, idx, param='F_dff', comp_array=None):
    """
    Plots components as traces and in a color-graded graph (pcolormesh)

    Args:
        cnm:        CNMF object that includes the estimates object
        idx:        indices of components to be plotted. Can also be 'all', in which case all traces are plotted.
        param:      Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                    detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.
        comp_array: Previously calculated data arrays with individual components in rows and measurements in columns
                    (same format as estimates object)

    Returns:
        Line plot and color-coded pcolormesh of individual component traces
    """
    # plots components in traces and color-graded
    # comp_array is a 2D array with single components in rows and measurements in columns
    if comp_array is not None:
        traces = comp_array
    else:
        try:
            if idx == 'all':
                traces = getattr(cnm.estimates, param)
                print('Plotting traces of all components... \n')
            elif:
                traces = getattr(cnm.estimates, param)
                traces = traces[idx]
        except NameError:
            print('Could find no component data! Run pipeline or load old results!\n')
            return

    # plot components
    trace_fig, trace_ax = plt.subplots(nrows=traces.shape[0], ncols=2, sharex=True, figsize=(20, 12))
    trace_fig.suptitle(f'Parameter {param} of selected components', fontsize=16)
    for i in range(traces.shape[0]):
        curr_trace = traces[i, np.newaxis]
        trace_ax[i, 1].pcolormesh(curr_trace)
        trace_ax[i, 0].plot(traces[i])
        if i == trace_ax[:, 0].size - 1:
            trace_ax[i, 0].spines['top'].set_visible(False)
            trace_ax[i, 0].spines['right'].set_visible(False)
            trace_ax[i, 0].set_yticks([])
            trace_ax[i, 1].spines['top'].set_visible(False)
            trace_ax[i, 1].spines['right'].set_visible(False)
        else:
            trace_ax[i, 0].axis('off')
            trace_ax[i, 1].axis('off')
        trace_ax[i, 0].set_title(f'{i+1}', x=-0.02, y=-0.4)
    trace_ax[i, 0].set_xlim(0, 1000)
    trace_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # trace_fig.tight_layout()
    plt.show()

#%% CORRELATION FUNCTIONS


def half_correlation_matrix(matrix):
    """
    Halves a full correlation matrix to remove double values

    Args:
        matrix: np.array or Pandas Dataframe of a correlation matrix
    Returns:
        Same matrix, with the top right half set to NaN
    """
    # halves a full correlation matrix to remove double values
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    masked_matrix = pd.DataFrame(np.ma.masked_array(matrix, mask=mask))
    return masked_matrix


def correlate_components(cnm, param='F_dff', half=True):
    """
    Calculates a simple correlation matrix.

    Args:
        cnm:    CNMF object that includes the estimates object
        param:  Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.
        half:   Flag whether the full matrix should be calculated, or if the redundant half can be omitted (default)

    Returns:
        Correlation matrix as Pandas Dataframe
    """

    traces = pd.DataFrame(np.transpose(getattr(cnm.estimates, param)))
    if half:
        trace_corr = half_correlation_matrix(traces.corr())
    else:
        trace_corr = traces.corr()
    return trace_corr


def plot_correlation_matrix(cnm, param='F_dff', half=True):
    """
    Plots a simple color-coded correlation matrix.

    Args:
        cnm:    CNMF object that includes the estimates object
        param:  Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.
        half:   Flag whether the full matrix should be plotted, or if the redundant half can be omitted (default)

    Returns:
        Color-coded seaborn heatmap plot of the correlation matrix.
    """

    trace_corr = correlate_components(cnm, param=param, half=half)
    sns.heatmap(trace_corr)


def plot_correlated_traces(cnm, thresh=0.7, param='F_dff',corr=''):
    """
    Plots components whose traces are highly correlating with overlaid traces and a heatmap

    Args:
        cnm:    CNMF object that includes the estimates object
        thresh: int or float, threshold value above which two traces are considered correlated. Default 0.4.
        param:  Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.
        corr:   a full or half correlation matrix. If not provided, it will be calculated with correlate_components()

    Returns:
        Plots highly correlating trace pairs overlaid as well as in a corresponding heatmap on top of each other.
    """

    if corr == '':
        corr = correlate_components(cnm, param)
        print("Using pairwise Pearson's correlation coefficient...\n")
    else:
        print('Using correlation provided by the user...\n')

    # load the component traces
    traces = getattr(cnm.estimates, param)

    # find all component pairs that have a correlation coefficient higher than thresh
    high_corr = np.where(corr >= thresh)
    pair_idx = tuple(zip(high_corr[0], high_corr[1]))

    # plot
    corr_plot, corr_ax = plt.subplots(nrows=len(high_corr[0]), ncols=2, sharex=True,
                                      figsize=(50, 12))  # create figure+subplots
    for i in range(len(high_corr[0])):
        # For each correlating component-pair...
        curr_idx_1 = pair_idx[i][0]
        curr_idx_2 = pair_idx[i][1]
        # ... first plot the calcium traces on the same figure
        corr_ax[i, 0].plot(traces[curr_idx_1], lw=1)
        corr_ax[i, 0].plot(traces[curr_idx_2], lw=1)
        plt.sca(corr_ax[i,0])
        plt.text(-200, 0, f'Comp {curr_idx_1} & {curr_idx_2}\n(r = {round(corr[curr_idx_1][curr_idx_2],2)})')
        # ... then plot the color-graded activity below each other
        trace_pair = np.vstack((traces[curr_idx_1], traces[curr_idx_2]))
        corr_ax[i, 1].pcolormesh(trace_pair)
        corr_ax[i, 1].set_ylim(0, 2)

        # adjust graph layout
        if i == corr_ax[:, 0].size - 1:
            corr_ax[i, 0].spines['top'].set_visible(False)
            corr_ax[i, 0].spines['right'].set_visible(False)
            corr_ax[i, 1].spines['top'].set_visible(False)
            corr_ax[i, 1].spines['right'].set_visible(False)
        else:
            corr_ax[i, 0].axis('off')
            corr_ax[i, 1].axis('off')
    corr_ax[i, 0].set_xlim(0, 1000)
    #corr_plot.constrained_layout()
    plt.show()


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of neural traces in C, controlling
    for the remaining traces in C.

    Args:
        C : array-like, shape (n, p)
            Array with the different component traces p with trace length n.

    Returns:
        P : array-like, shape (p, p)
            P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
            for the remaining traces in C.
    """

    C = np.asarray(C)
    C = np.column_stack([C, np.ones(C.shape[0])]) # add a column of 1s as an intercept
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_j)
            res_i = C[:, i] - C[:, idx].dot(beta_i)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def partial_cross_correlation(cnm, thresh=0.7, param='F_dff'):
    """
    Calculates and plots cross correlation of trace pairs, while correcting for activity of all other neurons to
    eliminate fluctuations in the general population.

    Args:
        cnm:    CNMF object that includes the estimates object
        thresh: int or float, threshold value above which two traces are considered correlated. Default 0.4.
        param:  Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.

    Returns:
        Plots a half-correlation matrix as well as overlaid traces of correlating trace pairs with corresponding heatmap.
    """
    # initialize correlation matrix
    traces = getattr(cnm.estimates, param)
    n_comp = traces.shape[0]
    part_corr = np.zeros((n_comp, n_comp))

    # calculate the partial correlation for each pair of neurons while controlling for the activity of all other neurons
    for i in range(n_comp):
        for j in range(n_comp):
            # this is done for every pair of neurons (also with itself)

            # get mean fluorescence of the remaining neuron population
            pop_mean = np.delete(traces, [i, j], axis=0).mean(axis=0)

            # calculate partial correlation of i and j, while controlling for pop_mean and put it in the corresponding
            # place in the array
            corr_in = np.transpose(np.vstack((traces[i], traces[j], pop_mean)))
            part_corr[i, j] = partial_corr(corr_in)[1, 0]

    sns.heatmap(half_correlation_matrix(part_corr))

    plot_correlated_traces(cnm, thresh=thresh, param=param, corr=np.asarray(half_correlation_matrix(part_corr)))


def plot_cross_correlation(cnm, thresh=0.7, param='F_dff'):
    """
    Plots cross correlation of trace pairs as a function of lag

    Args:
        cnm:    CNMF object that includes the estimates object
        thresh: int or float, threshold value above which two traces are considered correlated. Default 0.7.
        param:  Specific parameter of the estimates object that should be correlated. Defaults to 'F_dff' for
                detrended traces, can also be 'C' for pre-detrended or 'S' for deconvolved traces.

    Returns:
        Three-part plot for each highly correlating trace pair with trace 1 on top, trace 2 at the bottom and
        the cross correlation as a function of lag in the middle.
    """

    corr_coef = correlate_components(cnm, param=param)
    high_corr_idx = np.where(corr_coef >= thresh)
    corr_fig = plt.figure(figsize=(20,8))
    outer = gridspec.GridSpec(math.ceil(len(high_corr_idx[0])/2),2)

    for i in range(len(high_corr_idx[0])):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        trace_1 = cnm.estimates.F_dff[high_corr_idx[0][i]]
        trace_2 = cnm.estimates.F_dff[high_corr_idx[1][i]]

        # calculate the cross-correlation
        npts = len(trace_1)
        sample_rate = 1/30 # in Hz
        #lags = np.arange(start=-(npts*sample_rate)+sample_rate, stop=npts*sample_rate-sample_rate, step=sample_rate)
        lags = np.arange(-npts + 1, npts)
        # remove sample means
        trace_1_dm = trace_1 - trace_1.mean()
        trace_2_dm = trace_2 - trace_2.mean()
        # calculate correlation
        trace_cov = np.correlate(trace_1_dm,trace_2_dm,'full')
        # normalize against std
        trace_corr = trace_cov / (npts * trace_1.std() * trace_2.std())

        for j in range(3):
            ax = plt.Subplot(corr_fig, inner[j])
            if j == 0: # plot trace 1 on top
                ax.plot(trace_1)
                ax.axis('off')
            elif j == 2: # plot trace 2 on the bottom
                ax.plot(trace_2)
                ax.axis('off')
            elif j == 1: # plot correlation trace in the middle
                ax.plot(lags, trace_corr)

            corr_fig.add_subplot(ax)

    corr_fig.show()