import colorcet as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import seaborn.apionly as sns
import warnings


class Cut:
    def __init__(self, orientation, idx, color):
        self.orientation = orientation
        self.idx = idx
        self.color = color


class PlotPatch:
    def __init__(self, box, color):
        self.box = box
        self.color = color


def transversal_cuts(img, cuts, cmap='gray', normalize=False, axes=None):
    if normalize:
        vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = 0, 255

    if axes is None:
        ax = plt.subplot(121)
    else:
        ax = axes[0]
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    for c in cuts:
        if c.orientation == 'v':
            lines = ax.plot([c.idx, c.idx], [0, img.shape[0] - 0.5])
        if c.orientation == 'h':
            lines = ax.plot([0, img.shape[1] - 0.5], [c.idx, c.idx])
        plt.setp(lines, color=c.color)
        plt.setp(lines, linewidth=1)
        ax.axis('off')

    with sns.axes_style('whitegrid'):
        if axes is None:
            ax = plt.subplot(122)
        else:
            ax = axes[1]

        def test(x):
            return (x * 0.5).sum()

        for c in cuts:
            if c.orientation == 'v':
                curve = img[:, c.idx]
            if c.orientation == 'h':
                curve = img[c.idx, :]
            ax.plot(curve, color=c.color, alpha=0.5)


def plot_patches_overlay(img, patches, selection=[], cmap='gray',
                         normalize=False):
    if normalize:
        vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = 0, 255

    if len(selection) == 0:
        subplot_idx = 121
    else:
        subplot_idx = 111
    fig = plt.gcf()
    grid = ImageGrid(fig, subplot_idx,
                      nrows_ncols=(1, 1),
                      direction="row",
                      axes_pad=0.05,
                      add_all=True,
                      share_all=True)
    grid[0].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    grid[0].axis('off')
    for p in patches:
        if p.color == 'w':
            zorder = 1
        else:
            zorder = 2
        rect = mpatches.Rectangle((p.box[1], p.box[0]), p.box[2], p.box[3],
                                  linewidth=2, edgecolor=p.color,
                                  facecolor='none', zorder=zorder)
        grid[0].add_artist(rect)

    if len(selection) == 0:
        nrows = int(np.ceil(np.sqrt(len(selection))))
        ncols = int(np.ceil(np.sqrt(len(selection))))

        grid = ImageGrid(fig, 122,
                          nrows_ncols=(nrows, ncols),
                          direction="row",
                          axes_pad=0.15,
                          add_all=True,
                          share_all=True)

        for ax, idx in zip(grid, selection):
            p = patches[idx]
            crop = img[p.box[0]: p.box[0] + p.box[2], p.box[1]: p.box[1] + p.box[3]]

            plot_patch(crop, edgecolor=p.color, ax=ax, vmin=vmin, vmax=vmax,
                       cmap=cmap)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            plt.tight_layout()


def plot_patch(patch, edgecolor=None, ax=None, vmin=None, vmax=None, cmap=None):
    if ax is None:
        ax = plt.gca()

    im_plot = ax.imshow(patch, vmin=vmin, vmax=vmax, cmap=cmap)

    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_color(edgecolor)
        ax.spines[loc].set_linewidth(4)
    ax.tick_params(axis='both',
                   which='both',
                   left='off', right='off',
                   top='off', bottom='off',
                   labelleft='off', labelbottom='off')

    return im_plot


def plot_vst_accumulator_space(acc_space, cmap=cc.m_fire, ax=None,
                               plot_estimates=False, plot_focus=False):
    if ax is None:
        ax = plt.gca()

    alpha_step0 = acc_space.alpha_range[1] - acc_space.alpha_range[0]
    alpha_step1 = acc_space.alpha_range[-1] - acc_space.alpha_range[-2]
    sigma_step0 = acc_space.sigma_sq_range[1] - acc_space.sigma_sq_range[0]
    sigma_step1 = acc_space.sigma_sq_range[-1] - acc_space.sigma_sq_range[-2]
    im_plt = ax.imshow(acc_space.score, cmap=cmap,
                       extent=(acc_space.alpha_range[0] - alpha_step0 / 2,
                               acc_space.alpha_range[-1] + alpha_step1 / 2,
                               acc_space.sigma_sq_range[-1] + sigma_step1 / 2,
                               acc_space.sigma_sq_range[0] - sigma_step0 / 2)
                       )
    ax.axis('tight')
    ax.set_xlabel(r'$\alpha$', fontsize='xx-large')
    ax.set_ylabel(r'$\beta$', fontsize='xx-large')
    plt.colorbar(mappable=im_plt, ax=ax)

    if plot_focus:
        len_a = (acc_space.alpha_range[-1] - acc_space.alpha_range[0]) / 4
        len_s = (acc_space.sigma_sq_range[-1]
                 - acc_space.sigma_sq_range[0]) / 10
        rect = mpatches.Rectangle((acc_space.alpha - len_a / 2,
                                   acc_space.sigma_sq - len_s / 2),
                                  len_a, len_s, ec='#0000cd', fc='none',
                                  linewidth=3)
        ax.add_artist(rect)

    if plot_estimates:
        tag_str = r'$\alpha={:.2f}$, $\beta={:.2f}$'
        bbox_props = dict(boxstyle='round', fc='w', ec='#0000cd', alpha=0.5)
        ax.annotate(tag_str.format(acc_space.alpha, acc_space.sigma_sq),
                       xy=(acc_space.alpha, acc_space.sigma_sq), xycoords='data',
                       xytext=(0.95, 0.05), textcoords='axes fraction',
                       va='bottom', ha='right', color='#0000cd', size='xx-large',
                       bbox=bbox_props,
                       arrowprops=dict(facecolor='#0000cd', edgecolor='none',
                                       shrink=0., width=2, headwidth=3,
                                       headlength=3)
                       )
