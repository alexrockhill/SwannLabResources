import numpy as np
import os.path as op

import matplotlib.pyplot as plt
import seaborn as sns

from swann.utils import get_config, derivative_fname
from swann.preprocessing import (get_bads, set_bads,
                                 preproc_slowfast, slowfast_group)

from pactools import Comodulogram, raw_to_mask


# need to remove the computations?
def plot_pac(subject, raw, ch_names=None, events=None, tmin=None, tmax=None):
    config = get_config()
    if ch_names is None:
        ch_names = raw.ch_names
    for idx0, ch0 in enumerate(ch_names):
        for idx1, ch1 in enumerate(ch_names):
            if ch0 in raw.info['bads'] or ch1 in raw.info['bads']:
                continue
            figf = op.join(config['bids_dir'], 'derivatives',
                           'sub-%s' % subject,
                           'sub-%s_driver-%s_carrier-%s_DAR_PAC.png' %
                           (subject, ch0, ch1))
            # create the input array for Comodulogram.fit
            low_sig, high_sig, mask = raw_to_mask(raw, ixs=[idx0, idx1],
                                                  events=events, tmin=tmin,
                                                  tmax=tmax)
            # create the instance of Comodulogram
            estimator = \
                Comodulogram(fs=raw.info['sfreq'],
                             n_surrogates=int(10 / config['p_value']),
                             low_fq_range=np.linspace(1, 20, 20),
                             low_fq_width=2., method='duprelatour',
                             progress_bar=True)
            # compute the comodulogram
            try:
                estimator.fit(low_sig, high_sig, mask)
            except Exception as e:
                print(e)
                continue
            # plot the results
            fig, ax = plt.subplots()
            fig = estimator.plot(contour_method='comod_max',
                                 contour_level=config['p_value'],
                                 titles=['Driver %s, Carrier %s' % (ch0, ch1)],
                                 axs=[ax], tight_layout=False)
            fig.savefig(figf, dpi=300)


def plot_find_bads(rawf, raw):
    config = get_config()
    badsf = derivative_fname(rawf, 'bad_channels', 'tsv')
    if op.isfile(badsf):
        raw.info['bads'] = get_bads(rawf)
    else:
        psdf = derivative_fname(rawf, 'psd_w_bads', 'eps')
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch != 'Event'])
        fig.savefig(psdf, dpi=300)
        raw.plot()  # pick bad channels
        set_bads(rawf, raw)
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch != 'Event' and
                                  ch not in raw.info['bads']])
        fig.savefig(derivative_fname(rawf, 'psd', config['fig']))


def plot_slow_fast_group(behfs, name, overwrite=False):
    config = get_config()
    plotf = op.join(config['bids_dir'], 'derivatives',
                    'group_%s_slowfast.' % name + config['fig'])
    slow, fast, blocks, p, accuracy = slowfast_group(behfs)
    _plot_slow_fast(slow, fast, blocks, p, accuracy, plotf, config,
                    'Group %s' % name, overwrite)
    plt.close('all')


def plot_slow_fast(behf, overwrite=False):
    config = get_config()
    subject = behf.entities['subject']
    plotf = derivative_fname(behf, 'slowfast', config['fig'])
    slow, fast, blocks, p, accuracy = \
        preproc_slowfast(behf, return_saved=True)
    _plot_slow_fast(slow, fast, blocks, p, accuracy, plotf, config,
                    'Subject %s' % subject, overwrite)
    plt.close('all')


def _plot_slow_fast(slow, fast, blocks, p, accuracy, plotf,
                    config, name, overwrite):
    slow_fast = config['task_params']
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.set_size_inches(6, 12)
    sns.distplot(slow[config['response_col']], ax=ax0,
                 color='blue', label='Slow')
    sns.distplot(fast[config['response_col']], ax=ax0,
                 color='red', label='Fast')
    ax0.legend()
    ax0.set_xlabel('Response Time')
    ax0.set_ylabel('Count')
    ax0.set_xlim([0, max([slow[config['response_col']].max(),
                          fast[config['response_col']].max()]) * 1.1])
    if p < 0.001:
        ax0.set_title('%s, p < 0.001, accuracy = %.2f' % (name, accuracy))
    else:
        ax0.set_title('%s, p = %.3f, accuracy = %.2f' % (name, p, accuracy))

    for block in blocks:
        color = ('blue' if list(block[slow_fast['name_col']])[0] ==
                 slow_fast['slow'] else 'red')
        ax1.plot(block.index, block[config['response_col']],
                 color=color)
    ax1.plot(0, 0, color='blue', label='slow')
    ax1.plot(0, 0, color='red', label='fast')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Response Time')
    ax1.legend()
    fig.tight_layout()
    if not op.isfile(plotf) or overwrite:
        fig.savefig(plotf, dpi=300)
    return fig
