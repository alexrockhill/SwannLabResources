import numpy as np
import os.path as op

import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
from scipy.stats import ttest_ind

from swann.utils import get_config, make_derivatives_dir

from pactools import Comodulogram, raw_to_mask


# need to remove the computations?
def plot_pac(subject, raw, ch_names=None, events=None, tmin=None, tmax=None):
    config = get_config()
    make_derivatives_dir(subject)
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
            fig.savefig(figf)


def plot_find_bads(raw, subject):
    config = get_config()
    make_derivatives_dir(subject)
    badsf = op.join(config['bids_dir'], 'derivatives', 'sub-%s' % subject,
                    'sub-%s_bad_channels.tsv' % subject)
    if op.isfile(badsf):
        raw.info['bads'] = list(read_csv(badsf, sep='\t')['bads'])
    else:
        psdf = op.join(config['bids_dir'], 'derivatives',
                       'sub-%s' % subject, 'sub-%s_psd.png' % subject)
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch != 'Event'])
        fig.savefig(psdf)
        raw.plot()  # pick bad channels
        with open(badsf, 'w') as f:
            f.write('bads\n')
            for ch in raw.info['bads']:
                f.write('%s\n' % ch)
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch != 'Event' and
                                  ch not in raw.info['bads']])
        fig.savefig(psdf.replace('.png', '2.png'))


def plot_slow_fast_group(dfs):
    config = get_config()
    slow_fast = config['task_params']
    for subject in np.unique(dfs['Subject']):
        make_derivatives_dir(subject)
        df = dfs[dfs['Subject'] == subject]


def plot_slow_fast(behf, axes=None, overwrite=False):
    config = get_config()
    slow_fast = config['task_params']
    subject = behf.entities['subject']
    plotf = op.join(config['bids_dir'], 'derivatives',
                    'sub-%s' % subject,
                    'sub-%s_slowfast.png' % subject)
    if op.isfile(plotf) and not overwrite:
        if input('Slowfast plot already exists, overwrite? (Y/N)') != 'Y':
            return
    df = read_csv(behf.path, sep='\t')
    if axes is None:
        fig, (ax0, ax1) = plt.subplots(2, 1)
    else:
        if len(axes) != 2:
            raise ValueError('Three axes required for slow fast plots')
    fig.set_size_inches(6, 12)
    fig.tight_layout()
    fast = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'fast']]
    fast2 = fast[fast[config['response_col']] > 0.1]
    slow = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'slow']]
    slow2 = slow[slow[config['response_col']] > 0.1]
    n_to_exclude = (len(fast) - len(fast2)) - (len(slow) - len(slow2))
    if n_to_exclude < 0:
        raise ValueError('More no responses for subject %s ' % subject +
                         'on slow blocks than fast. This is not expected, ' +
                         'exclude subject')
    slow2 = slow2.reset_index()
    indices = list(np.argsort(slow2[config['response_col']])[:-n_to_exclude])
    slow2 = slow2.iloc[indices]
    sns.distplot(slow2[config['response_col']], ax=ax0,
                 color='blue', label='Slow')
    sns.distplot(fast2[config['response_col']], ax=ax0,
                 color='red', label='Fast')
    ax0.legend()
    ax0.set_xlabel('Response Time')
    ax0.set_ylabel('Count')
    ax0.set_xlim([0, max([slow2[config['response_col']].max(),
                          fast2[config['response_col']].max()]) * 1.1])
    t, p = ttest_ind(
        slow2[config['response_col']], fast2[config['response_col']])
    if p < 0.001:
        ax0.set_title('Subject %s, p < 0.001' % subject)
    else:
        ax0.set_title('Subject %s, p = %.3f' % (subject, p))

    slow2 = slow2.sort_values(by=config['trial_col'])

    for b in range(int(len(df) / config['n_trials_per_block'])):
        this_block = df[(b * config['n_trials_per_block'] <
                         df[config['trial_col']]) &
                        (df[config['trial_col']] <=
                         (b + 1) * config['n_trials_per_block'])]
        this_block = this_block.reset_index()
        this_block = this_block[this_block[config['response_col']] > 0.1]
        color = ('blue' if list(this_block[slow_fast['name_col']])[0] ==
                 slow_fast['slow'] else 'red')
        ax1.plot(this_block.index,
                 this_block[config['response_col']],
                 color=color)
    ax1.plot(0, 0, color='blue', label='slow')
    ax1.plot(0, 0, color='red', label='fast')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Response Time')
    ax1.legend()
    fig.savefig(plotf)
    return ax0.get_figure()
