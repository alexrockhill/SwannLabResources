import numpy as np
import os.path as op

import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
from scipy.stats import ttest_ind

from ..utils import get_config, make_derivatives_dir

from pactools import Comodulogram, raw_to_mask


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


def plot_slow_fast(dfs):
    config = get_config()
    slow_fast = config['task_params']
    for subject in np.unique(dfs['Subject']):
        make_derivatives_dir(subject)
        df = dfs[dfs['Subject'] == subject]
        fig, ax = plt.subplots()
        g = sns.catplot(x=slow_fast['name_col'], y=config['response_col'],
                        data=df, height=6, kind='bar', palette='muted', ax=ax)
        g.despine(left=True)
        ax.set_xlabel('Slow or Fast')
        ax.set_ylabel('Response Time')
        ax.set_xticklabels([slow_fast['0'], slow_fast['1']])
        t, p = ttest_ind(
            df[df[slow_fast['name_col']] == 0][config['response_col']],
            df[df[slow_fast['name_col']] == 1][config['response_col']])
        ax.set_title('Subject %s, p = %.3f' % (subject, p))
        fig.savefig(op.join(config['bids_dir'], 'derivatives',
                            'sub-%s' % subject, 'sub-%s_rt_bar.png' % subject))
        fig.show()

        n_blocks = len(np.unique(df[config['block_col']]))
        fig, axes = plt.subplots(1, n_blocks + 1)
        fig.set_size_inches(12, 4)
        fig.tight_layout()
        for i, block in enumerate(np.unique(df[config['block_col']])):
            axes[i].set_xlabel('Trials')
            axes[i].set_title('Block %s' % block)
            axes[i].set_ylabel('Response Time')
            axes[i].set_ylim([0.2, 1.5])
            this_block = df[df[config['block_col']] == block]
            this_slow_or_fast = \
                slow_fast[str(list(this_block[slow_fast['name_col']])[0])]
            if this_slow_or_fast == 'slow':
                color = 'b'  # (0, 0, 0.5 + (1 / n_blocks) * block / 2)
            else:
                color = 'r'  # (0.5 + (1 / n_blocks) * block / 2, 0, 0)
            axes[i].plot(list(this_block[config['response_col']]),
                         color=color)
        axes[-1].plot([0], [0], color='b', label='Slow')
        axes[-1].plot([0], [0], color='r', label='Fast')
        axes[-1].axis('off')
        axes[-1].legend()
        fig.savefig(op.join(config['bids_dir'], 'derivatives',
                            'sub-%s' % subject,
                            'sub-%s_rt_time_course.png' % subject))
        fig.show()

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        fig.tight_layout()
        slow = df.loc[[i for i, sf in
                       enumerate(list(df[slow_fast['name_col']]))
                       if slow_fast[str(sf)] == 'slow']]
        fast = df.loc[[i for i, sf in
                       enumerate(list(df[slow_fast['name_col']]))
                       if slow_fast[str(sf)] == 'fast']]
        ax.set_title('Subject %s' % subject)
        ax.set_ylabel('Response Time')
        sns.distplot(slow[config['response_col']], ax=ax, label='Slow')
        sns.distplot(fast[config['response_col']], ax=ax, label='Fast')
        ax.legend()
        fig.savefig(op.join(config['bids_dir'], 'derivatives',
                            'sub-%s' % subject,
                            'sub-%s_rt_dist.png' % subject))
