import numpy as np
import os.path as op

import matplotlib.pyplot as plt

from swann.utils import get_config, derivative_fname
from swann.preprocessing import preproc_slowfast, slowfast_group


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
    plotf = derivative_fname(behf, 'plots', 'slowfast', config['fig'])
    slow, fast, blocks, p, accuracy = \
        preproc_slowfast(behf, return_saved=True)
    _plot_slow_fast(slow, fast, blocks, p, accuracy, plotf, config,
                    'Subject %s' % subject, overwrite)
    plt.close('all')


def _plot_slow_fast(slow, fast, blocks, p, accuracy, plotf,
                    config, name, overwrite):
    slow_fast = config['task_params']
    bins = np.linspace(min([slow[config['response_col']].min(),
                            fast[config['response_col']].min()]),
                       max([slow[config['response_col']].max(),
                            fast[config['response_col']].max()]),
                       10)
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.set_size_inches(6, 12)
    ax0.hist(slow[config['response_col']], bins=bins,
             color='blue', label='Slow', alpha=0.5)
    ax0.hist(fast[config['response_col']], bins=bins,
             color='red', label='Fast', alpha=0.5)
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
