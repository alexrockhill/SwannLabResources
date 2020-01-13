import numpy as np
import os.path as op

import matplotlib.pyplot as plt

from swann.utils import (get_config, derivative_fname, read_raw,
                         get_events, get_no_responses)
from swann.preprocessing import (get_bads, set_bads, get_ica,
                                 preproc_slowfast, slowfast_group,
                                 set_ica_components, apply_ica,
                                 mark_autoreject, slowfast2epochs_indices)
from swann.analyses import decompose_tfr

from pactools import Comodulogram, raw_to_mask

from mne.viz import iter_topography


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


def plot_find_bads(rawf, overwrite=False):
    config = get_config()
    badsf = derivative_fname(rawf, 'bad_channels', 'tsv')
    if op.isfile(badsf) and not overwrite:
        print('Bad channels already marked, skipping plot raw for ' +
              'determining bad channels, use `overwrite=True` to plot')
        return
    else:
        raw = read_raw(rawf.path)
        print('Plotting PSD spectrogram and raw channels for bad channel ' +
              'selection, %s' % rawf.path)
        raw.info['bads'] += [ch for ch in get_bads(rawf) if
                             ch not in raw.info['bads']]
        psdf = derivative_fname(rawf, 'psd_w_bads', config['fig'])
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch not in ['Event', 'Status']],
                           show=False)
        fig.savefig(psdf, dpi=300)
        raw.plot(block=True)  # pick bad channels
        set_bads(rawf, raw.info['bads'])
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch not in ['Event', 'Status'] and
                                  ch not in raw.info['bads']])
        fig.savefig(derivative_fname(rawf, 'psd', config['fig']))


def plot_ica(rawf, method='fastica', n_components=None,
             overwrite=False):
    if (op.isfile(derivative_fname(rawf, 'ica_components', 'tsv')) and
            not overwrite):
        print('ICA component choices already saved, use `overwrite=True` ' +
              'to re-plot.')
        return
    raw = read_raw(rawf.path)
    raw.info['bads'] = get_bads(rawf)
    ica = get_ica(rawf)
    ica.plot_sources(raw, block=True, show=True,
                     title=rawf.entities['subject'])
    '''
    done = False
    while not done:
        # ica.plot_components(show=False) just click on the plot
        # ica.plot_properties(raw, show=False) too many plots
        ica.plot_sources(raw, block=True, show=True,
                         title=rawf.entities['subject'])
        aux_epochs = get_aux_epochs(ica, raw)
        for ch, epo in aux_epochs.items():
            fig = ica.plot_overlay(epo.average(), show=False)
            fig.suptitle('%s %s' % (rawf.entities['subject'], ch))
            fig = ica.plot_sources(epo.average(), exclude=ica.exclude,
                                   show=False)
            fig.suptitle('%s %s' % (rawf.entities['subject'], ch))
        done = input('Have all the artifact components been removed ' +
                     '(Y/N)?').upper() == 'Y'
    '''
    set_ica_components(rawf, ica.exclude)


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


def plot_beta_bursting(rawf, behf, verbose=True, overwrite=False):
    config = get_config()
    if all([op.isfile(
        derivative_fname(rawf, '%s_beta_bursts_%s' % (name, event),
                         config['fig']))
            for name in ['All', 'Slow', 'Fast']
            for event in config['event_id']]) and not overwrite:
        print('Beta bursting plot already exist, use `overwrite=True` ' +
              'to replot')
        return
    raw = apply_ica(rawf)
    tmin, tmax, sfreq = config['tmin'], config['tmax'], raw.info['sfreq']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    epo_reject_indices = {event: list() for event in config['event_id']}
    # mark_autoreject(rawf, return_saved=True)
    all_indices, slow_indices, fast_indices = slowfast2epochs_indices(behf)
    no_responses = get_no_responses(behf)
    tfr = decompose_tfr(rawf, return_saved=True)
    beta_burst_lim = \
        np.quantile([len(beta_bursts[beta_bursts['channel'] == ch])
                     for ch in raw.ch_names], 0.9)
    for name, indices in {'All': all_indices, 'Slow': slow_indices,
                          'Fast': fast_indices}.items():
        if verbose:
            print('Plotting beta bursting for %s trials' % name)
        indices = [i for i in indices if i not in epo_reject_indices]
        for event in config['event_id']:
            if verbose:
                print('Plotting beta bursting for the %s event' % event)
            events = get_events(raw, event, exclude_events=epo_reject_indices,
                                no_responses=(no_responses if
                                              event == 'Response' else None))
            plotf = derivative_fname(rawf, '%s_beta_bursts_%s' % (name, event),
                                     config['fig'])
            for ax, idx in iter_topography(raw.info, fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white'):
                if verbose:
                    print('Plotting channel %s' % raw.ch_names[idx])
                this_beta_bursts = \
                    beta_bursts[beta_bursts['channel'] == raw.ch_names[idx]]
                beta_burst_indices = set([i for start, stop in
                                          zip(this_beta_bursts['burst_start'],
                                              this_beta_bursts['burst_end'])
                                          for i in range(start, stop)])
                this_bins = np.zeros((len(bin_indices)))
                for event_index in events[:, 0]:
                    for j, offset in enumerate(bin_indices):
                        if event_index - offset in beta_burst_indices:
                            this_bins[j] += 1
                ax.plot(times, this_bins)
                ax.set_ylim([0, beta_burst_lim])
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            fig.suptitle('%s Trials Beta Bursting for ' % name +
                         'the %s Event from ' % event +
                         '%s to %s Seconds' % (tmin, tmax))
            fig.savefig(plotf)
