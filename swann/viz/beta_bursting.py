import os
import numpy as np
import os.path as op

import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
from scipy.stats import ttest_ind

from swann.preprocessing import get_info
from swann.utils import get_config, derivative_fname
from swann.analyses import decompose_tfr, find_bursts, get_bursts

from mne.viz import iter_topography
from mne.time_frequency import tfr_morlet
from mne import Epochs


def plot_spectrogram(rawf, raw, event, events, columns=3, lfreq=4,
                     hfreq=50, dfreq=1, n_cycles=7, use_fft=True,
                     plot_bursts=False, verbose=True, overwrite=False):
    ''' Plots a bar chart of beta bursts.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    raw : mne.io.Raw
        The raw data object.
    event : str
        The name of the event (e.g. `Response`).
    events : np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    columns : int
        The number of columns to use in the plot.
    lfreq : float
        The lowest frequency to use.
    hfreq : float
        The greatest frequency to use.
    dfreq : int
        The step size between lfreq and hfreq.
    n_cycles : int, np.array
        The number of cycles to use in the Morlet transform
    use_fft : bool
        Use Fast Fourier Transform see `mne.time_frequency.tfr.cwt`.
    plot_bursts : bool
        Whether to include vertical bars for when the bursts are detected.
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/spectrograms',
                             event + '_spectrogram_%s',
                             config['fig'])
    if op.isdir(op.dirname(plotf)) and not overwrite:
        print('Spectrogram plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    if plot_bursts:
        bursts = find_bursts(rawf, return_saved=True)
    epochs = Epochs(raw, events, tmin=config['tmin'] - 1,
                    tmax=config['tmax'] + 1)
    epochs_tfr = tfr_morlet(epochs, np.arange(lfreq, hfreq, dfreq),
                            n_cycles=n_cycles, use_fft=use_fft,
                            average=False, return_itc=False)
    epochs_tfr.crop(tmin=config['tmin'], tmax=config['tmax'])
    for i, ch_name in enumerate(epochs_tfr.ch_names):
        if verbose:
            print('Plotting spectrogram for channel %s' % ch_name)
            if plot_bursts:
                n_bursts = len(bursts[bursts['channel'] == ch_name])
                print('%i bursts for this channel total' % n_bursts)
        this_plot_f = plotf % ch_name
        rows = int(len(events) / columns)
        fig, axes = plt.subplots(rows, columns)
        axes = axes.flatten()
        fig.subplots_adjust(right=0.85)
        cax = fig.add_subplot(position=[0.9, 0.1, 0.05, .8])
        fig.set_size_inches(12, 8)
        vmin, vmax = epochs_tfr.data[:, i].min(), epochs_tfr.data[:, i].max()
        for j, this_tfr in enumerate(epochs_tfr):
            cmap = axes[j].imshow(this_tfr[i], aspect='auto', vmin=vmin,
                                  vmax=vmax, cmap='RdYlBu_r')
            axes[j].invert_yaxis()
            if j % columns == 0:
                axes[j].set_yticks(np.linspace(0, (hfreq - lfreq) / dfreq, 3))
                axes[j].set_yticklabels(['%i' % f for f in
                                         np.linspace(lfreq, hfreq, 3)])
            else:
                axes[j].set_yticklabels([])
            if j == int(len(events) / 2):
                axes[j].set_ylabel('Frequency (Hz)')
            axes[j].axvline(np.where(epochs_tfr.times == 0)[0][0], color='k')
            axes[j].set_xticks([])
            if plot_bursts:
                min_idx = events[j, 0] + raw.info['sfreq'] * config['tmin']
                max_idx = events[j, 0] + raw.info['sfreq'] * config['tmax']
                these_bursts = bursts[(bursts['channel'] == ch_name) &
                                      (bursts['burst_end'] > min_idx) &
                                      (bursts['burst_start'] < max_idx)]
                if these_bursts.size > 0:
                    for burst_idx in these_bursts.index:
                        for start_stop in ['burst_start', 'burst_end']:
                            if (max_idx > these_bursts.loc[burst_idx,
                                                           start_stop] >
                                    min_idx):
                                axes[j].axvline(
                                    x=these_bursts.loc[burst_idx,
                                                       start_stop] - min_idx,
                                    color='green')
        axes = axes.reshape(rows, columns)
        for col_idx in range(columns):
            axes[-1, col_idx].set_xlabel('Time (s)')
            axes[-1, col_idx].set_xticks(
                np.linspace(0, len(epochs_tfr.times), 5))
            axes[-1, col_idx].set_xticklabels(
                ['%.2f' % t for t in np.linspace(epochs_tfr.times[0],
                                                 epochs_tfr.times[-1], 5)])
        cax = fig.colorbar(cmap, cax=cax)
        cax.set_label('Power')
        fig.suptitle('Time Frequency Decomposition for the %s ' % event +
                     'Event, Channel %s' % ch_name)
        fig.savefig(this_plot_f, dpi=300)
        plt.close(fig)


def plot_group_bursting(rawfs, event, events, tfr_name='beta',
                        picks=None, method='peaks', ylim=None, rolling=0.25,
                        verbose=True, overwrite=False):
    ''' Plots bursts on topography or in graph overlayed.
    Parameters
    ----------
    rawfs : list of pybids.BIDSlayout file
        The object containing the raw data.
    event : str
        The name of the event (e.g. `Response`).
    events : dict(str=dict(str=np.array(n_events, 3)))
        A dict with keys that are name and secondary keys that are rawf
        paths and values that are events from mne.events_from_annotations or
        mne.find_events; e.g. {'All': {'sub-1...': events}}.
    tfr_name : str
        What to call the frequencies used e.g. beta
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    method : ('peaks', 'all', 'durations', 'shape')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    rolling : float
        Amount of time to using for the rolling average (default 250 ms)
    '''
    config = get_config()
    events = (events if isinstance(events[list(events.keys())[0]], dict) else
              {None: events})
    picks_str = ('' if picks is None else 'channels_' + '_and_'.join(picks))
    names_str = (None if list(events.keys())[0] is None else
                 '_and_'.join(events.keys()))
    basename = '%s_%s_bursting_%s_%s_%s.%s' % (names_str, tfr_name, method,
                                               event, picks_str, config['fig'])
    plotf = op.join(config['bids_dir'],
                    'derivatives/plots/%s_bursting' % tfr_name, basename)
    if not op.isdir(op.dirname(plotf)):
        os.makedirs(op.dirname(plotf))
    if op.isfile(plotf) and not overwrite:
        print('Group %s bursting plot for %s ' % (tfr_name.title(), event) +
              'already exists, use `overwrite=True` to replot')
        return
    burst_data = dict()
    for name in events:
        burst_data[name] = dict()
        for rawf in rawfs:
            burst_data[name][rawf.path] = \
                get_bursts(rawf, events[name][rawf.path], method, rolling)
    fig = _plot_bursting(burst_data, picks, method, ylim, rolling, verbose)
    fig.suptitle('' if names_str is None else '%s Trials ' %
                 names_str.replace('_', ' ') +
                 '%s Bursting ' % tfr_name +
                 'for the %s Event from ' % event +
                 '%s to %s Seconds' % (config['tmin'], config['tmax']))
    fig.savefig(plotf, dpi=300)


def plot_bursting(rawf, event, events, tfr_name='beta',
                  picks=None, method='peaks', ylim=None, rolling=0.25,
                  verbose=True, overwrite=False):
    ''' Plots bursts on topography or in graph overlayed.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    event : str
        The name of the event (e.g. `Response`).
    events : dict(str=np.array(n_events, 3))
        A dict with values that are events from mne.events_from_annotations or
        mne.find_events corresponding to the event and trials that are
        described by the names which are the keys (e.g. `Slow` or `All`).
    tfr_name : str
        What to call the frequencies used e.g. beta
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    method : ('peaks', 'all', 'durations', 'shape')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    rolling : float
        Amount of time to using for the rolling average (default 250 ms)
    '''
    config = get_config()
    picks_str = ('' if picks is None else 'channels_' + '_and_'.join(picks))
    names_str = '_and_'.join(events.keys())
    basename = '%s_%s_bursting_%s_%s_%s.%s' % (names_str, tfr_name, method,
                                               event, picks_str, config['fig'])
    plotf = derivative_fname(rawf, 'plots/%s_bursting' % tfr_name,
                             basename)
    if op.isfile(plotf) and not overwrite:
        print('%s bursting plot for %s ' % (tfr_name.title(), event) +
              'already exists, use `overwrite=True` to replot')
        return
    burst_data = dict()
    for name in events:
        burst_data[name] = {rawf.path: get_bursts(rawf, events[name],
                                                  method, rolling)}
    fig = _plot_bursting(burst_data, picks, method, ylim, rolling, verbose)
    fig.suptitle('' if names_str is None else '%s Trials ' %
                 names_str.replace('_', ' ') +
                 '%s Bursting ' % tfr_name +
                 'for the %s Event from ' % event +
                 '%s to %s Seconds' % (config['tmin'], config['tmax']))
    fig.savefig(plotf, dpi=300)


def _plot_bursting(burst_data, picks, method, ylim, rolling, verbose):
    if picks is None:
        infos = [get_info(rawf) for name in burst_data for
                 rawf in burst_data[name]]
        info_max_idx = np.argmax([len(info['ch_names']) for info in infos])
        info = infos[info_max_idx]
        for ax, idx in iter_topography(info, fig_facecolor='white',
                                       axis_facecolor='white',
                                       axis_spinecolor='white'):
            _plot_burst_data(burst_data, [info['ch_names'][idx]],
                             method, ylim, rolling, ax, verbose)
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
        _plot_burst_data(burst_data, picks, method, ylim, rolling, ax, verbose)
        if method in ('all', 'peaks'):
            ax.set_ylabel('Percent of Trials')
            ax.set_xlabel('Time (s)')
        elif method in ('durations', ):
            ax.set_ylabel('Duration (s)')
        else:
            raise ValueError('Unrecognized method %s' % method)
    fig.set_size_inches(10, 12)
    return fig


def _plot_burst_data(plot_data, ch_names, method, ylim, rolling, ax, verbose):
    if method in ('all', 'peaks'):
        times = None
        for name in plot_data:
            for ch in ch_names:
                bins = list()
                for rawf in plot_data[name]:
                    these_times, these_bins = plot_data[name][rawf][ch]
                    if times is not None and any(these_times != times):
                        raise ValueError('Times are not all the same ' +
                                         'for group averaging')
                    times = these_times
                    bins.append(these_bins)
                label = ch if name is None else name + '_' + ch
                bins_mean = np.mean(bins, axis=0)
                bins_std = np.std(bins, axis=0)
                ax.plot(times, bins_mean, label=label)
                ax.fill_between(times, np.max([bins_mean - bins_std,
                                               np.zeros(bins_mean.shape)],
                                              axis=0),
                                bins_mean + bins_std, alpha=0.1)
        if ylim is None:
            ax.set_ylim([0, ylim])
    elif method in ('durations',):
        all_durations = list()
        labels = list()
        colors = list()
        for name in plot_data:
            for ch_idx, ch in enumerate(ch_names):
                for rawf in plot_data[name][ch]:
                    times, durations = plot_data[name][ch][rawf]
                    all_durations.append(durations)
                    labels.append(ch if name is None else name + ' ' + ch)
                    colors.append(list(BASE_COLORS)[ch_idx % len(BASE_COLORS)])
        ax.bar(range(len(all_durations)),
               [np.mean(ds) for ds in all_durations],
               yerr=[np.std(ds) for ds in all_durations],
               color=colors, alpha=0.5)
        for i, durations in enumerate(all_durations):
            ax.scatter(np.repeat(i, len(durations)), durations,
                       color=colors[i], alpha=0.1)
        ax.set_xticks(range(len(all_durations)))
        for i, durations0 in enumerate(all_durations):
            for j, durations1 in enumerate(all_durations[i + 1:]):
                t, p = ttest_ind(durations0, durations1)
                if verbose:
                    print('%s-%s: p = %s' % (labels[i], labels[i + j + 1], p))
                '''
                if p < 0.05:
                    y = (max([max(durations0), max(durations1)]) + 0.01) * 1.1
                    ax.axhline(y=y, xmin=(i + 0.5) / len(all_durations),
                               xmax=(i + j + 1 + 0.5) / len(all_durations),
                               color='k')
                    ax.text(np.mean([i, i + j + 1]) - 0.5, (y + 0.01) * 1.05,
                            'p = %.3f' % p if p > 0.001 else 'p < 0.001')
                '''
        ax.set_xticklabels(labels)
    else:
        raise ValueError('Unrecognized method %s' % method)


def plot_group_power(rawf, name, event, events, infos, picks=None,
                     tfr_name='beta', verbose=True, overwrite=False):
    ''' Plots a line graph with one standard deviation shaded of power.
    Parameters
    ----------
    rawfs : pybids.BIDSlayout file
        The objects containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`).
    event : str
        The name of the event (e.g. `Response`).
    events : dict(str=dict(str=np.array(n_events, 3)))
        A dict with keys that are name and secondary keys that are rawf
        paths and values that are events from mne.events_from_annotations or
        mne.find_events; e.g. {'All': {'sub-1...': events}}.
    infos : list of instance of mne.io.Raw.info
        The info objects with channel names and positions.
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    tfr_name : str
        The keyword name of the tfr that was previously computed.
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/%s_power' % tfr_name,
                             '%s_%s_power_%s' % (name, tfr_name, event),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('%s power plot already exist, ' % tfr_name.capitalize() +
              'use `overwrite=True` to replot')
        return
    tfr, my_ch_names, sfreq = decompose_tfr(rawf, tfr_name, return_saved=True)
    tfr /= tfr.mean(axis=1)[:, np.newaxis]  # z-score power
    if sfreq != info['sfreq']:
        raise ValueError('Raw sampling frequency mismatch with tfr sfreq')
    if any(info['ch_names'] != my_ch_names):
        raise ValueError('Raw channel names mismatch with tfr channel names')
    if verbose:
        print('Plotting %s power for ' % tfr_name.capitalize() +
              '%s trials during the %s event' % (name, event))
    if picks is None:
        for ax, idx in iter_topography(info, fig_facecolor='white',
                                       axis_facecolor='white',
                                       axis_spinecolor='white'):
            _plot_power(info['ch_names'][idx], events, sfreq,
                        tfr[idx], np.quantile(tfr, 0.9), ax=ax)
    else:
        pick_indices = [list(my_ch_names).index(ch) for ch in picks]
        tfr = tfr[pick_indices]
        fig, ax = plt.subplots()
        for idx, ch_name in enumerate(picks):
            _plot_power(ch_name, events, sfreq, tfr[idx],
                        np.quantile(tfr, 0.9), ax=ax)
        ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.suptitle(('%s Trials z-scored %s Power for ' % (name, tfr_name) +
                  'the %s Event from ' % event +
                  '%s to %s Seconds' % (config['tmin'], config['tmax'])
                  ).title())
    fig.savefig(plotf, dpi=300)


def plot_power(rawf, event, events, info, picks=None,
               tfr_name='beta', verbose=True, overwrite=False):
    ''' Plots a line graph with one standard deviation shaded of power.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    event : str
        The name of the event (e.g. `Response`).
    events : dict(str=np.array(n_events, 3))
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    info : instance of mne.io.Raw.info
        The info object with channel names and positions.
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    tfr_name : str
        The keyword name of the tfr that was previously computed.
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/%s_power' % tfr_name,
                             '%s_%s_power_%s' % (name, tfr_name, event),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('%s power plot already exist, ' % tfr_name.capitalize() +
              'use `overwrite=True` to replot')
        return
    tfr, my_ch_names, sfreq = decompose_tfr(rawf, tfr_name, return_saved=True)
    tfr /= tfr.mean(axis=1)[:, np.newaxis]  # z-score power
    if sfreq != info['sfreq']:
        raise ValueError('Raw sampling frequency mismatch with tfr sfreq')
    if any(info['ch_names'] != my_ch_names):
        raise ValueError('Raw channel names mismatch with tfr channel names')
    if verbose:
        print('Plotting %s power for ' % tfr_name.capitalize() +
              '%s trials during the %s event' % (name, event))
    if picks is None:
        for ax, idx in iter_topography(info, fig_facecolor='white',
                                       axis_facecolor='white',
                                       axis_spinecolor='white'):
            _plot_power(info['ch_names'][idx], events, sfreq,
                        tfr[idx], np.quantile(tfr, 0.9), ax=ax)
    else:
        pick_indices = [list(my_ch_names).index(ch) for ch in picks]
        tfr = tfr[pick_indices]
        fig, ax = plt.subplots()
        for idx, ch_name in enumerate(picks):
            _plot_power(ch_name, events, sfreq, tfr[idx],
                        np.quantile(tfr, 0.9), ax=ax)
        ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.suptitle(('%s Trials z-scored %s Power for ' % (name, tfr_name) +
                  'the %s Event from ' % event +
                  '%s to %s Seconds' % (config['tmin'], config['tmax'])
                  ).title())
    fig.savefig(plotf, dpi=300)


def _plot_power(ch_name, events, sfreq,
                tfr_ch, ylim, ax=None, verbose=True):
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    if verbose:
        print('Plotting channel %s' % ch_name)
    if ax is None:
        fig, ax = plt.subplots()
    epochs_tfr = np.zeros((len(events), len(bin_indices)))
    for i, event_index in enumerate(events[:, 0]):
        epochs_tfr[i] = [tfr_ch[event_index + offset] for offset in
                         bin_indices]
    evoked_tfr = np.mean(epochs_tfr, axis=0)
    evoked_tfr_std = np.std(epochs_tfr, axis=0)
    ax.plot(times, evoked_tfr, label=ch_name)
    ax.fill_between(times, evoked_tfr - evoked_tfr_std,
                    evoked_tfr + evoked_tfr_std, alpha=0.25)
    ax.set_ylim([0, ylim])
    return plt.gcf()


def plot_burst_shape(rawf, name, event, info, events, tfr_name='beta',
                     picks=None, verbose=True, overwrite=False):
    ''' Plots a bar chart of  bursts.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`).
    event : str
        The name of the event (e.g. `Response`).
    events : np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/burst_shape',
                             '%s_beta_burst_%s' % (name, event) +
                             ('' if picks is None else '_'.join(picks)),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('Beta burst shape plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    ch_names = info['ch_names'] if picks is None else picks
    beta_bursts = find_bursts(rawf, return_saved=True)
    tfr, my_ch_names, sfreq = decompose_tfr(rawf, tfr_name, return_saved=True)
    if verbose:
        print('Plotting beta burst shape for %s trials during ' % name +
              'the %s event' % event)
    if picks is None:
        for ax, idx in iter_topography(info, fig_facecolor='white',
                                       axis_facecolor='white',
                                       axis_spinecolor='white'):
            _plot_beta_bursts(ch_names[idx], events, info['sfreq'],
                              beta_bursts[beta_bursts['channel'] ==
                                          ch_names[idx]], method,
                              beta_burst_lim, ax=ax)
    else:
        fig, ax = plt.subplots()
        for ch_name in picks:
            _plot_beta_bursts(ch_name, events, info['sfreq'],
                              beta_bursts[beta_bursts['channel'] ==
                                          ch_name], method,
                              beta_burst_lim, ax=ax)
        ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    fig.suptitle('%s Trials Beta Burst Shape for ' % name +
                 'the %s Event' % event)
    fig.savefig(plotf, dpi=300)


def _plot_beta_shape(ch_name, events, sfreq, my_beta_bursts,
                      method, ylim, ax=None, verbose=True):
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    if verbose:
        print('Plotting channel %s' % ch_name)
    if ax is None:
        fig, ax = plt.subplots()
    if method == 'peaks':
        beta_burst_indices = set([i for start, stop in
                                  zip(my_beta_bursts['burst_start'],
                                      my_beta_bursts['burst_end'])
                                  for i in range(start, stop)])
    elif method == 'all':
        beta_burst_indices = set(my_beta_bursts['burst_peak'])
    else:
        raise ValueError('Method of calculating beta bursts %s ' % method +
                         'not recognized.')
    bins = np.zeros((len(bin_indices)))
    for i, event_index in enumerate(events[:, 0]):
        for j, offset in enumerate(bin_indices):
            if event_index - offset in beta_burst_indices:
                bins[j] += 1
    ax.plot(times, bins, label=ch_name)
    ax.set_ylim([0, ylim])
    return plt.gcf()


def plot_group_burst_shape():
    pass


def plot_burst_topo():
    pass


def plot_group_burst_topo():
    pass
