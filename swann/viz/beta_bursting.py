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
from mne.time_frequency import tfr_morlet, EpochsTFR, AverageTFR
from mne import Epochs, EvokedArray


def plot_spectrogram(rawf, raw, event, events, bl_events,
                     method='raw', baseline='z-score',
                     lfreq=4, hfreq=50, dfreq=1, n_cycles=7, use_fft=True,
                     columns=3, plot_bursts=False, picks=None,
                     verbose=True, overwrite=False):
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
    method : `raw` | `phase-locked` | `non-phase-locked` | `total'
        How to plot the spectrograms:
            raw -- plot without averaging power (default)
            phase-locked -- just average the event-related potential (ERP)
            non-phase-locked -- subtract the ERP from each epoch, do time
                                frequency decomposition (TFR) then average
            total -- do TFR on each epoch and then average
    baseline : `z-score` | `gain`
        How to baseline specrogram data:
            z-score -- for each frequency, subtract the median and divide
                       by the standard deviation (default)
            gain -- divide by median
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
    columns : int
        The number of columns to use in the plot (for `method=raw`).
    plot_bursts : bool
        Whether to include vertical bars for when the bursts are detected
        (for `method=raw`).
    picks : None | list of str
        The names of the channels to plot
    '''
    config = get_config()
    if method not in ('raw', 'phase-locked', 'non-phase-locked', 'total'):
        raise ValueError('Unrecognized method %s' % method)
    if picks is None:
        picks = raw.ch_names
    else:
        raw = raw.pick_channels(picks)
    if method == 'raw' and len(picks) > 1:
        raise ValueError('Only one channel can be plotted at a time ' +
                         'for raw spectrograms')
    picks_str = '_'.join(picks).replace(' ', '_')
    plotf = derivative_fname(rawf, 'plots/spectrograms',
                             '%s_spectrogram' % event +
                             '%s_%s' % (method, picks_str) +
                             '_%s_power' % baseline,
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('Spectrogram plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    if method == 'raw' and plot_bursts:
        bursts = find_bursts(rawf, return_saved=True)
    freqs = np.arange(lfreq, hfreq + dfreq, dfreq)
    if isinstance(n_cycles, np.array) and len(freqs) != len(n_cycles):
        raise ValueError('Mismatch lengths n_cycles %s to freqs %s' %
                         (n_cycles, freqs))
    epochs = Epochs(raw, events, tmin=config['tmin'] - 1, baseline=None,
                    tmax=config['tmax'] + 1, preload=True)
    bl_epochs = Epochs(raw, bl_events, tmin=config['baseline_tmin'] - 1,
                       baseline=None, tmax=config['baseline_tmax'] + 1,
                       preload=True)
    if method == 'phase-locked':
        bl_evoked = EvokedArray(np.median(bl_epochs._data, axis=0),
                                info=bl_epochs.info, tmin=bl_epochs.tmin,
                                nave=len(bl_epochs))
        bl_evoked_tfr = tfr_morlet(bl_evoked, freqs, n_cycles=n_cycles,
                                   use_fft=use_fft, return_itc=False)
        bl_evoked_tfr.crop(tmin=config['baseline_tmin'],
                           tmax=config['baseline_tmax'])
        evoked = EvokedArray(np.median(epochs._data, axis=0),
                             info=epochs.info, tmin=epochs.tmin,
                             nave=len(epochs))
        evoked_tfr = tfr_morlet(evoked, freqs, n_cycles=n_cycles,
                                use_fft=use_fft, return_itc=False)
        evoked_tfr.crop(tmin=config['tmin'], tmax=config['tmax'])
        evoked_tfr.data = ((evoked_tfr.data -
                            np.median(bl_evoked_tfr.data,
                                      axis=2)[:, :, np.newaxis]) /
                           np.std(bl_evoked_tfr.data,
                                  axis=2)[:, :, np.newaxis])
    else:
        if method == 'non-phase-locked':
            epochs._data -= np.median(epochs._data, axis=0)
        cropped_epochs = epochs.copy().crop(tmin=config['tmin'],
                                            tmax=config['tmax'])
        cropped_bl_epochs = bl_epochs.copy().crop(
            tmin=config['baseline_tmin'], tmax=config['baseline_tmax'])
        epochs_data = np.zeros((len(epochs), len(epochs.ch_names), len(freqs),
                                len(cropped_epochs.times)))
        bl_epochs_data = np.zeros((len(bl_epochs), len(bl_epochs.ch_names),
                                   len(freqs), len(cropped_bl_epochs.times)))
        epochs_tfr = EpochsTFR(epochs.info, epochs_data, cropped_epochs.times,
                               freqs, verbose=False)
        bl_epochs_tfr = EpochsTFR(bl_epochs.info, bl_epochs_data,
                                  cropped_bl_epochs.times, freqs,
                                  verbose=False)
        if method != 'raw':
            evoked_tfr_data = np.zeros((len(epochs.ch_names), len(freqs),
                                        len(cropped_epochs.times)))
            evoked_tfr = AverageTFR(epochs.info, evoked_tfr_data,
                                    cropped_epochs.times, freqs,
                                    nave=len(epochs))
        for i, ch in enumerate(epochs.ch_names):
            if verbose:
                print('\nComputing TFR (%i/%i)' % (i, len(epochs.ch_names)) +
                      'for %s... Computing frequency' % ch,
                      end=' ', flush=True)
            this_epochs = epochs.copy().pick_channels([ch])
            this_bl_epochs = bl_epochs.copy().pick_channels([ch])
            for j, freq in enumerate(freqs):
                if verbose:
                    print(freq, end=' ', flush=True)
                this_n_cycles = (n_cycles if isinstance(n_cycles, int) else
                                 n_cycles[i])
                this_bl_epochs_tfr = \
                    tfr_morlet(this_bl_epochs, [freq], n_cycles=this_n_cycles,
                               use_fft=use_fft, average=False,
                               return_itc=False, verbose=False)
                this_bl_epochs_tfr = this_bl_epochs_tfr.crop(
                    tmin=config['baseline_tmin'], tmax=config['baseline_tmax'])
                this_epochs_tfr = \
                    tfr_morlet(this_epochs, [freq], n_cycles=this_n_cycles,
                               use_fft=use_fft, average=False,
                               return_itc=False, verbose=False)
                this_epochs_tfr = this_epochs_tfr.crop(
                    tmin=config['tmin'], tmax=config['tmax'])
                full_data = np.concatenate([this_bl_epochs_tfr.data,
                                            this_epochs_tfr.data], axis=3)
                epochs_tfr.data[:, i:i + 1, j:j + 1, :] = \
                    ((this_epochs_tfr.data -
                      np.median(full_data, axis=3)[:, :, :, np.newaxis]) /
                     np.std(full_data, axis=3)[:, :, :, np.newaxis])
                bl_epochs_tfr.data[:, i:i + 1, j:j + 1, :] = \
                    ((this_bl_epochs_tfr.data -
                      np.median(full_data, axis=3)[:, :, :, np.newaxis]) /
                     np.std(full_data, axis=3)[:, :, :, np.newaxis])
                if method != 'raw':
                    this_evoked_tfr = np.median(epochs_tfr.data[:, i, j],
                                                axis=0)
                    this_bl_evoked_tfr = np.median(bl_epochs_tfr.data[:, i, j],
                                                   axis=0)
                    evoked_tfr.data[i, j] = \
                        ((this_evoked_tfr - np.median(this_bl_evoked_tfr)) /
                         np.std(this_bl_evoked_tfr))
    if method == 'raw':
        ch_name = epochs_tfr.ch_names[0]
        vmin, vmax = np.min(epochs_tfr.data), np.max(epochs_tfr.data)
        if verbose:
            print('Plotting spectrogram for channel %s' % ch_name)
            if plot_bursts:
                n_bursts = len(bursts[bursts['channel'] == ch_name])
                print('%i bursts for this channel total' % n_bursts)
        rows = int(len(events) / columns)
        fig, axes = plt.subplots(rows, columns)
        fig.set_size_inches(columns * 4, len(events))
        axes = axes.flatten()
        for j, this_tfr in enumerate(epochs_tfr):
            cmap = _plot_spectrogram(axes[j], this_tfr[i],
                                     epochs_tfr.times, vmin, vmax, freqs,
                                     j % columns == 0,
                                     j == int(len(events) / 2), False)
            if plot_bursts:
                _plot_bursts(config, events, raw, bursts, j, axes, ch_name)
        axes = axes.reshape(rows, columns)
        for col_idx in range(columns):
            axes[-1, col_idx].set_xlabel('Time (s)')
            axes[-1, col_idx].set_xticks(
                np.linspace(0, len(epochs_tfr.times), 5))
            axes[-1, col_idx].set_xticklabels(
                ['%.2f' % t for t in np.linspace(epochs_tfr.times[0],
                                                 epochs_tfr.times[-1], 5)])
    else:
        vmin, vmax = np.min(evoked_tfr.data), np.max(evoked_tfr.data)
        if raw.info['dig'] is None:
            nrows = int(len(raw.ch_names) ** 0.5)
            ncols = int(len(raw.ch_names) / nrows) + 1
            fig, axes = plt.subplots(nrows, ncols)
            axes = axes.flatten()
            for idx, ax in enumerate(axes):
                if idx < len(picks):
                    cmap = _plot_spectrogram(
                        ax, evoked_tfr.data[idx], epochs_tfr.times,
                        vmin, vmax, freqs,
                        show_xticks=idx >= len(picks) - nrows - 1,
                        show_yticks=idx % ncols == 0,
                        show_ylabel=idx == int(len(axes) / 2))
                    ax.set_title(raw.ch_names[idx])
                else:
                    ax.axis('off')
        else:
            for ax, idx in iter_topography(raw.info, fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white'):
                cmap = _plot_spectrogram(ax, this_tfr, epochs_tfr.times,
                                         vmin, vmax, freqs)
        fig.subplots_adjust(right=0.85, hspace=0.3)
        cax = fig.add_subplot(position=[0.9, 0.1, 0.05, .8])
        cax = fig.colorbar(cmap, cax=cax)
        cax.set_label(('%s Power %s Normalized' % (method, baseline)
                       ).title())
        fig.set_size_inches(12, 8)
        fig.suptitle('Time Frequency Decomposition for the %s ' % event +
                     'Event, %s Power' % baseline.capitalize())
        fig.savefig(plotf, dpi=300)
        plt.close(fig)


def _plot_spectrogram(ax, this_tfr, times, vmin, vmax,
                      freqs, show_yticks=True,
                      show_ylabel=True, show_xticks=True):
    '''Plot a single spectrogram'''
    cmap = ax.imshow(this_tfr, aspect='auto', vmin=vmin,
                     vmax=vmax, cmap='RdYlBu_r')
    ax.invert_yaxis()
    if show_yticks:
        ax.set_yticks(np.linspace(0, len(freqs), 3))
        ax.set_yticklabels(['%i' % f for f in
                           np.linspace(freqs[0], freqs[-1], 3)])
    else:
        ax.set_yticklabels([])
    if show_ylabel:
        ax.set_ylabel('Frequency (Hz)')
    ax.axvline(np.where(times == 0)[0][0], color='k')
    if show_xticks:
        ax.set_xlabel('Time (s)')
        ax.set_xticks(np.linspace(0, len(times), 5))
        ax.set_xticklabels(['%.1f' % t for t in
                            np.linspace(times[0], times[-1], 5)])
    else:
        ax.set_xticks([])
    return cmap


def _plot_bursts(config, events, raw, bursts, j, axes, ch_name):
    '''Plot bursts on a single spectrogram'''
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


def plot_group_bursting(rawfs, event, events, tfr_name='beta',
                        picks=None, method='all', ylim=None, rolling=0.25,
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
            if rawf.path not in events[name]:
                continue
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
                  picks=None, method='all', ylim=None, rolling=0.25,
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
                             basename, config['fig'])
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
        ax.legend()
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
    info = infos[0]
    if any([(this_info['sfreq'] != info['sfreq'] or
             this_info['ch_names'] != info['ch_names'])
            for this_info in infos[1:]]):
        raise ValueError('Info objects in infos have a different sampling ' +
                         'frequency or channel names')
    if info['sfreq'] != sfreq:
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
    picks_str = ('' if picks is None else 'channels_' + '_and_'.join(picks))
    names_str = '_and_'.join(events.keys())
    basename = '%s_%s_power_%s_%s.%s' % (names_str, tfr_name,
                                         event, picks_str, config['fig'])
    plotf = derivative_fname(rawf, 'plots/%s_power' % tfr_name,
                             basename, config['fig'])
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
              '%s trials during the %s event' % (names_str, event))
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
        for name in events:
            for idx, ch_name in enumerate(picks):
                label = ch_name if name is None else name + ' ' + ch_name
                _plot_power(label, events[name], sfreq, tfr[idx],
                            np.quantile(tfr, 0.9), ax=ax)
        ax.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.suptitle(('%s Trials z-scored %s Power for ' % (name, tfr_name) +
                  'the %s Event from ' % event +
                  '%s to %s Seconds' % (config['tmin'], config['tmax'])
                  ).title())
    fig.savefig(plotf, dpi=300)


def _plot_power(name, events, sfreq,
                tfr_ch, ylim, ax=None, verbose=True):
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    if verbose:
        print('Plotting %s' % name)
    if ax is None:
        fig, ax = plt.subplots()
    epochs_tfr = np.zeros((len(events), len(bin_indices)))
    for i, event_index in enumerate(events[:, 0]):
        epochs_tfr[i] = [tfr_ch[event_index + offset] for offset in
                         bin_indices]
    evoked_tfr = np.mean(epochs_tfr, axis=0)
    evoked_tfr_std = np.std(epochs_tfr, axis=0)
    ax.plot(times, evoked_tfr, label=name)
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
