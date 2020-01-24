import numpy as np
import os.path as op
from pandas import concat

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from swann.utils import get_config, derivative_fname
from swann.analyses import decompose_tfr, find_bursts

from mne.viz import iter_topography
from mne.time_frequency import tfr_morlet
from mne import Epochs


def plot_spectrogram(rawf, raw, event, events, columns=3, lfreq=4,
                     hfreq=50, dfreq=1, n_cycles=7, use_fft=True,
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
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/spectrograms',
                             event + '_spectrogram_%s',
                             config['fig'])
    if op.isdir(op.dirname(plotf)) and not overwrite:
        print('Spectrogram plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    epochs = Epochs(raw, events, tmin=config['tmin'] - 1,
                    tmax=config['tmax'] + 1)
    epochs_tfr = tfr_morlet(epochs, np.arange(lfreq, hfreq, dfreq),
                            n_cycles=n_cycles, use_fft=use_fft,
                            average=False, return_itc=False)
    epochs_tfr.crop(tmin=config['tmin'], tmax=config['tmax'])
    for i, ch_name in enumerate(epochs_tfr.ch_names):
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
                                  vmax=vmax, cmap='coolwarm')
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
        axes = axes.reshape(rows, columns)
        for col_idx in range(columns):
            axes[-1, col_idx].set_xlabel('Time (s)')
            axes[-1, col_idx].set_xticks(
                np.linspace(0, len(epochs_tfr.times), 10))
            axes[-1, col_idx].set_xticklabels(
                ['%.2f' % t for t in np.linspace(epochs_tfr.times[0],
                                                 epochs_tfr.times[-1], 10)])
        cax = fig.colorbar(cmap, cax=cax)
        cax.set_label('Power')
        fig.suptitle('Time Frequency Decomposition for the %s ' % event +
                     'Event, Channel %s' % ch_name)
        fig.savefig(this_plot_f)
    plt.close('all')


def plot_group_bursting(rawfs, name, event, events, tfr_name='beta',
                        infos=None, picks=None, method='peaks', ylim=0.5,
                        verbose=True, overwrite=False):
    ''' Plots bursts on topography or in graph overlayed.
    Parameters
    ----------
    rawfs : list of pybids.BIDSlayout file
        The object containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`).
    event : str
        The name of the event (e.g. `Response`).
    events : dict keys: rawf.path, values: np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    tfr_name : str
        What to call the frequencies used e.g. beta
    info : mne.io.Raw.info
        The info for topographic plotting
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    method : ('peaks', 'all', 'durations')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    '''
    if infos is None and picks is None:
        raise ValueError('`infos` and `picks` are none, if no picks are ' +
                         'specified, infos are needed for topo plotting.')
    config = get_config()
    plotf = \
        op.join(config['bids_dir'], 'derivatives/plots/%s_bursting' % tfr_name,
                'group_%s_%s_burst_%s_%s' % (tfr_name, name, method, event) +
                ('' if picks is None else '_' + '_'.join(picks)) +
                '.' + config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('Group %s bursting plot for %s ' % (tfr_name.title(), event) +
              'already exists, use `overwrite=True` to replot')
        return
    if picks is None:
        ch_names = picks
    else:
        ch_names = set([ch for info in infos for ch in info['ch_names']])
    bursts = concat([find_bursts(rawf, return_saved=True) for rawf in rawfs])
    info_max_idx = np.argmax([len(info['ch_names']) for info in infos])
    info = infos[info_max_idx]
    fig = _plot_bursting(ch_names, bursts, name, event, events, tfr_name, info,
                         picks, method, ylim, verbose)
    fig.set_size_inches(10, 12)
    fig.suptitle('Group %s Trials %s Bursting for ' % (tfr_name, name) +
                 'the %s Event from ' % event +
                 '%s to %s Seconds' % (config['tmin'], config['tmax']))
    fig.savefig(plotf)


def plot_bursting(rawf, name, event, events, tfr_name='beta',
                  info=None, picks=None, method='peaks', ylim=0.5,
                  verbose=True, overwrite=False):
    ''' Plots bursts on topography or in graph overlayed.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`).
    event : str
        The name of the event (e.g. `Response`).
    events : np.array(n_events, 3) or dict with key: name values: events
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    tfr_name : str
        What to call the frequencies used e.g. beta
    info : mne.io.Raw.info
        The info for topographic plotting
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
    method : ('peaks', 'all', 'durations')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    '''
    if info is None and picks is None:
        raise ValueError('`info` and `picks` are none, if no picks are ' +
                         'specified, info is needed for topo plotting.')
    config = get_config()
    plotf = derivative_fname(rawf, 'plots/%s_bursting' % tfr_name,
                             '%s_%s_burst_%s_%s' % (tfr_name, name,
                                                    method, event) +
                             ('' if picks is None else '_' + '_'.join(picks)),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('%s bursting plot for %s ' % (tfr_name.title(), event) +
              'already exists, use `overwrite=True` to replot')
        return
    ch_names = info['ch_names'] if picks is None else picks
    events = events if isinstance(events, dict) else {None: events}
    bursts = find_bursts(rawf, return_saved=True)
    fig = _plot_bursting(ch_names, bursts, name, event, events, tfr_name, info,
                         picks, method, ylim, verbose)
    fig.set_size_inches(10, 12)
    fig.suptitle('%s Trials %s Bursting for ' % (tfr_name, name) +
                 'the %s Event from ' % event +
                 '%s to %s Seconds' % (config['tmin'], config['tmax']))
    fig.savefig(plotf)


def _plot_bursting(ch_names, bursts, name, event, events, tfr_name, info,
                   picks, method, ylim, verbose):
    if verbose:
        print('Plotting %s bursting for %s ' % (tfr_name, name) +
              'trials during the %s event' % event)
    if picks is None:
        for ax, idx in iter_topography(info, fig_facecolor='white',
                                       axis_facecolor='white',
                                       axis_spinecolor='white'):
            for name, events in events.items():
                name = (ch_names[idx] if name is None else
                        name + ' ' + ch_names[idx])
                _plot_bursts(name, events, info['sfreq'],
                             bursts[bursts['channel'] ==
                                    ch_names[idx]], method,
                             ylim, ax=ax)
    else:
        fig, ax = plt.subplots()
        for ch_name in picks:
            for name, events in events.items():
                name = (ch_name if name is None else
                        name + ' ' + ch_name)
                _plot_bursts(name, events, info['sfreq'],
                             bursts[bursts['channel'] ==
                                    ch_name], method,
                             ylim, ax=ax)
        ax.legend()
    return plt.gcf()


def _plot_bursts(name, events, sfreq, my_bursts,
                 method, ylim, ax=None, verbose=True):
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    if verbose:
        print('Plotting %s' % name)
    if ax is None:
        fig, ax = plt.subplots()
    if method == 'peaks':
        burst_indices = set([i for start, stop in
                             zip(my_bursts['burst_start'],
                                 my_bursts['burst_end'])
                             for i in range(start, stop)])
    elif method == 'all':
        burst_indices = set(my_bursts['burst_peak'])
    elif method == 'durations':
        durations = [(stop - start) / sfreq for start, stop in
                     zip(my_bursts['burst_start'], my_bursts['burst_end'])]
        n_bars = len([rect for rect in ax.get_children() if
                      isinstance(rect, Rectangle)]) - 1
        ax.bar(n_bars, np.mean(durations), yerr=np.std(durations))
        if n_bars == 0:
            ax.set_xticklabels([name])
        else:
            ax.set_xticklabels(ax.get_xticklabels() + [name])
    else:
        raise ValueError('Method of calculating bursts %s ' % method +
                         'not recognized.')
    bins = np.zeros((len(bin_indices)))
    for i, event_index in enumerate(events[:, 0]):
        for j, offset in enumerate(bin_indices):
            if event_index - offset in burst_indices:
                bins[j] += 1
    bins /= len(events)
    ax.plot(times, bins, label=name)
    ax.set_ylim([0, ylim])


def plot_group_power(rawf, name, event, info, ch_names, events,
                     tfr_name='beta', ylim=0.5,
                     verbose=True, overwrite=False):
    # find max number of channels
    # for each channel append tfr on sequentially, adjust event index
    pass


def plot_power(rawf, name, event, info, events, tfr_name='beta',
               picks=None, verbose=True, overwrite=False):
    ''' Plots a line graph with one standard deviation shaded of power.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`).
    event : str
        The name of the event (e.g. `Response`).
    info : instance of mne.io.Raw.info
        The info object with channel names and positions
    events : np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    tfr_name : str
        The keyword name of the tfr that was previously computed.
    picks : list(str) | None
        If None, all the channels will be plotted on the topo. If channels are
        given, they will be overlayed on one plot.
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
    fig.savefig(plotf)


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
    fig.savefig(plotf)


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
