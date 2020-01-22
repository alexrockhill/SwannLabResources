import numpy as np
import os.path as op

import matplotlib.pyplot as plt

from swann.utils import get_config, derivative_fname, my_events
from swann.analyses import decompose_tfr, find_bursts

from mne.viz import iter_topography


def plot_group_beta_bursting(rawf, name, event, info, ch_names, events,
                             method='peaks', ylim=0.5,
                             verbose=True, overwrite=False):
    pass


def plot_beta_bursting(rawf, name, event, info, ch_names, events,
                       method='peaks', ylim=0.5,
                       verbose=True, overwrite=False):
    ''' Plots a bar chart of beta bursts.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    name : str
        The name of the trials being passed (e.g. `Slow` or `All`)
    event : str
        The name of the event (e.g. `Response`)
    raw : mne.io.Raw.info
        The raw info for topography
    ch_names : list(str)
        The names of the channels
    events : np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    method : ('peaks', 'all')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    '''
    config = get_config()
    plotf = derivative_fname(rawf, 'plots',
                             '%s_beta_burst_%s_%s' % (name, method, event),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('Beta bursting plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    beta_bursts = find_bursts(rawf, return_saved=True)
    beta_burst_lim = \
        np.quantile([len(beta_bursts[beta_bursts['channel'] == ch])
                     for ch in ch_names], ylim)
    if verbose:
        print('Plotting beta bursting for %s trials during ' % name +
              'the %s event' % event)
    for ax, idx in iter_topography(info, fig_facecolor='white',
                                   axis_facecolor='white',
                                   axis_spinecolor='white'):
        _plot_beta_bursts(ch_names[idx], events, info['sfreq'],
                          beta_bursts[beta_bursts['channel'] ==
                                      ch_names[idx]], method,
                          beta_burst_lim, ax=ax)
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    fig.suptitle('%s Trials Beta Bursting for ' % name +
                 'the %s Event from ' % event +
                 '%s to %s Seconds' % (config['tmin'], config['tmax']))
    fig.savefig(plotf)


def plot_beta_bursts_comparison(rawf, behf, chs=['C3', 'C4'], ylim=0.75,
                                verbose=True, overwrite=True):
    ''' Plots the beta power overlayed with burst times for given channels.
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    behf : pybids.BIDSlayout file
        The object containing the behavioral data.
    ylim : 0 < float < 1
        The scale of the yaxis (relative to channel with the max number of
        bursts).
    '''
    config = get_config()
    if all([op.isfile(derivative_fname(rawf, 'plots',
                                       'beta_bursts_sf_comp_%s' % event,
                                       config['fig']))
            for event in my_events()]) and not overwrite:
        print('Beta bursting plot already exist, use `overwrite=True` ' +
              'to replot')
        return
        plotf = derivative_fname(rawf, 'plots',
                                 'beta_bursts_sf_comp_%s' % event,
                                 config['fig'])
        fig, ax = plt.subplots()
        _plot_beta_bursts(ch_names[idx], events[event], sfreq,
                          beta_bursts[beta_bursts['channel'] ==
                                      ch_names[idx]],
                          beta_burst_lim, tfr[idx], ax=ax)
        fig.set_size_inches(10, 10)
        fig.suptitle('%s Trials Beta Bursting for ' % name +
                     'the %s Event from ' % event +
                     '%s to %s Seconds' % (config['tmin'], config['tmax']))
        fig.savefig(plotf)


def _plot_beta_bursts(ch_name, my_events, sfreq, my_beta_bursts,
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
    for i, event_index in enumerate(my_events[:, 0]):
        for j, offset in enumerate(bin_indices):
            if event_index - offset in beta_burst_indices:
                bins[j] += 1
    ax.plot(times, bins)
    ax.set_ylim([0, ylim])
    return plt.gcf()


def plot_group_power(rawf, name, event, info, ch_names, events,
                     tfr_name='beta', ylim=0.5,
                     verbose=True, overwrite=False):
    pass


def plot_power(rawf, name, event, info, ch_names, events, tfr_name='beta',
               ylim=0.5, verbose=True, overwrite=False):
    config = get_config()
    plotf = derivative_fname(rawf, 'plots',
                             '%s_%s_power_%s' % (name, tfr_name, event),
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('%s power plot already exist, ' % tfr_name.capitalize() +
              'use `overwrite=True` to replot')
        return
    tfr, my_ch_names, sfreq = decompose_tfr(rawf, tfr_name, return_saved=True)
    tfr /= tfr.std(axis=1)[:, np.newaxis]  # z-score power
    if sfreq != info['sfreq']:
        raise ValueError('Raw sampling frequency mismatch with tfr sfreq')
    if any(ch_names != my_ch_names):
        raise ValueError('Raw channel names mismatch with tfr channel names')
    if verbose:
        print('Plotting %s power for ' % tfr_name.capitalize() +
              '%s trials during the %s event' % (name, event))
    for ax, idx in iter_topography(info, fig_facecolor='white',
                                   axis_facecolor='white',
                                   axis_spinecolor='white'):
        _plot_power(ch_names[idx], events, sfreq,
                    tfr[idx], ax=ax)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.suptitle(('%s Trials %s Power for ' % (name, tfr_name) +
                  'the %s Event from ' % event +
                  '%s to %s Seconds' % (config['tmin'], config['tmax'])
                  ).title())
    fig.savefig(plotf)


def _plot_power(ch_name, my_events, sfreq,
                tfr_ch, ax=None, verbose=True):
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    bin_indices = range(int(sfreq * tmin), int(sfreq * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    if verbose:
        print('Plotting channel %s' % ch_name)
    if ax is None:
        fig, ax = plt.subplots()
    epochs_tfr = np.zeros((len(my_events), len(bin_indices)))
    for i, event_index in enumerate(my_events[:, 0]):
        epochs_tfr[i] = [tfr_ch[event_index + offset] for offset in
                         bin_indices]
    evoked_tfr = np.mean(epochs_tfr, axis=0)
    evoked_tfr -= np.mean(evoked_tfr)
    evoked_tfr_std = np.std(epochs_tfr, axis=0)
    ax.plot(times, evoked_tfr, label='Power')
    ax.fill_between(times, evoked_tfr - evoked_tfr_std,
                    evoked_tfr + evoked_tfr_std, alpha=0.25)
    ax.set_ylim([0, 1])
    return plt.gcf()


def plot_beta_burst_shape():
    pass


def plot_group_beta_burst_shape():
    pass


def plot_beta_burst_topo():
    pass


def plot_group_beta_burst_topo():
    pass
