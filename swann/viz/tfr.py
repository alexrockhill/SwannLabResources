import os
import numpy as np
import os.path as op

import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS, SymLogNorm
from scipy.stats import ttest_ind

from swann.preprocessing import get_info
from swann.utils import get_config, derivative_fname
from swann.analyses import decompose_tfr, find_bursts, get_bursts

from mne.viz import iter_topography
from mne.time_frequency import tfr_morlet, EpochsTFR, AverageTFR
from mne import Epochs, EvokedArray


def plot_spectrogram(rawf, raw, event, events, bl_events,
                     method='raw', baseline='z-score',
                     freqs=np.logspace(np.log(4), np.log(250), 50, base=np.e),
                     n_cycles=7, use_fft=True, ncols=3, plot_erp=True,
                     plot_bursts=False, picks=None, verbose=True,
                     overwrite=False):
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
    freqs : np.array
        The frequencies over which to compute the spectral data.
    n_cycles : int, np.array
        The number of cycles to use in the Morlet transform
    use_fft : bool
        Use Fast Fourier Transform see `mne.time_frequency.tfr.cwt`.
    ncols : int
        The number of ncols to use in the plot (for `method=raw`).
    plot_erp : bool
        Whether to plot the event-related potential on top.
    plot_bursts : bool
        Whether to include vertical bars for when the bursts are detected
        (for `method=raw`).
    picks : None | list of str
        The names of the channels to plot
    '''
    config = get_config()
    raw = raw.copy()
    if method not in ('raw', 'phase-locked', 'non-phase-locked', 'total'):
        raise ValueError('Unrecognized method %s' % method)
    if picks is None:
        picks = raw.ch_names
    else:
        if isinstance(picks, str):
            picks = [picks]
        raw = raw.pick_channels(picks)
    if method == 'raw' and len(picks) > 1:
        raise ValueError('Only one channel can be plotted at a time ' +
                         'for raw spectrograms')
    picks_str = '_'.join(picks).replace(' ', '_')
    plotf = derivative_fname(rawf, 'plots/spectrograms',
                             'event-%s_spectrogram_' % event +
                             '%s_%s' % (method, picks_str) +
                             '_%s_power' % baseline,
                             config['fig'])
    if op.isfile(plotf) and not overwrite:
        print('Spectrogram plot for %s already exists, ' % event +
              'use `overwrite=True` to replot')
        return
    if method == 'raw' and plot_bursts:
        bursts = find_bursts(rawf, return_saved=True)
    if isinstance(n_cycles, np.ndarray) and len(freqs) != len(n_cycles):
        raise ValueError('Mismatch lengths n_cycles %s to freqs %s' %
                         (n_cycles, freqs))
    epochs = Epochs(raw, events, tmin=config['tmin'] - 1, baseline=None,
                    tmax=config['tmax'] + 1, preload=True)
    bl_epochs = Epochs(raw, bl_events, tmin=config['baseline_tmin'] - 1,
                       baseline=None, tmax=config['baseline_tmax'] + 1,
                       preload=True)
    cropped_epochs = epochs.copy().crop(tmin=config['tmin'],
                                        tmax=config['tmax'])
    cropped_bl_epochs = bl_epochs.copy().crop(
        tmin=config['baseline_tmin'], tmax=config['baseline_tmax'])
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
                      end=' ', flush=True)  # noqa
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
        emin, emax = np.min(cropped_epochs._data), np.max(cropped_epochs._data)
        if verbose:
            print('Plotting spectrogram for channel %s' % ch_name)
            if plot_bursts:
                n_bursts = len(bursts[bursts['channel'] == ch_name])
                print('%i bursts for this channel total' % n_bursts)
        nrows = int(np.ceil(len(events) / ncols))
        fig, axes = plt.subplots(nrows, ncols)
        fig.set_size_inches(ncols, nrows)
        axes = axes.flatten()
        for j, this_tfr in enumerate(epochs_tfr):
            evoked_data = (cropped_epochs._data[j, 0], emin, emax)
            cmap = _plot_spectrogram(
                axes[j], this_tfr[i], epochs_tfr.times,
                vmin, vmax, freqs, evoked_data,
                show_xticks=j >= len(events) - ncols,
                show_yticks=j % ncols == 0,
                show_ylabel=j == int(nrows / 2) * ncols)
            if plot_bursts:
                _plot_bursts(config, events, raw, bursts, j, axes, ch_name)
        for ax in axes[len(epochs_tfr):]:
            ax.axis('off')
    else:
        if plot_erp:
            evoked_data = np.median(cropped_epochs._data, axis=0)
            evoked_data -= np.median(evoked_data, axis=1)[:, np.newaxis]
            evoked = EvokedArray(evoked_data, info=epochs.info,
                                 tmin=epochs.tmin, nave=len(epochs))
            emin, emax = np.min(evoked.data), np.max(evoked.data)
        vmin, vmax = np.min(evoked_tfr.data), np.max(evoked_tfr.data)
        if raw.info['dig'] is None:
            nrows = int(len(raw.ch_names) ** 0.5)
            ncols = int(len(raw.ch_names) / nrows) + 1
            fig, axes = plt.subplots(nrows, ncols)
            fig.set_size_inches(12, 8)
            axes = axes.flatten()
            for idx, ax in enumerate(axes):
                if idx < len(picks):
                    cmap = _plot_spectrogram(
                        ax, evoked_tfr.data[idx], epochs_tfr.times,
                        vmin, vmax, freqs, ((evoked.data[idx], emin, emax) if
                                            plot_erp else None),
                        show_xticks=idx >= len(picks) - ncols,
                        show_yticks=idx % ncols == 0,
                        show_ylabel=idx % int(nrows / 2) * ncols)
                    ax.set_title(raw.ch_names[idx])
                else:
                    ax.axis('off')
        else:
            for ax, idx in iter_topography(raw.info, fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white'):
                cmap = _plot_spectrogram(
                    ax, this_tfr, epochs_tfr.times, vmin, vmax, freqs,
                    ((evoked.data[idx], emin, emax) if plot_erp else None))
        fig.subplots_adjust(right=0.85, hspace=0.3)
        cax = fig.add_subplot(position=[0.87, 0.1, 0.05, 0.8])
        cax = fig.colorbar(cmap, cax=cax, format='%.2f',
                           ticks=[vmin, vmin / 10, vmin / 100,
                                  vmax / 100, vmax / 10, vmax])
        cax.set_label(('Log %s Power %s Normalized' % (method, baseline)
                       ).title())
        fig.suptitle('Time Frequency Decomposition for the %s ' % event +
                     'Event, %s Power' % baseline.capitalize())
        fig.savefig(plotf, dpi=300)
        plt.close(fig)


def _plot_spectrogram(ax, this_tfr, times, vmin, vmax,
                      freqs, evoked_data, show_yticks=True,
                      show_ylabel=True, show_xticks=True):
    '''Plot a single spectrogram'''
    cmap = ax.imshow(this_tfr, cmap='RdYlBu_r', aspect='auto',
                     extent=[0, this_tfr.shape[1], 0, this_tfr.shape[0]],
                     norm=SymLogNorm(linthresh=(vmax - vmin) / 100,
                                     vmin=vmin, vmax=vmax))
    if evoked_data is not None:
        evoked, emin, emax = evoked_data
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.plot(range(this_tfr.shape[1]), evoked, alpha=0.25, color='k')
        ax2.set_ylim([emin, emax])
    ax.invert_yaxis()
    if show_yticks:
        ax.set_yticks(np.linspace(0, len(freqs), 5))
        ax.set_yticklabels(['%i' % f for f in
                           freqs[::-int(len(freqs) / 5)]])
    else:
        ax.set_yticklabels([])
    if show_ylabel:
        ax.set_ylabel('Frequency (Hz)')
    ax.axvline(np.where(times == 0)[0][0], color='k')
    if show_xticks:
        ax.set_xlabel('Time (s)')
        ax.set_xticks(np.linspace(0, len(times), 3))
        ax.set_xticklabels(['%.1f' % t for t in
                            np.linspace(times[0], times[-1], 3)])
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
