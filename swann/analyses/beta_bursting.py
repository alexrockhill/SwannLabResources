import numpy as np

from pandas import DataFrame, read_csv

from swann.utils import (get_config, derivative_fname,
                         check_overwrite_or_return, rolling_mean)

from swann.preprocessing import get_info

from mne.time_frequency.tfr import cwt
from mne.time_frequency import morlet


def get_bursts(rawf, events, method, rolling=0.25):
    ''' Threshold a signal to find bursts.
    Parameters
    ----------
    rawfs : pybids.BIDSlayout object
        The object pointing to the data file
    events : np.array(n_events, 3)
        Events from mne.events_from_annotations or mne.find_events.
    method : ('peaks', 'all', 'durations', 'shape')
        Plot only the peak values (`peaks`) of beta bursts or plot all
        time values (`all`) during which a beta burst is occuring.
    Returns
    -------
    times : np.array float
        The times over which the bursts are considered
    beta_data : dict
        The dictionary with keys for channel and values of the bursts
        calculated by the given method
    rolling : float
        Amount of time to using for the rolling average (default 250 ms)
        Note: this is ignored for durations
    '''
    config = get_config()
    tmin, tmax = config['tmin'], config['tmax']
    burst_data = dict()
    bursts = find_bursts(rawf, return_saved=True)
    info = get_info(rawf)
    sfreq = info['sfreq']
    bin_indices = range(int(info['sfreq'] * tmin),
                        int(info['sfreq'] * tmax))
    times = np.linspace(tmin, tmax, len(bin_indices))
    for ch in info['ch_names']:
        these_bursts = bursts[bursts['channel'] == ch]
        if method == 'all':
            burst_indices = set([i for start, stop in
                                 zip(these_bursts['burst_start'],
                                     these_bursts['burst_end'])
                                 for i in range(start, stop)])
        elif method == 'peaks':
            burst_indices = set(these_bursts['burst_peak'])
        elif method == 'durations':
            event_indices = set([i for e in events[:, 0]
                                 for i in bin_indices + e])
            durations = [(stop - start) / sfreq for start, stop in
                         zip(these_bursts['burst_start'],
                             these_bursts['burst_end'])
                         if start in event_indices or stop in event_indices]
            burst_data[ch] = durations
            continue
        else:
            raise ValueError('Method of calculating bursts %s ' % method +
                             'not recognized.')
        bins = np.zeros((len(bin_indices)))
        for i, event_index in enumerate(events[:, 0]):
            for j, offset in enumerate(bin_indices):
                if event_index - offset in burst_indices:
                    bins[j] += 1
        bins /= len(events)
        bins = rolling_mean(bins, n=int(sfreq * rolling))
        burst_data[ch] = (times, bins)
    return burst_data


def find_bursts(bf, signal=None, ch_names=None, thresh=6, return_saved=False,
                verbose=True, overwrite=False):
    ''' Threshold a signal to find bursts.
    Parameters
    ----------
    bf : pybids.BIDSlayout
        A behavioral/electrophysiology bids file for naming
    signal : np.ndarray(ch, times)
        The signal in the form channels x times.
    ch_names : np.array(ch)
        The names of the channels (for clarity, disambiguity)
    thresh : float
        The number of times greater than the median to count
        as a burst.
    Returns
    -------
    beta_bursts : dict
        The dictionary with beta bursts with keys channel, burst_start and
        burst_end.
    '''
    burstf = derivative_fname(bf, 'data', 'bursts', 'tsv')
    return_saved = \
        check_overwrite_or_return('beta bursts', burstf, return_saved,
                                  overwrite, verbose)
    if return_saved:
        return read_csv(burstf, sep='\t')
    bursts = dict(channel=list(), burst_start=list(), burst_peak=list(),
                  burst_end=list())
    threshs = [np.median(ch_sig) * thresh for ch_sig in signal]
    if verbose:
        print('Finding beta bursts for %s' % bf.path)
    for i, ch in enumerate(ch_names):
        if verbose:
            print('Finding bursts for channel %s' % ch)
        burst = False
        for t in range(signal.shape[1]):
            if signal[i, t] > threshs[i]:
                if not burst:
                    bursts['channel'].append(ch)
                    start_t = t
                    bursts['burst_start'].append(t)
                    burst = True
            else:
                if burst:
                    bursts['burst_end'].append(t)
                    bursts['burst_peak'].append(
                        start_t + np.argmax(signal[i, start_t:t]))
                    burst = False
        if burst:  # fencepost
            bursts['burst_end'].append(signal.shape[1])
            bursts['burst_peak'].append(
                start_t + np.argmax(signal[i, start_t:signal.shape[1]]))
    df = DataFrame(bursts)
    df.to_csv(burstf, sep='\t', index=False)
    return df


def decompose_tfr(this_data, sfreq, method=None, avg_freqs=False,
                  lfreq=15, hfreq=29, dfreq=1, n_cycles=7,
                  use_fft=True, mode='same', output='power',
                  verbose=True, overwrite=False):
    ''' Compute a time frequency decomposition (default beta).
    Parameters
    ----------
    this_data : np.array
        The data to be transformed with shape (n_epochs x n_chs x n_times)
        or (n_chs x n_times) for epochs or raw data respectively.
    sfreq : float
        The sampling frequency of the data (e.g. raw.info['sfreq'])
    method : str
        A string in ('phase-locked', 'non-phase-locked', 'total').
        This is only used for epoched data.
    avg_freqs : bool
        Whether or not to average the frequencies that were given.
        This would be useful in the case that you are analyzing a long
        recording of raw data and don't have room to store all the data.Z
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
    mode : ‘same’ | ‘valid’ | ‘full’
        Convention for convolution see see `mne.time_frequency.tfr.cwt`.
    output : `power` | `phase`
        Whether to output the power or phase of the TFR.
    Returns
    -------
    tfr : np.array(n_channels, n_times)
        An array of the data transformed by the Morlet method
    '''
    try:
        from tqdm import tqdm
        use_tqdm = True if verbose else False
    except ImportError as e:
        print(e, 'Install tqdm for progress bar')
        use_tqdm = False
    freqs = np.arange(lfreq, hfreq + dfreq, dfreq)
    n_freqs = len(freqs)
    if len(this_data.shape) not in (2, 3):
        raise ValueError('The data must be of shape (n_epochs x ' +
                         'n_ch x n_times) or (n_ch x n_times)')
    if len(this_data.shape) == 2:
        raw_used = True
        n_chs, n_times = this_data.shape
        n_epochs = 1
        this_data = this_data[np.newaxis, :, :]
    else:
        n_epochs, n_chs, n_times = this_data.shape
        raw_used = False
    collapse_axis = raw_used
    if raw_used:
        if method is not None:
            raise ValueError('Method given as %s, but this ' % method +
                             'parameter cannot be used for raw data.')
    elif method not in ('phase-locked', 'non-phase-locked', 'total'):
        raise ValueError('Unregcognized method %s, must be ' % method +
                         '`phase-locked`, `non-phase-locked` or `total`')
    elif method == 'phase-locked':
        collapse_axis = True
        this_data = np.mean(this_data, axis=0)[np.newaxis, :, :]
    elif method == 'non-phase-locked':
        this_data -= np.mean(this_data, axis=0)
    if avg_freqs:
        tfr = np.zeros(n_chs, n_times)
    else:
        tfr = np.zeros(n_freqs, n_chs, n_times)
    if output not in ('power', 'phase'):
        raise ValueError('Unregcognized ouput %s, must be ' % output +
                         '`power` or `phase`')
    if verbose:
        print('Computing time frequency decomposition')
    for epoch in tqdm(this_data) if use_tqdm else this_data:
        for i, freq in enumerate(tqdm(freqs) if use_tqdm else freqs):
            W = morlet(sfreq, [freq], n_cycles=n_cycles,
                       zero_mean=False)
            this_tfr = cwt(epoch, W, use_fft=use_fft, mode=mode)
            if output == 'power':
                this_tfr = abs(this_tfr[:, 0])
            elif output == 'phase':
                this_tfr = np.angle(this_tfr[:, 0])
            if avg_freqs:
                tfr += this_tfr
            else:
                tfr[i] += this_tfr
        if avg_freqs:
            tfr /= len(freqs)
    tfr /= n_epochs
    return tfr[0] if collapse_axis else tfr


'''
Followed the following code from
https://www.biorxiv.org/content/10.1101/644682v1
'''
