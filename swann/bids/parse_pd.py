import os.path as op
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pandas import read_csv
from tqdm import tqdm

from mne.io import read_raw_edf, read_raw_bdf
from mne import Annotations, events_from_annotations


def _find_pd_candidates(pd, sfreq, chunk, baseline, zscore, min_i, overlap,
                        verbose=True):
    if verbose:
        print('Finding photodiode events')
    ''' Move in chunks twice as long as your longest photodiode signal
        with the first 0.25 as baseline o test whether the signal goes
        above/below baseline and back below/above.
        The event onset must have a minimum length `min_i` and
        `overlap` should be set such that no events will be missed'''
    #  gets cut off, you'll find it the next chunk
    chunk_i = int(chunk * sfreq)
    baseline_i = int(chunk_i * baseline / 2)
    pd_candidates = set()
    for i in tqdm(range(baseline_i, len(pd) - chunk_i - baseline_i,
                        int(chunk_i * overlap))):
        b = pd[i - baseline_i:i]
        s = (pd[i:i + chunk_i] - np.median(b)) / np.std(b)
        for binary_s in [s > zscore, s < zscore]:
            if not binary_s[0]:  # must start off
                onset = np.where(binary_s)[0]
                if onset.size > min_i:
                    e = onset[0]
                    offset = np.where(1 - binary_s[e:])[0]
                    # must have an offset and no more events
                    if offset.size > 0 and not any(binary_s[e + offset[0]:]):
                        if not any([i + e + j in pd_candidates for j in
                                    range(-1, 2)]):  # off by one error
                            pd_candidates.add(i + e)
    return pd_candidates


def _nearest_pd_event(b_event, pd_candidates, max_index):
    j = 0
    while (int(b_event + j) not in pd_candidates and
           int(b_event - j) not in pd_candidates and
           b_event + j < max_index and b_event - j > 0):
        j += 1
    return j


def _find_best_alignment(beh_events, pd_candidates, first_pd_alignment_n,
                         verbose=True):
    if verbose:
        print('Finding best alignment with behavioral events using the ' +
              'first %i events' % first_pd_alignment_n)
    min_error = best_alignment = best_errors = None
    max_index = max(pd_candidates)
    sorted_pds = np.array(sorted(pd_candidates))
    # want first_pd_alignment to be small to avoid drift
    for i in tqdm(range(len(pd_candidates) - first_pd_alignment_n)):
        these_beh_events = beh_events.copy() + sorted_pds[i]
        errors = [_nearest_pd_event(b_event, pd_candidates, max_index) for
                  b_event in these_beh_events[:first_pd_alignment_n]]
        median_error = np.median(errors)
        if min_error is None or median_error < min_error:
            best_alignment = i
            min_error = median_error
            best_errors = errors
    if verbose:
        print('Best alignment starting with photodiode event ' +
              '#%i, min %i, q1 %i, med %i, q3 %i, max %i ' %
              (best_alignment, min(best_errors),
               np.quantile(best_errors, 0.25), np.median(best_errors),
               np.quantile(best_errors, 0.75), max(best_errors)))
    return best_alignment


def _exclude_ambiguous_events(beh_events, pd_candidates, pd, sfreq, chunk,
                              best_alignment, exclude_shift, verbose=True):
    if verbose:
        print('Excluding events that have zero close events or more than '
              'one photodiode event within `chunk` time')
    events = dict()
    errors = dict()
    chunk_i = int(chunk * sfreq)
    max_index = max(pd_candidates)
    sorted_pds = np.array(sorted(pd_candidates))
    beh_events_aligned = beh_events.copy() + sorted_pds[best_alignment]
    for i, b_event in enumerate(beh_events_aligned):
        j = _nearest_pd_event(b_event, pd_candidates, max_index)
        if j > sfreq * exclude_shift:
            if verbose:
                print('Excluding event %i off by %i samples, ' % (i, j))
                plt.plot(pd[int(b_event - 10 * sfreq):
                            int(b_event + 10 * sfreq)])
                plt.show()
        else:
            if int(b_event + j) in pd_candidates:
                beh_events_aligned += j
                events[i] = int(b_event + j)
                errors[i] = j
            else:
                beh_events_aligned -= j
                events[i] = int(b_event - j)
                errors[i] = -j
            pd_events = np.logical_and(sorted_pds < (events[i] + chunk_i),
                                       sorted_pds > (events[i] - chunk_i))
            if sum(pd_events) > 1:
                events.pop(i)
                errors.pop(i)
                if verbose:
                    print('%i events found for behvaior event %i, excluding' %
                          (sum(pd_events), i))
                    plt.plot(pd[int(b_event - 10 * sfreq):
                                int(b_event + 10 * sfreq)])
                    plt.show()
    if verbose:
        print(errors)
        print('Final behavior event-photodiode event differences '
              'min %i, q1 %i, med %i, q3 %i, max %i ' %
              (min(errors.values()), np.quantile(list(errors.values()), 0.25),
               np.median(list(errors.values())),
               np.quantile(list(errors.values()), 0.75), max(errors.values())))
        trials = sorted(errors.keys())
        plt.plot(trials, [errors[t] for t in trials])
        plt.ylabel('Difference (samples)')
        plt.xlabel('Trial')
        plt.title('Photodiode Events Compared to Behavior Events')
        plt.show()
    return events


def _add_events_to_raw(raw, events, pd_event_name, relative_events):
    onsets = np.array([events[i] for i in sorted(events.keys())])
    annot = Annotations(onset=raw.times[onsets],
                        duration=np.repeat(0.1, len(onsets)),
                        description=np.repeat(pd_event_name,
                                              len(onsets)))
    if relative_events is not None:
        for name, beh_col in relative_events.items():
            onsets = [events[i] + int(np.round(df[beh_col].loc[i] *
                                      raw.info['sfreq']))
                      for i in sorted(events.keys()) if
                      not np.isnan(df[beh_col].loc[i])]
            annot += Annotations(onset=raw.times[np.array(onsets)],
                                 duration=np.repeat(0.1, len(onsets)),
                                 description=np.repeat(name, len(onsets)))
    raw.set_annotations(annot)
    return raw


def parse_pd_events(eegf, beh_events, pd_event_name='Fixation', chunk=2,
                    baseline=0.25, overlap=0.25, exclude_shift=0.1, zscore=10,
                    min_i=10, first_pd_alignment_n=10, relative_events=None,
                    overwrite_raw=True, verbose=True):
    ''' Parses photodiode events from a likely very corrupted channel
        using behavioral data to sync events to determine which
        behavioral events don't have a match and are thus corrupted
        and should be excluded (while ignoring events that look like
        photodiode events but don't match behavior)
    Parameters
    ----------
    eegf: str
        The filepath to the eeg file.
    beh_events: np.array | list
        The events (in seconds) for the behavior that is matched
        to the photodiode.
    pd_event_name: str
        The name of the event corresponding to the photodiode.
    chunk: float
        The size of the window to chunk the photodiode events by
        should be larger than 2x the longest photodiode event
    baseline: float
        How much relative to the chunk to use to idenify the time before
        the photodiode event.
    overlap: float
        How much to overlap the windows of the photodiode event-finding
        process.
    exclude_shift: float
        How many seconds different than expected from the behavior events
        to exclude that event.
    zscore: float
        How large of a z-score difference to use to threshold photodiode
        events.
    min_i: int
        The minimum number of samples the photodiode event must be on for.
    first_pd_alignment_n : int
        How many samples to use to score the alignment to the first event.
        This number should be low to have an accurate score without the
        photodiode drifting and causing unexpected alignments (and it's
        faster).
    relative_events: dict | None
        Whether to add events with keys of the event name and values
        of the list of relative times (in seconds) compared to the
        photodiode event.1
    Returns
    -------
    events: DataFrame
        A DataFrame that has a column for to the (zero)
        indexed behavioral events and another column corresponding
        to the time stamp of the eeg file.
    '''
    if op.splitext(eegf)[-1] == '.edf':
        raw = read_raw_edf(eegf, preload=True)
    elif op.splitext(eegf)[-1] == '.bdf':
        raw = read_raw_bdf(eegf, preload=True)
    raw.plot()
    plt.show()
    pd = None
    pds = list()
    while pd != '' and len(pds) < 2 and pd not in raw.ch_names:
        pd = input('pd%i ch?\t' % len(pds))
        if pd not in raw.ch_names:
            print('Channel not in raw')
        else:
            pds.append(pd)
            pd = None
    if len(pds) == 2:
        pd = (raw._data[raw.ch_names.index(pds[0])] -
              raw._data[raw.ch_names.index(pds[1])])
    else:
        pd = raw._data[raw.ch_names.index(pds[0])]
    pd_candidates = _find_pd_candidates(pd, raw.info['sfreq'], chunk, baseline,
                                        zscore, min_i, overlap, verbose)
    first_pd_alignment_n = min([len(pd_candidates) / 2, first_pd_alignment_n])
    beh_events *= raw.info['sfreq']
    beh_events -= beh_events[0]
    best_alignment = _find_best_alignment(beh_events, pd_candidates,
                                          first_pd_alignment_n, verbose)
    events = _exclude_ambiguous_events(beh_events, pd_candidates, pd,
                                       raw.info['sfreq'], chunk,
                                       best_alignment, exclude_shift, verbose)
    raw = _add_events_to_raw(raw, events, pd_event_name, relative_events)
    return raw, pds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eegf', type=str, required=True,
                        help='The eeg filepath')
    parser.add_argument('--behf', type=str, required=True,
                        help='The behavioral tsv filepath')
    parser.add_argument('--beh_col', type=str, required=False,
                        default='fix_onset_time',
                        help='The name of the behavioral column '
                        'corresponding to the photodiode event timing')
    parser.add_argument('--pd_event_name', type=str, required=False,
                        default='Fixation',
                        help='The name of the photodiode event')
    parser.add_argument('--diff', type=bool, required=False,
                        default=False, help='Whether the behavior column is '
                        'an absolute time stamp or a difference (and thus '
                        'should be cumulatively summed')
    parser.add_argument('--chunk', type=float, required=False,
                        default=2, help='How large to window the '
                        'photodiode events, should be 2x longest event')
    parser.add_argument('--exclude_shift', type=float, required=False,
                        default=0.1, help='How many seconds off to exclude a '
                        'photodiode-behavioral event difference')
    parser.add_argument('--relative_event_cols', type=str, nargs='*',
                        required=False,
                        help='A behavioral column in the tsv file that has '
                        'the time relative to the photodiode events on the '
                        'same trial as in the `--beh_col event.')
    parser.add_argument('--relative_event_names', type=str, nargs='*',
                        required=False, help='The name of the corresponding '
                        '`--relative_event_cols` events')
    args = parser.parse_args()
    df = read_csv(args.behf, sep='\t')
    beh_events = np.array(df[args.beh_col])
    if args.diff:
        beh_events = np.concatentate([0], np.cumsum(beh_events))
    if args.relative_event_names is not None:
        if len(args.relative_event_cols) != len(args.relative_event_names):
            raise ValueError('Mismatched length of relative event behavior '
                             'file column names and names of the events')
        relative_events = [df[rel_event] for rel_event in
                           args.relative_event_cols]
        relative_events = {name: rel_events for rel_events, name in
                           zip(relative_events, args.relative_event_names)}
        '''e.g. relative_events = {'ISI Onset': 'fix_duration', 'Go Cue': 'go_time',
                                   'Response': 'response_time'}'''
    else:
        relative_events = None
    raw, pds = parse_pd_events(args.eegf, beh_events, args.pd_event_name,
                               chunk=args.chunk,
                               relative_events=relative_events,
                               exclude_shift=args.exclude_shift)
    events, event_id = events_from_annotations(raw)
    np.savetxt(op.splitext(args.eegf)[0] + '_pd_events.tsv', events, fmt='%i',
               delimiter='\t')
    with open(op.splitext(args.eegf)[0] + '_pd_event_id.tsv', 'w') as f:
        for name, e_id in event_id.items():
            f.write(name + '\t' + str(e_id) + '\n')
    raw.annotations.save(op.splitext(args.eegf)[0] + '_pd-annot.fif')
    with open(op.splitext(args.eegf)[0] + '_pd_channels.tsv', 'w') as f:
        f.write('\t'.join(pds))
