import os
import os.path as op
import numpy as np
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pandas import DataFrame
# from datetime import datetime, timedelta

from mne import Annotations, events_from_annotations
from mne.io import read_raw_edf, read_raw_bdf, Raw
from mne.utils import _TempDir
from mne_bids import write_raw_bids, write_anat


def _get_candidates(signal, thresh):
    baseline = list()
    trigger = list()
    events = list()
    thresh = np.std(signal) * thresh
    signal -= np.median(signal)
    for t, s in enumerate(signal):
        if abs(s) < thresh:
            if trigger:
                trigger = list()
            baseline.append(t)
        else:
            if baseline:
                trigger.append(t)
        if len(baseline) > 100 and len(trigger) == 5:
            events.append(trigger[0])
            baseline = list()
            trigger = list()
    return np.array(events)


def _find_events(df, pd, sfreq, aud=None):
    thresh = 0.25
    while thresh:
        pd_candidates = _get_candidates(pd, thresh=float(thresh))
        print('#candidates found %i' % len(pd_candidates))
        thresh = input('New threshold?\t')
    diffs = pd_candidates[1:] - pd_candidates[:-1]
    # m, b, r, p, se = linregress(diffs, np.array(df['trial_length'] * sfreq))
    for i, trial_length in df['trial_length'].items():
        j = np.argmin(abs(diffs - trial_length * sfreq))
        print('Trial %s: closest diff %s' % (i, j))
    for block in np.unique(df['block']):
        this_df = df[df['block'] == block]
        min_error = np.inf
        best_i = None
        for i in range(0, len(pd_candidates) - len(this_df)):
            this_error = sum(abs((pd_candidates[i + 1: i + 1 + len(this_df)] -
                                  pd_candidates[i: i + len(this_df)]) -
                                 this_df['trial_length'] * sfreq))
            if this_error < min_error:
                best_i = i
                min_error = this_error
        print('Block %i best offset: %i' % (block, best_i))
    skip_to_event = input('Skip to event?\t')
    events = dict()
    trial = 0
    pd_index = 0
    while trial < len(df):
        c = pd_candidates[pd_index]
        if skip_to_event and pd_index < int(skip_to_event):
            pd_index += 1
            continue
        if pd_index < len(pd_candidates) - 1:
            to_next = float(pd_candidates[pd_index + 1] - c) / sfreq
        else:
            to_next = 99999
        if trial < len(df):
            min_trial = max([0, trial - 2])
            max_trial = min([len(df), trial + 5])
            print(df['trial_length'].loc[min_trial:max([trial - 1, 0])])
            print()
            print('%i    %f    time to next %s, event %s' %
                  (trial, df['trial_length'].loc[trial], to_next, pd_index))
            print()
            print(df['trial_length'].loc[min([trial + 1, len(df)]): max_trial])
        if aud is None:
            fig, (ax0, ax1) = plt.subplots(2, 1)
        else:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        ax0.plot(pd[c - int(5 * sfreq): c + int(5 * sfreq)])
        ax1.plot(pd[c - int(50 * sfreq): c + int(50 * sfreq)])
        if aud:
            ax2.plot(aud[c - int(5 * sfreq): c + int(5 * sfreq)])
            ax3.plot(aud[c - int(50 * sfreq): c + int(50 * sfreq)])
        fig.show()
        trial_input = input('trial?\n')
        if trial_input != 'e':
            if 'pd+' in trial_input:
                pd_index += int(trial_input.replace('pd+', ''))
            elif 'pd-' in trial_input:
                pd_index -= int(trial_input.replace('pd-', ''))
            elif 't+' in trial_input:
                trial += int(trial_input.replace('t+', ''))
            elif 't-' in trial_input:
                trial -= int(trial_input.replace('t-', ''))
            elif trial_input:
                trial = int(trial_input)
            events[trial] = c
            trial += 1
        pd_index += 1
        plt.close(fig)
    return events


def edf2bids(bids_dir, sub, task, eegf, behf):
    bids_basename = 'sub-%s_task-%s' % (sub, task)
    if not op.isfile(op.join(bids_dir, 'beh')):
        os.makedirs(op.join(bids_dir, 'beh'))
    if op.splitext(eegf)[-1] == '.edf':
        raw = read_raw_edf(eegf, preload=True)
    elif op.splitext(eegf)[-1] == '.bdf':
        raw = read_raw_bdf(eegf, preload=True)
    raw.set_channel_types({ch: 'stim' if ch == 'Event' else 'seeg'
                           for ch in raw.ch_names})
    raw.plot()
    plt.show()
    pd0 = input('pd0 ch?\t')
    pd1 = input('pd1 ch?\t')
    a0 = input('a0 ch?\t')
    a1 = input('a1 ch?\t')
    sfreq = raw.info['sfreq']
    mat = loadmat(behf)
    df = DataFrame({key: value[0] for key, value in mat.items()
                    if '__' not in key})
    # s, mu_s = raw.info['meas_date']
    # t0 = timedelta(0, s, mu_s).seconds
    if pd1:
        pd = (raw._data[raw.ch_names.index(pd0)] -
              raw._data[raw.ch_names.index(pd1)])
    else:
        pd = raw._data[raw.ch_names.index(pd0)]
    if a0 and a1:
        aud = (raw._data[raw.ch_names.index(a0)] -
               raw._data[raw.ch_names.index(a1)])
    else:
        aud = None
    events = _find_events(df, pd, raw.info['sfreq'], aud=aud)
    if aud:
        raw = raw.drop_channels([pd0, pd1, a0, a1])
    else:
        raw = raw.drop_channels([pd0, pd1])
    tmin = raw.times[min(events.values())] - 5
    tmax = raw.times[max(events.values())] + 5
    raw2 = raw.copy().crop(tmin=tmax)
    tmp_dir = _TempDir()
    raw2.save(op.join(tmp_dir, 'resting_tmp-raw.fif'))
    raw2 = Raw(op.join(tmp_dir, 'resting_tmp-raw.fif'))
    write_raw_bids(raw2, bids_basename.replace(task, 'resting'), bids_dir)
    fix_onset = [events[i] for i in sorted(events.keys())]
    fix_annot = Annotations(onset=raw.times[np.array(fix_onset)],
                            duration=np.repeat(0.1, len(events)),
                            description=np.repeat('fix', len(events)))
    isi_onset = [events[i] + int(np.round(df['fix_duration'].loc[i] * sfreq))
                 for i in sorted(events.keys())]
    isi_annot = Annotations(onset=raw.times[np.array(isi_onset)],
                            duration=np.repeat(0.1, len(events)),
                            description=np.repeat('isi', len(events)))
    isi_onset = [events[i] + int(np.round(df['fix_duration'].loc[i] * sfreq))
                 for i in sorted(events.keys())]
    isi_annot = Annotations(onset=raw.times[np.array(isi_onset)],
                            duration=np.repeat(0.1, len(events)),
                            description=np.repeat('isi', len(events)))
    go_onset = [events[i] + int(np.round(df['go_time'].loc[i] * sfreq))
                for i in sorted(events.keys())]
    go_annot = Annotations(onset=raw.times[np.array(go_onset)],
                           duration=np.repeat(0.1, len(events)),
                           description=np.repeat('go', len(events)))
    resp_onset = [events[i] + int(np.round(df['response_time'].loc[i] * sfreq))
                  for i in sorted(events.keys())]
    resp_annot = Annotations(onset=raw.times[np.array(resp_onset)],
                             duration=np.repeat(0.1, len(events)),
                             description=np.repeat('resp', len(events)))
    raw.set_annotations(fix_annot + isi_annot + go_annot + resp_annot)
    raw = raw.crop(tmin=tmin, tmax=tmax)
    raw.save(op.join(tmp_dir, 'tmp-raw.fif'))
    raw = Raw(op.join(tmp_dir, 'tmp-raw.fif'))
    events2, event_id = events_from_annotations(raw)
    write_raw_bids(raw, bids_basename, bids_dir,
                   events_data=events2, event_id=event_id,
                   overwrite=True)
    df['exclude_trial'] = [0 if trial in events else 1 for trial in
                           df['trials']]
    df.to_csv(op.join(bids_dir, 'sub-%s' % sub, 'beh',
                      bids_basename + '_beh.tsv'),
              sep='\t', index=False)


def anat2bids(bids_dir, sub, anatf):
    write_anat(bids_dir, sub, anatf)


if __name__ == '__main__':
    edf2bids(*sys.argv[1:])

'''
a0 = 'LSTG 5'
a1 = 'LSTG 6'
pd0 = 'LSTG 7'
pd1 = 'LSTG 8'
'''
