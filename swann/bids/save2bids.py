import sys
import os.path as op
import mne
from mne_bids import write_raw_bids


def save2bids(subject, session, run, task, eegf, bids_dir):
    eegf_base, ext = op.splitext(eegf)
    if ext == '.bdf':
        raw = mne.io.read_raw_bdf(eegf, preload=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(eegf, preload=False)
    else:
        raise ValueError('Extention %s not yet implemented' % ext)
    events = mne.find_events(raw, min_duration=0.002)
    raw.set_channel_types({ch: 'stim' if ch == 'Event' else 'eeg'
                           for ch in raw.ch_names})
    bids_basename = ('sub-%s_ses-%s_task-%s_run-%s' %
                     (subject, session, task, run))
    write_raw_bids(raw, bids_basename, bids_dir, events_data=events,
                   overwrite=True)


if __name__ == '__main__':
    save2bids(*sys.argv[1:])
