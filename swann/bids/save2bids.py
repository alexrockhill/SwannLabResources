import sys
import mne
from mne_bids import write_raw_bids


def save2bids(subject, session, run, task, eegf, bids_dir):
    raw = mne.io.read_raw_bdf(eegf, preload=False)
    events = mne.find_events(raw, min_duration=0.002)
    raw.set_channel_types({ch: 'stim' if ch == 'Event' else 'eeg'
                           for ch in raw.ch_names})
    bids_basename = ('sub-%s_ses-%s_task-%s_run-%s' %
                     (subject, session, task, run))
    write_raw_bids(raw, bids_basename, bids_dir, events_data=events,
                   overwrite=True)


if __name__ == '__main__':
    save2bids(*sys.argv[1:])
