import sys
import mne
from mne_bids import write_raw_bids

from swann.utils import read_raw


def save2bids(basename, eegf, bids_dir, digf=None):
    raw = read_raw(eegf)
    events = mne.find_events(raw, min_duration=0.002)
    raw.set_channel_types({ch: 'stim' if ch == 'Event' else 'eeg'
                           for ch in raw.ch_names})
    write_raw_bids(raw, basename, bids_dir, events_data=events,
                   overwrite=True)


if __name__ == '__main__':
    save2bids(*sys.argv[1:])
