import os
import os.path as op
import sys
import argparse

from shutil import copyfile

from mne import read_annotations, events_from_annotations
from mne.io import read_raw_edf, read_raw_bdf, Raw
from mne.utils import _TempDir
from mne_bids import write_raw_bids, write_anat


def edf2bids(bids_dir, sub, task, eegf, behf, annot, pd_channels):
    """Convert iEEG data collected at OHSU to BIDS format
    Parameters
    ----------
    bids_dir : str
        The subject directory in the bids directory where the data
        should be saved.
    sub : str
        The name of the subject.
    task : str
        The name of the task.
    eegf : str
        The filepath where the file containing the eeg data lives.
    annot : str
        The filepath to an mne.Annotations object to encode events
        for the raw data.
    pd_channels: list
        The names of the photodiode channels (to drop).
    """
    bids_basename = 'sub-%s_task-%s' % (sub, task)
    if not op.isdir(op.join(bids_dir, 'beh')):
        os.makedirs(op.join(bids_dir, 'beh'))
    if op.splitext(eegf)[-1] == '.edf':
        raw = read_raw_edf(eegf, preload=True)
    elif op.splitext(eegf)[-1] == '.bdf':
        raw = read_raw_bdf(eegf, preload=True)
    raw.set_channel_types({ch: 'stim' if ch == 'Event' else 'seeg'
                           for ch in raw.ch_names})
    raw.set_annotations(read_annotations(annot))
    events, event_id = events_from_annotations(raw)
    raw = raw.drop_channels(pd_channels)
    tmin = raw.times[min(events.values())] - 5
    tmax = raw.times[max(events.values())] + 5
    raw2 = raw.copy().crop(tmin=tmax)
    tmp_dir = _TempDir()
    raw2.save(op.join(tmp_dir, 'resting_tmp-raw.fif'))
    raw2 = Raw(op.join(tmp_dir, 'resting_tmp-raw.fif'))
    write_raw_bids(raw2, bids_basename.replace(task, 'resting'), bids_dir)
    raw = raw.crop(tmin=tmin, tmax=tmax)
    raw.save(op.join(tmp_dir, 'tmp-raw.fif'))
    raw = Raw(op.join(tmp_dir, 'tmp-raw.fif'))
    write_raw_bids(raw, bids_basename, bids_dir,
                   events_data=events, event_id=event_id,
                   overwrite=True)
    copyfile(behf, op.join(bids_dir, 'sub-%s' % sub, 'beh',
                           bids_basename + '_beh.tsv'))


def anat2bids(bids_dir, sub, anatf):
    write_anat(bids_dir, sub, anatf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_dir', type=str, required=True,
                        help='The bids directory filepath')
    parser.add_argument('--sub', type=str, required=True,
                        help='The subject/patient identifier')
    parser.add_argument('--task', type=str, required=False,
                        help='The name of the task')
    parser.add_argument('--eegf', type=str, required=False,
                        help='The eeg filepath')
    parser.add_argument('--behf', type=str, required=False,
                        help='The behavioral tsv filepath')
    parser.add_argument('--annot', type=str, required=False,
                        help='The annotations filepath')
    parser.add_argument('--pd_channels', type=str, nargs='*', required=False,
                        help='The photodiode channels or a filepath to '
                        'the tsv where they are')
    parser.add_argument('--anatf', type=str, required=False,
                        help='The T1 image filepath')
    args = parser.parse_args()
    if args.anatf is None:
        if len(args.pd_channels) == 1 and op.isfile(args.pd_channels[0]):
            pd_channels = open(args.pd_channels[0], 'r'
                               ).readline().rstrip().split('\t')
        else:
            pd_channels = args.pd_channels
        edf2bids(args.bids_dir, args.sub, args.task, args.eegf,
                 args.behf, args.annot, pd_channels)
    else:
        anat2bids(args.bids_dir, args.sub, args.anatf)
