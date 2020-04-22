import os
import os.path as op
import argparse

from shutil import copyfile

from swann.utils import read_raw

from mne import read_annotations, events_from_annotations, find_events
from mne_bids import write_raw_bids, write_anat


def save2bids(bids_dir, sub, task, eegf, behf, ses=None, run=None,
              annot=None, pd_channels=None, data_ch_type='eeg',
              eogs=None, ecgs=None, emgs=None):
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
        for the raw data. If None, events in raw already are used.
    pd_channels: list
        The names of the photodiode channels (to drop).
    data_ch_type: str
        The type of the channels containing data, i.e. 'eeg' or 'seeg'.
    eogs: list | None
        The channels recording eye electrophysiology.
    ecgs: list | None
        The channels recording heart electrophysiology.
    emgs: list | None
        The channels recording muscle electrophysiology.
    """
    bids_basename = 'sub-%s' % sub
    bids_dir = op.join(bids_dir, 'sub-%s' % sub)
    if ses is not None:
        bids_basename += '_ses-%s' % ses
        bids_dir = op.join(bids_dir, 'ses-%s' % ses)
    bids_basename += '_task-%s' % task
    if run is not None:
        bids_basename += '_run-%s' % run
    if not op.isdir(op.join(bids_dir, 'beh')):
        os.makedirs(op.join(bids_dir, 'beh'))
    raw = read_raw(eegf, data_ch_type, list() if eogs is None else eogs,
                   list() if ecgs is None else ecgs, list() if
                   emgs is None else emgs)
    if annot is None:
        events = find_events(raw, min_duration=0.002)
    else:
        raw.set_annotations(read_annotations(annot))
        events, event_id = events_from_annotations(raw)
    raw = raw.drop_channels(pd_channels)
    write_raw_bids(raw, bids_basename, bids_dir,
                   events_data=events, event_id=event_id,
                   overwrite=True)
    copyfile(behf, op.join(bids_dir, 'beh',
                           bids_basename + '_beh.tsv'))


def anat2bids(bids_dir, sub, anatf):
    write_anat(bids_dir, sub, anatf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_dir', type=str, required=True,
                        help='The bids directory filepath')
    parser.add_argument('--sub', type=str, required=True,
                        help='The subject/patient identifier')
    parser.add_argument('--ses', type=str, required=False,
                        help='The session identifier')
    parser.add_argument('--run', type=str, required=False,
                        help='The run identifier')
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
    parser.add_argument('--data_ch_type', type=str, required=False,
                        default='eeg', help='The type of data i.e. '
                        '`eeg` or `seeg`')
    parser.add_argument('--eogs', type=str, nargs='*', required=False,
                        help='EOG channels')
    parser.add_argument('--ecgs', type=str, nargs='*', required=False,
                        help='ECG channels')
    parser.add_argument('--emgs', type=str, nargs='*', required=False,
                        help='EMG channels')
    parser.add_argument('--anatf', type=str, required=False,
                        help='The T1 image filepath')
    args = parser.parse_args()
    if args.anatf is None:
        if len(args.pd_channels) == 1 and op.isfile(args.pd_channels[0]):
            pd_channels = open(args.pd_channels[0], 'r'
                               ).readline().rstrip().split('\t')
        else:
            pd_channels = args.pd_channels
        save2bids(args.bids_dir, args.sub, args.task, args.eegf,
                  args.behf, args.ses, args.run, args.annot,
                  pd_channels, args.data_ch_type,
                  args.eogs, args.ecgs, args.emgs)
    else:
        anat2bids(args.bids_dir, args.sub, args.anatf)
