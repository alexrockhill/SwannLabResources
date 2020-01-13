import os
import os.path as op
import numpy as np
import json
from pandas import concat, read_csv
from bids import BIDSLayout
import mne


def get_config():
    with open('./params.json', 'r') as f:
        config = json.load(f)
    return config


def get_participants():
    config = get_config()
    return read_csv(op.join(config['bids_dir'], 'participants.tsv'), sep='\t')


def get_sidecar(fname, sidecar_ext):
    ext = op.splitext(fname)[-1]
    if sidecar_ext[0] != '.':
        sidecar_ext = '.' + sidecar_ext
    sidecarf = fname.replace(ext, sidecar_ext)
    if not op.isfile(sidecarf):
        raise ValueError('Sidecar file %s not found' % sidecarf)
    return sidecarf


def get_behf(bidsf):
    config = get_config()
    layout = get_layout()
    behf = layout.get(task=config['task'],
                      extension='tsv', suffix='beh',
                      subject=bidsf.entities['subject'],
                      session=(bidsf.entities['session'] if
                               'session' in bidsf.entities else None),
                      run=(bidsf.entities['run'] if
                           'run' in bidsf.entities else None))
    if len(behf) > 1:
        raise ValueError('More than one matching behavior file ' +
                         'found for %s' % bidsf.path)
    elif len(behf) == 0:
        raise ValueError('No matching behavior file ' +
                         'found for %s' % bidsf.path)
    return behf[0]


def get_layout():
    config = get_config()
    return BIDSLayout(config['bids_dir'])


def derivative_fname(bf, suffix, extention):
    config = get_config()
    out_dir = op.join(config['bids_dir'], 'derivatives',
                      'sub-%s' % bf.entities['subject'])
    if 'session' in bf.entities:
        out_dir = op.join(out_dir, 'ses-%s' % bf.entities['session'])
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
    outf = op.join(out_dir, 'sub-%s' % bf.entities['subject'])
    if 'session' in bf.entities:
        outf += '_ses-%s' % bf.entities['session']
    if 'task' in bf.entities:
        outf += '_task-%s' % bf.entities['task']
    if 'run' in bf.entities:
        outf += '_run-%s' % bf.entities['run']
    return outf + '_%s.%s' % (suffix, extention)


def get_overwrite(fname, name, verbose=True):
    is_file = '.' in fname
    if (is_file and op.isfile(fname)) or (not is_file and op.isdir(fname)):
        overwrite = \
            input('%s already exists, overwrite? (Y/N)' % name).upper() != 'Y'
        if overwrite:
            print('Overwriting %s' % fname)
        return overwrite
    return True


def get_dfs(behfs):
    df_list = []
    for behf in behfs:
        df = read_csv(behf.path, sep='\t')
        df['Subject'] = behf.entities['subject']
        df_list.append(df)
    dfs = concat(df_list)
    return dfs


def exclude_subjects(bids_list):
    config = get_config()
    bids_list = [bids_item for bids_item in bids_list if
                 bids_item.entities['subject'] not in
                 config['exclude_subjects']]
    return bids_list


'''
def get_epochs(raw, event, condition, value):
    """ Exclude trials, if event is response also exclude no response."""
    config = get_config()
    this_events = events[events[:, 2] == config['event_id'][event]['id']]
    event = "Stimulus/S  %i" % config['event_id'][event]['id']
    response = event == config['response_col']
    indices = dict()
    i = 0
    for trial in df.index:
        if not df.loc[trial, config['exclude_trial_col']]:
            if response:
                val = df.loc[i, config['response_col']]
                if not np.isnan(val) and val > 0 and val != 99:
                    indices[trial] = i
            else:
                indices[trial] = i
            i += 1
    if len(indices) != len(this_events):
        raise ValueError('Event behavior mismatch e %i b %i' %
                         (len(this_events), len(indices)))
    condition_indices = df[df[condition] == value].index
    return this_events[[trial for j, trial in indices.items()
                        if j in condition_indices]]
'''


def get_no_responses(behf):
    config = get_config()
    df = read_csv(behf.path, sep='\t')
    no_responses = list()
    for trial in df.index:
        val = df.loc[trial, config['response_col']]
        if np.isnan(val) or val < 1e-6 or val == 99:
            no_responses.append(trial)
    return no_responses


def read_raw(raw_path, preload=False):
    config = get_config()
    rawf, ext = op.splitext(raw_path)
    if ext == '.bdf':
        raw = mne.io.read_raw_bdf(raw_path, preload=preload)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(raw_path, preload=preload)
    elif ext == '.fif':
        raw = mne.io.Raw(raw_path, preload=preload)
    else:
        raise ValueError('Extention %s not yet implemented' % ext)
    for ch in ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']:
        if ch in raw.ch_names and ch not in raw.info['bads']:
            raw.info['bads'].append(ch)
    eogs, ecgs, emgs = config['eogs'], config['ecgs'], config['emgs']
    raw.set_channel_types({ch: ch_type for ch_type, chs in
                           {'eog': eogs, 'ecg': ecgs, 'emg': emgs}.items()
                           for ch in chs})
    aux = eogs + ecgs + emgs + ['Status']
    # TO DO: implement read montage from sidecar data
    # for data with digitization
    if raw.info['dig'] is None:
        montage = default_montage(raw)
        raw.set_channel_types({ch: 'misc' for ch in raw.ch_names if
                               ch not in montage.ch_names + aux})
        raw.set_montage(montage)
    return raw


def get_events(raw, event):
    config = get_config()
    events, _ = mne.events_from_annotations(raw)
    if events.size == 0:
        events = mne.find_events(raw)
    this_event = config['event_id'][event]
    if isinstance(this_event, list):
        indices = [i for i, e in enumerate(events[:, 2]) if
                   e in this_event]
    else:
        indices = [i for i, e in enumerate(events[:, 2]) if
                   e == this_event]
    return events[indices]


def default_montage(raw):
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_pos = dict()
    pos_indices = mne.pick_types(raw.info, meg=True, eeg=True,
                                 seeg=True, ecog=True)
    for i, ch in enumerate(raw.ch_names):
        if i in pos_indices:
            if ch in montage.ch_names:
                ch_pos[ch] = montage.dig[montage.ch_names.index(ch) + 3]['r']
    lpa = montage.dig[0]['r']
    nasion = montage.dig[1]['r']
    rpa = montage.dig[2]['r']
    return mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nasion,
                                         lpa=lpa, rpa=rpa, coord_frame='head')


'''
def read_dig_montage(dig_path):
    if '.bvct' in op.basename(corf):
        montage = mne.channels.read_dig_montage(bvct=corf)
    elif '.csv' in op.basename(corf):
        montage = mne.channels.read_dig_montage(csv=corf)
'''
