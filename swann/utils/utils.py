import os
import os.path as op
import numpy as np
import json
from pandas import concat, read_csv
from bids import BIDSLayout
import mne
from warnings import warn


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
        raise ValueError('More than one matching behavior file '
                         'found for %s' % bidsf.path)
    elif len(behf) == 0:
        raise ValueError('No matching behavior file '
                         'found for %s' % bidsf.path)
    return behf[0]


def get_layout():
    config = get_config()
    return BIDSLayout(config['bids_dir'])


def derivative_fname(bf, dir_name, suffix, extention):
    config = get_config()
    out_dir = op.join(config['bids_dir'], 'derivatives',
                      'sub-%s' % bf.entities['subject'])
    if 'session' in bf.entities:
        out_dir = op.join(out_dir, 'ses-%s' % bf.entities['session'])
    out_dir = op.join(out_dir, dir_name)
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
    this_events = events[events[:, 2] == config['events'][event]['id']]
    event = "Stimulus/S  %i" % config['events'][event]['id']
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


def check_overwrite_or_return(name, fname, return_saved, overwrite, verbose):
    if op.isfile(fname) and overwrite:
        if verbose:
            print('Overwriting existing %s' % name)
    elif not op.isfile(fname):
        if return_saved:
            raise ValueError('%s' % name.capitalize() + ' have not been '
                             'calculated cannot return saved')
    elif not overwrite:
        if not return_saved:
            print('%s' % name.capitalize() + ' have already been calculated, '
                  'use `overwrite=True` to recompute')
        return True
    return False


def my_events():
    config = get_config()
    events = dict()
    events.update(config['stimuli'])
    events.update(config['responses'])
    events.update(config['feedback'])
    return events


def get_no_responses(behf):
    config = get_config()
    df = read_csv(behf.path, sep='\t')
    no_responses = list()
    for trial in df.index:
        val = df.loc[trial, config['response_col']]
        if np.isnan(val) or val < 0.05 or val == 99:  # RT < ~0.03 not recorded
            no_responses.append(trial)
    return no_responses


def read_raw(raw_path, data_ch_type=None, eogs=None, ecgs=None, emgs=None,
             preload=False):
    rawf, ext = op.splitext(raw_path)
    if ext == '.bdf':
        raw = mne.io.read_raw_bdf(raw_path, preload=preload)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(raw_path, preload=preload)
    elif ext == '.fif':
        raw = mne.io.Raw(raw_path, preload=preload)
    elif ext in ('.eeg', '.vhdr'):
        if ext == '.eeg':
            raw_path = raw_path.replace('.eeg', '.vhdr')
        raw = mne.io.read_raw_brainvision(raw_path, preload=preload)
    else:
        raise ValueError('Extention %s not yet implemented' % ext)
    for ch in ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']:
        if ch in raw.ch_names and ch not in raw.info['bads']:
            raw.info['bads'].append(ch)
    if eogs is None or ecgs is None or emgs is None:
        config = get_config()
        if eogs is None:
            eogs = config['eogs']
        if ecgs is None:
            ecgs = config['ecgs']
        if emgs is None:
            emgs = config['emgs']
    raw.set_channel_types({ch: ch_type for ch_type, chs in
                           {'eog': eogs, 'ecg': ecgs, 'emg': emgs}.items()
                           for ch in chs})
    if 'Event' in raw.ch_names:
        raw.set_channel_types({'Event': 'stim'})
    if data_ch_type is not None:
        raw.set_channel_types({ch: data_ch_type for ch in raw.ch_names if
                               ch not in (eogs + ecgs + emgs + ['Event'])})
    # aux = eogs + ecgs + emgs + ['Status']
    # TO DO: implement read montage from sidecar data
    # for data with digitization
    if raw.info['dig'] is None:
        try:
            montage = default_montage(raw)
            # raw.set_channel_types({ch: 'misc' for ch in raw.ch_names if
            #                        ch not in montage.ch_names + aux})
            raw.set_montage(montage)
        except Exception as e:
            print('Unable to find default montage', e)
    return raw


def get_events(raw, exclude_events=None):
    config = get_config()
    all_events = dict()
    events, _ = mne.events_from_annotations(raw)
    if events.size == 0:
        events = mne.find_events(raw)
    if 128 in events[:, 2]:  # weird but where biosemi can loop
        events = events[7:]  # through trigger bits at the start of a file
    n_events = None
    for event in (list(config['stimuli'].keys()) + list(
                  config['feedback'].keys())):
        this_events = _this_events(events, event)
        if n_events is None:
            n_events = len(this_events)
        elif n_events != len(this_events):
            raise ValueError('%i' % n_events + ' events found previously, '
                             '%i events for %s' % (len(this_events), event))
        this_events[:, 2] = np.arange(n_events)
        all_events[event] = this_events
    for event in config['responses']:
        response_events = _this_events(events, event)
        response_ts = set(response_events[:, 0])  # time stamps
        response_indices = list()
        excluded = list()
        for i in range(n_events):
            these_response_events = list()
            min_i = max([all_events[stim_event][i, 0]
                         for stim_event in config['stimuli']])
            min_i_check = min([all_events[stim_event][i, 0]
                               for stim_event in config['stimuli']])
            if i == n_events - 1:
                max_i = max(events[:, 0]) + 1
            else:
                max_i = min([all_events[stim_event][i + 1, 0]
                             for stim_event in config['stimuli']])
            for event_ts in range(min_i, max_i):
                if event_ts in response_ts:
                    these_response_events.append(event_ts)
            for event_ts in range(min_i_check, min_i):
                if event_ts in response_ts:
                    excluded.append(np.where(
                        response_events[:, 0] == event_ts)[0][0])
                    stim0 = [stim_event for stim_event in config['stimuli']
                             if all_events[stim_event][i, 0] == min_i_check][0]
                    stim1 = [stim_event for stim_event in config['stimuli']
                             if all_events[stim_event][i, 0] == min_i][0]
                    warn('{} event found between {} and {} stimuli, '
                         'for trial {} excluding'.format(event, stim0,
                                                         stim1, i))
            if len(these_response_events) > 2:
                warn('{} response events found for response event {}'
                     'for stimulus {}'.format(len(these_response_events),
                                              event, i))
            elif len(these_response_events) == 1:
                response_indices.append(np.where(
                    response_events[:, 0] == these_response_events[0])[0][0])
        for i, e in enumerate(response_events):
            if i not in response_indices and i not in excluded:
                warn('{} {} not between stimuli, excluding'.format(event, i))
        response_events = response_events[response_indices]
        response_events[:, 2] = response_indices
        all_events[event] = response_events
    if exclude_events is not None:
        for events in all_events:
            this_exclude_events = [i for i, e in
                                   enumerate(all_events[event][:, 2])
                                   if e in exclude_events[event]]
            all_events[event] = np.delete(all_events[event],
                                          this_exclude_events, axis=0)
    return all_events


def select_events(events, indices):
    return events[[i for i, e in enumerate(events[:, 2]) if
                   e in indices]]


def _this_events(events, event):
    event_id = my_events()[event]
    if isinstance(event_id, list):
        indices = [i for i, e in enumerate(events[:, 2]) if
                   e in event_id]
    else:
        indices = [i for i, e in enumerate(events[:, 2]) if
                   e == event_id]
    return events[indices]


def pick_data(raw):
    return raw.pick_types(meg=True, eeg=True, seeg=True, ecog=True, csd=True)


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


def most_square(n_plots):
    ''' Finds the most square axes shape for n plots.
    Parameters
    ----------
    n_plots : int
        The number of plots to show
    Returns
    -------
    dim0 : int
        The first dimension of the plot
    dim1 : int
        The second dimension of the plot
    '''
    dim0 = int(n_plots ** 0.5)
    dim1 = int(n_plots / dim0) + 1
    return dim0, dim1


def rolling_mean(signal, n=10):
    rolling_signal = np.zeros(signal.shape)
    half_n = int(n / 2)
    for offset in range(-half_n, half_n):
        if offset < 0:
            rolling_signal[-offset:] += signal[:offset]
        elif offset > 0:
            rolling_signal[:-offset] += signal[offset:]
        else:
            rolling_signal += signal
    denomenators = np.concatenate([np.arange(half_n, 2 * half_n),
                                   np.ones((len(signal) - half_n * 2)
                                           ) * half_n * 2,
                                   np.flip(np.arange(half_n, 2 * half_n))])
    return rolling_signal / denomenators


def xval_inds(n, prop):
    config = get_config()
    np.random.seed(config['seed'])
    m = np.round(n * prop).as_type(int)
    return np.random.choice(np.arange(n), size=m, replace=False)


'''
def read_dig_montage(dig_path):
    if '.bvct' in op.basename(corf):
        montage = mne.channels.read_dig_montage(bvct=corf)
    elif '.csv' in op.basename(corf):
        montage = mne.channels.read_dig_montage(csv=corf)
'''
'''
    if no_responses is None:
        raise ValueError('A `response_event` was defined in the config ' +
                         'file thus a no responses list must be provided')
    response_events = _this_events(events, config['response_event'])
    if len(response_events) + len(no_responses) != n_events:
        raise ValueError('%i events for other events ' % n_events +
                         'compared to %i ' % len(response_events) +
                         'response events + %i ' % len(no_responses) +
                         'no response trials: these do not add up.')
    response_indices = np.setdiff1d(np.arange(n_events), no_responses)
    response_events[:, 2] = response_indices
    response_exclude_events = [i for i, e in enumerate(response_indices) if
                               e in
                               exclude_events[config['response_event']]]
    response_events = np.delete(response_events, response_exclude_events,
                                axis=0)
    all_events[config['response_event']] = response_events
'''
