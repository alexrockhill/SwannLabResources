import os
import os.path as op
import numpy as np
import json
from pandas import concat, read_csv
from bids import BIDSLayout


def get_config():
    with open('./params.json', 'r') as f:
        config = json.load(f)
    return config


def get_participants():
    config = get_config()
    return read_csv(op.join(config['bids_dir'], 'participants.tsv'), sep='\t')


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


def get_dfs(behs):
    df_list = []
    for beh in behs:
        df = read_csv(beh.path, sep='\t')
        df['Subject'] = beh.entities['subject']
        df_list.append(df)
    dfs = concat(df_list)
    return dfs


def exclude_subjects(bids_list):
    config = get_config()
    bids_list = [bids_item for bids_item in bids_list if
                 bids_item.entities['subject'] not in
                 config['exclude_subjects']]
    return bids_list


def get_events(df, events, event, condition, value):
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
