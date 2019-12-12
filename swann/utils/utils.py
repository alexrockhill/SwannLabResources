import os
import os.path as op
import numpy as np
import json
from pandas import concat, read_csv


def get_config():
    with open('./params.json', 'r') as f:
        config = json.load(f)
    return config


def make_derivatives_dir(subject):
    config = get_config()
    if not op.isdir(op.join(config['bids_dir'], 'derivatives',
                    'sub-%s' % subject)):
        os.makedirs(op.join(config['bids_dir'], 'derivatives',
                            'sub-%s' % subject))


def get_dfs(behs):
    df_list = []
    for beh in behs:
        df = read_csv(beh.path, sep='\t')
        df['Subject'] = beh.entities['subject']
        df_list.append(df)
    dfs = concat(df_list)
    return dfs


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
