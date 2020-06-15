import os.path as op
import numpy as np
import json
from pandas import read_csv, concat, DataFrame
from scipy.stats import ttest_ind

from swann.utils import get_config, derivative_fname, get_sidecar


def preproc_slowfast(behf, min_resp_t=0.1, fast_cutoff=False,
                     overwrite=False, return_saved=False, verbose=True):
    """Preprocess slowfast data so that it's counter-balanced.
    Parameters
    ----------
    behf : BIDSDataFile
        The behavior data file from the pybids layout object.
    min_resp_t : float
        The minimum amount of time acceptable to include a response.
    fast_cutoff : bool
        Whether the responses on fast blocks were cut off.
    overwrite : bool
        Whether to overwrite saved behavior files if they exist.
    return_saved : bool
        Whether to only return the saved values and not recompute new.
    verbose : bool
        Whether to display text output.
    """
    config = get_config()
    np.random.seed(config['seed'])
    slow_fast = config['task_params']
    subject = behf.entities['subject']
    slowf = derivative_fname(behf, 'data', 'slow', 'tsv')
    fastf = derivative_fname(behf, 'data', 'fast', 'tsv')
    dataf = derivative_fname(behf, 'data', 'data', 'tsv')
    blockf = derivative_fname(behf, 'data', 'block_%i', 'tsv')
    if (all([op.isfile(thisf) for thisf in [slowf, fastf, dataf]]
            ) and not overwrite):
        slow = read_csv(slowf, sep='\t')
        fast = read_csv(fastf, sep='\t')
        data = read_csv(dataf, sep='\t')
        blocks = list()
        for i in range(data['n_blocks'].loc[0]):
            blocks.append(read_csv(blockf % i, sep='\t'))
        return slow, fast, blocks, data['p'].loc[0], data['accuracy'].loc[0]
    elif return_saved:
        raise ValueError('Behavior must first be computed')
    if verbose:
        print('preprocessing %s slowfast behavior' % subject
              (', session %s' % behf.entities['session']
               if 'session' in behf.entities else ''))
    df = read_csv(behf.path, sep='\t')
    with open(get_sidecar(behf.path, 'json'), 'r') as f:
        df_sidecar = json.load(f)
    left_right = \
        {config['left']: df_sidecar[config['left_key_col']],
         config['right']: df_sidecar[config['right_key_col']]}
    fast = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'fast']]
    fast['answer_key'] = [left_right[ans] for ans in
                          fast[config['answer_col']]]
    fast2 = fast[(fast[config['response_col']] > min_resp_t
                  ) & (abs(fast[config['button_col']
                                ] - fast['answer_key']) < 0.001)]
    slow = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'slow']]
    slow['answer_key'] = [left_right[ans] for ans in
                          slow[config['answer_col']]]
    slow2 = slow[(slow[config['response_col']] > min_resp_t
                  ) & (abs(slow[config['button_col']
                                ] - slow['answer_key']) < 0.001)]
    if verbose:
        print('%i ' % (len(fast) - len(fast2)) + 'no/wrong '
              'response trials on fast blocks, '
              '%i on slow blocks' % (len(slow) - len(slow2)))
    accuracy = (len(slow2) + len(fast2)) / len(df)
    n_to_exclude = (len(fast) - len(fast2)) - (len(slow) - len(slow2))
    if fast_cutoff and n_to_exclude >= 0:
        if verbose:
            print('%i ' % n_to_exclude + 'slowest trials in the slow block '
                  'excluded to balance for trials in the fast block '
                  'that were cut off')
        slow2 = slow2.reset_index()
        indices = \
            list(np.argsort(slow2[config['response_col']])[:-n_to_exclude])
        slow2 = slow2.iloc[indices]
        slow2 = slow2.sort_values(by=config['trial_col'])
    else:
        if fast_cutoff and n_to_exclude < 0:
            print('WARNING: More no/wrong responses for subject '
                  '%s ' % subject + 'on slow blocks than fast. '
                  'This is not expected, consider excluding subject')
        more = 'fast' if n_to_exclude > 0 else 'slow'
        less = 'slow' if n_to_exclude > 0 else 'fast'
        if verbose:
            print('%i ' % abs(n_to_exclude) + 'more trials on '
                  '%s ' % more + 'blocks missed than on '
                  '%s ' % less + 'blocks, counterbalancing '
                  'by excluding that many %s trials' % less)
        if n_to_exclude > 0:
            indices = np.random.choice(range(len(slow2)), len(fast2),
                                       replace=False)
            slow2 = slow2.iloc[indices]
            slow2 = slow2.sort_values(by=config['trial_col'])
        else:
            indices = np.random.choice(range(len(fast2)), len(slow2),
                                       replace=False)
            fast2 = fast2.iloc[indices]
            fast2 = fast2.sort_values(by=config['trial_col'])
    if verbose:
        print('Number of slow trials: %i, Number of fast trials: %i' %
              (len(slow2), len(fast2)))
    t, p = ttest_ind(
        slow2[config['response_col']], fast2[config['response_col']])
    blocks = list()
    for b in range(int(len(df) / config['n_trials_per_block'])):
        this_block = df[
            (b * config['n_trials_per_block'] < df[config['trial_col']]
             ) & (df[config['trial_col']
                     ] <= (b + 1) * config['n_trials_per_block'])]
        this_block = this_block.reset_index()
        this_block = this_block[
            this_block[config['response_col']] > min_resp_t]
        blocks.append(this_block)
    slow2.to_csv(slowf, sep='\t', index=False)
    fast2.to_csv(fastf, sep='\t', index=False)
    for i, block in enumerate(blocks):
        block.to_csv(blockf % i, sep='\t', index=False)
    with open(dataf, 'w') as f:
        f.write('n_blocks\tp\taccuracy\n%i\t%s\t%s' %
                (len(blocks), p, accuracy))
    return slow2, fast2, blocks, p, accuracy


def slowfast2epochs_indices(behf):
    config = get_config()
    df = read_csv(behf.path, sep='\t')
    slow, fast, blocks, p, accuracy = \
        preproc_slowfast(behf, return_saved=True)
    slow_trials = list(slow[config['trial_col']])
    fast_trials = list(fast[config['trial_col']])
    all_indices = np.arange(len(df))
    slow_indices = [i for i, trial in enumerate(df[config['trial_col']]) if
                    trial in slow_trials]
    fast_indices = [i for i, trial in enumerate(df[config['trial_col']]) if
                    trial in fast_trials]
    return all_indices, slow_indices, fast_indices


def slowfast_group(behfs, overwrite=False):
    config = get_config()
    slows = list()
    fasts = list()
    blocks_group = list()
    accuracies = list()
    for behf in behfs:
        slow, fast, blocks, _, accuracy = \
            preproc_slowfast(behf, return_saved=True)
        slows.append(slow)
        fasts.append(fast)
        for block in blocks:
            blocks_group.append(block)
        accuracies.append(accuracy)
    slow = concat(slows, axis=0, sort=False)
    fast = concat(fasts, axis=0, sort=False)
    t, p = ttest_ind(
        slow[config['response_col']], fast[config['response_col']])
    return slow, fast, blocks_group, p, np.mean(accuracies)


def slowfast_group_stats(behfs, name, overwrite=False):
    config = get_config()
    statsf = op.join(config['bids_dir'], 'derivatives',
                     'slowfast_group_stats_%s_slowfast.tsv' % name)
    if op.isfile(statsf) and not overwrite:
        print('Skipping groups stats, already exists and overwrite is False')
        return
    stats = DataFrame(columns=['name', 'slow_mean', 'slow_std', 'fast_mean',
                               'fast_std', 'accuracy', 'p'])
    for i, behf in enumerate(behfs):
        this_name = op.splitext(op.basename(behf.path))[0].replace('_beh', '')
        slow, fast, blocks, p, accuracy = \
            preproc_slowfast(behf, return_saved=True)
        stats.loc[i] = [this_name, np.mean(slow[config['response_col']]),
                        np.std(slow[config['response_col']]),
                        np.mean(fast[config['response_col']]),
                        np.std(fast[config['response_col']]),
                        accuracy, p]
    slow, fast, blocks, p, accuracy = slowfast_group(behfs)
    stats.loc[len(stats)] = ['group %s' % name,
                             np.mean(slow[config['response_col']]),
                             np.std(slow[config['response_col']]),
                             np.mean(fast[config['response_col']]),
                             np.std(fast[config['response_col']]),
                             accuracy, p]
    stats.to_csv(statsf, sep='\t', index=False)


def comparison_stats(behfs, group_by, group0, group1, condition,
                     overwrite=False):
    config = get_config()
    statsf = \
        op.join(config['bids_dir'], 'derivatives', 'comparison_stats'
                '_%s-%s_vs_%s' % (group_by, group0, group1) + '_'
                '%s_slowfast.tsv' % condition)
    if op.isfile(statsf) and not overwrite:
        print('Skipping comparison stats, already exists '
              'and overwrite is False')
        return
    behfs0 = list()
    behfs1 = list()
    for behf in behfs:
        if '%s-%s' % (group_by, group0) in behf.path:
            behfs0.append(behf)
        elif '%s-%s' % (group_by, group1) in behf.path:
            behfs1.append(behf)
        else:
            print('Group %s not found for behavior file %s' %
                  (group_by, behf.path))
    stats = DataFrame(columns=['name',
                               '%s_%s_mean' % (condition, group0),
                               '%s_%s_std' % (condition, group0),
                               '%s_%s_mean' % (condition, group1),
                               '%s_%s_std' % (condition, group1),
                               '%s_p' % condition,
                               '%s_%s_slow_mean' % (condition, group0),
                               '%s_%s_slow_std' % (condition, group0),
                               '%s_%s_slow_mean' % (condition, group1),
                               '%s_%s_slow_std' % (condition, group1),
                               '%s_slow_p' % condition,
                               '%s_%s_fast_mean' % (condition, group0),
                               '%s_%s_fast_std' % (condition, group0),
                               '%s_%s_fast_mean' % (condition, group1),
                               '%s_%s_fast_std' % (condition, group1),
                               '%s_fast_p' % condition])
    slow0, fast0, blocks0, p0, accuracy0 = slowfast_group(behfs0)
    sf0 = slow0.append(fast0)
    slow1, fast1, blocks1, p1, accuracy1 = slowfast_group(behfs1)
    sf1 = slow1.append(fast1)
    t_all, p_all = ttest_ind(sf0[condition], sf1[condition])
    t_slow, p_slow = ttest_ind(slow0[condition], slow1[condition])
    t_fast, p_fast = ttest_ind(fast0[condition], fast1[condition])
    stats.loc[0] = ['Comparison %s: %s vs %s' % (group_by, group0, group1),
                    np.mean(sf0[condition]),
                    np.std(sf0[condition]),
                    np.mean(sf1[condition]),
                    np.std(sf1[condition]),
                    p_all,
                    np.mean(slow0[condition]),
                    np.std(slow0[condition]),
                    np.mean(slow1[condition]),
                    np.std(slow1[condition]),
                    p_slow,
                    np.mean(fast0[condition]),
                    np.std(fast0[condition]),
                    np.mean(fast1[condition]),
                    np.std(fast1[condition]),
                    p_fast]
    stats.to_csv(statsf, sep='\t', index=False)
