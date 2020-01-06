import os.path as op
import numpy as np
from pandas import read_csv, concat, DataFrame
from scipy.stats import ttest_ind

from swann.utils import get_config, get_participants, derivative_fname


def preproc_slowfast(behf, min_resp_t=0.1, fast_cutoff=False,
                     overwrite=False, return_saved=False, verbose=True):
    """Preprocess slowfast data so that it's counter-balanced
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
    participants = get_participants()
    left_right = \
        {config['left']: participants[config['left_key_col']].loc[0],
         config['right']: participants[config['right_key_col']].loc[0]}
    np.random.seed(config['seed'])
    slow_fast = config['task_params']
    subject = behf.entities['subject']
    if (all([op.isfile(derivative_fname(behf, name, 'tsv')) for name in
             ['slow', 'fast', 'data']]) and not overwrite):
        slow = read_csv(derivative_fname(behf, 'slow', 'tsv'), sep='\t')
        fast = read_csv(derivative_fname(behf, 'fast', 'tsv'), sep='\t')
        data = read_csv(derivative_fname(behf, 'data', 'tsv'), sep='\t')
        blocks = list()
        for i in range(data['n_blocks'].loc[0]):
            blocks.append(read_csv(derivative_fname(behf, 'block_%i' % i,
                                                    'tsv'), sep='\t'))
        return slow, fast, blocks, data['p'].loc[0], data['accuracy'].loc[0]
    elif return_saved:
        raise ValueError('Behavior must first be computed')
    if verbose:
        print('preprocessing %s slowfast behavior' % subject +
              (', session %s' % behf.entities['session']
               if 'session' in behf.entities else ''))
    df = read_csv(behf.path, sep='\t')
    fast = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'fast']]
    fast['answer_key'] = [left_right[ans] for ans in
                          fast[config['answer_col']]]
    fast2 = fast[(fast[config['response_col']] > min_resp_t) &
                 (abs(fast[config['button_col']] -
                  fast['answer_key']) < 0.001)]
    slow = df.loc[[i for i, sf in
                   enumerate(list(df[slow_fast['name_col']]))
                   if slow_fast[str(sf)] == 'slow']]
    slow['answer_key'] = [left_right[ans] for ans in
                          slow[config['answer_col']]]
    slow2 = slow[(slow[config['response_col']] > min_resp_t) &
                 (abs(slow[config['button_col']] -
                  slow['answer_key']) < 0.001)]
    if verbose:
        print('%i no/wrong ' % (len(fast) - len(fast2)) +
              'response trials on fast blocks, ' +
              '%i on slow blocks' % (len(slow) - len(slow2)))
    accuracy = (len(slow2) + len(fast2)) / len(df)
    n_to_exclude = (len(fast) - len(fast2)) - (len(slow) - len(slow2))
    if fast_cutoff and n_to_exclude >= 0:
        if verbose:
            print('%i slowest trials in the slow block ' % n_to_exclude +
                  'excluded to balance for trials in the fast block ' +
                  'that were cut off')
        slow2 = slow2.reset_index()
        indices = \
            list(np.argsort(slow2[config['response_col']])[:-n_to_exclude])
        slow2 = slow2.iloc[indices]
        slow2 = slow2.sort_values(by=config['trial_col'])
    else:
        if fast_cutoff and n_to_exclude < 0:
            print('WARNING: More no/wrong responses for subject ' +
                  '%s on slow blocks than fast. ' % subject +
                  'This is not expected, consider excluding subject')
        more = 'fast' if n_to_exclude > 0 else 'slow'
        less = 'slow' if n_to_exclude > 0 else 'fast'
        if verbose:
            print('%i more trials on %s blocks ' % (abs(n_to_exclude), more) +
                  'missed than on %s blocks, counterbalancing ' % less +
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
        this_block = df[(b * config['n_trials_per_block'] <
                         df[config['trial_col']]) &
                        (df[config['trial_col']] <=
                         (b + 1) * config['n_trials_per_block'])]
        this_block = this_block.reset_index()
        this_block = this_block[this_block[config['response_col']] >
                                min_resp_t]
        blocks.append(this_block)
    slow2.to_csv(derivative_fname(behf, 'slow', 'tsv'), sep='\t', index=False)
    fast2.to_csv(derivative_fname(behf, 'fast', 'tsv'), sep='\t', index=False)
    for i, block in enumerate(blocks):
        block.to_csv(derivative_fname(behf, 'block_%i' % i, 'tsv'),
                     sep='\t', index=False)
    with open(derivative_fname(behf, 'data', 'tsv'), 'w') as f:
        f.write('n_blocks\tp\taccuracy\n%i\t%s\t%s' %
                (len(blocks), p, accuracy))
    return slow2, fast2, blocks, p, accuracy


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
    slow = concat(slows, sort=False)
    fast = concat(fasts, sort=False)
    t, p = ttest_ind(
        slow[config['response_col']], fast[config['response_col']])
    return slow, fast, blocks_group, p, np.mean(accuracies)


def slowfast_group_stats(behfs, name, overwrite=False):
    config = get_config()
    statsf = op.join(config['bids_dir'], 'derivatives',
                     'slowfast_group_stats_%s_slowfast.tsv' % name)
    if op.isfile(statsf) and not overwrite:
        return
    stats = DataFrame(columns=['name', 'slow_mean', 'slow_std', 'fast_mean',
                               'fast_std', 'accuracy'])
    for i, behf in enumerate(behfs):
        this_name = op.splitext(op.basename(behf.path))[0].replace('_beh', '')
        slow, fast, blocks, p, accuracy = \
            preproc_slowfast(behf, return_saved=True)
        stats.loc[i] = [this_name, np.mean(slow[config['response_col']]),
                        np.std(slow[config['response_col']]),
                        np.mean(fast[config['response_col']]),
                        np.std(fast[config['response_col']]),
                        accuracy]
    slow, fast, blocks, p, accuracy = slowfast_group(behfs)
    stats.loc[len(stats)] = ['group %s' % name,
                             np.mean(slow[config['response_col']]),
                             np.std(slow[config['response_col']]),
                             np.mean(fast[config['response_col']]),
                             np.std(fast[config['response_col']]),
                             accuracy]
    stats.to_csv(statsf, sep='\t', index=False)
