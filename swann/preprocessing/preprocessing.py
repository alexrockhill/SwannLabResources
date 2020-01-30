import os.path as op
import numpy as np
from pandas import read_csv

from swann.utils import (get_config, derivative_fname, read_raw,
                         get_events)

from autoreject import AutoReject
from mne.preprocessing import (ICA, read_ica, create_eog_epochs,
                               create_ecg_epochs)
from mne import Epochs


def get_raw(rawf):
    raw = read_raw(rawf.path)
    raw.info['bads'] += [ch for ch in get_bads(rawf) if
                         ch not in raw.info['bads']]
    return raw


def get_info(rawf):
    return get_raw(rawf).info


def get_bads(rawf):
    badsf = derivative_fname(rawf, 'data', 'bad_channels', 'tsv')
    if op.isfile(badsf):
        return list(read_csv(badsf, sep='\t')['bads'])
    return []


def set_bads(rawf, bads):
    badsf = derivative_fname(rawf, 'data', 'bad_channels', 'tsv')
    with open(badsf, 'w') as f:
        f.write('bads\n')
        for ch in bads:
            f.write('%s\n' % ch)


def find_ica(rawf, method='fastica', n_components=None, overwrite=False):
    if (op.isfile(derivative_fname(rawf, 'data', 'ica', 'fif')) and
            not overwrite):
        print('ICA already computed, use `overwrite=True` to recompute')
        return
    config = get_config()
    raw = read_raw(rawf.path)
    raw.info['bads'] += [ch for ch in get_bads(rawf) if
                         ch not in raw.info['bads']]
    ica = ICA(method=method, n_components=n_components,
              random_state=config['seed'])
    ica.fit(raw)
    set_ica(rawf, ica)


def get_aux_epochs(ica, raw):
    config = get_config()
    eogs, ecgs, emgs = config['eogs'], config['ecgs'], config['emgs']
    aux_epochs = dict()
    for eog in eogs:
        aux_epochs[eog] = create_eog_epochs(raw, ch_name=eog, h_freq=8)
    for ecg in ecgs + emgs:  # use emgs to detect heartbeat
        aux_epochs[ecg] = create_ecg_epochs(raw, ch_name=ecg)
    return aux_epochs


def apply_ica(rawf):
    raw = get_raw(rawf)
    ica = get_ica(rawf)
    components = get_ica_components(rawf)
    raw.load_data()
    raw = ica.apply(raw, exclude=components)
    return raw


def get_ica(rawf):
    icaf = derivative_fname(rawf, 'data', 'ica', 'fif')
    if not op.isfile(icaf):
        raise ValueError('ICA not computed, this must be done first')
    return read_ica(icaf)


def get_ica_components(rawf):
    componentsf = derivative_fname(rawf, 'data', 'ica_components', 'tsv')
    if op.isfile(componentsf):
        return list(read_csv(componentsf, sep='\t')['components'])
    else:
        return []


def set_ica(rawf, ica):
    ica.save(derivative_fname(rawf, 'data', 'ica', 'fif'))


def set_ica_components(rawf, components):
    componentsf = derivative_fname(rawf, 'data', 'ica_components', 'tsv')
    with open(componentsf, 'w') as f:
        f.write('components\n')
        for component in components:
            f.write('%i\n' % component)


def mark_autoreject(rawf, event, raw=None,
                    n_interpolates=[1, 2, 3, 5, 7, 10, 20],
                    consensus_percs=np.linspace(0, 1.0, 11),
                    return_saved=False, verbose=True, overwrite=False):
    config = get_config()
    autorejectf = derivative_fname(rawf, 'data', 'rejected_epochs_%s' % event,
                                   'tsv')
    if not op.isfile(autorejectf):
        if return_saved:
            raise ValueError('Autoreject not yet run, cannot return saved')
    elif not overwrite:
        if not return_saved:
            print('Autoreject already computed for %s for ' % event +
                  '%s, use `overwrite=True` to rerun' % rawf.path)
        return list(read_csv(autorejectf, sep='\t')['rejected_epochs'])
    events = get_events(raw)
    if verbose:
        print('Computing autoreject on %s' % event)
    epochs = Epochs(raw, events[event],
                    tmin=config['tmin'], tmax=config['tmax'])
    ar = AutoReject(n_interpolates, consensus_percs,
                    random_state=config['seed'],
                    n_jobs=config['n_jobs'])
    epochs.load_data()
    epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)
    with open(autorejectf, 'w') as f:
        f.write('rejected_epochs\n')
        for i, rejected in enumerate(reject_log.bad_epochs):
            if rejected:
                f.write('%s\n' % i)
    return list(read_csv(autorejectf, sep='\t')['rejected_epochs'])


'''
def default_aux(inst):
    inds = pick_types(inst.info, meg=False, eog=True)
    eogs = [inst.ch_names[ind] for ind in inds]
    if eogs:
        print('Using ' + ', '.join(eogs) + ' as eogs')
    else:
        inst.plot()
        eogs = input('Pick eog channels from: %s' % ', '.join(inst.ch_names) +
                     'using `, ` to separate\n').split(', ')
    inds = pick_types(inst.info, meg=False, ecg=True)
    ecgs = [inst.ch_names[ind] for ind in inds]
    if ecgs:
        print('Using ' + ', '.join(eogs) + ' as eogs')
    else:
        inst.plot()
        ecgs = input('Pick ecg channels from: %s' % ', '.join(inst.ch_names) +
                     'using `, ` to separate\n').split(', ')
    return eogs, ecgs
'''
