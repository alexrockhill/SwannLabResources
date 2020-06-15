import os.path as op

from swann.utils import get_config, derivative_fname, read_raw
from swann.preprocessing import (get_bads, set_bads, get_ica,
                                 set_ica_components)


def plot_find_bads(rawf, overwrite=False):
    config = get_config()
    badsf = derivative_fname(rawf, 'data', 'bad_channels', 'tsv')
    if op.isfile(badsf) and not overwrite:
        print('Bad channels already marked, skipping plot raw for '
              'determining bad channels, use `overwrite=True` to plot')
        return
    else:
        raw = read_raw(rawf.path)
        print('Plotting PSD spectrogram and raw channels for bad channel '
              'selection, %s' % rawf.path)
        raw.info['bads'] += [ch for ch in get_bads(rawf) if
                             ch not in raw.info['bads']]
        psdf = derivative_fname(rawf, 'plots', 'psd_w_bads', config['fig'])
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch not in ['Event', 'Status']],
                           show=False)
        fig.savefig(psdf, dpi=300)
        raw.plot(block=True)  # pick bad channels
        set_bads(rawf, raw.info['bads'])
        fig = raw.plot_psd(picks=[i for i, ch in enumerate(raw.ch_names)
                                  if ch not in ['Event', 'Status'] and
                                  ch not in raw.info['bads']])
        fig.savefig(derivative_fname(rawf, 'plots', 'psd', config['fig']))


def plot_ica(rawf, method='fastica', n_components=None,
             overwrite=False):
    if (op.isfile(derivative_fname(rawf, 'data', 'ica_components', 'tsv')) and
            not overwrite):
        print('ICA component choices already saved, use `overwrite=True` '
              'to re-plot.')
        return
    raw = read_raw(rawf.path)
    raw.info['bads'] = get_bads(rawf)
    ica = get_ica(rawf)
    ica.plot_sources(raw, block=True, show=True,
                     title=rawf.entities['subject'])
    '''
    done = False
    while not done:
        # ica.plot_components(show=False) just click on the plot
        # ica.plot_properties(raw, show=False) too many plots
        ica.plot_sources(raw, block=True, show=True,
                         title=rawf.entities['subject'])
        aux_epochs = get_aux_epochs(ica, raw)
        for ch, epo in aux_epochs.items():
            fig = ica.plot_overlay(epo.average(), show=False)
            fig.suptitle('%s %s' % (rawf.entities['subject'], ch))
            fig = ica.plot_sources(epo.average(), exclude=ica.exclude,
                                   show=False)
            fig.suptitle('%s %s' % (rawf.entities['subject'], ch))
        done = input('Have all the artifact components been removed ' +
                     '(Y/N)?').upper() == 'Y'
    '''
    set_ica_components(rawf, ica.exclude)
