import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from swann.utils import get_config, derivative_fname

from mne import Epochs


def plot_erp(rawf, raw, event, events, bl_events, l_freq=None, h_freq=30,
             picks=None, overwrite=False):
    ''' Plots event-related potentials for given data
    Parameters
    ----------
    rawf : pybids.BIDSlayout file
        The object containing the raw data.
    raw : mne.io.Raw
        The raw data object.
    event : str
        The name of the event (e.g. `Response`).
    events : np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the event and trials that are described by the name.
    bl_events: np.array(n_events, 3)
        The events from mne.events_from_annotations or mne.find_events
        corresponding to the baseline for the event and trials
        that are described by the name.
    l_freq : float
        Low frequency to use to filter the data.
    h_freq : float
        High frequency to use to filter the data.
    picks : None | list of str
        The names of the channels to plot.
    '''
    config = get_config()
    raw = raw.copy()
    plotf = derivative_fname(rawf, 'plots/erps',
                             'event-{}_erp'.format(event), config['fig'])
    if not op.isdir(op.dirname(plotf)):
        os.makedirs(op.dirname(plotf))
    if op.isfile(plotf) and not overwrite:
        print('erp plot for {} already exists, '
              'use `overwrite=True` to replot'.format(event))
        return
    epochs = Epochs(raw, events, tmin=config['tmin'], baseline=None,
                    tmax=config['tmax'], preload=True)
    bl_epochs = Epochs(raw, bl_events, tmin=config['baseline_tmin'],
                       baseline=None, tmax=config['baseline_tmax'],
                       preload=True)
    evoked_data = np.median(bl_epochs._data, axis=0)
    epochs._data -= np.median(evoked_data, axis=1)[:, np.newaxis]
    evoked = epochs.average()
    if l_freq is not None and h_freq is not None:
        evoked = evoked.filter(l_freq=l_freq, h_freq=h_freq)
    fig = evoked.plot(picks=picks, show=False)
    fig.suptitle('Event-Related Potential for the {} Event'.format(event))
    fig.savefig(plotf, dpi=300)
    plt.close(fig)
