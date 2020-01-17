

from pactools import Comodulogram, raw_to_mask


# need to remove the computations?
def plot_pac(subject, raw, ch_names=None, events=None, tmin=None, tmax=None):
    config = get_config()
    if ch_names is None:
        ch_names = raw.ch_names
    for idx0, ch0 in enumerate(ch_names):
        for idx1, ch1 in enumerate(ch_names):
            if ch0 in raw.info['bads'] or ch1 in raw.info['bads']:
                continue
            figf = op.join(config['bids_dir'], 'derivatives',
                           'sub-%s' % subject,
                           'sub-%s_driver-%s_carrier-%s_DAR_PAC.png' %
                           (subject, ch0, ch1))
            # create the input array for Comodulogram.fit
            low_sig, high_sig, mask = raw_to_mask(raw, ixs=[idx0, idx1],
                                                  events=events, tmin=tmin,
                                                  tmax=tmax)
            # create the instance of Comodulogram
            estimator = \
                Comodulogram(fs=raw.info['sfreq'],
                             n_surrogates=int(10 / config['p_value']),
                             low_fq_range=np.linspace(1, 20, 20),
                             low_fq_width=2., method='duprelatour',
                             progress_bar=True)
            # compute the comodulogram
            try:
                estimator.fit(low_sig, high_sig, mask)
            except Exception as e:
                print(e)
                continue
            # plot the results
            fig, ax = plt.subplots()
            fig = estimator.plot(contour_method='comod_max',
                                 contour_level=config['p_value'],
                                 titles=['Driver %s, Carrier %s' % (ch0, ch1)],
                                 axs=[ax], tight_layout=False)
            fig.savefig(figf, dpi=300)
