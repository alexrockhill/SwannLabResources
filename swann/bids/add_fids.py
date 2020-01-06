# Authors: Alex Rockhill <aprockhill206@gmail.com>
# License: BSD (3-clause)

import os.path as op
import importlib
import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting import plot_anat

import mne

from mne_bids import write_anat

'''
mne setup_source_space --subject sub-1
mkheadsurf -subjid sub-1
mne flash_bem -s sub-1 -3 -n
mne setup_forward_model -s sub-1
'''

output_path = '/Users/alexrockhill/education/UO/SwannLab/EMU_data_BIDS'
subject = 'sub-1'
subjects_dir = '/Users/alexrockhill/education/UO/SwannLab/EMU_data/'
t1wf = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
fname = ('/Users/alexrockhill/education/UO/SwannLab/' +
         'EMU_data/sub-1_raw/sub-1_eeg.edf')
raw = mne.io.read_raw_edf(fname)

ch_names = raw.ch_names
elec = np.zeros((len(ch_names), 3))

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='head',
                                        lpa=np.zeros((3,)),
                                        nasion=np.zeros((3,)),
                                        rpa=np.zeros((3,)))

mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir,
                       inst=fname)

fids, coord_frame = mne.io.read_fiducials(
    op.join(subjects_dir, subject, 'bem', '%s-fiducials.fif' % subject))

montage2 = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                         coord_frame='head',
                                         lpa=fids[0]['r'],
                                         nasion=fids[1]['r'],
                                         rpa=fids[2]['r'])

raw.set_montage(montage2)

# No trans specified thus the identity matrix is our transform
trans = mne.transforms.Transform(fro='head', to='mri')

fig = mne.viz.plot_alignment(raw.info, trans, subject=subject, dig='fiducials',
                             eeg=['original', 'projected'], meg=[],
                             coord_frame='head', subjects_dir=subjects_dir)

ses = 'test'

anat_dir = write_anat(bids_root=output_path,  # the BIDS dir we wrote earlier
                      subject=subject.replace('sub-', ''),
                      t1w=t1wf,  # path to the MRI scan
                      session=ses,
                      raw=raw,  # the raw MEG data file connected to the MRI
                      trans=trans,  # our transformation matrix
                      deface=True)

t1_nii_fname = op.join(anat_dir, '%s_ses-%s_T1w.nii.gz' % (subject, ses))

# Plot it
importlib.reload(plt)  # bug due to mne.gui
fig, ax = plt.subplots()
plot_anat(t1_nii_fname, axes=ax, title='Defaced')
plt.show()
