import os.path as op
from pandas import read_csv

from swann.utils import derivative_fname


def get_bads(rawf):
    badsf = derivative_fname(rawf, 'bad_channels', 'tsv')
    if op.isfile(badsf):
        return list(read_csv(badsf, sep='\t')['bads'])
    return []


def set_bads(rawf, raw):
    badsf = derivative_fname(rawf, 'bad_channels', 'tsv')
    with open(badsf, 'w') as f:
        f.write('bads\n')
        for ch in raw.info['bads']:
            f.write('%s\n' % ch)
