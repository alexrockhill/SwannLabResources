import sys
import numpy as np
import json
from pandas import read_csv


def add_beh_sidecar(bids_name, paramf, bids_dir):
    df = read_csv(paramf, sep='\t')
    beh_metadataf = '%s_beh.json' % bids_name
    beh_metadata = dict()
    for column in df:
        this_metadata = df[column].loc[0]
        if (not this_metadata or this_metadata is None or
                (isinstance(this_metadata, np.floating) and
                 np.isnan(this_metadata))):
            beh_metadata[column] = 'n/a'
        else:
            if isinstance(this_metadata, np.integer):
                this_metadata = int(this_metadata)
            elif isinstance(this_metadata, np.floating):
                this_metadata = float(this_metadata)
            elif isinstance(this_metadata, np.ndarray):
                this_metadata = this_metadata.tolist()
            beh_metadata[column] = this_metadata
    with open(beh_metadataf, 'w') as f:
        json.dump(beh_metadata, f, indent=4, separators=(',', ': '),
                  sort_keys=True)


if __name__ == '__main__':
    add_beh_sidecar(*sys.argv[1:])
