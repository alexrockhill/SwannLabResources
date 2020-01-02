import sys
import os.path as op
from pandas import DataFrame, read_csv


def add_subject_info(subject, bids_dir, paramf):
    participantsf = op.join(bids_dir, 'participants.tsv')
    if op.isfile(participantsf):
        df = read_csv(participantsf, sep='\t')
    else:
        df = DataFrame({'participant_id': ['sub-' + subject], 'age': ['n/a'],
                        'sex': ['n/a']})
    df2 = read_csv(paramf, sep='\t')
    index = df.index[df['participant_id'] == 'sub-' + subject][0]
    for column in df2:
        if column in df:
            df.loc[index, column] = df2.loc[0, column]
        else:
            df[column] = [df2.loc[0, column] if i == index else 'n/a'
                          for i in df.index]
    df.fillna('n/a', inplace=True)
    df.to_csv(participantsf, sep='\t', index=False)


if __name__ == '__main__':
    subject, bids_dir, paramf = sys.argv[1:]
    add_subject_info(subject, bids_dir, paramf)
