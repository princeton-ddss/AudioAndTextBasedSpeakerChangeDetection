import os
import pandas as pd

labelled_data_path = '/Users/jf3375/Desktop/evaluation_data/VoxConverse/test_rttm'
labelled_data_csv_path = '/Users/jf3375/Desktop/evaluation_data/VoxConverse/test_csv'

os.chdir(labelled_data_path)

rttm_files = [file for file in os.listdir(labelled_data_path) if file.endswith('.rttm')]

for filename in rttm_files:
    print(filename)
    df = pd.read_csv(filename, sep=' ', names = ['type', 'filename', 'channelid',
                                          'bgn', 'duration', 'orth', 'subtype', 'speaker', 'cscore', 'slat'])
    df.to_csv(os.path.join(labelled_data_csv_path, filename.split('.')[0] + '.csv'), index = False)



