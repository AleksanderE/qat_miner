import pandas as pd
from datetime import datetime
import os


def create_directory_if_not_exist(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)


def read_dataset(path):
    if path[-5:] == '.json':
        df = pd.read_json(path, lines=True)
        df.rename(columns={'date': 'date_'}, inplace=True)
    else:
        df = pd.read_csv(path)
        df['date_'] = df['date'].apply(lambda date_: datetime.strptime(date_, '%Y-%m-%d %H:%M:%S'))
        df['is_root'].fillna(0, inplace=True)
    df.sort_values('date_', inplace=True)
    df['message'].fillna('', inplace=True)
    return df


def filter_dataset(data):
    print("Before:", data.shape[0])
    dialogs_numbers = set(data[data['is_root'] == 1]['dialog_number'])
    data_ = data[data['dialog_number'].isin(dialogs_numbers)]
    print("After removing bad dialogs", data_.shape[0])
    data_ = data_[data_['reply_to_msg_id'].isnull()].reset_index()
    print("Only reply_to_msg_id where is null", data_.shape[0])
    
    no_flood_dialogs = data_[(data_['is_root'] == 1) & (data_['flood'] == 0) & (data_['dialog_type'] == 0)]['dialog_number']
    data_ = data_[data_['dialog_number'].isin(no_flood_dialogs)].reset_index()
    print("without flood", data_.shape[0])
    return data_
