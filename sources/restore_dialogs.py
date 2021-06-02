import pandas as pd
import random
from sklearn.utils import shuffle


def set_should_predict(df):
    labels = set(df[df['is_root'] == 1]['dialog_number'])
    print('labels', len(labels))

    df_labeled = df[(df['dialog_number'].isin(labels))]
    print('Nodes in dialogs', df_labeled.shape[0])

    df_to_predict = df_labeled[((df_labeled['is_root'] == 0) & (df_labeled['reply_to_msg_id'].isnull()))]
    print('Nodes in predict', df_to_predict.shape[0])

    df['should_predict'] = 0
    df.loc[df['id'].isin(df_to_predict['id']), 'should_predict'] = 1

    
def make_reply_chains(df):
    ids = set(df.id)
    leaves = ids - set(df.reply_to_msg_id)
    pairs = {row['id'] : row['reply_to_msg_id'] for _, row in df.iterrows()}
    chains = []
    for id_ in leaves:
        chain = []
        while id_ in ids:
            chain.append(id_)
            id_ = pairs[id_]
        chains.append(tuple(chain[::-1]))
    return chains


def sort_reply_chains_by_root(chains, msg_id_to_date_dict):
    result = [(msg_id_to_date_dict[chain[0]], chain) for chain in chains]
    result.sort()
    return [pair[1] for pair in result]


def create_positives(reply_chains):
    res = {'msg_ids1': [],
           'msg_ids2': []}

    for chain in reply_chains:
        assert len(chain) > 0
        if len(chain) == 1:
            continue
        split_index = random.randint(0, len(chain) - 2)
        res['msg_ids1'].append(tuple(chain[:split_index + 1]))
        res['msg_ids2'].append(tuple(chain[split_index + 1:]))
        
#         assert len(result[-1][0]) > 0 and len(result[-1][1]) > 0
#         assert type(result[-1][1][-1]) == str
    return pd.DataFrame.from_dict(res)


def create_negatives(first_reply_chains, second_reply_chains):
    res = {'msg_ids1': [],
           'msg_ids2': []}
    
    for first_chain in first_reply_chains:
        for second_chain in second_reply_chains:
            res['msg_ids1'].append(first_chain)
            res['msg_ids2'].append(second_chain)

    return pd.DataFrame.from_dict(res)


def join_positives_and_negatives(positives, negatives):
    positives['label'] = 1
    negatives['label'] = 0
    return shuffle(pd.concat([positives, negatives])).reset_index()


# Features

def add_from_ids(msg_ids, msg_id_to_from_id_dict):
    return set(msg_id_to_from_id_dict[id_] for id_ in msg_ids)


def add_min_date(msg_ids, msg_id_to_date_dict):
    return min(msg_id_to_date_dict[id_] for id_ in msg_ids)


def add_max_date(msg_ids, msg_id_to_date_dict):
    return max(msg_id_to_date_dict[id_] for id_ in msg_ids)


def add_dialog_number_label(msg_ids1, msg_ids2, msg_id_to_dialog_number_dict):
    return int(msg_id_to_dialog_number_dict[msg_ids1[0]] == msg_id_to_dialog_number_dict[msg_ids2[0]])


def count_msgs_between(msg_ids1, msg_ids2):
    ids1 = list(int(id_) for id1 in msg_ids1 for id_ in id1.split('__'))
    ids2 = list(int(id_) for id2 in msg_ids2 for id_ in id2.split('__'))
    res = 999999999999999
    for id1 in ids1:
        for id2 in ids2:
#             assert id2 > id1
            if res > abs(id2 - id1):
                res = abs(id2 - id1)
    return res


def count_seconds_between(msg_ids1, msg_ids2, msg_id_to_date_dict):
#     ids1 = list(int(id_) for id1 in msg_ids1 for id_ in id1.split('__'))
#     ids2 = list(int(id_) for id2 in msg_ids2 for id_ in id2.split('__'))
#     ids1 = [int(id_) for id_ in msg_ids1]
#     ids2 = [int(id_) for id_ in msg_ids2]
    res = 999999999999999
    for id1 in msg_ids1:
        for id2 in msg_ids2:
#             assert id2 > id1
            time_span = abs(msg_id_to_date_dict[id2] - msg_id_to_date_dict[id1]).total_seconds()
            if res > time_span:
                res = time_span
    return res


def create_pairs(reply_chains):
    res = {'msg_ids1': [],
           'msg_ids2': []}
    for i, second_reply_chain in enumerate(reply_chains):
        for first_reply_chain in reply_chains[:i]:
            res['msg_ids1'].append(first_reply_chain)
            res['msg_ids2'].append(second_reply_chain)
    return pd.DataFrame.from_dict(res)


def add_features(df, msg_id_to_from_id_dict, msg_id_to_date_dict):
    df['from_ids1'] = df['msg_ids1'].apply(lambda msg_ids: add_from_ids(msg_ids, msg_id_to_from_id_dict))
    df['from_ids2'] = df['msg_ids2'].apply(lambda msg_ids: add_from_ids(msg_ids, msg_id_to_from_id_dict))

    df['min_date1'] = df['msg_ids1'].apply(lambda msg_ids: add_min_date(msg_ids, msg_id_to_date_dict))
    df['min_date2'] = df['msg_ids2'].apply(lambda msg_ids: add_min_date(msg_ids, msg_id_to_date_dict))

    df['max_date1'] = df['msg_ids1'].apply(lambda msg_ids: add_max_date(msg_ids, msg_id_to_date_dict))
    df['max_date2'] = df['msg_ids2'].apply(lambda msg_ids: add_max_date(msg_ids, msg_id_to_date_dict))

    df['msgs_between'] = df.apply(
        lambda row: count_msgs_between(row['msg_ids1'], row['msg_ids2']), axis=1)

    df['seconds_between'] = df.apply(
        lambda row: count_seconds_between(row['msg_ids1'], row['msg_ids2'], msg_id_to_date_dict), axis=1)


def add_label(df, msg_id_to_dialog_number_dict):
    df['label'] = df.apply(
        lambda row: add_dialog_number_label(row['msg_ids1'], row['msg_ids2'], msg_id_to_dialog_number_dict), axis=1)


def join_texts(message1, message2):
    return message1 + ' [SEP] ' + message2


def join_mesages(texts):
    return ' '.join(texts)


def make_dataframe(df, msg_id_to_from_id_dict, msg_id_to_message_dict, msg_id_to_date_dict):
    result = {'msg_ids1':[], 'msg_ids2':[], 
              'message1':[], 'message2':[],
              'from_ids1':[], 'from_ids2':[],
              'min_date1':[], 'max_date1':[],
              'min_date2':[], 'max_date2':[],
              'messages_number_between': [], 'seconds_between': [],
              'label': []}
    for tuple_ in df.itertuples():
        result['msg_ids1'].append(tuple_.msg_ids1)
        result['msg_ids2'].append(tuple_.msg_ids2)
        result['message1'].append(join_mesages(msg_id_to_message_dict[id_] for id_ in tuple_.msg_ids1))
        result['message2'].append(join_mesages(msg_id_to_message_dict[id_] for id_ in tuple_.msg_ids2))
        result['from_ids1'].append(tuple(msg_id_to_from_id_dict[id_] for id_ in tuple_.msg_ids1))
        result['from_ids2'].append(tuple(msg_id_to_from_id_dict[id_] for id_ in tuple_.msg_ids2))
        result['min_date1'].append(min(msg_id_to_date_dict[id_] for id_ in tuple_.msg_ids1))
        result['max_date1'].append(max(msg_id_to_date_dict[id_] for id_ in tuple_.msg_ids1))
        result['min_date2'].append(min(msg_id_to_date_dict[id_] for id_ in tuple_.msg_ids2))
        result['max_date2'].append(max(msg_id_to_date_dict[id_] for id_ in tuple_.msg_ids2))
        result['messages_number_between'].append(count_msgs_between(tuple_.msg_ids1, tuple_.msg_ids2))
        result['seconds_between'].append(tuple_.seconds_between)
        result['label'].append(tuple_.label)
    return pd.DataFrame.from_dict(result)
