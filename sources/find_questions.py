import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

# For finding URL
import re

# For finding code fragments
import nltk
nltk.download('punkt')
from string import punctuation


from sklearn import svm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from metrics import *
from sklearn.metrics import precision_recall_fscore_support

from tqdm.notebook import tqdm
from collections import defaultdict

# For teoretical best
from bitsets import bitset
from copy import copy
from collections import Counter
import numpy as np
from itertools import product

# For MLP
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow_addons as tfa
import tensorflow as tf


# Features


meta_features = ['was_active_last_day', 'was_active_last_hour', 'was_active_last_minute', 
                 'is_new_author', 'is_previous_message_him']


binary_features = ['was_question_mark', 'was_greeting', 'was_error', 'ask_help',
                   'with_empty_message',
                   'with_url', 
                   'with_code',
                  ]


features_names = meta_features + binary_features


output_features = ['id', 'date', 'reply_to_msg_id', 'from_id', 'message', 'reply_to_msg_id_human_choice', 'is_root', 
                   'is_root_predict', 'dialog_number', 'is_answer', 
                   'not_solved', 'satisfaction', 'new_subtopic', 'flood', 'dialog_type']


def was_question_mark(message):
    return int('?' in message)


def was_active_last_day(from_id, current_date, full_dataset):
    time_threshold = 24 * 60 * 60
    date_from = current_date - timedelta(seconds=time_threshold)
    return int(full_dataset[(full_dataset['from_id'] == from_id) & 
                            (full_dataset['date_'] >= date_from) & (full_dataset['date_'] < current_date)
                           ].shape[0] > 0)


def was_active_last_hour(from_id, current_date, full_dataset):
    time_threshold = 60 * 60
    date_from = current_date - timedelta(seconds=time_threshold)
    return int(full_dataset[(full_dataset['from_id'] == from_id) & 
                            (full_dataset['date_'] >= date_from) & (full_dataset['date_'] < current_date)
                           ].shape[0] > 0)


def was_active_last_minute(from_id, current_date, full_dataset):
    time_threshold = 60
    date_from = current_date - timedelta(seconds=time_threshold)
    return int(full_dataset[(full_dataset['from_id'] == from_id) & 
                            (full_dataset['date_'] >= date_from) & (full_dataset['date_'] < current_date)
                           ].shape[0] > 0)


def was_greeting(message):
    for text in ['привет', 'здравствуйте', 'доброе утро', 'добрый день', 'добрый вечер', 'доброй ночи']:
        if text in message.lower():
            return 1
    return 0


def was_error(message):
    for text in ['error', 'exception']:
        if text in message.lower():
            return 1
    return 0


def ask_help(message):
    for text in ["помогите", "help", "подскажите"]:
        if text in message.lower():
            return 1
    return 0


def is_new_author(from_id, date_, full_dataset):
    df = full_dataset[(full_dataset['from_id'] == from_id) & (full_dataset['date_'] < date_)]
    return int(df.shape[0] == 0)


def fill_previous_message_him(test_sample, full_dataset):
    full_dataset['prev_from_id'] = full_dataset['from_id'].shift(1)
    full_dataset['is_previous_message_him'] = (full_dataset['prev_from_id'] == full_dataset['from_id']).astype(int)
    df = full_dataset[['id', 'is_previous_message_him']]
    res = pd.merge(test_sample, df, on='id')
    assert res[res['is_previous_message_him'].isnull()].shape[0] == 0
    assert res.shape[0] == test_sample.shape[0]
    return res['is_previous_message_him']


def with_empty_message(combine_message):
    messages = combine_message.split(' [JOIN] ')
    for message in messages:
        if len(message) == 0:
            return 1
    return 0


def with_url(clean_message):
#     return int(('http' in message) or ('https' in message))
    return int('<URL>' in clean_message)


def with_code(clean_message):
    return int('<CODE>' in clean_message)


# Clean text
def _remove_pattern(text, pattern, url_tag='<URL>'):
    res = re.findall(pattern, text)
    if len(res) == 0:
        return text
    for link in res:
        text = text.replace(link, url_tag)
    return text


def _remove_links(text):
    url_tag = '<URL>'
    text = text.replace("https:\\/\\russia\\/sochi\\/", url_tag)
    text = _remove_pattern(text, r'(http?://[^\s]+)')
    text = _remove_pattern(text, r'(https?://[^\s]+)')
    text = _remove_pattern(text, r'(http?:/[^\s]+)')
    text = _remove_pattern(text, r'(https?:/[^\s]+)')
    text = text.replace('http:', '')

    assert 'http:/' not in text, text
    assert 'https:/' not in text, text
    assert 'http:' not in text, text
    assert 'https:' not in text, text
    return text


# find code fragments

_code_start_with = ['select', 'update', 'insert', 'from', 'with', 'alter', 'create', 'modify',
                   'building', 'error', 'code', 'unix', 'pull', 'partition', 'order']
_special_tokens = ['_join_', '_url_']


def _code_word(unistr):
    if unistr in _special_tokens:
        return False
    return all('a' <= uchr <= 'z' or '0' <= uchr <= '9' or uchr == '_' for uchr in unistr)

def _russian_word(unistr):
    return all('а' <= uchr <= 'я' for uchr in unistr)


def _remove_code(text):
    text = text.replace('[JOIN]', '_JOIN_')
    text = text.replace('<URL>', '_URL_')

    tokens = nltk.word_tokenize(text)
    token_lower = [token.lower() for token in tokens]
    
    res = []
    i = 0
    while i < len(tokens):
        if token_lower[i] not in _code_start_with:
            res.append(tokens[i])
            i += 1
            continue
        
        # skip punctuation or special_tokens
        j = i + 1
        while j < len(tokens) and (token_lower[j] in punctuation or token_lower[j] in _special_tokens):
            j += 1
        
        # not code
        if j < len(tokens) and not _code_word(token_lower[j]):
            res.append(tokens[i])
            i += 1
            continue
        
        res.append('<CODE>')
        while j < len(tokens) and (not _russian_word(token_lower[j])):
            j += 1
        i = j
    res = ' '.join(res)
    res = res.replace('_JOIN_', '[JOIN]')
    res = res.replace('_URL_', '<URL>')
    return res


def add_features(test_sample, full_dataset):
    test_sample['clean_message'] = test_sample['message'].apply(_remove_links).apply(_remove_code)
    
    test_sample['was_question_mark'] = test_sample['clean_message'].apply(was_question_mark)
    test_sample['was_active_last_day'] = test_sample.apply(
        lambda row: was_active_last_day(row['from_id'], row['date_'], full_dataset), 1
    )
    test_sample['was_active_last_hour'] = test_sample.apply(
        lambda row: was_active_last_hour(row['from_id'], row['date_'], full_dataset), 1
    )
    test_sample['was_active_last_minute'] = test_sample.apply(
        lambda row: was_active_last_minute(row['from_id'], row['date_'], full_dataset), 1
    )
    test_sample['was_greeting'] = test_sample['message'].apply(was_greeting)
    test_sample['was_error'] = test_sample['message'].apply(was_error)
    test_sample['ask_help'] = test_sample['message'].apply(ask_help)
    test_sample['is_new_author'] = test_sample.apply(
        lambda row: is_new_author(row['from_id'], row['date_'], full_dataset), 1
    )
    test_sample['is_previous_message_him'] = fill_previous_message_him(test_sample, full_dataset)
    test_sample['with_empty_message'] = test_sample['message'].apply(with_empty_message)
    test_sample['with_url'] = test_sample['clean_message'].apply(with_url)
    test_sample['with_code'] = test_sample['clean_message'].apply(with_code)


def get_embedding_features(n_features):
    return [f'feature_embbeding_{i}' for i in range(n_features)]


# Prediction


def cross_val_predict(df, fabric, fabric_arguments=None, features=features_names, kf=LeaveOneOut(), n_splits=5):
    metrics_list = []
    df['predict'] = 0
    for train_idx, test_idx in tqdm(kf.split(df, df['label'])):
        X_train, X_test = df[features].loc[train_idx], df[features].loc[test_idx]
        assert X_train.shape[1] == len(features)
        assert X_test.shape[1] == len(features)
        y_train, y_test = df.loc[train_idx, 'label'],  df.loc[test_idx, 'label']

        clf = fabric(fabric_arguments)
        clf.fit(X_train, y_train)
        df.loc[test_idx, 'predict'] = clf.predict(X_test)
    df['predict'] = df['predict'].astype(int)
    metrics = precision_recall_fscore_support(df['label'], df['predict'])
    metrics_df = pd.DataFrame.from_dict({'class': [0, 1],
                                         'precision': metrics[0],
                                         'recall': metrics[1],
                                         'f1': metrics[2],
                                         'support': metrics[3]})
    metrics_list.append(metrics_df)
    metrics = pd.concat(metrics_list).groupby('class').mean()
    return df['predict'].copy(), metrics


class FindQuestionsHeuristicNaive:
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return X['was_question_mark']


class FindQuestionsHeuristic:
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return (X['was_question_mark'].astype(bool) & (X['was_active_last_hour'] == 0)).astype(int)


class FindQuestionsMLP:
    def __init__(self, n_features):
        num_labels = 1
        hidden_units = [160, 160, 160]
        dropout_rates = [0.2, 0.2, 0.2, 0.2]
        label_smoothing = 1e-2
        learning_rate = 1e-3
        self.threshold = 0.5
        self.model = self.__create_mlp(n_features, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate)
        
    def __create_mlp(self, num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate):
        inp = tf.keras.layers.Input(shape=(num_columns,))
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dropout(dropout_rates[0])(x)
        for i in range(len(hidden_units)):
            x = tf.keras.layers.Dense(hidden_units[i])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
            x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)

        x = tf.keras.layers.Dense(num_labels)(x)
        out = tf.keras.layers.Activation("sigmoid")(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=tf.keras.metrics.AUC(name="AUC"),
        )
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    
    def predict(self, X):
        y_pred = self.model.predict(X).flatten()
        return (y_pred > self.threshold).astype(int)


def find_questions_heuristic_naive_fabric(args=None):
    return FindQuestionsHeuristicNaive()


def find_questions_heuristic_fabric(args=None):
    return FindQuestionsHeuristic()


def logistic_regression_fabric(args=None):
    return LogisticRegression()


def svm_fabric(args=None):
    return svm.SVC(kernel='linear', C=1)


def xgb_fabric(args=None):
    return xgb.XGBClassifier(n_estimators=100, 
#                              subsample=0.6, colsample_bytree=0.6,
#                              objective= 'binary:logistic',

#                              max_depth=10,
#                              learning_rate=0.03,
#                              min_child_weight=1,
                             
                             n_jobs=-1)


def mlp_fabric(args):
    return FindQuestionsMLP(args['feature_n'])


# Teoretical best classifier


def create_dataset_with_all_features():
    return pd.DataFrame(np.array(list(product([0, 1], repeat=len(features_names)))),
                        columns=features_names
                       )


def _create_labels(counter, n_bits):
    return [int(x) for x in '{:0{size}b}'.format(counter, size=n_bits)]


def _create_classifier(bad_examples, counter):
    labels = _create_labels(counter, len(bad_examples))
    return dict(zip(bad_examples, labels))


def _brutforce_clf_for_bad_examples(df, bad_examples, features):
    indices_to_predict = df[df['best_predict'].isnull()].index
    
    best_counter = 0
    best_f1 = 0
    print('Iterations', 2**len(bad_examples))
    for counter in range(2**len(bad_examples)):
        best_classifier = _create_classifier(bad_examples, counter)
#         print(best_classifier)
        
        df.loc[indices_to_predict, 'best_predict'] = df\
            .loc[indices_to_predict]\
            .apply(lambda row: best_classifier[tuple(row[features])], axis=1)
        
        metrics = precision_recall_fscore_support(df['label'], df['best_predict'])
        if metrics[2][1] > best_f1:
            best_counter = counter
            best_f1 = metrics[2][1]
            print(best_counter, best_f1)
        counter += 1
        
#         if counter % 10 == 0:
#             print(counter)
    print(best_counter)
    return _create_classifier(bad_examples, best_counter)


def find_best_classifier(df, features):
    counters = df\
        .groupby(features)\
        .agg(Counter)['label']\
        .to_dict()
    
    clf_from_good_examples = {k: int(list(v.keys())[0]) for k, v in counters.items() if len(v) == 1}
    df['best_predict'] = df.apply(
        lambda row: clf_from_good_examples.get(tuple(row[features]), None), axis=1)
    
    bad_examples = [k for k, v in counters.items() if len(v) == 2]
    clf_from_bad_examples = _brutforce_clf_for_bad_examples(df, bad_examples, features)
    
    return {**clf_from_good_examples, **clf_from_bad_examples}


def set_tensorflow():
    # num_cores = 20
    # num_CPU = 3
    # num_GPU = 1


    # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
    #                                   inter_op_parallelism_threads=num_cores,
    #                                   allow_soft_placement=True,
    #                                   device_count = {'CPU' : num_CPU,
    #                                                   'GPU' : num_GPU})
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True


    session = tf.compat.v1.Session(config=config)

    # from tensorflow.keras.backend.tensorflow_backend import set_session
    # set_session(session)
    tf.compat.v1.keras.backend.set_session(session)
