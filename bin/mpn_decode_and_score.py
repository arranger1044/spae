import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from collections import defaultdict

import numpy

import datetime

import os

import sys

import pickle
import gzip

import logging

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn import neighbors
from sklearn import decomposition
from sklearn import manifold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import average_precision_score


from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import SPLIT_NAMES

from spn.linked.representation import decode_embeddings_mpn
from spn.linked.representation import extract_features_nodes_mpn
from spn.linked.representation import var_log_val
from spn.linked.representation import filter_all_nodes
from spn.linked.representation import filter_non_leaf_nodes
from spn.linked.representation import load_feature_info

MAX_N_INSTANCES = 10000
COMPRESSED_MODEL_EXT = 'model.gz'
INFO_FILE_EXT = 'features.info'
PICKLE_SPLIT_EXT = 'pickle'
PREDS_PATH = 'preds'

SCORE_NAMES = {'accuracy': 'acc',
               'hamming': 'ham',
               'exact': 'exc',
               'jaccard': 'jac'}


def compute_scores(y_true, y_preds, score='accuracy'):

    if score == 'accuracy':
        return accuracy_score(y_true, y_preds)
    elif score == 'hamming':
        return 1 - hamming_loss(y_true, y_preds)
    elif score == 'exact':
        return 1 - zero_one_loss(y_true, y_preds)
    elif score == 'jaccard':
        return jaccard_similarity_score(y_true, y_preds)
    elif score == 'micro-f1':
        return f1_score(y_true, y_preds, average='micro')
    elif score == 'macro-f1':
        return f1_score(y_true, y_preds, average='macro')
    elif score == 'micro-auc-pr':
        return average_precision_score(y_true, y_preds, average='micro')
    elif score == 'macro-auc-pr':
        return average_precision_score(y_true, y_preds, average='macro')


def load_spn_model(model_path, fold=None, compress_ext='.gz'):
    #
    # loading a particular fold model (COMPRESSED ONLY)
    if fold is not None:
        model_path = '{}.{}.{}'.format(model_path, fold, COMPRESSED_MODEL_EXT)

    logging.info('Loading spn model from {}'.format(model_path))

    #
    # compressed?
    model_file = None
    if model_path.endswith(compress_ext):
        model_file = gzip.open(model_path, 'rb')
    else:
        model_file = open(model_path, 'rb')

    load_start_t = perf_counter()
    spn = pickle.load(model_file)
    load_end_t = perf_counter()
    logging.info('\tdone in {}'.format(load_end_t - load_start_t))

    model_file.close()
    return spn


def load_feature_info_fold(feature_info_path, fold=None):
    #
    # this is brutal
    feature_info_path = feature_info_path.replace('.pklz', '')
    feature_info_path = feature_info_path.replace('.pkl', '')

    if fold is not None:
        feature_info_path = '{}.{}.{}'.format(feature_info_path, fold, INFO_FILE_EXT)
    else:
        feature_info_path = '{}.{}'.format(feature_info_path, INFO_FILE_EXT)

    logging.info('Loading feature info from {}'.format(feature_info_path))

    feature_info = load_feature_info(feature_info_path)

    return feature_info


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
                    help='Loaded dataset type')

parser.add_argument('--repr-x', type=str,
                    default=None,
                    help='Specify a learned representation for the X (file path)')

parser.add_argument('--repr-x-exts', type=str, nargs='+',
                    default=None,
                    help='Learned representations split extensions')

parser.add_argument('--repr-x-dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded representation type')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

parser.add_argument('--scores', type=str, nargs='+',
                    default=['accuracy'],
                    help='Scores for the classifiers ("accuracy"|"hamming"|"exact")')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--reev', action='store_true',
                    help='Whether to reevaluate bottom-up the MPN')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')

parser.add_argument('--model', type=str,
                    help='Spn model file path or (path to spn models dir when --cv)')

parser.add_argument('--y-only', action='store_true',
                    help='Whether to load only the Y from the model pickle file')

parser.add_argument('--no-leaves', action='store_true',
                    help='Whether to filter out leaves from embeddings')

parser.add_argument('--emb-type', type=str, nargs='?',
                    default='activations',
                    help='Type of embedding to decode' +
                    '("activations"| "latent_categorical" | "latent_binary")')


#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)

#
# loading the dataset splits
#
logging.info('Loading datasets: %s', args.dataset)
dataset_name = args.dataset.split('/')[-1]
#
# replacing  suffixes names
dataset_name = dataset_name.replace('.pklz', '')
dataset_name = dataset_name.replace('.pkl', '')
dataset_name = dataset_name.replace('.pickle', '')

fold_splits = None
repr_fold_x_splits = None

train_ext = None
valid_ext = None
test_ext = None
repr_train_x_ext = None
repr_valid_x_ext = None
repr_test_x_ext = None

if args.data_exts is not None:
    if len(args.data_exts) == 1:
        train_ext, = args.data_exts
    elif len(args.data_exts) == 2:
        train_ext, test_ext = args.data_exts
    elif len(args.data_exts) == 3:
        train_ext, valid_ext, test_ext = args.data_exts
    else:
        raise ValueError('Up to 3 data extenstions can be specified')

if args.repr_x_exts is not None:
    if len(args.repr_exts) == 1:
        repr_train_x_ext, = args.repr_x_exts
    elif len(args.repr_exts) == 2:
        repr_train_x_ext, repr_test_x_ext = args.repr_x_exts
    elif len(args.repr_exts) == 3:
        repr_train_x_ext, repr_valid_x_ext, repr_test_x_ext = args.repr_x_exts
    else:
        raise ValueError('Up to 3 repr data extenstions can be specified')


n_folds = args.cv if args.cv is not None else 1

x_only = None
y_only = None
if args.y_only:
    x_only = False
    y_only = True
else:
    x_only = True
    y_only = False

#
# loading data and learned representations
if args.cv is not None:

    fold_splits = load_cv_splits(args.dataset,
                                 dataset_name,
                                 n_folds,
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
                                 x_only=x_only,
                                 y_only=y_only,
                                 test_ext=test_ext,
                                 dtype=args.dtype)
    repr_fold_x_splits = load_cv_splits(args.repr_x,
                                        dataset_name,
                                        n_folds,
                                        x_only=True,
                                        train_ext=repr_train_x_ext,
                                        valid_ext=repr_valid_x_ext,
                                        test_ext=repr_test_x_ext,
                                        dtype=args.repr_x_dtype)


else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             x_only=x_only,
                                             y_only=y_only,
                                             test_ext=test_ext,
                                             dtype=args.dtype)
    repr_fold_x_splits = load_train_val_test_splits(args.repr_x,
                                                    dataset_name,
                                                    x_only=True,
                                                    train_ext=repr_train_x_ext,
                                                    valid_ext=repr_valid_x_ext,
                                                    test_ext=repr_test_x_ext,
                                                    dtype=args.repr_x_dtype)


#
# printing
print_fold_splits_shapes(fold_splits)
print_fold_splits_shapes(repr_fold_x_splits)

n_scores = len(args.scores)
n_splits = 3
fold_score_tensor = numpy.zeros((n_folds, n_splits, n_scores))

for f, (splits, repr_splits) in enumerate(zip(fold_splits, repr_fold_x_splits)):

    logging.info('*************\nProcessing fold {}\n**************'.format(f))
    fold_id = f if n_folds > 1 else None
    fold_str = '.{}'.format(f) if n_folds > 1 else ''

    spn = load_spn_model(args.model, fold_id)
    for s, split in enumerate(splits):
        if split is not None:
            # feature_info_file = '{}.{}.{}.features.info'.format(dataset_name, f, s)
            repr_split = repr_splits[s]
            assert split.shape[0] == repr_split.shape[0], (split.shape[0], repr_split.shape[0])
            # #
            # # encode
            # filter_func = None
            # if not args.no_leaves:
            #     filter_func = filter_all_nodes
            # else:
            #     filter_func = filter_non_leaf_nodes
            # repr_split = extract_features_nodes_mpn(spn,
            #                                         split,
            #                                         filter_node_func=filter_func,
            #                                         retrieve_func=var_log_val,
            #                                         dtype=numpy.float,
            #                                         output_feature_info=feature_info_file)
            # logging.info('Old split shape {}'.format(split.shape))
            logging.info('New repr shape {}'.format(split.shape))
            #
            # decode
            feature_info = load_feature_info_fold(args.repr_x, fold=f)
            dec_split = decode_embeddings_mpn(spn,
                                              repr_split,
                                              feature_info,
                                              n_features=split.shape[1],
                                              embedding_type=args.emb_type)

            assert split.shape[1] == dec_split.shape[1]
            assert split.shape[0] == dec_split.shape[0]

            for k, score in enumerate(args.scores):
                split_score = compute_scores(split, dec_split, score)
                fold_score_tensor[f, s, k] = split_score


#
# averaging over folds
avg_fold_scores = fold_score_tensor.mean(axis=0)
std_fold_scores = fold_score_tensor.std(axis=0)

for s in range(n_splits):
    logging.info('split: {}'.format(s))
    logging.info('avg fold scores {}'.format(avg_fold_scores[s]))
    logging.info('std fold scores {}'.format(std_fold_scores[s]))

header_str = '{}'.format('\t'.join(SCORE_NAMES[s] for s in args.scores))
score_str = '\n'.join('{}\t{}'.format(SPLIT_NAMES[s],
                                      '\t'.join(str(avg_fold_scores[s, k]) for k in range(n_scores)))
                      for s in range(n_splits) if fold_splits[0][s] is not None)
logging.info('\n\t{}\n{}'.format(header_str, score_str))
