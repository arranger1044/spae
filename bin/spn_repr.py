import sys
sys.setrecursionlimit(1000000000)

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import functools
import numpy

import datetime

import os

import sys

import logging

import pickle
import gzip

import dataset
from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import SPLIT_NAMES

from spn.utils import stats_format
from spn import MARG_IND

from spn.linked.representation import extract_features_nodes_mpe
from spn.linked.representation import extract_features_nodes_mpn
from spn.linked.representation import extract_features_nodes_der
from spn.linked.representation import node_in_path_feature
from spn.linked.representation import acc_node_in_path_feature
from spn.linked.representation import max_child_id_feature
from spn.linked.representation import max_child_id_feature_cat
from spn.linked.representation import max_hidden_var_feature, filter_hidden_var_nodes
from spn.linked.representation import filter_hidden_var_cat_nodes
from spn.linked.representation import hidden_var_val, hidden_var_log_val
from spn.linked.representation import max_hidden_var_val, max_hidden_var_log_val
from spn.linked.representation import extract_features_nodes
from spn.linked.representation import child_var_val, child_var_log_val
from spn.linked.representation import var_val, var_log_val

from spn.linked.representation import filter_non_leaf_nodes
from spn.linked.representation import filter_all_nodes
from spn.linked.representation import filter_sum_nodes
from spn.linked.representation import filter_product_nodes
from spn.linked.representation import filter_non_sum_nodes
from spn.linked.representation import filter_non_prod_nodes

from spn.linked.representation import extract_features_nodes_by_scope
from spn.linked.representation import aggr_scopes_by_sum
from spn.linked.representation import aggr_scopes_by_mean
from spn.linked.representation import aggr_scopes_by_logsumexp
from spn.linked.representation import aggr_scopes_by_uni_mixture

from spn.linked.representation import extract_feature_marginalization_from_masks
from spn.linked.representation import extract_feature_marginalization_from_masks_theanok
from spn.linked.representation import extract_feature_marginalization_from_masks_opt_unique
from spn.linked.representation import extract_feature_marginalization_from_masks_theanok_opt_unique
from spn.linked.representation import extract_features_marginalization_rand
from spn.linked.representation import extract_features_marginalization_rectangles

from spn.linked.representation import extract_features_all_marginals_spn
from spn.linked.representation import extract_features_all_marginals_ml
from spn.linked.representation import all_single_marginals_ml
from spn.linked.representation import all_single_marginals_spn

from spn.linked.representation import extract_features_node_activations

from spn.linked.representation import load_features_from_file

from spn.factory import build_theanok_spn_from_block_linked

from spn.theanok.spn import BlockLayeredSpn


PREDS_EXT = 'lls'

TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

THEANO_MODEL_EXT = 'theano_model'

COMPRESSED_MODEL_EXT = 'model.gz'

PICKLE_SPLIT_EXT = 'pickle'
COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'

RETRIEVE_FUNC_DICT = {
    'in-path': node_in_path_feature,
    'acc-path': acc_node_in_path_feature,
    'max-var': max_hidden_var_feature,
    'hid-cat': max_child_id_feature_cat,
    'hid-val': hidden_var_val,
    'hid-log-val': hidden_var_log_val,
    'ch-val': child_var_val,
    'ch-log-val': child_var_log_val,
    'var-val': var_val,
    'var-log-val': var_log_val
}

FILTER_FUNC_DICT = {
    'non-lea': filter_non_leaf_nodes,
    'non-sum': filter_non_sum_nodes,
    'non-prod': filter_non_prod_nodes,
    'hid-var': filter_hidden_var_nodes,
    'hid-cat': filter_hidden_var_cat_nodes,
    'all': filter_all_nodes,
    'sum': filter_sum_nodes,
    'prod': filter_product_nodes
}

SCOPE_AGGR_FUNC_DICT = {
    'sum': aggr_scopes_by_sum,
    'mean': aggr_scopes_by_mean,
    'logsumexp': aggr_scopes_by_logsumexp,
    'mix': aggr_scopes_by_uni_mixture
}

DTYPE_DICT = {
    'int': numpy.int32,
    'float': numpy.float32,
    'float.8': numpy.float32,
}

FMT_DICT = {
    'int': '%d',
    'int32': '%d',
    'float': '%.18e',
    'float32': '%.18e',
    'float.8': '%.8e',
}

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode


def filter_sum_nodes(spn):
    return [node for node in spn.top_down_nodes() if isinstance(node, SumNode)]


def filter_product_nodes(spn):
    return [node for node in spn.top_down_nodes() if isinstance(node, ProductNode)]


def filter_leaf_nodes(spn):
    return [node for node in spn.top_down_nodes()
            if not isinstance(node, ProductNode) and not isinstance(node, SumNode)]


def filter_nodes_by_layer(spn, layer_id):
    return [node for i, layer in enumerate(spn.bottom_up_layers())
            for node in layer.nodes() if layer_id == i]


def filter_nodes_by_scope_length(spn, min_scope_len, max_scope_len):
    return [node for node in spn.top_down_nodes()
            if ((hasattr(node, 'var_scope') and
                 len(node.var_scope) >= min_scope_len and
                 len(node.var_scope) < max_scope_len)
                or
                (hasattr(node, 'var') and
                 len(node.var) >= min_scope_len and
                 len(node.var) < max_scope_len))]


def filter_nodes(spn, filter_str):

    nodes = None

    if filter_str == 'all':
        nodes = list(spn.top_down_nodes())

    elif filter_str == 'sum':
        nodes = filter_sum_nodes(spn)

    elif filter_str == 'prod':
        nodes = filter_product_nodes(spn)

    elif filter_str == 'leaves':
        nodes = filter_leaf_nodes(spn)

    elif 'layer' in filter_str:
        layer_id = int(filter_str.replace('layer', ''))
        nodes = filter_nodes_by_layer(spn, layer_id)

    elif 'scope' in filter_str:
        scope_ids = int(filter_str.replace('scope', ''))
        min_scope, max_scope = scope_ids.split(',')
        min_scope, max_scope = int(min_scope), int(max_scope)
        nodes = filter_nodes_by_scope_length(spn, min_scope, max_scope)

    return nodes


def evaluate_on_dataset(spn, data):

    n_instances = data.shape[0]
    pred_lls = numpy.zeros(n_instances)

    for i, instance in enumerate(data):
        (pred_ll, ) = spn.single_eval(instance)
        pred_lls[i] = pred_ll

    return pred_lls


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset dir')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=None,
                        help='Dataset split extensions')

    parser.add_argument('--model', type=str,
                        help='Spn model file path or (path to spn models dir when --cv)')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/repr/',
                        help='Output dir path')

    parser.add_argument('--ret-func', type=str, nargs='?',
                        default='max-var',
                        help='Node value retrieve func in creating representations')

    parser.add_argument('--filter-func', type=str, nargs='?',
                        default='hid-var',
                        help='Node filter func in creating representations')

    parser.add_argument('--scope-aggr', type=str,
                        default=None,
                        help='Aggregate by scope (mean|sum)')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--node-activations', type=str, nargs='+',
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument('--sep', type=str, nargs='?',
                        default=',',
                        help='Dataset output separator')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='float',
                        help='Dataset output number formatter')

    parser.add_argument('--shuffle-ext', type=int, nargs='?',
                        default=None,
                        help='Whether to shuffle stacked features')

    parser.add_argument('--theano', type=int, nargs='?',
                        default=None,
                        help='Whether to use theano for marginal feature eval (batch size)')

    parser.add_argument('--max-nodes-layer', type=int,
                        default=None,
                        help='Max number of nodes per layer in a theano representation')

    # parser.add_argument('--rand-marg-rect', type=int, nargs='+',
    #                     default=None,
    #                     help='Generating features by marginalization over random rectangles')

    # parser.add_argument('--rand-marg', type=int, nargs='+',
    #                     default=None,
    #                     help='Generating features by marginalization over random subsets')

    parser.add_argument('--features', type=str, nargs='?',
                        default=None,
                        help='Loading feature masks from file')

    parser.add_argument('--no-ext', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--save-features', action='store_true',
                        help='Saving the generated features')

    parser.add_argument('--save-text', action='store_true',
                        help='Saving the repr data to text as well')

    parser.add_argument('--rand-features', type=float, nargs='+',
                        default=None,
                        help='Using only random features, generated as a binomial with param p')

    parser.add_argument('--no-mpe', action='store_true',
                        help='Whether not to use MPE inference in the upward pass')

    parser.add_argument('--mpn', action='store_true',
                        help='Whether to transform an SPN to MPN')

    parser.add_argument('--der', action='store_true',
                        help='Whether to use node derivatives as embeddings')

    parser.add_argument('--sing-marg', action='store_true',
                        help='Whether to evaluate all single marginals')

    parser.add_argument('--sing-marg-ml', action='store_true',
                        help='Whether to evaluate all single marginals with ML estimator')

    parser.add_argument('--alpha', type=float,
                        default=0.0,
                        help='Smoothing parameter')

    parser.add_argument('--opt-unique', action='store_true',
                        help='Whether to activate the unique patches opt while computing marg features')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    parser.add_argument('--cv', type=int,
                        default=None,
                        help='Folds for cross validation for model selection')

    parser.add_argument('--feature-scheme', type=str,
                        default=None,
                        help='Path to feature scheme file')

    parser.add_argument('--gzip', action='store_true',
                        help='Whether to compress the repr out file')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

    parser.add_argument('--y-only', action='store_true',
                        help='Whether to load only the Y from the model pickle file')

    parser.add_argument('--sparsify-mpe', type=float,
                        default=None,
                        help='Whether to have sparse embeddings where non-zero entries correspond to nodes in an MPE descend path')

    parser.add_argument('--remove-zero-feature', type=float,
                        default=None,
                        help='Whether to remove zero features (the value can be provided)')

    parser.add_argument('--repr-dtype', type=str, nargs='?',
                        default='float',
                        help='Loaded representation type')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # fixing a seed
    rand_gen = numpy.random.RandomState(args.seed)

    #
    # creating output dir if it does not exist yet
    os.makedirs(args.output, exist_ok=True)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    #
    # loading dataset folds
    logging.info('Loading datasets: %s', args.dataset)
    dataset_name = args.dataset.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name = dataset_name.replace('.pklz', '')
    dataset_name = dataset_name.replace('.pkl', '')
    dataset_name = dataset_name.replace('.pickle', '')

    # dtype = DTYPE_DICT[args.fmt]

    fold_splits = None
    repr_fold_splits = []

    train_ext = None
    valid_ext = None
    test_ext = None
    repr_train_ext = None
    repr_valid_ext = None
    repr_test_ext = None

    if args.data_exts is not None:
        if len(args.data_exts) == 1:
            train_ext, = args.data_exts
        elif len(args.data_exts) == 2:
            train_ext, test_ext = args.data_exts
        elif len(args.data_exts) == 3:
            train_ext, valid_ext, test_ext = args.data_exts
        else:
            raise ValueError('Up to 3 data extenstions can be specified')

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
                                     x_only=x_only,
                                     y_only=y_only,
                                     train_ext=train_ext,
                                     valid_ext=valid_ext,
                                     test_ext=test_ext,
                                     dtype=args.dtype)

    else:
        fold_splits = load_train_val_test_splits(args.dataset,
                                                 dataset_name,
                                                 x_only=x_only,
                                                 y_only=y_only,
                                                 train_ext=train_ext,
                                                 valid_ext=valid_ext,
                                                 test_ext=test_ext,
                                                 dtype=args.dtype)

    #
    # printing
    print_fold_splits_shapes(fold_splits)

    #
    # estimating the frequencies for the features
    logging.info('Estimating features on training set...')
    # freqs, features = dataset.data_2_freqs(train)
    n_features = fold_splits[0][0].shape[1]
    feature_vals = None
    if args.feature_scheme is None:
        feature_vals = numpy.array([2 for i in range(n_features)])
    else:
        raise ValueError('Loading feature schema not implemented yet')

    for f, (train, valid, test) in enumerate(fold_splits):

        #
        # printing fold stats
        logging.info('\n\n**************\nProcessing fold {}\n**************\n\n'.format(f))

        if train is not None and valid is not None and test is not None:
            logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
                                                                       valid.shape,
                                                                       test.shape))
        elif train is not None and valid is not None:
            logging.info('\ttrain: {}\n\tvalid: {}'.format(train.shape,
                                                           valid.shape))
        elif train is not None and test is not None:
            logging.info('\ttrain: {}\n\ttest: {}'.format(train.shape,
                                                          test.shape))
        elif train is not None:
            logging.info('\tonly train: {}'.format(train.shape))

        fold_id = f if n_folds > 1 else None
        fold_str = '.{}'.format(f) if n_folds > 1 else ''

        repr_train = None
        repr_valid = None
        repr_test = None

        if args.features:

            # logging.info('\nLoading spn model from: {}'.format(args.model))

            spn = load_spn_model(args.model, fold_id)

            # with open(args.model, 'rb') as model_file:
            #     load_start_t = perf_counter()
            #     spn = pickle.load(model_file)
            #     load_end_t = perf_counter()
            #     logging.info('done in {}'.format(load_end_t - load_start_t))

            #
            # loading features from file
            feature_file_path = args.features
            feature_masks = load_features_from_file(feature_file_path)
            logging.info('Loaded {} feature masks from {}'.format(len(feature_masks),
                                                                  feature_file_path))

            if args.theano is not None:

                train_out_path = os.path.join(args.output,
                                              '{}.{}{}'.format(args.suffix,
                                                               fold_str,
                                                               args.train_ext))
                valid_out_path = os.path.join(args.output,
                                              '{}.{}{}'.format(args.suffix,
                                                               fold_str,
                                                               args.valid_ext))
                test_out_path = os.path.join(args.output,
                                             '{}.{}{}'.format(args.suffix,
                                                              fold_str,
                                                              args.test_ext))

                if args.gzip:
                    train_out_path += '.gz'
                    valid_out_path += '.gz'
                    test_out_path += '.gz'

                #
                # if it is 0 then we set it to None to evaluate it in a single batch
                batch_size = args.theano if args.theano > 0 else None
                logging.info('Evaluation with theano')

                feat_s_t = perf_counter()
                ind_train = dataset.one_hot_encoding(train, feature_vals)
                feat_e_t = perf_counter()
                logging.info('Train one hot encoding done in {}'.format(feat_e_t - feat_s_t))

                if valid is not None:
                    feat_s_t = perf_counter()
                    ind_valid = dataset.one_hot_encoding(valid, feature_vals)
                    feat_e_t = perf_counter()
                    logging.info('Valid one hot encoding done in {}'.format(feat_e_t - feat_s_t))

                if test is not None:
                    feat_s_t = perf_counter()
                    ind_test = dataset.one_hot_encoding(test, feature_vals)
                    feat_e_t = perf_counter()
                    logging.info('Test one hot encoding done in {}'.format(feat_e_t - feat_s_t))

                theano_model_path = os.path.join(args.output,
                                                 '{}.{}.{}{}'.format(args.suffix,
                                                                     dataset_name,
                                                                     fold_str,
                                                                     THEANO_MODEL_EXT))
                theanok_spn = None
                logging.info('Looking for theano spn model in {}'.format(theano_model_path))
                #
                # TODO: allow for compressed pickle
                if os.path.exists(theano_model_path):
                    logging.info('Loading theanok pickle model')

                    with open(theano_model_path, 'rb') as mfile:
                        pic_s_t = perf_counter()
                        theanok_spn = BlockLayeredSpn.load(mfile)
                        pic_e_t = perf_counter()
                        logging.info('\tLoaded in {} secs'.format(pic_e_t - pic_s_t))
                else:
                    feat_s_t = perf_counter()
                    theanok_spn = build_theanok_spn_from_block_linked(spn,
                                                                      ind_train.shape[1],
                                                                      feature_vals,
                                                                      max_n_nodes_layer=args.max_nodes_layer)
                    feat_e_t = perf_counter()
                    logging.info('Spn transformed in theano in {}'.format(feat_e_t - feat_s_t))
                    with open(theano_model_path, 'wb') as mfile:
                        pic_s_t = perf_counter()
                        print('rec lim', sys.getrecursionlimit())
                        theanok_spn.dump(mfile)
                        pic_e_t = perf_counter()

                    logging.info('Serialized into {}\n\tdone in {}'.format(theano_model_path,
                                                                           pic_e_t - pic_s_t))

                extract_feature_func = None
                if args.opt_unique:
                    logging.info('Using unique opt')
                    extract_feature_func = extract_feature_marginalization_from_masks_theanok_opt_unique
                else:
                    extract_feature_func = extract_feature_marginalization_from_masks_theanok

                logging.info('\nConverting training set')
                feat_s_t = perf_counter()
                repr_train = extract_feature_func(theanok_spn,
                                                  ind_train,
                                                  feature_masks,
                                                  feature_vals=feature_vals,
                                                  batch_size=batch_size,
                                                  marg_value=MARG_IND,
                                                  # rand_gen=rand_gen,
                                                  dtype=float)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

                #
                # saving it to disk asap
                logging.info('\nSaving training set to: {}'.format(train_out_path))
                numpy.savetxt(train_out_path,
                              repr_train,
                              delimiter=args.sep,
                              fmt=FMT_DICT[args.fmt])

                if valid is not None:
                    logging.info('\nConverting validation set')
                    feat_s_t = perf_counter()
                    repr_valid = extract_feature_func(theanok_spn,
                                                      ind_valid,
                                                      feature_masks,
                                                      feature_vals=feature_vals,
                                                      batch_size=batch_size,
                                                      marg_value=MARG_IND,
                                                      # rand_gen=rand_gen,
                                                      dtype=float)
                    feat_e_t = perf_counter()
                    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

                    logging.info('Saving validation set to: {}'.format(valid_out_path))
                    numpy.savetxt(valid_out_path,
                                  repr_valid,
                                  delimiter=args.sep,
                                  fmt=FMT_DICT[args.fmt])

                if test is not None:
                    logging.info('\nConverting test set')
                    feat_s_t = perf_counter()
                    repr_test = extract_feature_func(theanok_spn,
                                                     ind_test,
                                                     feature_masks,
                                                     feature_vals=feature_vals,
                                                     batch_size=batch_size,
                                                     marg_value=MARG_IND,
                                                     # rand_gen=rand_gen,
                                                     dtype=float)
                    feat_e_t = perf_counter()
                    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

                    logging.info('Saving test set to: {}'.format(test_out_path))
                    numpy.savetxt(test_out_path,
                                  repr_test,
                                  delimiter=args.sep,
                                  fmt=FMT_DICT[args.fmt])
            else:
                #
                # not using theano, but linked representation

                extract_feature_func = None
                if args.opt_unique:
                    logging.info('Using unique opt')
                    extract_feature_func = extract_feature_marginalization_from_masks_opt_unique
                else:
                    extract_feature_func = extract_feature_marginalization_from_masks

                logging.info('\nConverting training set')
                feat_s_t = perf_counter()
                repr_train = extract_feature_func(spn,
                                                  train,
                                                  feature_masks,
                                                  marg_value=MARG_IND,
                                                  # rand_gen=rand_gen,
                                                  dtype=float)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

                if valid is not None:
                    logging.info('Converting validation set')
                    feat_s_t = perf_counter()
                    repr_valid = extract_feature_func(spn,
                                                      valid,
                                                      feature_masks,
                                                      marg_value=MARG_IND,
                                                      # rand_gen=rand_gen,
                                                      dtype=float)
                    feat_e_t = perf_counter()
                    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

                if test is not None:
                    logging.info('Converting test set')
                    feat_s_t = perf_counter()
                    repr_test = extract_feature_func(spn,
                                                     test,
                                                     feature_masks,
                                                     marg_value=MARG_IND,
                                                     # rand_gen=rand_gen,
                                                     dtype=float)
                    feat_e_t = perf_counter()
                    logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        elif args.rand_features is not None:

            rand_n_features, rand_perc = args.rand_features
            rand_n_features = int(rand_n_features)
            logging.info('\nGenerating {0} random features (with perc {1})'.format(rand_n_features,
                                                                                   rand_perc))
            #
            # adding random features
            repr_train = dataset.random_binary_dataset(train.shape[0],
                                                       rand_n_features,
                                                       perc=rand_perc,
                                                       rand_gen=rand_gen)
            if valid is not None:
                repr_valid = dataset.random_binary_dataset(valid.shape[0],
                                                           rand_n_features,
                                                           perc=rand_perc,
                                                           rand_gen=rand_gen)
            if test is not None:
                repr_test = dataset.random_binary_dataset(test.shape[0],
                                                          rand_n_features,
                                                          perc=rand_perc,
                                                          rand_gen=rand_gen)

        elif args.scope_aggr is not None:
            logging.info('Aggregating by scope')
            # logging.info('\nLoading spn model from: {}'.format(args.model))
            spn = load_spn_model(args.model, fold_id)
            # spn = None
            # with open(args.model, 'rb') as model_file:
            #     load_start_t = perf_counter()
            #     spn = pickle.load(model_file)
            #     load_end_t = perf_counter()
            #     logging.info('done in {}'.format(load_end_t - load_start_t))

            ret_func = RETRIEVE_FUNC_DICT[args.ret_func]
            filter_func = FILTER_FUNC_DICT[args.filter_func]

            aggr_func = SCOPE_AGGR_FUNC_DICT[args.scope_aggr]
            logging.info('Using {} for aggregating by scope'.format(aggr_func))

            feature_info_path = os.path.join(args.output, '{}.{}{}.{}'.format(args.suffix,
                                                                              dataset_name,
                                                                              fold_str,
                                                                              INFO_FILE_EXT))

            # feature_info_path = os.path.join(args.output, '{}.{}.{}'.format(args.suffix,
            #                                                                 dataset_name,
            #                                                                 INFO_FILE_EXT))
            # logging.info('Using function {}'.format(extract_repr_func))

            logging.info('\nConverting training set')
            feat_s_t = perf_counter()
            repr_train = extract_features_nodes_by_scope(spn,
                                                         train,
                                                         filter_node_func=filter_func,
                                                         remove_zero_features=False,
                                                         output_feature_info=feature_info_path,
                                                         aggr_func=aggr_func,
                                                         dtype=args.repr_dtype,
                                                         verbose=False)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if valid is not None:
                logging.info('Converting validation set')
                feat_s_t = perf_counter()
                repr_valid = extract_features_nodes_by_scope(spn,
                                                             valid,
                                                             filter_node_func=filter_func,
                                                             remove_zero_features=False,
                                                             output_feature_info=None,
                                                             aggr_func=aggr_func,
                                                             dtype=args.repr_dtype,
                                                             verbose=False)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if test is not None:
                logging.info('Converting test set')
                feat_s_t = perf_counter()
                repr_test = extract_features_nodes_by_scope(spn,
                                                            test,
                                                            filter_node_func=filter_func,
                                                            remove_zero_features=False,
                                                            output_feature_info=None,
                                                            aggr_func=aggr_func,
                                                            dtype=args.repr_dtype,
                                                            verbose=False)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        elif args.sing_marg:

            # logging.info('\nLoading spn model from: {}'.format(args.model))
            spn = load_spn_model(args.model, fold_id)

            # with open(args.model, 'rb') as model_file:
            #     load_start_t = perf_counter()
            #     spn = pickle.load(model_file)
            #     load_end_t = perf_counter()
            #     logging.info('done in {}'.format(load_end_t - load_start_t))

            logging.info('Extracting single marginals')

            all_marginals = all_single_marginals_spn(spn,
                                                     feature_vals,
                                                     dtype=numpy.int32)
            logging.info('Converting train set')
            feat_s_t = perf_counter()
            repr_train = extract_features_all_marginals_spn(spn,
                                                            train,
                                                            feature_vals,
                                                            all_marginals,
                                                            dtype=numpy.int32)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if valid is not None:
                logging.info('Converting valid set')
                feat_s_t = perf_counter()
                repr_valid = extract_features_all_marginals_spn(spn,
                                                                valid,
                                                                feature_vals,
                                                                all_marginals,
                                                                dtype=numpy.int32)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if test is not None:
                logging.info('Converting test set')
                feat_s_t = perf_counter()
                repr_test = extract_features_all_marginals_spn(spn,
                                                               test,
                                                               feature_vals,
                                                               all_marginals,
                                                               dtype=numpy.int32)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        elif args.sing_marg_ml:
            logging.info('Extracting single marginals with an ML estimator')

            alpha = args.alpha
            all_marginals = all_single_marginals_ml(train,
                                                    feature_vals,
                                                    alpha=alpha)

            logging.info('Converting train set')
            feat_s_t = perf_counter()
            repr_train = extract_features_all_marginals_ml(None,
                                                           train,
                                                           feature_vals,
                                                           alpha=alpha,
                                                           all_marginals=all_marginals,
                                                           dtype=numpy.int32)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))
            logging.info('Converting valid set')
            feat_s_t = perf_counter()
            repr_valid = extract_features_all_marginals_ml(None,
                                                           valid,
                                                           feature_vals,
                                                           alpha=alpha,
                                                           all_marginals=all_marginals,
                                                           dtype=numpy.int32)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))
            logging.info('Converting test set')
            feat_s_t = perf_counter()
            repr_test = extract_features_all_marginals_ml(None,
                                                          test,
                                                          feature_vals,
                                                          alpha=alpha,
                                                          all_marginals=all_marginals,
                                                          dtype=numpy.int32)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        elif args.node_activations:

            # logging.info('\nLoading spn model from: {}'.format(args.model))
            spn = load_spn_model(args.model, fold_id)

            # with open(args.model, 'rb') as model_file:
            #     load_start_t = perf_counter()
            #     spn = pickle.load(model_file)
            #     load_end_t = perf_counter()
            #     logging.info('done in {}'.format(load_end_t - load_start_t))

            logging.info('Extracting node activations features')

            node_filter_str = args.node_activations[0]
            mean = False
            if len(args.node_activations) > 1:
                mean = bool(int(args.node_activations[1]))
            logging.info('Using mean: {}'.format(mean))

            nodes = filter_nodes(spn, node_filter_str)
            logging.info('Considering nodes: {} ({})'.format(node_filter_str, len(nodes)))
            logging.info('Converting train set')
            feat_s_t = perf_counter()
            repr_train = extract_features_node_activations(spn,
                                                           nodes,
                                                           train,
                                                           marg_mask=None,
                                                           mean=mean,
                                                           log=False,
                                                           hard=False,
                                                           dtype=float)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if valid is not None:
                logging.info('Converting valid set')
                feat_s_t = perf_counter()
                repr_valid = extract_features_node_activations(spn,
                                                               nodes,
                                                               valid,
                                                               marg_mask=None,
                                                               mean=mean,
                                                               log=False,
                                                               hard=False,
                                                               dtype=float)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if test is not None:
                logging.info('Converting test set')
                feat_s_t = perf_counter()
                repr_test = extract_features_node_activations(spn,
                                                              nodes,
                                                              test,
                                                              marg_mask=None,
                                                              mean=mean,
                                                              log=False,
                                                              hard=False,
                                                              dtype=float)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        else:
            logging.info('Eval repr')
            # logging.info('\nLoading spn model from: {}'.format(args.model))
            spn = load_spn_model(args.model, fold_id)
            # with open(args.model, 'rb') as model_file:
            #     load_start_t = perf_counter()
            #     spn = pickle.load(model_file)
            #     load_end_t = perf_counter()
            #     logging.info('done in {}'.format(load_end_t - load_start_t))

            ret_func = RETRIEVE_FUNC_DICT[args.ret_func]
            filter_func = FILTER_FUNC_DICT[args.filter_func]

            extract_repr_func = None

            if args.mpn:
                extract_repr_func = functools.partial(extract_features_nodes_mpn,
                                                      sparsify_mpe=args.sparsify_mpe)
            elif args.no_mpe:
                extract_repr_func = extract_features_nodes
            elif args.der:
                extract_repr_func = extract_features_nodes_der
            else:
                extract_repr_func = functools.partial(extract_features_nodes_mpe,
                                                      sparsify_mpe=args.sparsify_mpe)

            feature_info_path = os.path.join(args.output, '{}.{}{}.{}'.format(args.suffix,
                                                                              dataset_name,
                                                                              fold_str,
                                                                              INFO_FILE_EXT))
            logging.info('Using function {}'.format(extract_repr_func))

            logging.info('\nConverting training set')
            feat_s_t = perf_counter()
            repr_train, train_features = extract_repr_func(spn,
                                                           train,
                                                           nodes_id_assoc=None,
                                                           filter_node_func=filter_func,
                                                           retrieve_func=ret_func,
                                                           remove_zero_features=args.remove_zero_feature,
                                                           output_feature_info=feature_info_path,
                                                           dtype=args.repr_dtype,
                                                           verbose=False)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if valid is not None:
                logging.info('Converting validation set')
                feat_s_t = perf_counter()
                repr_valid, _valid_features = extract_repr_func(spn,
                                                                valid,
                                                                nodes_id_assoc=train_features,
                                                                filter_node_func=filter_func,
                                                                retrieve_func=ret_func,
                                                                remove_zero_features=None,
                                                                output_feature_info=None,
                                                                dtype=args.repr_dtype,
                                                                verbose=False)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

            if test is not None:
                logging.info('Converting test set')
                feat_s_t = perf_counter()
                repr_test, test_features = extract_repr_func(spn,
                                                             test,
                                                             nodes_id_assoc=train_features,
                                                             filter_node_func=filter_func,
                                                             retrieve_func=ret_func,
                                                             remove_zero_features=None,
                                                             output_feature_info=None,
                                                             dtype=args.repr_dtype,
                                                             verbose=False)
                feat_e_t = perf_counter()
                logging.info('\t done in {}'.format(feat_e_t - feat_s_t))

        assert train.shape[0] == repr_train.shape[0]
        logging.info('New train shape {}'.format(repr_train.shape))
        if valid is not None:
            assert valid.shape[0] == repr_valid.shape[0]
            assert repr_train.shape[1] == repr_valid.shape[1]
            logging.info('New valid shape {}'.format(repr_valid.shape))

        if test is not None:
            assert test.shape[0] == repr_test.shape[0]
            logging.info('New test shape {}'.format(repr_test.shape))
            assert repr_train.shape[1] == repr_test.shape[1]
        # logging.info('New shapes {0} {1} {2}'.format(repr_train.shape,
        #                                              repr_valid.shape,
        #                                              repr_test.shape))

        #
        # shuffling?
        if args.shuffle_ext is not None:
            logging.info('\n\nShuffling data features')

            #
            # shuffling k times
            for k in range(args.shuffle_ext):
                repr_train = dataset.shuffle_columns(repr_train, rand_gen)
                assert train.shape[0] == repr_train.shape[0]
                logging.info('Train shape after shuffling {}'.format(repr_train.shape))

                if valid is not None:
                    repr_valid = dataset.shuffle_columns(repr_valid, rand_gen)
                    assert valid.shape[0] == repr_valid.shape[0]
                    logging.info('Valid shape after shuffling {}'.format(repr_valid.shape))
                if test is not None:
                    repr_test = dataset.shuffle_columns(repr_test, rand_gen)
                    assert test.shape[0] == repr_test.shape[0]
                    logging.info('Test shape after shuffling {}'.format(repr_test.shape))
            # repr_train = dataset.shuffle_columns(repr_train, numpy_rand_gen)
            # repr_valid = dataset.shuffle_columns(repr_valid, numpy_rand_gen)
            # repr_test = dataset.shuffle_columns(repr_test, numpy_rand_gen)

            # logging.info('Shape checking {0} {1} {2}\n'.format(repr_train.shape,
            #                                                    repr_valid.shape,
            #                                                    repr_test.shape))

        #
        # extending the original dataset
        ext_train = None
        ext_valid = None
        ext_test = None

        if args.no_ext:
            ext_train = repr_train
            if valid is not None:
                ext_valid = repr_valid
            if test is not None:
                ext_test = repr_test

        else:
            logging.info('\nConcatenating datasets')
            ext_train = numpy.concatenate((train, repr_train), axis=1)
            assert train.shape[0] == ext_train.shape[0]
            assert ext_train.shape[1] == train.shape[1] + repr_train.shape[1]
            logging.info('Ext train shape {}'.format(ext_train.shape))

            if valid is not None:
                ext_valid = numpy.concatenate((valid, repr_valid), axis=1)
                assert valid.shape[0] == ext_valid.shape[0]
                assert ext_valid.shape[1] == valid.shape[1] + repr_valid.shape[1]
                logging.info('Ext valid shape {}'.format(ext_valid.shape))

            if test is not None:
                ext_test = numpy.concatenate((test, repr_test), axis=1)
                assert test.shape[0] == ext_test.shape[0]
                assert ext_test.shape[1] == test.shape[1] + repr_test.shape[1]
                logging.info('Ext test shape {}'.format(ext_test.shape))

        #
        #
        # collecting new representations
        repr_fold_splits.append((ext_train, ext_valid, ext_test))

    #
    #
    # SERIALIZING REPRESENTATIONS
    for f, (ext_train, ext_valid, ext_test) in enumerate(repr_fold_splits):

        fold_str = '.{}'.format(f) if n_folds > 1 else ''

        # logging.info('New shapes {0} {1} {2}'.format(ext_train.shape,
        #                                              ext_valid.shape,
        #                                              ext_test.shape))

        #
        # storing them
        if args.save_text:
            train_out_path = os.path.join(args.output, '{}{}.{}'.format(args.suffix,
                                                                        fold_str,
                                                                        args.train_ext))
            if args.gzip:
                train_out_path += '.gz'
            logging.info('\nSaving training set to: {}'.format(train_out_path))
            numpy.savetxt(train_out_path,
                          ext_train,
                          delimiter=args.sep,
                          fmt=FMT_DICT[args.fmt])

            if valid is not None:
                valid_out_path = os.path.join(args.output, '{}{}.{}'.format(args.suffix,
                                                                            fold_str,
                                                                            args.valid_ext))
                if args.gzip:
                    valid_out_path += '.gz'
                logging.info('Saving validation set to: {}'.format(valid_out_path))
                numpy.savetxt(valid_out_path,
                              ext_valid,
                              delimiter=args.sep,
                              fmt=FMT_DICT[args.fmt])

            if test is not None:
                test_out_path = os.path.join(args.output, '{}{}.{}'.format(args.suffix,
                                                                           fold_str,
                                                                           args.test_ext))
                if args.gzip:
                    test_out_path += '.gz'
                logging.info('Saving test set to: {}'.format(test_out_path))
                numpy.savetxt(test_out_path,
                              ext_test,
                              delimiter=args.sep,
                              fmt=FMT_DICT[args.fmt])

    #
    # saving to pickle
    split_file_path = None
    split_file = None
    if args.gzip:
        split_file_path = os.path.join(args.output, '{}.{}.{}'.format(args.suffix,
                                                                      dataset_name,
                                                                      COMPRESSED_PICKLE_SPLIT_EXT))
        logging.info('Saving pickle data splits to: {}'.format(split_file_path))
        split_file = gzip.open(split_file_path, 'wb')
    else:
        split_file_path = os.path.join(args.output, '{}.{}.{}'.format(args.suffix,
                                                                      dataset_name,
                                                                      PICKLE_SPLIT_EXT))
        logging.info('Saving pickle data splits to: {}'.format(split_file_path))
        split_file = open(split_file_path, 'wb')

    if n_folds > 1:
        pickle.dump(repr_fold_splits, split_file, protocol=4)
    else:
        ext_train, ext_valid, ext_test = repr_fold_splits[0]
        pickle.dump((ext_train, ext_valid, ext_test), split_file, protocol=4)

    split_file.close()
