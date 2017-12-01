import numpy

import arff

import argparse

import logging

import pickle
import gzip

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


import os

DATASETS = ['Arts1500',
            'birds',
            'Business1500',
            'CAL500',
            'emotions',
            'flags',
            'Health1500',
            'human3106',
            'plant978',
            'rcv1subset1500', 'rcv1subset2500', 'rcv1subset3500',
            'rcv1subset4500', 'rcv1subset5500',
            'scene',
            'yeast']

DATASET_LABELS = {'Arts1500': 26,
                  'birds': 19,
                  'Business1500': 30,
                  'CAL500': 174,
                  'emotions': 6,
                  'flags': 7,
                  'Health1500': 32,
                  'human3106': 14,
                  'plant978': 12,
                  'rcv1subset1500': 101, 'rcv1subset2500': 101, 'rcv1subset3500': 101,
                  'rcv1subset4500': 101, 'rcv1subset5500': 101,
                  'scene': 6,
                  'yeast': 14}


def arff_file_to_numpy(filename, n_labels, big_endian=True, input_feature_type='int', encode_nominal=True):
    """
    Load ARFF files as numpy array

    Parameters
    ______
    filename : string
        Path to ARFF file
    n_labels: integer
        Number of labels in the ARFF file
    big_endian: boolean
        Whether the ARFF file contains labels at the beginning of the attributes list
        ("big" endianness, MEKA format) or at the end ("little" endianness, MULAN format)
    input_feature_type: numpy.type as string
        The desire type of the contents of the return 'X' array-likes, default 'i8',
        should be a numpy type, see http://docs.scipy.org/doc/numpy/user/basics.types.html
    encode_nominal: boolean
        Whether convert categorical data into numeric factors,
        required for some scikit classifiers that can't handle non-numeric input featuers.

    Returns
    -------
    x: numpy matrix with input_feature_type elements,
    y: numpy matrix of binary (int8) label vectors
    """

    matrix = None
    with open(filename, 'r') as f:
        arff_frame = arff.load(f, encode_nominal=encode_nominal, return_type=arff.DENSE)
        matrix = numpy.array(arff_frame['data']).astype(input_feature_type)

    x, y = None, None

    if big_endian:
        x, y = matrix[:, n_labels:], matrix[:, :n_labels].astype(numpy.int8)
    else:
        x, y = matrix[:, :-n_labels], matrix[:, -n_labels:].astype(numpy.int8)

    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == matrix.shape[0]
    assert x.shape[1] + y.shape[1] == matrix.shape[1]

    logging.info('\n\t\tx shape: {}\ty shape: {}'.format(x.shape, y.shape))

    return x, y


def arff_cv_splits_to_pickle(dataset,
                             arff_file_path,
                             output_path,
                             n_labels,
                             big_endian=True,
                             n_folds=5,
                             fold_ext='f',
                             compress=False):

    fold_splits = []
    logging.info('\n')
    for i in range(n_folds):

        fold_train_split_path = os.path.join(arff_file_path,
                                             '{}.{}{}.train.arff'.format(dataset,
                                                                         fold_ext,
                                                                         i))
        logging.info('Loading train dataset path {}'.format(fold_train_split_path))
        load_start_t = perf_counter()
        train_x, train_y = arff_file_to_numpy(fold_train_split_path,
                                              n_labels,
                                              big_endian=big_endian)
        load_end_t = perf_counter()
        logging.info('\tdone in {} secs\n'.format(load_end_t - load_start_t))

        fold_test_split_path = os.path.join(arff_file_path,
                                            '{}.{}{}.test.arff'.format(dataset,
                                                                       fold_ext,
                                                                       i))
        logging.info('Loading test dataset path {}'.format(fold_test_split_path))
        load_start_t = perf_counter()
        test_x, test_y = arff_file_to_numpy(fold_test_split_path,
                                            n_labels,
                                            big_endian=big_endian)
        load_end_t = perf_counter()
        logging.info('\tdone in {} secs\n'.format(load_end_t - load_start_t))

        fold_splits.append(((train_x, train_y), (test_x, test_y)))

    os.makedirs(output_path, exist_ok=True)

    dataset = dataset.lower()

    if compress:
        pickle_out_path = os.path.join(output_path, '{}.{}.folds.pklz'.format(dataset, n_folds))
        with gzip.open(pickle_out_path, 'wb') as f:
            pickle_start_t = perf_counter()
            pickle.dump(fold_splits, f)
            pickle_end_t = perf_counter()
            logging.info('Dumped pickle to {} in {} secs'.format(pickle_out_path,
                                                                 pickle_end_t - pickle_start_t))
    else:
        pickle_out_path = os.path.join(output_path, '{}.{}.folds.pkl'.format(dataset, n_folds))
        with open(pickle_out_path, 'wb') as f:
            pickle_start_t = perf_counter()
            pickle.dump(fold_splits, f)
            pickle_end_t = perf_counter()
            logging.info('Dumped pickle to {} in {} secs'.format(pickle_out_path,
                                                                 pickle_end_t - pickle_start_t))

    return fold_splits


def filter_split_x(dataset_name, fold_splits, output_path, compress=False, save_txt=False, save_pickle=False):

    dataset_name = dataset_name.lower()

    folds_x = []
    for i, ((train_x, train_y), (test_x, test_y)) in enumerate(fold_splits):
        train_x = train_x.astype('int')
        test_x = test_x.astype('int')

        folds_x.append((train_x, test_x))
        if save_txt:
            if compress:
                train_fold_split_path = os.path.join(output_path,
                                                     '{}.{}.train.data.gz'.format(dataset_name, i))
            else:
                train_fold_split_path = os.path.join(output_path,
                                                     '{}.{}.train.data'.format(dataset_name, i))
            numpy.savetxt(train_fold_split_path, train_x, fmt='%d', delimiter=',')
            logging.info('Dumped train x for fold {} to {}'.format(i, train_fold_split_path))

            if compress:
                test_fold_split_path = os.path.join(output_path,
                                                    '{}.{}.test.data.gz'.format(dataset_name, i))
            else:
                test_fold_split_path = os.path.join(output_path,
                                                    '{}.{}.test.data'.format(dataset_name, i))
            numpy.savetxt(test_fold_split_path, test_x, fmt='%d', delimiter=',')
            logging.info('Dumped test x for fold {} to {}'.format(i, test_fold_split_path))

    #
    # create pickle

    if save_pickle:
        if compress:
            pickle_out_path = os.path.join(output_path, '{}.x.pklz'.format(dataset_name))
            with gzip.open(pickle_out_path, 'wb') as fp:
                pickle.dump(folds_x, fp)
        else:
            pickle_out_path = os.path.join(output_path, '{}.x.pkl'.format(dataset_name))
            with open(pickle_out_path, 'wb') as fp:
                pickle.dump(folds_x, fp)

#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset name')

parser.add_argument('-d', '--data-dir', type=str,
                    help='Data file path')

parser.add_argument('-f', '--fold-ext', type=str,
                    default='f',
                    help='Fold extension')

parser.add_argument('-k', '--n-folds', type=int,
                    default=5,
                    help='Number of folds to expect')

parser.add_argument('-l', '--n-labels', type=int,
                    help='Number of labels')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./data/multilabel',
                    help='Output dir path')

parser.add_argument('--gzip', action='store_true',
                    help='Whether to compress the files with pickle')

parser.add_argument('--save-x-txt', action='store_true',
                    help='Whether to save txt data files for x')

parser.add_argument('--save-x-pkl', action='store_true',
                    help='Whether to save txt data files for x')
#
# parsing the args
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info("Starting with arguments:\n%s", args)

conv_start_t = perf_counter()
fold_splits = arff_cv_splits_to_pickle(args.dataset,
                                       args.data_dir,
                                       args.output,
                                       n_labels=args.n_labels,
                                       n_folds=args.n_folds,
                                       fold_ext=args.fold_ext,
                                       compress=args.gzip)
conv_end_t = perf_counter()
logging.info('\t\tdone in {} secs'.format(conv_end_t - conv_start_t))

conv_start_t = perf_counter()
filter_split_x(args.dataset, fold_splits, args.output,
               compress=args.gzip,
               save_txt=args.save_x_txt,
               save_pickle=args.save_x_pkl)
conv_end_t = perf_counter()
logging.info('\t\tdone in {} secs'.format(conv_end_t - conv_start_t))
