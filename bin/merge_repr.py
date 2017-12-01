from spn.linked.representation import load_features_from_file
from spn.linked.representation import save_features_to_file

import numpy
from numpy.testing import assert_array_equal

import os

import logging

import argparse

import pickle

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'ts.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

SPLITS_EXT = [TRAIN_DATA_EXT,
              VALID_DATA_EXT,
              TEST_DATA_EXT]

PICKLE_SPLIT_EXT = 'pickle'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'

FMT_DICT = {
    'int': '%d',
    'float': '%.18e',
    'float.8': '%.8e',
}


def merging_features(feature_base_paths,
                     dtype=float):

    train_splits = []
    valid_splits = []
    test_splits = []

    for feature_base_path in feature_base_paths:
        logging.info('Considering path {}'.format(feature_base_path))

        #
        # check for a single pickle file first
        pickle_path = '{}.{}'.format(feature_base_path,
                                     PICKLE_SPLIT_EXT)

        train = None
        valid = None
        test = None
        print('Looking for {}'.format(pickle_path))
        if os.path.exists(pickle_path):
            logging.info('Loading from pickle {}'.format(pickle_path))
            with open(pickle_path, 'rb') as split_file:
                train, valid, test = pickle.load(split_file)

        else:
            repr_splits = []
            for s in SPLITS_EXT:
                split_path = '{}.{}'.format(feature_base_path, s)
                repr_splits.append(numpy.loadtxt(split_path, dtype=dtype, delimiter=','))

            train, valid, test = repr_splits

        train_splits.append(train)
        valid_splits.append(valid)
        test_splits.append(test)

    #
    # composing them
    train_n_features = train_splits[0].shape[1]
    for s in train_splits[1:]:
        assert train_splits[0].shape[0] == s.shape[0]
        train_n_features += s.shape[1]

    valid_n_features = valid_splits[0].shape[1]
    for s in valid_splits[1:]:
        assert valid_splits[0].shape[0] == s.shape[0]
        valid_n_features += s.shape[1]

    test_n_features = test_splits[0].shape[1]
    for s in test_splits[1:]:
        assert test_splits[0].shape[0] == s.shape[0]
        test_n_features += s.shape[1]

    ext_train = numpy.concatenate(train_splits, axis=1)
    ext_valid = numpy.concatenate(valid_splits, axis=1)
    ext_test = numpy.concatenate(test_splits, axis=1)

    assert ext_train.shape[0] == train_splits[0].shape[0]
    assert ext_valid.shape[0] == valid_splits[0].shape[0]
    assert ext_test.shape[0] == test_splits[0].shape[0]

    logging.info('\tAll train shape: {}'.format(ext_train.shape))
    logging.info('\tAll valid shape: {}'.format(ext_valid.shape))
    logging.info('\tAll test shape: {}'.format(ext_test.shape))

    return ext_train, ext_valid, ext_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("paths", type=str, nargs='+',
                        help='Features file base paths')

    parser.add_argument('--dtype', type=str,
                        default=float,
                        help='Batch split size')

    parser.add_argument('-o', '--output', type=str,
                        default='repr/rect/',
                        help='Dataset output suffix')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='float',
                        help='Dataset output number formatter')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--save-text', action='store_true',
                        help='Saving the repr data to text as well')
    #
    # parsing the args
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting with arguments:\n%s", args)

    ext_train, ext_valid, ext_test = merging_features(args.paths, args.dtype)

    #
    # saving them into a single pickle file
    if args.save_text:
        train_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.train_ext))
        valid_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.valid_ext))
        test_out_path = os.path.join(args.output, '{}.{}'.format(args.suffix, args.test_ext))

        logging.info('\nSaving training set to: {}'.format(train_out_path))
        numpy.savetxt(train_out_path, ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving validation set to: {}'.format(valid_out_path))
        numpy.savetxt(valid_out_path, ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

        logging.info('Saving test set to: {}'.format(test_out_path))
        numpy.savetxt(test_out_path, ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    split_file_path = os.path.join(args.output, '{}.{}'.format(args.suffix,
                                                               PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    with open(split_file_path, 'wb') as split_file:
        pickle.dump((ext_train, ext_valid, ext_test), split_file, protocol=4)

    logging.info('All done.')
