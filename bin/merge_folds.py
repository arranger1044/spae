import argparse
import logging
import sys
import os
import gzip
import pickle

import numpy

from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import one_hot_encoding

FOLD_SUFFIX = '5.folds'
MERGED_SUFFIX = 'x+y'.format(FOLD_SUFFIX)
PICKLE_SPLIT_EXT = 'pickle'
COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
SPLIT_NAMES = ['train', 'valid', 'test']

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset template file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='.',
                    help='Output dir path')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded dataset type')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--gzip', action='store_true',
                    help='Whether to compress the repr out file')

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

rand_gen = numpy.random.RandomState(args.seed)

os.makedirs(args.output, exist_ok=True)
#
# loading the dataset splits
#
logging.info('Loading dataset splits from template: %s', args.dataset)
dataset_name = args.dataset.split('/')[-1]
#
# replacing  suffixes names
dataset_name = dataset_name.replace('.pklz', '')
dataset_name = dataset_name.replace('.pkl', '')
dataset_name = dataset_name.replace('.pickle', '')
dataset_name = dataset_name.replace('.{}'.format(FOLD_SUFFIX), '')

n_folds = args.cv if args.cv is not None else 1

merged_fold_splits = []
for i in range(n_folds):
    #
    # assuming just one placeholder in the template
    fold_path = args.dataset.format(i)

    #
    # loading the splits as a single fold
    train_ext = None
    valid_ext = None
    test_ext = None

    if args.data_exts is not None:
        if len(args.data_exts) == 1:
            train_ext, = args.data_exts
        elif len(args.data_exts) == 2:
            train_ext, test_ext = args.data_exts
        elif len(args.data_exts) == 3:
            train_ext, valid_ext, test_ext = args.data_exts
        else:
            raise ValueError('Up to 3 data extenstions can be specified')

    fold_splits = load_train_val_test_splits(fold_path,
                                             dataset_name,
                                             x_only=False,
                                             y_only=False,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             dtype=args.dtype)

    print_fold_splits_shapes(fold_splits)

    #
    # storing all the folds together
    for f in fold_splits:
        merged_fold_splits.append(f)

#
# now serializing them again
n_merged_folds = len(merged_fold_splits)

logging.info('\n\n')
logging.info('<< Merged folds >>')
print_fold_splits_shapes(merged_fold_splits)

#
# saving to pickle
merged_split_file_path = None
merged_split_file = None
fold_suffix = '.{}'.format(FOLD_SUFFIX) if n_merged_folds > 1 else ''
if args.gzip:
    merged_split_file_path = os.path.join(args.output, '{}.{}'.format(dataset_name.format(n_merged_folds),
                                                                      COMPRESSED_PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(merged_split_file_path))
    merged_split_file = gzip.open(merged_split_file_path, 'wb')
else:
    merged_split_file_path = os.path.join(args.output, '{}.{}'.format(dataset_name.format(n_merged_folds),
                                                                      PICKLE_SPLIT_EXT))
    logging.info('Saving pickle data splits to: {}'.format(merged_split_file_path))
    merged_split_file = open(merged_split_file_path, 'wb')

if n_merged_folds > 1:
    pickle.dump(merged_fold_splits, merged_split_file, protocol=4)
else:
    train, valid, test = merged_fold_splits[0]
    pickle.dump((train, valid, test), merged_split_file, protocol=4)
