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
                    help='Specify a dataset file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('-k', '--n-classes', type=int, nargs='?',
                    help='Number of classes')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./multilabel/data/',
                    help='Output dir path')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
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
logging.info('Loading datasets: %s', args.dataset)
dataset_name = args.dataset.split('/')[-1]
#
# replacing  suffixes names
dataset_name = dataset_name.replace('.pklz', '')
dataset_name = dataset_name.replace('.pkl', '')
dataset_name = dataset_name.replace('.pickle', '')
dataset_name = dataset_name.replace('.{}'.format(FOLD_SUFFIX), '')

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

n_folds = args.cv if args.cv is not None else 1

if n_folds > 1:
    fold_splits = load_cv_splits(args.dataset,
                                 dataset_name,
                                 n_folds,
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
                                 test_ext=test_ext,
                                 dtype=args.dtype)
else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             x_only=False,
                                             y_only=False,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             dtype=args.dtype)

print_fold_splits_shapes(fold_splits)

merged_fold_splits = []

for i, splits in enumerate(fold_splits):
    logging.info('Processing fold {}\n'.format(i))

    merged_splits = []
    for j, split in enumerate(splits):
        if split is not None:
            split_x, split_y = split
            logging.info('\tProcessing split {} ({}, {})'.format(SPLIT_NAMES[j],
                                                                 split_x.shape,
                                                                 split_y.shape))
            if args.n_classes:

                logging.info('classes : {} -> {}'.format(split_y.min(), split_y.max()))
                if split_y.ndim == 1:
                    split_y = split_y[:, numpy.newaxis]

                logging.info('\t\tOne hot encoding target var ({} classes) ({})'.format(args.n_classes,
                                                                                        split_y.shape))

                split_y = one_hot_encoding(split_y,
                                           feature_values=[args.n_classes],
                                           dtype=split_y.dtype)
            #
            # merging
            merged_split = numpy.hstack((split_x, split_y))
            logging.info('\n\tMerged split --> new shape {}\n'.format(merged_split.shape))

            merged_splits.append(merged_split)
        else:
            merged_splits.append(split)
    merged_fold_splits.append(tuple(merged_splits))


print_fold_splits_shapes(merged_fold_splits)

#
# saving to pickle
split_file_path = None
split_file = None
fold_suffix = '.{}'.format(FOLD_SUFFIX) if n_folds > 1 else ''
if args.gzip:
    split_file_path = os.path.join(args.output, '{}.{}.{}{}'.format(dataset_name,
                                                                    MERGED_SUFFIX,
                                                                    COMPRESSED_PICKLE_SPLIT_EXT,
                                                                    fold_suffix))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    split_file = gzip.open(split_file_path, 'wb')
else:
    split_file_path = os.path.join(args.output, '{}.{}.{}{}'.format(dataset_name,
                                                                    MERGED_SUFFIX,
                                                                    PICKLE_SPLIT_EXT,
                                                                    fold_suffix))
    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    split_file = open(split_file_path, 'wb')

if n_folds > 1:
    pickle.dump(merged_fold_splits, split_file, protocol=4)
else:
    train, valid, test = merged_fold_splits[0]
    pickle.dump((train, valid, test), split_file, protocol=4)
