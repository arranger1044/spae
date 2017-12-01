import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy

import datetime

import os

import logging

from spn.utils import stats_format
from spn.linked.spn import evaluate_on_dataset

from spn.theanok.spn import evaluate_on_dataset_batch

import pickle

PREDS_EXT = 'lls'
TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)


# def evaluate_on_dataset(spn, data):

#     n_instances = data.shape[0]
#     pred_lls = numpy.zeros(n_instances)

#     for i, instance in enumerate(data):
#         (pred_ll, ) = spn.single_eval(instance)
#         pred_lls[i] = pred_ll

#     return pred_lls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Specify a dataset name from data/ (es. nltcs)')

    parser.add_argument('--train-ext', type=str,
                        default=None,
                        help='Training set extension')

    parser.add_argument('--valid-ext', type=str,
                        default=None,
                        help='Validation set extension')

    parser.add_argument('--test-ext', type=str,
                        default=None,
                        help='Test set extension')

    parser.add_argument('--model', type=str,
                        help='Spn model file path')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./exp/learnspn-b/',
                        help='Output dir path')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    train = None
    valid = None
    test = None
    if args.train_ext is not None:

        dataset_name = args.dataset.split('/')[-1]

        train_path = '{}.{}'.format(args.dataset, args.train_ext)
        logging.info('Loading training csv file {}'.format(train_path))
        train = numpy.loadtxt(train_path, dtype='int32', delimiter=',')

        if args.valid_ext is not None:
            valid_path = '{}.{}'.format(args.dataset, args.valid_ext)
            logging.info('Loading valid csv file {}'.format(valid_path))
            valid = numpy.loadtxt(valid_path, dtype='int32', delimiter=',')
            assert train.shape[1] == valid.shape[1]

        if args.test_ext is not None:
            test_path = '{}.{}'.format(args.dataset, args.test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype='int32', delimiter=',')
            assert train.shape[1] == test.shape[1]

    else:

        #
        # loading dataset splits
        logging.info('Loading datasets: %s', args.dataset)
        dataset_name = args.dataset
        train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
        n_instances = train.shape[0]
        n_test_instances = test.shape[0]
        logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
                                                                   valid.shape,
                                                                   test.shape))

    if args.exp_name:
        out_path = os.path.join(args.output, dataset_name + '_' + args.exp_name)
    else:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(args.output, dataset_name + '_' + date_string)
    out_log_path = os.path.join(out_path, 'exp.log')
    os.makedirs(out_path, exist_ok=True)

    logging.info('Opening log file {}...'.format(out_log_path))

    preamble = ("""train_ll\tvalid_ll\ttest_ll\n""")

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.write(preamble)
        out_log.flush()

        train_avg_ll = None
        valid_avg_ll = None
        test_avg_ll = None

        logging.info('\nLoading spn model from: {}'.format(args.model))
        spn = None
        with open(args.model, 'rb') as model_file:
            load_start_t = perf_counter()
            spn = pickle.load(model_file)
            load_end_t = perf_counter()
            logging.info('done in {}'.format(load_end_t - load_start_t))

        logging.info('\nEvaluating on training set')
        train_preds = evaluate_on_dataset(spn, train)
        assert train_preds.shape[0] == train.shape[0]
        train_avg_ll = numpy.mean(train_preds)
        logging.info('\t{}'.format(train_avg_ll))

        if valid is not None:
            logging.info('Evaluating on validation set')
            valid_preds = evaluate_on_dataset(spn, valid)
            assert valid_preds.shape[0] == valid.shape[0]
            valid_avg_ll = numpy.mean(valid_preds)
            logging.info('\t{}'.format(valid_avg_ll))

        if test is not None:
            logging.info('Evaluating on test set')
            test_preds = evaluate_on_dataset(spn, test)
            assert test_preds.shape[0] == test.shape[0]
            test_avg_ll = numpy.mean(test_preds)
            logging.info('\t{}'.format(test_avg_ll))

        #
        # writing to file
        stats = stats_format([train_avg_ll,
                              valid_avg_ll,
                              test_avg_ll],
                             '\t',
                             digits=5)
        out_log.write(stats + '\n')
        out_log.flush()

        #
        # also serializing the split predictions
        train_lls_path = os.path.join(out_path, TRAIN_PREDS_EXT)
        numpy.savetxt(train_lls_path, train_preds, delimiter='\n')

        if valid is not None:
            valid_lls_path = os.path.join(out_path, VALID_PREDS_EXT)
            numpy.savetxt(valid_lls_path, valid_preds, delimiter='\n')

        if test is not None:
            test_lls_path = os.path.join(out_path, TEST_PREDS_EXT)
            numpy.savetxt(test_lls_path, test_preds, delimiter='\n')

        logging.info('Saved predictions to disk')
