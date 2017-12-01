try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import argparse
import dataset
import sys
import datetime
import os
import logging
import pickle

import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_similarity_score


from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes

from spn.utils import stats_format
from spn.linked.spn import evaluate_on_dataset


FOLD_SUFFIX = '5.folds'
PREDS_EXT = 'preds'
TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)
SPLIT_NAMES = ['train', 'valid', 'test']
SCORE_NAMES = {'accuracy': 'acc',
               'hamming': 'ham',
               'exact': 'exc',
               'jaccard': 'jac'}


def compute_scores(y_true, y_preds, score='accuracy'):

    if score == 'accuracy':
        # print(accuracy_score(numpy.where(y_true == 1)[1],
        #                      numpy.where(y_preds == 1)[1]))
        return accuracy_score(y_true, y_preds)
    elif score == 'hamming':
        return 1 - hamming_loss(y_true, y_preds)
    elif score == 'exact':
        return 1 - zero_one_loss(y_true, y_preds)
    elif score == 'jaccard':
        return jaccard_similarity_score(y_true, y_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Dataset file path')

    parser.add_argument('--train-ext', type=str,
                        default=None,
                        help='Training set extension')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=None,
                        help='Dataset split extensions')

    parser.add_argument('--model', type=str,
                        help='Spn model file path or (path to spn models dir when --cv)')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./exp/predict-spn/',
                        help='Output dir path')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

    parser.add_argument('-k', '--n-targets', type=int, nargs='?',
                        default=None,
                        help='Number of target vars, assumed to be the last features')

    parser.add_argument('--target-ids', type=int, nargs='+',
                        default=None,
                        help='Sequence of target var ids (e.g. 1 10 40)')

    parser.add_argument('--scores', type=str, nargs='+',
                        default=['accuracy'],
                        help='Scores for the classifiers ("accuracy"|"hamming"|"exact")')

    parser.add_argument('--cv', type=int,
                        default=None,
                        help='Folds for cross validation for model selection')

    parser.add_argument('--mpe', action='store_true',
                        help='Whether to use (approximate) MPE inference for prediction')

    parser.add_argument('--save-preds', action='store_true',
                        help='Whether to save predictions')

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

    if args.exp_name:
        out_path = os.path.join(args.output, dataset_name + '_' + args.exp_name)
    else:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(args.output, dataset_name + '_' + date_string)
    out_log_path = os.path.join(out_path, 'exp.log')
    os.makedirs(out_path, exist_ok=True)

    logging.info('Opening log file {}...'.format(out_log_path))

    preamble = ("""metric\tfold\ttrain\tvalid\ttest\n""")

    #
    # collect target var ids
    first_fold_first_split = fold_splits[0][0]
    n_features = first_fold_first_split[0].shape[1] if isinstance(first_fold_first_split,
                                                                  tuple) else first_fold_first_split.shape[1]
    target_var_ids = None

    if args.target_ids is not None:
        target_var_ids = numpy.array(args.target_ids)
    elif args.n_targets is not None:
        target_var_ids = numpy.arange(n_features - 1, n_features - args.n_targets - 1, -1)
    else:
        raise ValueError('Specify the target var ids!')

    target_var_ids = numpy.sort(target_var_ids)

    feature_var_ids = numpy.ones(n_features, dtype=bool)
    feature_var_ids[target_var_ids] = 0

    logging.info('feature var ids {}'.format(feature_var_ids))
    logging.info('target var ids {}'.format(target_var_ids))

    logging.info('\nLoading spn model from: {}'.format(args.model))
    spn = None
    with open(args.model, 'rb') as model_file:
        load_start_t = perf_counter()
        spn = pickle.load(model_file)
        load_end_t = perf_counter()
        logging.info('done in {}'.format(load_end_t - load_start_t))

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.write(preamble)
        out_log.flush()

        n_splits = 3
        n_scores = len(args.scores)
        score_tensor = numpy.zeros((n_folds, n_splits, n_scores))
        score_tensor[:] = None

        if args.save_preds:
            preds_path = os.path.join(out_path, PREDS_EXT)
            os.makedirs(preds_path, exist_ok=True)
        #
        # for each fold in the data
        for i, splits in enumerate(fold_splits):

            logging.info('>> Processing fold {}'.format(i))

            for j, split in enumerate(splits):

                if split is not None:
                    logging.info('\t processing {}'.format(SPLIT_NAMES[j]))

                    #
                    # assuming it to be a single split, if a tuple, take just the first
                    if isinstance(split, tuple):
                        split = split[0]

                    #
                    # separate feature from target vars
                    target = split[:, target_var_ids]
                    n_instances = split.shape[0]
                    split_preds = []

                    pred_start_t = perf_counter()
                    if args.mpe:
                        raise ValueError('MPE implementation yet to come')
                    else:
                        logging.info('\t\t(brute force enumeration for multi-class)\n')

                        for k in range(n_instances):

                            eval_start_t = perf_counter()
                            best_class = None
                            best_class_ll = -numpy.inf
                            for c in range(args.n_targets):
                                #
                                # create one instance copy with class one hot encoded
                                instance = numpy.copy(split[k])
                                instance[target_var_ids] = 0
                                instance[target_var_ids[c]] = 1

                                # print('class {}\ninstance {}\noriginal {}'.format(c,
                                #                                                   instance[-100:],
                                # split[k, -100:]))

                                #
                                # evaluating
                                (pred_ll, ) = spn.single_eval(instance)
                                if pred_ll > best_class_ll:
                                    # logging.info('new best ll {} {}'.format(pred_ll,
                                    # best_class_ll))
                                    best_class_ll = pred_ll
                                    best_class = c
                            # print('best class {}'.format(best_class))

                            eval_end_t = perf_counter()
                            #
                            # storing the prediction as a hoe array
                            pred = numpy.zeros(args.n_targets, dtype=int)
                            pred[best_class] = 1
                            split_preds.append(pred)

                            print('\t\tprocessed instance {}/{} in {}'.format(k + 1,
                                                                              n_instances,
                                                                              eval_end_t -
                                                                              eval_start_t),
                                  end='           \r')

                    pred_end_t = perf_counter()
                    logging.info('\t\tAll instances processed in {} secs'.format(pred_end_t -
                                                                                 pred_start_t))
                    #
                    # comparing split preds and target vars
                    split_preds = numpy.array(split_preds)
                    for s, score in enumerate(args.scores):
                        print(target[:10])
                        print(split_preds[:10])
                        split_score = compute_scores(target, split_preds, score)
                        logging.info('{} {}: {}'.format(SPLIT_NAMES[j],
                                                        SCORE_NAMES[score],
                                                        split_score))

                        score_tensor[i, j, s] = split_score

                    #
                    # saving preds?
                    if args.save_preds:
                        preds_file_path = os.path.join(preds_path,
                                                       '{}.{}.gz'.format(SPLIT_NAMES[j],
                                                                         PREDS_EXT))
                        numpy.savetxt(preds_file_path, split_preds, fmt='%d')

            #
            # saving to log file
            for s, score in enumerate(args.scores):
                out_log.write('{}\t{}\t{}\n'.format(SCORE_NAMES[score],
                                                    i,
                                                    '\t'.join('{:.6f}'.format(score_tensor[i, j, s])
                                                              for j in range(n_splits))))
                out_log.flush()

        fold_avg_score_tensor = score_tensor.mean(axis=0)
        fold_std_score_tensor = score_tensor.std(axis=0)

        logging.info('\n')
        out_log.write('\n')
        #
        # saving to log file the avg scores
        out_log.write("metric\tavg-train\tstd-train\tavg-valid\tstd-valid\tavg-test\tstd-test\n")
        for s, score in enumerate(args.scores):
            out_log.write('{}\t{}\n'.format(SCORE_NAMES[score],
                                            '\t'.join('{:.6f}\t{:.6f}'.format(fold_avg_score_tensor[j, s],
                                                                              fold_std_score_tensor[j, s])
                                                      for j in range(n_splits))))
            out_log.flush()
            logging.info('{}:\t{}\n'.format(SCORE_NAMES[score],
                                            '\t'.join('{:.6f} (+/-{:.6f})'.format(fold_avg_score_tensor[j, s],
                                                                                  fold_std_score_tensor[j, s])
                                                      for j in range(n_splits))))
