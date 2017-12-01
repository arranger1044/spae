import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset
from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import SPLIT_NAMES

import numpy

import datetime

import os

import sys

import logging

from sklearn import neural_network

from spn.utils import stats_format
from spn import MARG_IND

import pickle
import gzip

import matplotlib.pyplot as pyplot

PREDS_PATH = 'preds'

COMPRESSED_MODEL_EXT = 'model.gz'
MODEL_EXT = 'model'
PREDS_EXT = 'lls'

TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
PICKLE_SPLIT_EXT = 'pickle'
FEATURE_FILE_EXT = 'features'
SCOPE_FILE_EXT = 'scopes'

FMT_DICT = {'int': '%d',
            'float': '%.18e',
            'float.8': '%.8e',
            }


def sample_from_rbm(rbm,
                    n_samples,
                    init_V=None,
                    gibbs_burn=1000,
                    restarts=False,
                    leap=None,
                    dtype=numpy.int32,
                    rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    n_visibles = rbm.components_.shape[1]
    n_hiddens = rbm.components_.shape[0]
    samples = numpy.zeros((n_samples, n_visibles), dtype=dtype)

    #
    # restarting
    n_restarts = None
    if restarts:
        n_restarts = n_samples
        n_samples = 1
    else:
        n_restarts = 1

    logging.info('Sampling {} samples from best model'.format(n_samples))
    logging.info('\t with init:\t{}\n\tburn in:\t{}\n\trestarts:\t{}\n\tleap:\t{}'.format(init_V,
                                                                                          gibbs_burn,
                                                                                          restarts,
                                                                                          leap))
    n_leaps = 1 if leap is None else leap
    for j in range(n_restarts):

        print('Restart: {}/{}'.format(j + 1, n_restarts))

        #
        # initing
        V = None
        if init_V is None:
            logging.info('Random initing V')
            V = rand_gen.binomial(n=1, p=0.5, size=n_visibles)
        elif init_V == 'zeros':
            V = numpy.zeros(n_visibles, dtype=dtype)
        #
        # use provided samples
        elif isinstance(init_V, numpy.ndarray):
            assert len(init_V) == n_restarts, len(init_V)
            V = init_V[j]
        else:
            raise ValueError('Unrecognized init scheme {}'.format(init_V))
        print('V inited:', V.shape, V[:20])

        for i in range(gibbs_burn):
            print('Burning gibbs iterations {}/{}'.format(i + 1, gibbs_burn), end='           \r')
            V = rbm.gibbs(V)
        print("")

        n_iters = n_samples * n_leaps
        k = j
        for i in range(n_iters):
            print('Sampling {}/{}'.format(i + 1, n_iters), end='           \r')
            V = rbm.gibbs(V)
            if i % n_leaps == 0:
                samples[k, :] = V
                k += 1
        print("")

    return samples


def save_samples(samples,
                 n_rows,
                 output_dir,
                 dataset_name,
                 cmap,
                 img_rows=28,
                 img_cols=28):
    n_samples = samples.shape[0]
    n_cols = max(n_samples // n_rows, 1)
    total_final_array = []
    for w in range(n_cols):
        total_final_array += [numpy.hstack([numpy.pad(zz.reshape((img_rows, img_cols)), 1,
                                                      mode='constant', constant_values=1)
                                            for zz in samples[w * n_rows:w * n_rows + n_rows]])]

    out_img_path = os.path.join(output_dir,
                                'samples_{}x{}_{}.png'.format(n_rows, n_cols,
                                                              dataset_name))
    out_samples_path = os.path.join(output_dir,
                                    '{}_samples_{}'.format(n_samples,
                                                           dataset_name))

    numpy.save(out_samples_path, samples)
    print('Saved samples to {}'.format(out_samples_path))
    pyplot.imsave(out_img_path, numpy.vstack(total_final_array), cmap=cmap)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset dir')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=None,
                        help='Dataset split extensions')

    # parser.add_argument('--train-ext', type=str,
    #                     help='Training set name regex')

    # parser.add_argument('--valid-ext', type=str,
    #                     help='Validation set name regex')

    # parser.add_argument('--test-ext', type=str,
    #                     help='Test set name regex')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./exp/rbm/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--sep', type=str, nargs='?',
                        default=',',
                        help='Dataset output separator')

    parser.add_argument('--fmt', type=str, nargs='?',
                        default='int',
                        help='Dataset output number formatter')

    parser.add_argument('--n-hidden', type=int, nargs='+',
                        default=[500],
                        help='Number of hidden units')

    parser.add_argument('--l-rate', type=float, nargs='+',
                        default=[0.1],
                        help='Learning rate for training')

    parser.add_argument('--batch-size', type=int, nargs='+',
                        default=[10],
                        help='Batch size during learning')

    parser.add_argument('--n-iters', type=int, nargs='+',
                        default=[10],
                        help='Number of epochs')

    parser.add_argument('--no-ext', action='store_true',
                        help='Whether to concatenate the new representation to the old dataset')

    parser.add_argument('--log', action='store_true',
                        help='Transforming the repr data with log')

    parser.add_argument('--save-model', action='store_true',
                        help='Whether to store the model file as a pickle file')

    parser.add_argument('--gzip', action='store_true',
                        help='Whether to compress the repr out file')

    parser.add_argument('--cv', type=int,
                        default=None,
                        help='Folds for cross validation for model selection')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    parser.add_argument('--save-text', action='store_true',
                        help='Saving the repr data to text as well')

    # parser.add_argument('--save-model', action='store_true',
    #                     help='Saving the best training model')

    parser.add_argument('--sample', type=int,
                        default=None,
                        help='Number to samples to generate')

    parser.add_argument('--gibbs-burn', type=int,
                        default=None,
                        help='Number to gibbs iterations to burn')

    parser.add_argument('--y-only', action='store_true',
                        help='Whether to load only the Y from the model pickle file')

    # parser.add_argument('--save-probs', action='store_true',
    #                     help='Whether to save predictions as probabilities')

    # parser.add_argument('--save-preds', action='store_true',
    #                     help='Whether to save predictions')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

    parser.add_argument('--init', type=str, nargs='?',
                        default=None,
                        help='Gibbs sampling visible units init scheme')

    parser.add_argument('--nb-rows', type=int,
                        default=10,
                        help='Number of rows of images')

    parser.add_argument('--img-rows', type=int,
                        default=28,
                        help='Image pixel height')

    parser.add_argument('--img-cols', type=int,
                        default=28,
                        help='Image pixel width')

    parser.add_argument('--restart', action='store_true',
                        help='Whether to restart the gibbs chain')

    parser.add_argument('--leap', type=int,
                        default=None,
                        help='Leaping with Gibbs sampling')

    parser.add_argument('--cmap', type=str,
                        default='Greys_r',
                        help='Colormap name')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # fixing a seed
    rand_gen = numpy.random.RandomState(args.seed)

    os.makedirs(args.output, exist_ok=True)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    # #
    # # loading dataset splits
    # logging.info('Loading datasets: %s', args.dataset)
    # dataset_path = args.dataset
    # train, valid, test = dataset.load_dataset_splits(dataset_path,
    #                                                  filter_regex=[args.train_ext,
    #                                                                args.valid_ext,
    #                                                                args.test_ext])
    # dataset_name = args.train_ext.split('.')[0]

    # n_instances = train.shape[0]
    # n_test_instances = test.shape[0]
    # logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
    #                                                            valid.shape,
    #                                                            test.shape))
    # freqs, feature_vals = dataset.data_2_freqs(train)

    #
    # loading dataset folds
    logging.info('Loading datasets: %s', args.dataset)
    dataset_name = args.dataset.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name = dataset_name.replace('.pklz', '')
    dataset_name = dataset_name.replace('.pkl', '')
    dataset_name = dataset_name.replace('.pickle', '')

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

    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.suffix:
        out_path = os.path.join(args.output, args.suffix)
    else:
        out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))

    out_log_path = os.path.join(out_path,  'exp.log')
    logging.info('Opening log file... {}'.format(out_log_path))

    # test_lls_path = os.path.join(out_path, 'test.lls')
    os.makedirs(out_path, exist_ok=True)

    #
    #
    # performing a grid search along the hyperparameter space
    best_fold_avg_pll = -numpy.inf
    best_params = {}
    best_models = None

    n_hidden_values = args.n_hidden
    learning_rate_values = args.l_rate
    batch_size_values = args.batch_size
    n_iter_values = args.n_iters

    preamble = ("""n-hidden\tlearning-rate\tbatch-size\tn-iters\tfold""" +
                """\ttrain_pll\tvalid_pll\ttest_pll\n""")

    #
    # we are always expecting each fold to have 3 splits
    n_splits = 3

    with open(out_log_path, 'w') as out_log:

        out_log.write("parameters:\n{0}\n\n".format(args))
        out_log.write(preamble)
        out_log.flush()
        #
        # looping over all parameters combinations
        for n_hidden in n_hidden_values:
            for l_rate in learning_rate_values:
                for batch_size in batch_size_values:
                    for n_iters in n_iter_values:

                        fold_scores = numpy.zeros((n_folds, n_splits))
                        fold_scores[:] = None

                        fold_models = []

                        for f, splits in enumerate(fold_splits):

                            train, valid, test = splits

                            #
                            # printing fold stats
                            logging.info(
                                '\n\n**************\nProcessing fold {}\n**************\n\n'.format(f))

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

                            logging.info('Learning RBM for {} {} {} {}'.format(n_hidden,
                                                                               l_rate,
                                                                               batch_size,
                                                                               n_iters))
                            #
                            # learning
                            rbm = neural_network.BernoulliRBM(n_components=n_hidden,
                                                              learning_rate=l_rate,
                                                              batch_size=batch_size,
                                                              n_iter=n_iters,
                                                              verbose=args.verbose - 1,
                                                              random_state=rand_gen)
                            fit_s_t = perf_counter()
                            rbm.fit(train)
                            fit_e_t = perf_counter()
                            logging.info('Trained in {} secs'.format(fit_e_t - fit_s_t))

                            fold_models.append(rbm)

                            #
                            # evaluating on splits
                            for i, split in enumerate(splits):
                                if split is not None:
                                    eval_s_t = perf_counter()
                                    split_plls = rbm.score_samples(split)
                                    eval_e_t = perf_counter()
                                    split_avg_pll = numpy.mean(split_plls)
                                    logging.info('\t{} avg PLL: {} ({})'.format(SPLIT_NAMES[i],
                                                                                split_avg_pll,
                                                                                eval_e_t - eval_s_t))

                                    fold_scores[f, i] = split_avg_pll

                            #
                            # writing to file a line for the grid
                            stats = stats_format([n_hidden,
                                                  l_rate,
                                                  batch_size,
                                                  n_iters,
                                                  f,
                                                  fold_scores[f, 0],
                                                  fold_scores[f, 1],
                                                  fold_scores[f, 2]],
                                                 '\t',
                                                 digits=5)
                            out_log.write(stats + '\n')
                            out_log.flush()

                            # eval_s_t = perf_counter()
                            # train_plls = rbm.score_samples(train)
                            # eval_e_t = perf_counter()
                            # train_avg_pll = numpy.mean(train_plls)
                            # logging.info('\tTrain avg PLL: {} ({})'.format(train_avg_pll,
                            #                                                eval_e_t - eval_s_t))

                            # #
                            # # evaluating validation
                            # eval_s_t = perf_counter()
                            # valid_plls = rbm.score_samples(valid)
                            # eval_e_t = perf_counter()
                            # valid_avg_pll = numpy.mean(valid_plls)
                            # logging.info('\tValid avg PLL: {} ({})'.format(valid_avg_pll,
                            #                                                eval_e_t - eval_s_t))

                            # #
                            # # evaluating test
                            # eval_s_t = perf_counter()
                            # test_plls = rbm.score_samples(test)
                            # eval_e_t = perf_counter()
                            # test_avg_pll = numpy.mean(test_plls)
                            # logging.info('\tTest avg PLL: {} ({})'.format(test_avg_pll,
                            #                                               eval_e_t - eval_s_t))
                        best_split = 1 if not numpy.isnan(fold_scores[0, 1]) else 2
                        eval_split = 2
                        logging.info('\t\t(best split: {} score split: {})'.format(SPLIT_NAMES[best_split],
                                                                                   SPLIT_NAMES[eval_split]))
                        fold_avg_scores = fold_scores.mean(axis=0)
                        fold_std_scores = fold_scores.std(axis=0)
                        logging.info('Avg scores: {}'.format(fold_avg_scores))

                        #
                        # checking for improvements on validation
                        if fold_avg_scores[best_split] > best_fold_avg_pll:
                            best_fold_avg_pll = fold_avg_scores[best_split]
                            best_models = fold_models
                            best_params['n-hidden'] = n_hidden
                            best_params['learning-rate'] = l_rate
                            best_params['batch-size'] = batch_size
                            best_params['n-iters'] = n_iters
                            # NOTE: not saving anymore PLLs, we can get them from the models
                            # best_test_plls = test_plls

                            #
                            # saving the model
                            if args.save_model:
                                for f in range(n_folds):
                                    fold_str = '.{}'.format(f) if n_folds > 1 else ''
                                    prefix_str = '{}_{}_{}_{}'.format(n_hidden,
                                                                      l_rate,
                                                                      batch_size,
                                                                      n_iters)
                                    model_ext = COMPRESSED_MODEL_EXT if args.gzip else MODEL_EXT
                                    model_path = os.path.join(out_path,
                                                              'best.{}{}.{}'.format(dataset_name,
                                                                                    fold_str,
                                                                                    model_ext))
                                    model_file = None
                                    if args.gzip:
                                        model_file = gzip.open(model_path, 'wb')
                                    else:
                                        model_file = open(model_path, 'wb')

                                    pickle.dump(rbm, model_file)
                                    model_file.close()
                                    logging.info('Dumped RBM to {}'.format(model_path))

            #
            # writing as last line the best params
            best_params_str = ', '.join(['{}: {}'.format(k, best_params[k])
                                         for k in sorted(best_params)])
            out_log.write("{0}".format(best_params_str))
            out_log.flush()

    logging.info('Grid search ended.')
    logging.info('Best params:\n\t%s', best_params)

    ext_fold_splits = []
    log_ext_fold_splits = []

    logging.info('\n')
    preds_path = os.path.join(out_path, PREDS_PATH)
    os.makedirs(preds_path, exist_ok=True)

    #
    # using the best models to transform the data
    for f, (best_model, splits) in enumerate(zip(best_models, fold_splits)):

        fold_str = '.{}'.format(f) if n_folds > 1 else ''

        train, valid, test = splits

        repr_train = None
        repr_valid = None
        repr_test = None

        # #
        # # saving the best test_lls
        # numpy.savetxt(test_lls_path, best_test_plls, delimiter='\n')

        # if args.save_probs:
        #     for i in range(n_splits):
        #         split = splits[i]
        #         if split is not None:
        #             x = split[0]
        #             best_probs = best_model.predict_proba(x)
        #             probs_file_path = os.path.join(preds_path,
        #                                            '{}{}.probs.gz'.format(SPLIT_NAMES[i],
        #                                                                   fold_str))
        #             logging.info('Dumping probs to {}'.format(probs_file_path))
        #             numpy.savetxt(probs_file_path, best_probs)

        # if args.save_preds:
        #     for i in range(n_splits):
        #         split = splits[i]
        #         if split is not None:
        #             x = split[0]
        #             best_preds = best_model.predict(x)
        #             preds_file_path = os.path.join(preds_path,
        #                                            '{}{}.preds.gz'.format(SPLIT_NAMES[i],
        #                                                                   fold_str))
        #             logging.info('Dumping preds to {}'.format(preds_file_path))
        #             numpy.savetxt(preds_file_path, best_preds)

        #
        # now creating the new datasets from best model
        logging.info('\nConverting training set')
        feat_s_t = perf_counter()
        repr_train = best_model.transform(train)
        feat_e_t = perf_counter()
        logging.info('\t done in {}'.format(feat_e_t - feat_s_t))
        assert train.shape[0] == repr_train.shape[0]
        logging.info('New shapes {}->{}'.format(train.shape,
                                                repr_train.shape))

        if valid is not None:
            logging.info('Converting validation set')
            feat_s_t = perf_counter()
            repr_valid = best_model.transform(valid)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))
            logging.info('New shapes {}->{}'.format(valid.shape,
                                                    repr_valid.shape))
            assert valid.shape[0] == repr_valid.shape[0]
            assert repr_train.shape[1] == repr_valid.shape[1]

        if test is not None:
            logging.info('Converting test set')
            feat_s_t = perf_counter()
            repr_test = best_model.transform(test)
            feat_e_t = perf_counter()
            logging.info('\t done in {}'.format(feat_e_t - feat_s_t))
            logging.info('New shapes {}->{}'.format(test.shape,
                                                    repr_test.shape))
            assert test.shape[0] == repr_test.shape[0]
            assert repr_train.shape[1] == repr_test.shape[1]

        #
        # log transform as well?
        log_repr_train = None
        log_repr_valid = None
        log_repr_test = None

        if args.log:
            log_repr_train = numpy.log(repr_train)
            if repr_valid is not None:
                log_repr_valid = numpy.log(repr_valid)
            if repr_test is not None:
                log_repr_test = numpy.log(repr_test)

        # extending the original dataset
        ext_train = None
        ext_valid = None
        ext_test = None
        log_ext_train = None
        log_ext_valid = None
        log_ext_test = None

        if args.no_ext:
            ext_train = repr_train
            ext_valid = repr_valid
            ext_test = repr_test

            if args.log:
                log_ext_train = log_repr_train
                log_ext_valid = log_repr_valid
                log_ext_test = log_repr_test

        else:
            logging.info('\nConcatenating datasets')
            ext_train = numpy.concatenate((train, repr_train), axis=1)
            assert train.shape[0] == ext_train.shape[0]
            assert ext_train.shape[1] == train.shape[1] + repr_train.shape[1]

            if repr_valid is not None:
                ext_valid = numpy.concatenate((valid, repr_valid), axis=1)
                assert valid.shape[0] == ext_valid.shape[0]
                assert ext_valid.shape[1] == valid.shape[1] + repr_valid.shape[1]

            if repr_test is not None:
                ext_test = numpy.concatenate((test, repr_test), axis=1)
                assert test.shape[0] == ext_test.shape[0]
                assert ext_test.shape[1] == test.shape[1] + repr_test.shape[1]

            if args.log:
                log_ext_train = numpy.concatenate((train, log_repr_train), axis=1)
                assert train.shape[0] == log_ext_train.shape[0]
                assert ext_train.shape[1] == train.shape[1] + log_repr_train.shape[1]

                if log_repr_valid is not None:
                    log_ext_valid = numpy.concatenate((valid, log_repr_valid), axis=1)
                    assert valid.shape[0] == log_ext_valid.shape[0]
                    assert ext_valid.shape[1] == valid.shape[1] + log_repr_valid.shape[1]

                if log_repr_test is not None:
                    log_ext_test = numpy.concatenate((test, log_repr_test), axis=1)
                    assert test.shape[0] == log_ext_test.shape[0]
                    assert ext_test.shape[1] == test.shape[1] + log_repr_test.shape[1]

        ext_fold_splits.append((train_ext, valid_ext, test_ext))

        if args.log:
            log_ext_fold_splits.append((log_ext_train, log_ext_valid, log_ext_test))

        if args.save_text:
            #
            # storing them
            train_out_path = os.path.join(out_path, '{}{}.{}'.format(args.suffix,
                                                                     fold_str,
                                                                     args.train_ext))
            if args.gzip:
                train_out_path += '.gz'

            valid_out_path = os.path.join(out_path, '{}{}.{}'.format(args.suffix,
                                                                     fold_str,
                                                                     args.valid_ext))
            if args.gzip:
                valid_out_path += '.gz'

            test_out_path = os.path.join(out_path, '{}{}.{}'.format(args.suffix,
                                                                    fold_str,
                                                                    args.test_ext))
            if args.gzip:
                test_out_path += '.gz'

            logging.info('\nSaving training set to: {}'.format(train_out_path))
            numpy.savetxt(train_out_path, ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

            if ext_valid is not None:
                logging.info('Saving validation set to: {}'.format(valid_out_path))
                numpy.savetxt(
                    valid_out_path, ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

            if ext_test is not None:
                logging.info('Saving test set to: {}'.format(test_out_path))
                numpy.savetxt(test_out_path, ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

            if args.log:
                # storing them
                log_train_out_path = os.path.join(out_path,
                                                  'log.{}{}.{}'.format(args.suffix,
                                                                       fold_str,
                                                                       args.train_ext))
                log_valid_out_path = os.path.join(out_path,
                                                  'log.{}{}.{}'.format(args.suffix,
                                                                       fold_str,
                                                                       args.valid_ext))
                log_test_out_path = os.path.join(out_path,
                                                 'log.{}{}.{}'.format(args.suffix,
                                                                      fold_str,
                                                                      args.test_ext))

                logging.info('\nSaving log training set to: {}'.format(log_train_out_path))
                numpy.savetxt(log_train_out_path,
                              log_ext_train, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

                logging.info('Saving log validation set to: {}'.format(log_valid_out_path))
                numpy.savetxt(log_valid_out_path,
                              log_ext_valid, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

                logging.info('Saving log test set to: {}'.format(log_test_out_path))
                numpy.savetxt(log_test_out_path,
                              log_ext_test, delimiter=args.sep, fmt=FMT_DICT[args.fmt])

    #
    # saving to pickle
    model_ext = COMPRESSED_PICKLE_SPLIT_EXT if args.gzip else PICKLE_SPLIT_EXT
    split_file_path = os.path.join(out_path, '{}.{}.{}'.format(args.suffix,
                                                               dataset_name,
                                                               model_ext))

    logging.info('Saving pickle data splits to: {}'.format(split_file_path))
    split_file = None
    if args.gzip:
        split_file = gzip.open(split_file_path, 'wb')
    else:
        split_file = open(split_file_path, 'wb')

    if n_folds > 1:
        pickle.dump(ext_fold_splits, split_file)
    else:
        pickle.dump(ext_fold_splits[0], split_file)
    split_file.close()

    if args.log:
        log_split_file_path = os.path.join(out_path, 'log.{}.{}.{}'.format(args.suffix,
                                                                           dataset_name,
                                                                           model_ext))

        logging.info('Saving pickle log data splits to: {}'.format(log_split_file_path))
        log_split_file = None
        if args.gzip:
            log_split_file = gzip.open(log_split_file_path, 'wb')
        else:
            log_split_file = open(log_split_file_path, 'wb')

        if n_folds > 1:
            pickle.dump(log_ext_fold_splits, log_split_file)
        else:
            pickle.dump(log_ext_fold_splits[0], log_split_file)
        log_split_file.close()

    # if args.save_model:
    #     best_model_path = os.path.join(out_path, '{}.{}.model.pklz'.format(args.suffix,
    #                                                                        dataset_name))
    #     logging.info('Saving best model to: {}'.format(best_model_path))
    #     with gzip.open(best_model_path, 'wb') as mf:
    #         pickle.dump(best_models, mf, protocol=4)

    if args.sample is not None:
        n_samples = args.sample

        train, valid, test = fold_splits[0]
        # train_x, train_y = train
        init_V = None
        n_restarts = n_samples if args.restart else 1

        if args.init == 'train':
            rand_samples = rand_gen.choice(train.shape[0],
                                           size=n_restarts,
                                           replace=False)

            init_V = train[rand_samples]
        else:
            init_V = args.init
        samples = sample_from_rbm(best_model,
                                  n_samples,
                                  init_V=init_V,
                                  gibbs_burn=args.gibbs_burn,
                                  restarts=args.restart,
                                  leap=args.leap,
                                  dtype=args.dtype,
                                  rand_gen=rand_gen)

        # best_model_path = os.path.join(out_path, '{}.{}.model.pklz'.format(args.suffix,
        #                                                                    dataset_name))
        init_str = args.init if args.init is not None else 'rand'
        restart_str = 'restart' if args.restart else ''
        save_samples(samples,
                     args.nb_rows,
                     out_path,
                     'rbm_samples_{}_{}_{}_{}'.format(dataset_name,
                                                      init_str,
                                                      restart_str,
                                                      args.leap),
                     cmap=pyplot.get_cmap(args.cmap),
                     img_rows=args.img_rows,
                     img_cols=args.img_cols)
