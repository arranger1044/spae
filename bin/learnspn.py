import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from collections import defaultdict

import numpy
# from numpy.testing import assert_almost_equal

import datetime

import os

import sys

import logging

from algo.learnspn import LearnSPN

from spn import NEG_INF
from spn.utils import stats_format
from spn.linked.spn import evaluate_on_dataset
from spn import evaluate_on_dataset_batch


from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import load_feature_scheme

import pickle
import gzip

MODEL_EXT = 'model'

TEST_PREDS_FILE = 'test.lls'

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
                    help='Loaded dataset type')

# parser.add_argument("--data-dir", type=str,
#                     default='data/',
#                     help='Specify dataset dir (default data/)')

# parser.add_argument('--train-ext', type=str,
#                     default=None,
#                     help='Training set extension')

# parser.add_argument('--valid-ext', type=str,
#                     default=None,
#                     help='Validation set extension')

# parser.add_argument('--test-ext', type=str,
#                     default=None,
#                     help='Test set extension')

parser.add_argument('-k', '--n-row-clusters', type=int, nargs='?',
                    default=2,
                    help='Number of clusters to split rows into' +
                    ' (for DPGMM it is the max num of clusters)')

parser.add_argument('-c', '--cluster-method', type=str, nargs='?',
                    default='GMM',
                    help='Cluster method to apply on rows' +
                    ' ["GMM"|"DPGMM"|"HOEM"]')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

parser.add_argument('-g', '--g-factor', type=float, nargs='+',
                    default=[1.0],
                    help='The "p-value like" for G-Test on columns')

parser.add_argument('-i', '--n-iters', type=int, nargs='?',
                    default=100,
                    help='Number of iterates for the row clustering algo')

parser.add_argument('-r', '--n-restarts', type=int, nargs='?',
                    default=4,
                    help='Number of restarts for the row clustering algo' +
                    ' (only for GMM)')

parser.add_argument('-p', '--cluster-penalty', type=float, nargs='+',
                    default=[1.0],
                    help='Penalty for the cluster number' +
                    ' (i.e. alpha in DPGMM and rho in HOEM, not used in GMM)')

parser.add_argument('-s', '--sklearn-args', type=str, nargs='?',
                    help='Additional sklearn parameters in the for of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('-m', '--min-inst-slice', type=int, nargs='+',
                    default=[50],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[0.1],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--clt-leaves', action='store_true',
                    help='Whether to use Chow-Liu trees as leaves')

parser.add_argument('--kde-leaves', action='store_true',
                    help='Whether to use kernel density estimations as leaves')

parser.add_argument('--save-model', action='store_true',
                    help='Whether to store the model file as a pickle file')

parser.add_argument('--gzip', action='store_true',
                    help='Whether to compress the model pickle file')

parser.add_argument('--suffix', type=str,
                    help='Dataset output suffix')

parser.add_argument('--feature-scheme', type=str,
                    default=None,
                    help='Path to feature scheme file')

parser.add_argument('--cv', type=int,
                    help='Folds for cross validation for model selection')

parser.add_argument('--y-only', action='store_true',
                    help='Whether to load only the Y from the model pickle file')

parser.add_argument('--keras', action='store_true',
                    help='Whether to use Keras for inference')

parser.add_argument('--batch', type=int,
                    default=100,
                    help='Mini batch size (meaningful with --keras)')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')
#
# parsing the args
args = parser.parse_args()

if args.clt_leaves and args.kde_leaves:
    raise ValueError('Cannot use both CLT and KDE leaves!')

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)

#
# gathering parameters
alphas = args.alpha
min_inst_slices = args.min_inst_slice
g_factors = args.g_factor
cluster_penalties = args.cluster_penalty

cltree_leaves = args.clt_leaves
kde_leaves = args.kde_leaves

sklearn_args = None
if args.sklearn_args is not None:
    sklearn_key_value_pairs = args.sklearn_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    sklearn_args = {key.strip(): value.strip() for key, value in
                    [pair.strip().split('=')
                     for pair in sklearn_key_value_pairs]}
else:
    sklearn_args = {}
logging.info(sklearn_args)

# initing the random generators
seed = args.seed
MAX_RAND_SEED = 99999999  # sys.maxsize
rand_gen = numpy.random.RandomState(seed)

#
# elaborating the dataset
#
fold_splits = None
n_splits = None
train = None
valid = None
test = None

dataset_name = args.dataset.split('/')[-1]
#
# replacing  suffixes names
dataset_name = dataset_name.replace('.pklz', '')
dataset_name = dataset_name.replace('.pkl', '')
dataset_name = dataset_name.replace('.pickle', '')


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
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
                                 test_ext=test_ext,
                                 x_only=x_only,
                                 y_only=y_only,
                                 dtype=args.dtype)

else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             x_only=x_only,
                                             y_only=y_only,
                                             dtype=args.dtype)


#
# printing
print_fold_splits_shapes(fold_splits)

# n_instances = train.shape[0]
# n_test_instances = test.shape[0]
#
# estimating the frequencies for the features
logging.info('Estimating features on training set...')
# freqs, features = dataset.data_2_freqs(train)
n_features = fold_splits[0][0].shape[1]
features = None
if args.feature_scheme is None:
    features = numpy.array([2 for i in range(n_features)])
else:
    # raise ValueError(a'Loading feature schema not implemented yet')
    features = load_feature_scheme(args.feature_scheme)
    assert len(features) == n_features
print(features)


#
# Opening the file for prediction
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.suffix:
    out_path = os.path.join(args.output, args.suffix)
else:
    out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))

out_log_path = os.path.join(out_path,  'exp.log')
logging.info('Opening log file... {}'.format(out_log_path))
#
# creating dir if non-existant
if not os.path.exists(os.path.dirname(out_log_path)):
    os.makedirs(os.path.dirname(out_log_path))


best_avg_ll = NEG_INF
best_state = {}
best_test_lls = None

preamble = ("""g-factor\tclu-pen\tmin-ins\talpha\tfold\tn_edges""" +
            """\tdepth\tn_weights\tn_leaves\tn_sums\tn_prods""" +
            """\tn_scopes\tlearn-time""" +
            """\ttrain-inf-time\tvalid-inf-time\ttest-inf-time""" +
            """\ttrain_ll\tvalid_ll\ttest_ll\n""")

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.write(preamble)
    out_log.flush()
    #
    # looping over all parameters combinations
    for g_factor in g_factors:
        for cluster_penalty in cluster_penalties:
            for min_inst_slice in min_inst_slices:

                train_alpha_lls = defaultdict(list)
                valid_alpha_lls = defaultdict(list)
                test_alpha_lls = defaultdict(list)

                train_alpha_times = defaultdict(list)
                valid_alpha_times = defaultdict(list)
                test_alpha_times = defaultdict(list)

                test_preds_list = defaultdict(list)
                best_models = defaultdict(list)

                train_avg_ll = NEG_INF
                valid_avg_ll = NEG_INF
                test_avg_ll = NEG_INF

                fold_models = []
                fold_params = defaultdict(dict)

                for i, (train, valid, test) in enumerate(fold_splits):
                    #
                    # fixing the seed
                    rand_gen = numpy.random.RandomState(seed)

                    #
                    # Creating the structure learner
                    learner = LearnSPN(g_factor=g_factor,
                                       min_instances_slice=min_inst_slice,
                                       alpha=alphas[0],
                                       row_cluster_method=args.cluster_method,
                                       cluster_penalty=cluster_penalty,
                                       n_cluster_splits=args.n_row_clusters,
                                       n_iters=args.n_iters,
                                       n_restarts=args.n_restarts,
                                       sklearn_args=sklearn_args,
                                       cltree_leaves=cltree_leaves,
                                       kde_leaves=kde_leaves,
                                       rand_gen=rand_gen)

                    learn_start_t = perf_counter()
                    #
                    # build an spn on the training set
                    spn = learner.fit_structure(data=train,
                                                feature_sizes=features)
                    learn_end_t = perf_counter()
                    l_time = learn_end_t - learn_start_t
                    logging.info('Structure learned in {} secs'.format(l_time))
                    fold_models.append(spn)

                    #
                    # compiling to keras?
                    if args.keras:
                        spn.build_k()
                    #
                    # print(spn)

                    #
                    # gathering statistics
                    n_edges = spn.n_edges()
                    n_levels = spn.n_layers()
                    n_weights = spn.n_weights()
                    n_leaves = spn.n_leaves()
                    n_sums = spn.n_sum_nodes()
                    n_prods = spn.n_product_nodes()
                    n_scopes = spn.n_unique_scopes()

                    fold_params[i]['n_edges'] = n_edges
                    fold_params[i]['n_levels'] = n_levels
                    fold_params[i]['n_weights'] = n_weights
                    fold_params[i]['n_leaves'] = n_leaves
                    fold_params[i]['n_sums'] = n_sums
                    fold_params[i]['n_prods'] = n_prods
                    fold_params[i]['n_scopes'] = n_scopes
                    fold_params[i]['time'] = l_time

                    # #
                    # # saving just in case
                    # if args.save_model:
                    #     prefix_str = stats_format([g_factor,
                    #                                cluster_penalty,
                    #                                min_inst_slice,
                    #                                learner._alpha],
                    #                               '_',
                    #                               digits=5)
                    #     model_path = os.path.join(out_path,
                    #                               'best.{0}.{1}'.format(dataset_name,
                    #                                                     MODEL_EXT))
                    #     with open(model_path, 'wb') as model_file:
                    #         pickle.dump(spn, model_file)
                    #         logging.info('Dumped spn to {}'.format(model_path))

                    #
                    # smoothing can be done after the spn has been built
                    for alpha in alphas:
                        logging.info('\n')
                        logging.info('Smoothing leaves with alpha = %f', alpha)
                        spn.smooth_leaves(alpha)

                        #
                        # Compute LL on training set
                        logging.info('\tEvaluating on training set...')
                        eval_start_t = perf_counter()
                        train_preds = evaluate_on_dataset_batch(spn, train,
                                                                batch_size=args.batch, _keras=args.keras)
                        eval_end_t = perf_counter()
                        assert train_preds.shape[0] == train.shape[0]
                        train_avg_ll = numpy.mean(train_preds)

                        train_inf_time = eval_end_t - eval_start_t
                        train_alpha_times[alpha].append(train_inf_time)
                        logging.info('\t\t{}\n\t\tdone in {} secs'.format(train_avg_ll,
                                                                          train_inf_time))

                        # eval_start_t = perf_counter()
                        # train_preds = evaluate_on_dataset_batch(spn, train,
                        #                                         batch_size=args.batch, _keras=args.keras)
                        # eval_end_t = perf_counter()
                        # assert train_preds.shape[0] == train.shape[0]
                        # train_avg_ll = numpy.mean(train_preds)

                        # train_inf_time = eval_end_t - eval_start_t
                        # train_alpha_times[alpha].append(train_inf_time)
                        # logging.info('\t\t{}\n\t\tdone in {} secs'.format(train_avg_ll,
                        #                                                   train_inf_time))

                        train_alpha_lls[alpha].append(train_avg_ll)

                        #
                        # Compute LL on validation set
                        if valid is not None:
                            logging.info('\tEvaluating on validation set')
                            eval_start_t = perf_counter()
                            valid_preds = evaluate_on_dataset_batch(spn, valid,
                                                                    batch_size=args.batch, _keras=args.keras)
                            eval_end_t = perf_counter()
                            assert valid_preds.shape[0] == valid.shape[0]
                            valid_avg_ll = numpy.mean(valid_preds)

                            valid_inf_time = eval_end_t - eval_start_t
                            valid_alpha_times[alpha].append(valid_inf_time)
                            logging.info('\t\t{}\n\t\tdone in {} secs'.format(valid_avg_ll,
                                                                              valid_inf_time))

                            valid_alpha_lls[alpha].append(valid_avg_ll)

                        #
                        # Compute LL on test set
                        if test is not None:
                            logging.info('\tEvaluating on test set')
                            eval_start_t = perf_counter()
                            test_preds = evaluate_on_dataset_batch(spn, test,
                                                                   batch_size=args.batch, _keras=args.keras)
                            eval_end_t = perf_counter()
                            assert test_preds.shape[0] == test.shape[0]
                            test_avg_ll = numpy.mean(test_preds)

                            test_inf_time = eval_end_t - eval_start_t
                            test_alpha_times[alpha].append(test_inf_time)
                            logging.info('\t\t{}\n\t\tdone in {} secs'.format(test_avg_ll,
                                                                              test_inf_time))

                            test_alpha_lls[alpha].append(test_avg_ll)
                            test_preds_list[alpha].append(test_preds)

                #
                # updating best stats according to the best avg ll
                for alpha in alphas:
                    train_a_lls = train_alpha_lls[alpha]
                    valid_a_lls = valid_alpha_lls[alpha]
                    test_a_lls = test_alpha_lls[alpha]

                    train_avg_ll = numpy.mean(train_a_lls)
                    valid_avg_ll = numpy.mean(valid_a_lls)
                    test_avg_ll = numpy.mean(test_a_lls)

                    for i in range(len(fold_splits)):
                        train_score = train_a_lls[i] if train_a_lls else NEG_INF
                        valid_score = valid_a_lls[i] if valid_a_lls else NEG_INF
                        test_score = test_a_lls[i] if test_a_lls else NEG_INF
                        #
                        # writing to file a line for the grid
                        stats = stats_format([g_factor,
                                              cluster_penalty,
                                              min_inst_slice,
                                              alpha,
                                              i,
                                              fold_params[i]['n_edges'],
                                              fold_params[i]['n_levels'],
                                              fold_params[i]['n_weights'],
                                              fold_params[i]['n_leaves'],
                                              fold_params[i]['n_sums'],
                                              fold_params[i]['n_prods'],
                                              fold_params[i]['n_scopes'],
                                              fold_params[i]['time'],
                                              train_alpha_times[alpha][i],
                                              valid_alpha_times[alpha][i],
                                              test_alpha_times[alpha][i],
                                              train_score,
                                              valid_score,
                                              test_score],
                                             '\t',
                                             digits=5)
                        out_log.write(stats + '\n')
                        out_log.flush()

                    if args.cv is not None:
                        valid_avg_ll = test_avg_ll

                    if valid_avg_ll > best_avg_ll:
                        best_avg_ll = valid_avg_ll
                        best_state['alpha'] = alpha
                        best_state['min-inst-slice'] = min_inst_slice
                        best_state['g-factor'] = g_factor
                        best_state['cluster-penalty'] = cluster_penalty
                        best_state['train_ll'] = train_avg_ll
                        best_state['valid_ll'] = valid_avg_ll
                        best_state['test_ll'] = test_avg_ll
                        best_test_lls = test_preds_list[alpha]

                        #
                        # This now overwrites the old best model
                        if args.save_model:
                            for i, model in enumerate(fold_models):
                                prefix_str = stats_format([g_factor,
                                                           cluster_penalty,
                                                           min_inst_slice,
                                                           alpha],
                                                          '_',
                                                          digits=5)

                                if len(fold_models) > 1:
                                    model_path = os.path.join(out_path,
                                                              'best.{}.{}.{}'.format(dataset_name,
                                                                                     i,
                                                                                     MODEL_EXT))
                                else:
                                    model_path = os.path.join(out_path,
                                                              'best.{}.{}'.format(dataset_name,
                                                                                  MODEL_EXT))

                                if args.gzip:
                                    model_path += '.gz'
                                #
                                # resetting the alpha
                                model.smooth_leaves(alpha)
                                model_file = None
                                if args.gzip:
                                    model_file = gzip.open(model_path, 'wb')
                                    logging.info('Compressing model file...')
                                else:
                                    model_file = open(model_path, 'wb')

                                pickle.dump(model, model_file)
                                logging.info('Dumped spn to {}'.format(model_path))
                                model_file.close()

    #
    # writing as last line the best params
    best_state_str = ', '.join(['{}: {}'.format(k, best_state[k]) for k in sorted(best_state)])
    out_log.write("{0}".format(best_state_str))
    out_log.flush()

    #
    # saving the best test_lls
    if args.cv is None:
        if test is not None:
            if best_test_lls:
                assert len(best_test_lls) == 1
                test_lls_path = os.path.join(out_path, TEST_PREDS_FILE)
                if args.gzip:
                    test_lls_path += '.gz'
                logging.info('Saving best test preds to {}'.format(test_lls_path))
                numpy.savetxt(test_lls_path, best_test_lls[0], delimiter='\n')
    else:
        for i, test_preds in enumerate(best_test_lls):
            test_lls_path = os.path.join(out_path, '{}.{}'.format(i, TEST_PREDS_FILE))
            if args.gzip:
                test_lls_path += '.gz'
            logging.info('Saving best test preds to {}'.format(test_lls_path))
            numpy.savetxt(test_lls_path, test_preds, delimiter='\n')


logging.info('Grid search ended.')
logging.info('Best params:\n\t%s', best_state_str)
