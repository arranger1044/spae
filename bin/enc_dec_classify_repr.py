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

import logging

import pickle
import gzip

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from multioutput import MultiOutputRegressor
from multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import average_precision_score

from pystruct.models import MultiLabelClf
from pystruct.learners import NSlackSSVM
from pystruct.learners import FrankWolfeSSVM
from pystruct.models import GraphCRF


from dataset import load_cv_splits
from dataset import load_train_val_test_splits
from dataset import print_fold_splits_shapes
from dataset import SPLIT_NAMES

from spn.linked.representation import decode_embeddings_mpn
from spn.linked.representation import load_feature_info
from spn.linked.representation import generate_missing_masks
from spn.linked.representation import filter_2darray_2dmask

from dataset import dataset_to_instances_set

MAX_N_INSTANCES = 10000

PICKLE_SPLIT_EXT = 'pickle'
PREDS_PATH = 'preds'
PICKLE_SPLIT_EXT = 'pickle'
COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'
COMPRESSED_MODEL_EXT = 'model.gz'


SCORE_NAMES = {'accuracy': 'acc',
               'hamming': 'ham',
               'exact': 'exc',
               'jaccard': 'jac',
               'micro-f1': 'mif',
               'macro-f1': 'maf',
               'micro-auc-pr': 'mipr',
               'macro-auc-pr': 'mapr', }

PREPROCESS_DICT = {
    'std-scl': StandardScaler,
    'min-max': MinMaxScaler,
    'l2-norm': Normalizer
}

CLASSIFIER_DICT = {
    'lr-l2-ovr-bal': lambda c:
    OneVsRestClassifier(LogisticRegression(C=c,
                                           penalty='l2',
                                           tol=0.0001,
                                           fit_intercept=True,
                                           class_weight='balanced',
                                           solver='liblinear')),
    'lr-l2-mlo-bal': lambda c:
    MultiOutputClassifier(LogisticRegression(C=c,
                                             penalty='l2',
                                             tol=0.0001,
                                             fit_intercept=True,
                                             multi_class='ovr',
                                             class_weight='balanced',
                                             solver='liblinear')),
    'sgd-hinge-l1': lambda c:
    OneVsRestClassifier(SGDClassifier(loss='hinge',
                                      penalty='l1', alpha=c, l1_ratio=1.0,
                                      fit_intercept=True,
                                      # n_iter=5,
                                      shuffle=True,
                                      verbose=0,
                                      learning_rate='optimal',
                                           class_weight='balanced')),
    'ls-rbf': lambda c:
    LabelSpreading(n_neighbors=c,
                   kernel='rbf',
                   gamma=20,
                   alpha=0.8,
                   max_iter=100,
                   tol=0.001,
                   n_jobs=1),
    'ls-knn': lambda c:
    LabelSpreading(n_neighbors=int(c),
                   kernel='knn',
                   gamma=20,
                   alpha=0.8,
                   max_iter=100,
                   tol=0.001,
                   n_jobs=-1),
    'lp-knn': lambda c:
    LabelPropagation(n_neighbors=int(c),
                     kernel='knn',
                     gamma=20,
                     max_iter=30,
                     tol=0.001,
                     n_jobs=-1),
    'rr-l2-ovr-bal': lambda c:
    MultiOutputRegressor(Ridge(alpha=c,
                               tol=0.0001, )),
    'rc-l2-ovr-bal': lambda c:
    OneVsRestClassifier(RidgeClassifier(alpha=c,
                                        tol=0.0001,
                                        fit_intercept=True,
                                        class_weight='balanced',)),

    'rr-l2-bal': lambda c: Ridge(alpha=c,
                                 tol=0.0001,
                                 fit_intercept=True),
    'mtl': lambda c: MultiTaskLasso(alpha=c,
                                    tol=0.0001,
                                    fit_intercept=True),
    'rfr': lambda c: RandomForestRegressor(n_estimators=int(c),
                                           max_depth=4,
                                           # max_features='sqrt',
                                           ),
    'rt': lambda c: RandomForestRegressor(criterion='mse',
                                          # splitter='best',
                                          max_depth=int(c),
                                          min_samples_split=2,
                                          min_samples_leaf=1,
                                          min_weight_fraction_leaf=0.0,
                                          max_features=None,
                                          random_state=None,
                                          max_leaf_nodes=None,
                                          min_impurity_split=1e-07,
                                          # presort=False
                                          ),
    'rfc': lambda c: RandomForestClassifier(n_estimators=int(c),
                                            max_depth=4,
                                            # max_features='sqrt',
                                            ),
    'svm-crf': lambda c: FrankWolfeSSVM(model=GraphCRF(directed=True,
                                                       inference_method="ad3"),
                                        C=c,
                                        max_iter=10),
}


def unlabel_instances(X, y, n_labelled=100, unlabel=-1, rand_gen=None, max_labelled_prop=0.9):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    n_instances = X.shape[0]
    assert y.shape[0] == n_instances, "Non-matching number of instances {} {}".format(y.shape[0])

    processed_y = numpy.zeros(y.shape, dtype=numpy.int32)
    processed_y[:] = unlabel

    labelled_prop = min(max_labelled_prop, n_labelled / n_instances)
    if labelled_prop > 0.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=labelled_prop, random_state=rand_gen)
        unlabelled_ids, labelled_ids = list(sss.split(X, y))[0]

        processed_y[labelled_ids] = y[labelled_ids]

    logging.info('Number of labelled instances {}'.format((processed_y != unlabel).sum()))

    return processed_y


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


def compute_multiclass_multioutput_scores(y_true, y_preds, score='accuracy'):
    score_p = 0.0
    n_features = y_true.shape[1]
    assert y_preds.shape[1] == n_features

    for i in range(n_features):
        score_p += compute_scores(y_true[:, i], y_preds[:, i], score)

    return score_p / n_features


def decode_predictions(preds, node_feature_assoc, mpn_model, n_features,
                       missing_masks=None, re_evaluate_mpe=False,
                       embedding_type='activations',
                       missing_emb_val=-1):
    dec_preds = decode_embeddings_mpn(mpn_model,
                                      preds,
                                      node_feature_assoc,
                                      n_features=n_features,
                                      embedding_type=embedding_type,
                                      missing_masks=missing_masks,
                                      re_evaluate_mpe=re_evaluate_mpe,
                                      missing_emb_val=missing_emb_val)
    return dec_preds


def decode_predictions_knn(preds, embeds, embeds_labels, missing_masks=None, **knn_wargs):
    knnc = KNeighborsClassifier(**knn_wargs)

    print('knn e', preds.shape, embeds.shape, embeds_labels.shape)
    # assert preds.shape[0] == embeds.shape[0]
    assert preds.shape[1] == embeds.shape[1]

    #
    # masking the embeddings
    if missing_masks is not None:
        preds = filter_2darray_2dmask(preds, missing_masks)
        embeds = filter_2darray_2dmask(embeds, missing_masks)

        # assert preds.shape[0] == embeds.shape[0]
        assert preds.shape[1] == embeds.shape[1]

    fit_start_t = perf_counter()
    knnc.fit(embeds, embeds_labels)
    fit_end_t = perf_counter()
    logging.info('\n\t\tKNN fitted in {} secs'.format(fit_end_t - fit_start_t))

    predict_start_t = perf_counter()
    dec_preds = knnc.predict(preds)
    predict_end_t = perf_counter()
    logging.info('\t\t\tprediction done in {} secs'.format(predict_end_t - predict_start_t))

    return dec_preds


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

parser.add_argument('--repr-x', type=str,
                    default=None,
                    help='Specify a learned representation for the X (file path)')

parser.add_argument('--repr-x-exts', type=str, nargs='+',
                    default=None,
                    help='Learned representations split extensions')

parser.add_argument('--repr-y', type=str,
                    default=None,
                    help='Specify a learned representation for the Y (file path)')

parser.add_argument('--repr-y-exts', type=str, nargs='+',
                    default=None,
                    help='Learned representations split extensions')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
                    help='Loaded dataset type')

parser.add_argument('--repr-x-dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded representation type')

parser.add_argument('--repr-y-dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded representation type')

parser.add_argument('--decode-model', type=str,
                    help='Spn model file path for decoding (or path to spn models dir when --cv)')

parser.add_argument('--knn-decode', type=str, nargs='?',
                    help='Additional sklearn knn parameters in the for of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('--emb-type', type=str, nargs='?',
                    default='activations',
                    help='Type of embedding to decode' +
                    '("activations"| "latent_categorical" | "latent_binary")')

parser.add_argument('--missing-emb-val', type=int, nargs='?',
                    default=-1,
                    help='Missing emb value (-1)')

parser.add_argument('--reev', action='store_true',
                    help='Whether to reevaluate bottom-up the MPN')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

parser.add_argument('--scores', type=str, nargs='+',
                    default=['accuracy'],
                    help='Scores for the classifiers ("accuracy"|"hamming"|"exact"|"micro-f1")')

parser.add_argument('--preprocess', type=str, nargs='+',
                    default=[],
                    help='Algorithms to preprocess data')

parser.add_argument('--classifier', type=str, nargs='?',
                    default=None,
                    help='Parametrized version of the logistic regression')

#
# TODO: to generalize to different classifiers
parser.add_argument('--log-c', type=float, nargs='+',
                    default=[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
                    help='Regularization coefficient (inverse for logistic regression)')

parser.add_argument('--missing-percs', type=float, nargs='+',
                    default=[None],
                    help='')

parser.add_argument('--feature-inc', type=int, nargs='+',
                    default=None,
                    help='Considering features in batches')

parser.add_argument('--semi-super-labels', type=int,
                    default=None,
                    help='Number of labelled training instances in the semi supervised case')

parser.add_argument('--exp-name', type=str, nargs='?',
                    default=None,
                    help='Experiment name, if not present a date will be used')

parser.add_argument('--concat', action='store_true',
                    help='Whether to concatenate the new representation to the old dataset')

parser.add_argument('--expify', action='store_true',
                    help='Whether to exp transform the data')

parser.add_argument('--save-probs', action='store_true',
                    help='Whether to save predictions as probabilities')

parser.add_argument('--save-preds', action='store_true',
                    help='Whether to save predictions')

parser.add_argument('--save-model', action='store_true',
                    help='Whether to store the model file as a pickle file')

parser.add_argument('--x-orig', action='store_true',
                    help='Whether to evaluate only the original data')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')

parser.add_argument('--save-text', action='store_true',
                    help='Saving the repr data to text as well')


#
# parsing the args
args = parser.parse_args()

seed = args.seed
rand_gen = numpy.random.RandomState(seed)

decode = False
knn_sklearn_args = None

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)


if args.repr_y is not None:
    decode = True

    if args.decode_model is None:
        raise ValueError('Missing model to decode data')

    if args.knn_decode is not None:
        sklearn_key_value_pairs = args.knn_decode.translate(
            {ord('['): '', ord(']'): ''}).split(',')
        knn_sklearn_args = {key.strip(): value.strip() for key, value in
                            [pair.strip().split('=')
                             for pair in sklearn_key_value_pairs]}
        if 'n_neighbors' in knn_sklearn_args:
            knn_sklearn_args['n_neighbors'] = int(knn_sklearn_args['n_neighbors'])
        if 'n_jobs' in knn_sklearn_args:
            knn_sklearn_args['n_jobs'] = int(knn_sklearn_args['n_jobs'])
        if 'p' in knn_sklearn_args:
            knn_sklearn_args['p'] = int(knn_sklearn_args['p'])
    else:
        knn_sklearn_args = {}
    logging.info('KNN sklearn args: {}'.format(knn_sklearn_args))


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
repr_fold_y_splits = None

train_ext = None
valid_ext = None
test_ext = None
repr_train_x_ext = None
repr_valid_x_ext = None
repr_test_x_ext = None
repr_train_y_ext = None
repr_valid_y_ext = None
repr_test_y_ext = None


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

if args.repr_y_exts is not None:
    if len(args.repr_exts) == 1:
        repr_train_y_ext, = args.repr_y_exts
    elif len(args.repr_exts) == 2:
        repr_train_y_ext, repr_test_y_ext = args.repr_y_exts
    elif len(args.repr_exts) == 3:
        repr_train_y_ext, repr_valid_y_ext, repr_test_y_ext = args.repr_y_exts
    else:
        raise ValueError('Up to 3 repr data extenstions can be specified')


n_folds = args.cv if args.cv is not None else 1

#
# loading data and learned representations
if args.cv is not None:

    fold_splits = load_cv_splits(args.dataset,
                                 dataset_name,
                                 n_folds,
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
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
    if decode:
        repr_fold_y_splits = load_cv_splits(args.repr_y,
                                            dataset_name,
                                            n_folds,
                                            y_only=True,
                                            train_ext=repr_train_y_ext,
                                            valid_ext=repr_valid_y_ext,
                                            test_ext=repr_test_y_ext,
                                            dtype=args.repr_y_dtype)

else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             dtype=args.dtype)
    repr_fold_x_splits = load_train_val_test_splits(args.repr_x,
                                                    dataset_name,
                                                    x_only=True,
                                                    train_ext=repr_train_x_ext,
                                                    valid_ext=repr_valid_x_ext,
                                                    test_ext=repr_test_x_ext,
                                                    dtype=args.repr_x_dtype)
    if decode:
        repr_fold_y_splits = load_train_val_test_splits(args.repr_y,
                                                        dataset_name,
                                                        y_only=True,
                                                        train_ext=repr_train_y_ext,
                                                        valid_ext=repr_valid_y_ext,
                                                        test_ext=repr_test_y_ext,
                                                        dtype=args.repr_y_dtype)


# for fold in fold_splits:
#     for split in fold:
#         if split:
#             split_x, split_y = split
#             print('#in x', len(dataset_to_instances_set(split_x)))
#             print('#in y', len(dataset_to_instances_set(split_y)))

#
# printing
logging.info('Original folds')
print_fold_splits_shapes(fold_splits)
logging.info('Repr X folds')
print_fold_splits_shapes(repr_fold_x_splits)
if decode:
    logging.info('Repr Y folds')
    print_fold_splits_shapes(repr_fold_y_splits)

full_fold_train_y = None
if args.semi_super_labels is not None:

    assert args.classifier in {'ls-rbf', 'ls-knn', 'lp-rbf', 'lp-knn'}

    full_fold_train_y = []
    un_fold_splits = []
    for fold in fold_splits:
        train, valid, test = fold
        train_x, train_y = train
        unlabel_train_y = unlabel_instances(train_x, train_y,
                                            n_labelled=args.semi_super_labels,
                                            unlabel=-1,
                                            rand_gen=rand_gen)
        un_fold = ((train_x, unlabel_train_y), valid, test)
        un_fold_splits.append(un_fold)
        full_fold_train_y.append(train_y)
    fold_splits = un_fold_splits
    logging.info('Modified training labels folds')
    print_fold_splits_shapes(fold_splits)
#
# Opening the file for test prediction
#
if args.exp_name:
    out_path = os.path.join(args.output, dataset_name + '_' + args.exp_name)
else:
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output, dataset_name + '_' + date_string)
out_log_path = os.path.join(out_path, 'exp.log')
os.makedirs(out_path, exist_ok=True)

logging.info('Opening log file {} ...'.format(out_log_path))

assert len(fold_splits) == len(repr_fold_x_splits)
if decode:
    assert len(fold_splits) == len(repr_fold_y_splits)

#
labelled_splits = []

for i in range(len(fold_splits)):

    repr_x_fold = repr_fold_x_splits[i]
    fold = fold_splits[i]
    repr_y_fold = None
    if decode:
        repr_y_fold = repr_fold_y_splits[i]

    labelled_fold = []

    for s, (repr_x, split) in enumerate(zip(repr_x_fold, fold)):

        split_x = None
        split_y = None
        repr_y = None

        if decode:
            repr_y = repr_y_fold[s]

        if split is not None:
            split_x, split_y = split

        if repr_x is not None:
            if args.concat:
                new_repr_x = numpy.concatenate((split_x, repr_x), axis=1)
                assert new_repr_x.shape[0] == split_x.shape[0]
                assert new_repr_x.shape[1] == split_x.shape[1] + repr_x.shape[1]
                logging.info('Concatenated representations: {} -> {}\n'.format(repr_x.shape,
                                                                               new_repr_x.shape))
                if decode:
                    labelled_fold.append([new_repr_x, repr_y])
                else:
                    labelled_fold.append([new_repr_x, split_y])

            else:
                if args.x_orig:
                    if decode:
                        logging.info('fold {}: {} (X |-> Y\') ({} |-> {})\n'.format(i,
                                                                                    SPLIT_NAMES[s],
                                                                                    split_x.shape,
                                                                                    repr_y.shape))
                        labelled_fold.append([split_x, repr_y])
                    else:
                        logging.info('fold {}: {} (X |-> Y) ({} |-> {})\n'.format(i,
                                                                                  SPLIT_NAMES[s],
                                                                                  split_x.shape,
                                                                                  split_y.shape))
                        labelled_fold.append([split_x, split_y])
                else:
                    if decode:
                        logging.info('fold {}: {} (X\' |-> Y\') ({} |-> {})\n'.format(i,
                                                                                      SPLIT_NAMES[
                                                                                          s],
                                                                                      repr_x.shape,
                                                                                      repr_y.shape))
                        labelled_fold.append([repr_x, repr_y])
                    else:
                        logging.info('fold {}: {} (X\' |-> Y) ({} |-> {})\n'.format(i,
                                                                                    SPLIT_NAMES[s],
                                                                                    repr_x.shape,
                                                                                    split_y.shape))
                        labelled_fold.append([repr_x, split_y])
        else:
            labelled_fold.append(None)

    labelled_splits.append(labelled_fold)


if args.expify:
    logging.info('Turning into exponentials\n')
    for f in range(len(labelled_splits)):
        for i in range(len(labelled_splits[f])):
            if labelled_splits[f][i] is not None:
                labelled_splits[f][i][0] = numpy.exp(labelled_splits[f][i][0])
#
# preprocessing
if args.preprocess:
    raise ValueError('Preprocessing not implemented yet')
    # for prep in args.preprocess:
    #     preprocessor = PREPROCESS_DICT[prep]()
    #     logging.info('Preprocessing with {}:'.format(preprocessor))
    #     #
    #     # assuming the first split is the training set
    #     preprocessor.fit(labelled_splits[0][0])
    #     for i in range(len(labelled_splits)):
    #         labelled_splits[i][0] = preprocessor.transform(labelled_splits[i][0])

mpn_fold_models = None
feature_fold_infos = None
if decode:
    if not args.knn_decode:
        mpn_fold_models = [load_spn_model(args.decode_model, f)
                           for f in range(len(labelled_splits))]

        feature_fold_infos = [load_feature_info_fold(args.repr_y, f)
                              for f in range(len(labelled_splits))]

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n".format(args))
    out_log.flush()

    if args.feature_inc:
        header_str = "\t\t\t{}\n\t\t\t{}\n".format('\t'.join(SPLIT_NAMES * len(args.scores)),
                                                   '\t'.join([SCORE_NAMES[s] for s in args.scores
                                                              for n in SPLIT_NAMES]))
        out_log.write('{}'.format(header_str))
        out_log.flush()

        min_feature = 0
        max_feature = labelled_splits[0][0][0].shape[1]

        increment = None
        if len(args.feature_inc) == 1:
            increment = args.feature_inc[0]
        elif len(args.feature_inc) == 2:
            min_feature = args.feature_inc[0]
            increment = args.feature_inc[1]
        elif len(args.feature_inc) == 3:
            min_feature = args.feature_inc[0]
            max_feature = args.feature_inc[1]
            increment = args.feature_inc[2]
        else:
            raise ValueError('More than three values specified for --feature-inc')

        increments_range = range(min_feature + increment, max_feature + increment, increment)
        n_increments = len(list(increments_range))
        n_params = len(args.log_c)
        n_folds = len(labelled_splits)
        n_splits = 3
        n_scores = len(args.scores)
        score_tensor = numpy.zeros((n_increments, n_params, n_folds, n_splits, n_scores))
        score_tensor[:] = None

        fold_preds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fold_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for t, m in enumerate(increments_range):
            #
            # selecting subset features
            logging.info('Considering features {}:{}'.format(min_feature, m))

            sel_labelled_splits = []
            for f in range(len(labelled_splits)):

                sel_labelled_splits.append([])
                for i in range(len(labelled_splits[f])):

                    sel_labelled_splits[f].append((labelled_splits[f][i][0][:, min_feature:m],
                                                   labelled_splits[f][i][1]))

                for i in range(len(labelled_splits[f])):
                    logging.info('shapes {} {}'.format(sel_labelled_splits[f][i][0].shape,
                                                       sel_labelled_splits[f][i][1].shape))

            train_x, train_y = sel_labelled_splits[0][0]
            for p, c in enumerate(args.log_c):
                logging.info('\n\nC: {}'.format(c))

                for f in range(len(labelled_splits)):

                    # log_res = linear_model.LogisticRegression(C=c,
                    #                                           **LOGISTIC_MOD_DICT_PARAMS[args.classifier])
                    # clf = OneVsRestClassifier(log_res)
                    clf = CLASSIFIER_DICT[args.classifier](c)

                    train_x, train_y = sel_labelled_splits[f][0]
                    true_train_x, true_train_y = fold_splits[f][0]
                    #
                    # fitting
                    fit_s_t = perf_counter()
                    clf.fit(train_x, train_y)
                    fit_e_t = perf_counter()

                    logging.info('\tfold: {} ({})'.format(f, fit_e_t - fit_s_t))
                    #
                    # scoring
                    for i, split in enumerate(sel_labelled_splits[f]):
                        if split is not None:
                            split_x, split_y = split
                            _, true_split_y = fold_splits[f][i]
                            if args.semi_super_labels is not None and i == 0:
                                true_split_y = full_fold_train_y[f]

                            split_s_t = perf_counter()
                            # split_acc = log_res.score(split_x, split_y)
                            split_preds = clf.predict(split_x)
                            split_e_t = perf_counter()

                            if decode:

                                # logging.info('RMSE:\t{}\n'.format(compute_scores(split_y, split_preds,'rmse')))
                                logging.info('EXA:\t{}\n'.format(compute_multiclass_multioutput_scores(split_y,
                                                                                                       split_preds,
                                                                                                       'exact')))

                                node_feature_assoc_f = None
                                if feature_fold_infos is not None:
                                    node_feature_assoc_f = {info.node_id: info.feature_id
                                                            for info in feature_fold_infos[f]}

                                #
                                # generate missing masks
                                missing_masks = None
                                missing_perc = None
                                if missing_perc is not None:
                                    missing_masks = generate_missing_masks(split_preds.shape[1],
                                                                           split_x.shape[0],
                                                                           node_feature_assoc_f,
                                                                           missing_perc,
                                                                           one_mask=False,
                                                                           rand_gen=rand_gen)

                                if args.knn_decode:
                                    split_preds = decode_predictions_knn(split_preds,
                                                                         train_y,
                                                                         true_train_y,
                                                                         missing_masks,
                                                                         **knn_sklearn_args)
                                else:
                                    split_preds = decode_predictions(split_preds,
                                                                     # feature_fold_infos[f],
                                                                     node_feature_assoc_f,
                                                                     mpn_fold_models[f],
                                                                     true_split_y.shape[1],
                                                                     missing_masks,
                                                                     re_evaluate_mpe=args.reev,
                                                                     embedding_type=args.emb_type,
                                                                     missing_emb_val=args.missing_emb_val)
                                assert split_preds.shape[0] == split_x.shape[0]
                                assert split_preds.shape[1] == true_split_y.shape[1]

                            fold_preds[t][p][i].append(split_preds)
                            if hasattr(clf, 'predict_proba'):
                                split_probs = clf.predict_proba(split_x)
                                fold_probs[t][p][i].append(split_probs)

                            print(true_split_y[:3], split_preds[:3])
                            for s, score_func in enumerate(args.scores):

                                split_score = compute_scores(true_split_y, split_preds, score_func)
                                score_tensor[t, p, f, i, s] = split_score

                            scores_str = '\t'.join(['{}:{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                       score_tensor[t, p, f, i, s])
                                                    for s in range(n_scores)])
                            logging.info('\t\t{}\t{}\t({})'.format(SPLIT_NAMES[i],
                                                                   scores_str,
                                                                   split_e_t - split_s_t))

                    #
                    # saving to file
                    scores_str = '\t'.join('{:.6f}'.format(score_tensor[t, p, f, i, s])
                                           for s in range(n_scores)
                                           for i in range(n_splits))
                    out_log.write('{}\t{}\t{}\t{}\n'.format(m, c, f, scores_str))
                    out_log.flush()

        #
        # computing statistics along folds
        fold_avg_score_tensor = score_tensor.mean(axis=2)
        fold_std_score_tensor = score_tensor.std(axis=2)

        logging.info('\n')
        out_log.write('\n')

        out_log.write('inc\tc\t{}\n'.format('\t'.join('avg-{0}-{1}\tstd-{0}-{1}'.format(SPLIT_NAMES[i],
                                                                                        SCORE_NAMES[args.scores[s]])
                                                      for s in range(n_scores)
                                                      for i in range(n_splits))))
        for m in range(n_increments):
            logging.info('{}'.format(list(increments_range)[m]))
            for p in range(n_params):
                logging.info('\t{}'.format(args.log_c[p]))
                for i in range(n_splits):
                    score_str = '\t'.join('{}:{:.6f} +/-{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                       fold_avg_score_tensor[
                                                                           m, p, i, s],
                                                                       fold_std_score_tensor[m, p, i, s])
                                          for s in range(n_scores))
                    logging.info('\t\t{}\t{}'.format(SPLIT_NAMES[i], score_str))
                out_log.write('{}\t{}\t{}\n'.format(list(increments_range)[m],
                                                    args.log_c[p],
                                                    '\t'.join('{:.6f}\t{:.6f}'.format(fold_avg_score_tensor[m, p, i, s],
                                                                                      fold_std_score_tensor[m, p, i, s])
                                                              for s in range(n_scores)
                                                              for i in range(n_splits))))
                out_log.flush()

        #
        # getting best parameters
        out_log.write('\n')
        logging.info('\n\tBest params: ->(best avg value)')
        best_split = 1 if not numpy.isnan(fold_avg_score_tensor[0, 0, 1, 0]) else 2
        eval_split = 2
        logging.info('\t\t(best split: {} score split: {})'.format(SPLIT_NAMES[best_split],
                                                                   SPLIT_NAMES[eval_split]))
        for m in range(n_increments):
            logging.info('{}'.format(list(increments_range)[m]))

            res_list = []

            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                res_list.append('\t{}:\t{}\t-> {} (+/-{})'.format(SCORE_NAMES[args.scores[s]],
                                                                  args.log_c[best_p],
                                                                  fold_avg_score_tensor[
                                                                      m, best_p, best_split, s],
                                                                  fold_std_score_tensor[m, best_p, eval_split, s]))

            res_str = '\n'.join(res_list)
            logging.info('\n{}'.format(res_str))

            out_log.write('{}\t\t{}\n'.format(list(increments_range)[m],
                                              '\t'.join(SCORE_NAMES[args.scores[s]]
                                                        for s in range(n_scores))))
            for p in range(n_params):
                scores_str = '\t'.join('{}'.format(fold_avg_score_tensor[m, p, best_split, s])
                                       for s in range(n_scores))
                out_log.write('\t{}\t{}\n'.format(args.log_c[p], scores_str))
                out_log.flush()

            out_log.write('\n')

        out_log.write('inc\tscore\tbest-c\tbest-avg\tbest-std\n')
        for m in range(n_increments):
            res_list_p = []
            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                res_list_p.append('{}\t{}\t{}\t{}\t{}'.format(list(increments_range)[m],
                                                              SCORE_NAMES[args.scores[s]],
                                                              args.log_c[best_p],
                                                              fold_avg_score_tensor[
                    m, best_p, eval_split, s],
                    fold_std_score_tensor[m, best_p, best_split, s]))
            res_str = '\n'.join(res_list_p)
            out_log.write('{}\n'.format(res_str))
            out_log.flush()

        #
        # saving predictions?
        if args.save_probs or args.save_preds:
            logging.info('\n')
            preds_path = os.path.join(out_path, PREDS_PATH)
            os.makedirs(preds_path, exist_ok=True)
            for m in range(n_increments):
                for s in range(n_scores):
                    best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                    best_p_str = '{}'.format(args.log_c[best_p])

                    if args.save_preds:
                        for i in range(n_splits):
                            best_preds_list = fold_preds[m][best_p][i]
                            for f, best_preds in enumerate(best_preds_list):
                                preds_file_path = os.path.join(preds_path,
                                                               '{}-{}-f{}.{}.preds.gz'.format(list(increments_range)[m],
                                                                                              best_p_str,
                                                                                              f,
                                                                                              SPLIT_NAMES[i]))
                                logging.info('Dumping preds to {}'.format(preds_file_path))
                                numpy.savetxt(preds_file_path, best_preds, fmt='%d')

                    if args.save_probs:
                        for i in range(n_splits):
                            best_probs_list = fold_probs[m][best_p][i]
                            for f, best_probs in enumerate(best_probs_list):
                                probs_file_path = os.path.join(preds_path,
                                                               '{}-{}-f{}.{}.probs.gz'.format(list(increments_range)[m],
                                                                                              best_p_str,
                                                                                              f,
                                                                                              SPLIT_NAMES[i]))
                                logging.info('Dumping probs to {}'.format(probs_file_path))
                                numpy.savetxt(probs_file_path, best_probs)

    else:

        header_str = "\t\t{}\n\t\t{}\n".format('\t'.join(SPLIT_NAMES * len(args.scores)),
                                               '\t'.join([SCORE_NAMES[s] for s in args.scores
                                                          for n in SPLIT_NAMES]))
        out_log.write('{}'.format(header_str))
        out_log.flush()

        n_params = len(args.log_c)
        n_folds = len(labelled_splits)
        n_splits = 3
        n_scores = len(args.scores)
        n_percs = len(args.missing_percs)
        score_tensor = numpy.zeros((n_percs, n_params, n_folds, n_splits, n_scores))
        score_tensor[:] = None

        fold_models = defaultdict(lambda: defaultdict(list))

        for m, missing_perc in enumerate(args.missing_percs):
            logging.info('\n\nmissing perc: {}'.format(missing_perc))

            for p, c in enumerate(args.log_c):
                logging.info('\n\nC: {}'.format(c))

                # log_res = linear_model.LogisticRegression(C=c,
                #                                           **LOGISTIC_MOD_DICT_PARAMS[args.classifier])

                #
                # for each fold
                for f, fold in enumerate(labelled_splits):

                    # clf = OneVsRestClassifier(log_res)
                    clf = CLASSIFIER_DICT[args.classifier](c)
                    fold_models[m][p].append(clf)

                    train_x, train_y = fold[0]
                    true_train_x, true_train_y = fold_splits[f][0]

                    #
                    # fitting the classifier
                    fit_s_t = perf_counter()
                    clf.fit(train_x, train_y)
                    fit_e_t = perf_counter()
                    logging.info('fold {} ({})'.format(f, fit_e_t - fit_s_t))

                    #
                    # scoring
                    for i, split in enumerate(labelled_splits[f]):

                        if split is not None:
                            split_x, split_y = split
                            _, true_split_y = fold_splits[f][i]
                            if args.semi_super_labels is not None and i == 0:
                                true_split_y = full_fold_train_y[f]

                            split_s_t = perf_counter()
                            split_preds = clf.predict(split_x)
                            split_e_t = perf_counter()

                            if decode:

                                # logging.info('RMSE:\t{}\n'.format(compute_scores(split_y, split_preds,'rmse')))
                                # logging.info('EXA:\t{}\n'.format(compute_multiclass_multioutput_scores(split_y,
                                #                                                                        split_preds,
                                #                                                                        'exact')))

                                node_feature_assoc_f = None
                                if feature_fold_infos is not None:
                                    node_feature_assoc_f = {info.node_id: info.feature_id
                                                            for info in feature_fold_infos[f]}
                                #
                                # generate missing masks
                                missing_masks = None
                                # missing_perc = 0.0

                                if args.knn_decode:
                                    if missing_perc is not None:
                                        missing_masks = generate_missing_masks(split_preds.shape[1],
                                                                               max(train_y.shape[0],
                                                                                   split_x.shape[0]),
                                                                               node_feature_assoc_f,
                                                                               missing_perc,
                                                                               one_mask=True,
                                                                               rand_gen=rand_gen)
                                    split_preds = decode_predictions_knn(split_preds,
                                                                         train_y,
                                                                         true_train_y,
                                                                         missing_masks,
                                                                         **knn_sklearn_args)
                                else:
                                    if missing_perc is not None:
                                        missing_masks = generate_missing_masks(split_preds.shape[1],
                                                                               split_x.shape[0],
                                                                               node_feature_assoc_f,
                                                                               missing_perc,
                                                                               one_mask=False,
                                                                               rand_gen=rand_gen)

                                    print(split_y[:3], split_preds[:3])
                                    split_preds = decode_predictions(split_preds,
                                                                     # feature_fold_infos[f],
                                                                     node_feature_assoc_f,
                                                                     mpn_fold_models[f],
                                                                     true_split_y.shape[1],
                                                                     missing_masks,
                                                                     re_evaluate_mpe=args.reev,
                                                                     embedding_type=args.emb_type,
                                                                     missing_emb_val=args.missing_emb_val)
                                assert split_preds.shape[0] == split_x.shape[0]
                                assert split_preds.shape[1] == true_split_y.shape[1]
                                # if args.knn_decode:
                                #     split_preds = decode_predictions_knn(split_preds,
                                #                                          train_y,
                                #                                          true_train_y,
                                #                                          **knn_sklearn_args)
                                # else:
                                #     split_preds = decode_predictions(split_preds,
                                #                                      feature_fold_infos[f],
                                #                                      mpn_fold_models[f],
                                #                                      true_split_y.shape[1])
                                # assert split_preds.shape[0] == split_x.shape[0]
                                # assert split_preds.shape[1] == true_split_y.shape[1]

                            print(true_split_y[:3], split_preds[:3])
                            for s, score_func in enumerate(args.scores):
                                print(true_split_y.shape, split_preds.shape)
                                split_score = compute_scores(true_split_y, split_preds, score_func)
                                score_tensor[m, p, f, i, s] = split_score

                            score_str = '\t'.join(['{}:{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                      score_tensor[m, p, f, i, s])
                                                   for s in range(n_scores)])
                            logging.info('\t{}\t{}\t({})'.format(SPLIT_NAMES[i],
                                                                 score_str,
                                                                 split_e_t - split_s_t))

                        # else:
                        #     for score_func in args.scores:
                        #         scores[score_func].append(None)

                    out_log.write('{}\t{}\t{}\t{}\n'.format(missing_perc, c, f,
                                                            '\t'.join('{:.6f}'.format(score_tensor[m, p, f, i, s])
                                                                      for s in range(n_scores)
                                                                      for i in range(n_splits))))
                    out_log.flush()

        #
        # computing statistics along folds
        fold_avg_score_tensor = score_tensor.mean(axis=2)
        fold_std_score_tensor = score_tensor.std(axis=2)

        logging.info('\n')
        out_log.write('\n')

        for m in range(n_percs):
            logging.info('\nmissing perc: {}'.format(args.missing_percs[m]))
            out_log.write('\nmissing perc: {}\n'.format(args.missing_percs[m]))
            for p in range(n_params):
                logging.info('{}'.format(args.log_c[p]))
                for i in range(n_splits):
                    score_str = '\t'.join('{}:{:.6f} +/-{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                       fold_avg_score_tensor[
                                                                           m, p, i, s],
                                                                       fold_std_score_tensor[m, p, i, s])
                                          for s in range(n_scores))
                    logging.info('{}\t{}'.format(SPLIT_NAMES[i], score_str))
                out_log.write('{}\t{}\n'.format(args.log_c[p],
                                                '\t'.join('{:.6f}\t{:.6f}'.format(fold_avg_score_tensor[m, p, i, s],
                                                                                  fold_std_score_tensor[m, p, i, s])
                                                          for s in range(n_scores)
                                                          for i in range(n_splits))))
                out_log.flush()

        logging.info('\n')
        out_log.write('\n')

        #
        # getting best parameters
        logging.info('\n\tBest params: ->(best avg value)')
        best_split = 1 if not numpy.isnan(fold_avg_score_tensor[0, 0, 1, 0]) else 2
        eval_split = 2
        logging.info('\t\t(best split: {} score split: {})'.format(SPLIT_NAMES[best_split],
                                                                   SPLIT_NAMES[eval_split]))

        res_list = []
        res_list_p = []
        for m in range(n_percs):
            # logging.info('\nmissing perc: {}'.format(args.missing_percs[m]))
            # out_log.write('\nmissing perc: {}\n'.format(args.missing_percs[m]))
            res_list.append('\nmissing perc: {}'.format(args.missing_percs[m]))
            res_list_p.append('\nmissing perc: {}'.format(args.missing_percs[m]))
            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                # print(fold_avg_score_tensor[:, best_split, s])
                res_list.append('{}:\t{}\t-> {} (+/-{})'.format(SCORE_NAMES[args.scores[s]],
                                                                args.log_c[best_p],
                                                                fold_avg_score_tensor[m,
                                                                                      best_p, eval_split, s],
                                                                fold_std_score_tensor[m,
                                                                                      best_p, best_split, s]))
                res_list_p.append('{}\t{}\t{}\t{}'.format(SCORE_NAMES[args.scores[s]],
                                                          args.log_c[best_p],
                                                          fold_avg_score_tensor[m,
                                                                                best_p, eval_split, s],
                                                          fold_std_score_tensor[m, best_p, eval_split, s]))
        res_str = '\n'.join(res_list)
        logging.info('\n{}'.format(res_str))

        # out_log.write('\t{}\n'.format('\t'.join(SCORE_NAMES[args.scores[s]]
        #                                        for s in range(n_scores))))
        for m in range(n_percs):
            # logging.info('\nmissing perc: {}'.format(args.missing_percs[m]))
            out_log.write('\nmissing perc: {}\n'.format(args.missing_percs[m]))
            out_log.write('\t{}\n'.format('\t'.join(SCORE_NAMES[args.scores[s]]
                                                    for s in range(n_scores))))
            for p in range(n_params):
                scores_str = '\t'.join('{}'.format(fold_avg_score_tensor[m, p, eval_split, s])
                                       for s in range(n_scores))
                out_log.write('{}\t{}\n'.format(args.log_c[p], scores_str))
                out_log.flush()

        out_log.write('\n')
        res_str = '\n'.join(res_list_p)
        out_log.write('{}\n'.format(res_str))
        out_log.flush()

        #
        # saving predictions?
        if args.save_probs or args.save_preds:
            logging.info('\n')
            preds_path = os.path.join(out_path, PREDS_PATH)
            os.makedirs(preds_path, exist_ok=True)

            for m in range(n_percs):
                missing_perc = args.missing_percs[m] if args.missing_percs[m] is not None else 0.0
                for s in range(n_scores):
                    best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                    best_p_str = '{}'.format(args.log_c[best_p])
                    best_models = fold_models[m][best_p]

                    if args.save_preds:
                        for i in range(n_splits):
                            for f, model in enumerate(best_models):
                                split = labelled_splits[f][i]
                                if split is not None:
                                    x = split[0]
                                    best_preds = model.predict(x)
                                    preds_file_path = os.path.join(preds_path,
                                                                   '{}-{}-f{}.{}.preds.gz'.format(missing_perc,
                                                                                                  best_p_str,
                                                                                                  f,
                                                                                                  SPLIT_NAMES[i]))
                                    logging.info('Dumping preds to {}'.format(preds_file_path))
                                    numpy.savetxt(preds_file_path, best_preds, fmt='%d')

                    if args.save_probs:
                        for i in range(n_splits):
                            for f, model in enumerate(best_models):
                                split = labelled_splits[f][i]
                                if split is not None and hasattr(model, 'predict_proba'):
                                    x = split[0]
                                    best_probs = model.predict_proba(x)
                                    probs_file_path = os.path.join(preds_path,
                                                                   '{}-{}-f{}.{}.probs.gz'.format(missing_perc,
                                                                                                  best_p_str,
                                                                                                  f,
                                                                                                  SPLIT_NAMES[i]))
                                    logging.info('Dumping probs to {}'.format(probs_file_path))
                                    numpy.savetxt(probs_file_path, best_probs)
