import numpy

import csv

import re

import logging

import pickle

import gzip

SPLIT_NAMES = ['train', 'valid', 'test']

DATA_PATH = "data/"
DATA_FULL_PATH = DATA_PATH + 'full/'

DATASET_NAMES = ['accidents',
                 'ad',
                 'baudio',
                 'bbc',
                 'bnetflix',
                 'book',
                 'c20ng',
                 'cr52',
                 'cwebkb',
                 'dna',
                 'jester',
                 'kdd',
                 'msnbc',
                 'msweb',
                 'nltcs',
                 'plants',
                 'pumsb_star',
                 'tmovie',
                 'tretail']
import os

from spn import RND_SEED
from spn import MARG_IND


def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int8'):
    """
    WRITEME
    """
    file_path = os.path.join(path, file)
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset


def load_train_val_test_csvs(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int32',
                             suffixes=['.ts.data',
                                       '.valid.data',
                                       '.test.data']):
    """
    WRITEME
    """
    csv_files = [dataset + ext for ext in suffixes]
    return [csv_2_numpy(file, path, sep, type) for file in csv_files]


def load_dataset_splits(path=DATA_PATH,
                        filter_regex=['\.ts\.data',
                                      '\.valid\.data',
                                      '\.test\.data'],
                        sep=',',
                        type='int32',):
    dataset_paths = []
    for pattern in filter_regex:
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and pattern in f:
                dataset_paths.append(f)
                break

    return [csv_2_numpy(file_path, path, sep, type) for file_path in dataset_paths]


def save_splits_to_csv(dataset_name,
                       output_path,
                       dataset_splits,
                       splits_names=['train',
                                     'valid',
                                     'test'],
                       ext='data'):

    assert len(splits_names) == len(dataset_splits)

    n_features = dataset_splits[0].shape[1]

    os.makedirs(output_path, exist_ok=True)
    for split, name in zip(dataset_splits, splits_names):

        assert split.shape[1] == n_features
        print('\t{0} shape: {1}'.format(name, split.shape))

        split_file_name = '.'.join([dataset_name, name, ext])
        split_out_path = os.path.join(output_path, split_file_name)

        numpy.savetxt(split_out_path, split, delimiter=',', fmt='%d')
        print('\t\tSaved split to {}'.format(split_out_path))


def load_cv_splits(dataset_path,
                   dataset_name,
                   n_folds,
                   train_ext=None, valid_ext=None, test_ext=None,
                   x_only=False,
                   y_only=False,
                   dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Expecting dataset into {} folds for {}'.format(n_folds, dataset_name))
    fold_splits = []

    if (train_ext is not None and test_ext is not None):
        #
        # NOTE: this applies only to x-only/y-only data files
        for i in range(n_folds):
            logging.info('Looking for train-test split {}'.format(i))

            train_path = '{}.{}.{}'.format(dataset_path, i, train_ext)
            logging.info('Loading training csv file {}'.format(train_path))
            train = numpy.loadtxt(train_path, dtype=dtype, delimiter=',')

            test_path = '{}.{}.{}'.format(dataset_path, i, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype=dtype, delimiter=',')

            assert train.shape[1] == test.shape[1]

            fold_splits.append((train, None, test))
    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle file containint k = n_splits
        # [((train_x,  train_y), (test_x, test_y))_1, ... ((train_x, train_y), (test_x, test_y))_k]

        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        folds = pickle.load(fsplit)
        fsplit.close()

        assert len(folds) == n_folds

        for splits in folds:

            if len(splits) == 1:
                raise ValueError('Not expecting a fold made by a single split')
            elif len(splits) == 2:
                train_split, test_split = splits
                #
                # do they contain label information?
                if x_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_x, None, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_y, None, test_y))
                else:
                    fold_splits.append((train_split, None, test_split))
            elif len(splits) == 3:
                train_split, valid_split, test_split = splits
                if x_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_x, valid_x, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_y, valid_y, test_y))
                else:
                    fold_splits.append((train_split, valid_split, test_split))

    assert len(fold_splits) == n_folds
    # logging.info('Loaded folds for {}'.format(dataset_name))
    # for i, (train, valid, test) in enumerate(fold_splits):
    #     logging.info('\tfold:\t{} {} {} {} '.format(i, len(train), len(test), valid))
    #     if len(train) == 2 and len(test) == 2:
    #         logging.info('\t\ttrain x:\tsize: {}\ttrain y:\tsize: {}'.format(train[0].shape,
    #                                                                          train[1].shape))
    #         logging.info('\t\ttest:\tsize: {}\ttest:\tsize: {}'.format(test[0].shape,
    #                                                                    test[1].shape))
    #     else:
    #         logging.info('\t\ttrain:\tsize: {}'.format(train.shape))
    #         logging.info('\t\ttest:\tsize: {}'.format(test.shape))

    return fold_splits


def load_train_val_test_splits(dataset_path,
                               dataset_name,
                               train_ext=None, valid_ext=None, test_ext=None,
                               x_only=False,
                               y_only=False,
                               dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Looking for (train/valid/test) dataset splits: %s', dataset_path)

    if train_ext is not None:
        #
        # NOTE this works only with x-only data files
        train_path = '{}.{}'.format(dataset_path, train_ext)
        logging.info('Loading training csv file {}'.format(train_path))
        train = numpy.loadtxt(train_path, dtype='int32', delimiter=',')

        if valid_ext is not None:
            valid_path = '{}.{}'.format(dataset_path, valid_ext)
            logging.info('Loading valid csv file {}'.format(valid_path))
            valid = numpy.loadtxt(valid_path, dtype='int32', delimiter=',')
            assert train.shape[1] == valid.shape[1]

        if test_ext is not None:
            test_path = '{}.{}'.format(dataset_path, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype='int32', delimiter=',')
            assert train.shape[1] == test.shape[1]

    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle containing (train_x) | (train_x, test_x) |
        # (train_x, valid_x, test_x)
        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        splits = pickle.load(fsplit)
        fsplit.close()

        valid, test = None, None

        if len(splits) == 1:
            logging.info('Only training set')
            train = splits
            if x_only and isinstance(train, tuple):
                logging.info('\tonly x')
                train = train[0]

        elif len(splits) == 2:
            logging.info('Found training and test set')
            train, test = splits

            if len(train) == 2 and len(test) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[1].shape[1] == test[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
            else:
                assert train.shape[1] == test.shape[1]

            if x_only:
                logging.info('\tonly x')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[0]
                    test = test[0]
                else:
                    raise ValueError('Cannot get x only for train and test splits')
            elif y_only:
                logging.info('\tonly y')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[1]
                    test = test[1]
                else:
                    raise ValueError('Cannot get y only for train and test splits')

        elif len(splits) == 3:
            logging.info('Found training, validation and test set')
            train, valid, test = splits

            if len(train) == 2 and len(test) == 2 and len(valid) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[0].shape[1] == valid[0].shape[1]
                if train[1].ndim > 1 and test[1].ndim > 1 and valid[1].ndim > 1:
                    assert train[1].shape[1] == test[1].shape[1]
                    assert train[1].shape[1] == valid[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
                assert valid[0].shape[0] == valid[1].shape[0]

                if x_only:
                    logging.info('\tonly x')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[0]
                        valid = valid[0]
                        test = test[0]
                    else:
                        raise ValueError('Cannot get x only for train, valid and test splits')
                elif y_only:
                    logging.info('\tonly y')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[1]
                        valid = valid[1]
                        test = test[1]
                    else:
                        raise ValueError('Cannot get y only for train, valid and test splits')
            else:
                assert train.shape[1] == test.shape[1]
                assert train.shape[1] == valid.shape[1]

        else:
            raise ValueError('More than 3 splits, check pkl file {}'.format(dataset_path))

    fold_splits = [(train, valid, test)]

    logging.info('Loaded dataset {}'.format(dataset_name))

    return fold_splits


def print_fold_splits_shapes(fold_splits):
    for f, fold in enumerate(fold_splits):
        logging.info('\tfold {}'.format(f))
        for s, split in enumerate(fold):
            if split is not None:
                split_name = SPLIT_NAMES[s]
                if len(split) == 2:
                    split_x, split_y = split
                    logging.info('\t\t{}\tx: {}\ty: {}'.format(split_name,
                                                               split_x.shape, split_y.shape))
                else:
                    logging.info('\t\t{}\t(x/y): {}'.format(split_name,
                                                            split.shape))


def sample_indexes(indexes, perc, replace=False, rand_gen=None):
    """
    index sampling
    """
    n_indices = indexes.shape[0]
    sample_size = int(n_indices * perc)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    sampled_indices = rand_gen.choice(  # n_indices,
        indexes,
        size=sample_size,
        replace=replace)

    return sampled_indices


def sample_instances(dataset, perc, replace=False, rndState=None):
    """
    Little utility to sample instances (rows) from
    a dataset (2d numpy array)
    """
    n_instances = dataset.shape[0]
    sample_size = int(n_instances * perc)
    if rndState is None:
        row_indexes = numpy.random.choice(n_instances,
                                          sample_size,
                                          replace)
    else:
        row_indexes = rndState.choice(n_instances,
                                      sample_size,
                                      replace)
    # print(row_indexes)
    return dataset[row_indexes, :]


def sample_sets(datasets, perc, replace=False, rndState=None):
    """
    WRITEME
    """
    sampled_datasets = [sample_instances(dataset, perc, replace, rndState)
                        for dataset in datasets]
    return sampled_datasets


def dataset_to_instances_set(dataset):
    #
    # from numpy arrays to python tuples
    instances = [tuple(x) for x in dataset]
    #
    # removing duplicates
    instances = set(instances)
    return instances


from time import perf_counter


def one_hot_encoding(data, feature_values=None, n_features=None, dtype=numpy.float32, _check=True):
    if feature_values and n_features:
        assert len(feature_values) == n_features

    #
    # if values are not specified, assuming all of them to be binary
    if not feature_values and n_features:
        feature_values = numpy.array([2 for i in range(n_features)])

    if feature_values and not n_features:
        n_features = len(feature_values)

    if not feature_values and not n_features:
        raise ValueError('Specify feature values or n_features')

    #
    # computing the new number of features
    n_features_ohe = numpy.sum(feature_values)

    n_instances = data.shape[0]

    transformed_data = numpy.zeros((n_instances, n_features_ohe), dtype=dtype)

    enc_start_t = perf_counter()
    for i in range(n_instances):
        for j in range(n_features):
            value = data[i, j]

            if value != MARG_IND:
                ohe_feature_id = int(numpy.sum(feature_values[:j]) + data[i, j])
                transformed_data[i, ohe_feature_id] = 1
            else:
                ohe_feature_id = int(numpy.sum(feature_values[:j]))
                # print(ohe_feature_id, ohe_feature_id + feature_values[j])
                ohe_feature_ids = [i for i in range(ohe_feature_id,
                                                    ohe_feature_id + feature_values[j])]
                transformed_data[i, ohe_feature_ids] = 1

    enc_end_t = perf_counter()

    if _check:
        for i in range(n_instances):
            for j in range(n_features):
                j_id_start = int(numpy.sum(feature_values[:j]))
                j_id_end = int(numpy.sum(feature_values[:j + 1]))
                assert data[i, j] == numpy.where(transformed_data[i,
                                                                  j_id_start:j_id_end] == 1)[0][0]
    print('New dataset ({0} x {1}) encoded in {2}'.format(transformed_data.shape[0],
                                                          transformed_data.shape[1],
                                                          enc_end_t - enc_start_t))
    return transformed_data


def data_2_freqs(dataset):
    """
    WRITEME
    """
    freqs = []
    features = []
    for j, col in enumerate(dataset.T):
        freq_dict = {'var': j}
        # transforming into a set to get the feature value
        # this is assuming not missing values features
        # feature_values = max(2, len(set(col)))
        feature_values = max(2, max(set(col)) + 1)
        features.append(feature_values)
        # create a list whose length is the number of feature values
        freq_list = [0 for i in range(feature_values)]
        # populate it with the seen values
        for val in col:
            freq_list[val] += 1
        # update the dictionary and the resulting list
        freq_dict['freqs'] = freq_list
        freqs.append(freq_dict)

    return freqs, features


def update_feature_count(old_freqs, new_freqs):
    if not old_freqs:
        return new_freqs
    else:
        for i, frew in enumerate(old_freqs):
            old_freqs[i] = max(old_freqs[i], new_freqs[i])
        return old_freqs


def data_clust_freqs(dataset,
                     n_clusters,
                     rand_state=None):
    """
    WRITEME
    """
    freqs = []
    features = []

    n_instances = dataset.shape[0]

    # assign clusters randomly to instances
    if rand_state is None:
        rand_state = numpy.random.RandomState(RND_SEED)

    # inst_2_clusters = numpy.random.randint(0, n_clusters, n_instances)

    # getting the indices for each cluster
    # this all stuff could be done with a single loop
    clusters = [[] for i in range(n_clusters)]
    # for instance in range(n_instances):
    #     rand_cluster = rand_state.randint(0, n_clusters)
    #     clusters[rand_cluster].append(instance)

    instance_ids = numpy.arange(n_instances)
    rand_state.shuffle(instance_ids)
    print(instance_ids)
    for i in range(n_instances):
        clusters[i % n_clusters].append(instance_ids[i])

    # now we can operate cluster-wise
    for cluster_ids in clusters:
        # collecting all the data for the cluster
        cluster_data = dataset[cluster_ids, :]
        # count the frequencies for the var values
        cluster_freqs, cluster_features = data_2_freqs(cluster_data)
        # updating stats
        features = update_feature_count(features, cluster_features)
        freqs.extend(cluster_freqs)

    return freqs, features


def merge_datasets(dataset_name,
                   shuffle=True,
                   path=DATA_PATH,
                   sep=',',
                   type='int32',
                   suffixes=['.ts.data',
                             '.valid.data',
                             '.test.data'],
                   savetxt=True,
                   out_path=DATA_FULL_PATH,
                   output_suffix='.all.data',
                   rand_gen=None):
    """
    Merging portions of a dataset
    Loading them from file and optionally writing them to file
    """
    dataset_parts = load_train_val_test_csvs(dataset_name,
                                             path,
                                             sep,
                                             type,
                                             suffixes)

    print('Loaded dataset parts for', dataset_name)

    #
    # checking features
    assert len(dataset_parts) > 0
    first_dataset = dataset_parts[0]
    n_features = first_dataset.shape[1]
    for dataset_p in dataset_parts:
        assert dataset_p.shape[1] == n_features

    print('\tFeatures are conform')

    #
    # storing instances
    n_instances = [dataset_p.shape[0]
                   for dataset_p in dataset_parts]

    #
    # merging
    merged_dataset = numpy.concatenate(dataset_parts)
    print('\tParts merged')

    #
    # shuffling
    if shuffle:
        if rand_gen is None:
            rand_gen = numpy.random.RandomState(RND_SEED)
        rand_gen.shuffle(merged_dataset)
        print('\tShuffled')

    #
    #
    tot_n_instances = sum(n_instances)
    assert merged_dataset.shape[0] == tot_n_instances

    #
    # writing out
    if savetxt:
        out_path = out_path + dataset_name + output_suffix
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        fmt = '%.8e'
        if 'int' in type:
            fmt = '%d'

        numpy.savetxt(out_path, merged_dataset, delimiter=sep, fmt=fmt)
        print('\tMerged Dataset saved to', out_path)

    return merged_dataset


def shuffle_columns(data, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    data = numpy.array(data)
    n_features = data.shape[1]
    for i in range(n_features):
        rand_gen.shuffle(data[:, i])

    return data


def random_binary_dataset(n_instances, n_features, perc=0.5, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    data = rand_gen.binomial(1, p=perc, size=(n_instances, n_features))
    return data


def split_into_folds(dataset,
                     n_folds=10,
                     percentages=[0.81, 0.09, 0.1]):
    """
    Splitting a dataset into N folds (e.g. for cv)
    and optionally each fold into train-valid-test
    """
    raise NotImplementedError('split_inot_folds not implemented yet')


def add_background_binary_noise(dataset_split,
                                theta=0.2,
                                invert=False,
                                rand_gen=None,
                                int_type='int32'):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_instances = dataset_split.shape[0]
    n_features = dataset_split.shape[1]

    #
    # binary noise comes from a bernoulli with parameter theta
    N = rand_gen.binomial(1, theta, (n_instances, n_features))

    noised_split = None
    if invert:
        noised_split = numpy.logical_and(dataset_split, N).astype(int_type)
    else:
        noised_split = numpy.logical_or(dataset_split, N).astype(int_type)

    return noised_split


def add_background_binary_noise_splits(dataset,
                                       theta=0.2,
                                       invert=False,
                                       rand_gen=None,
                                       int_type='int32'):
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    noised_dataset = []
    for split in dataset:
        noised_split = None
        #
        # labels are present?
        if len(split) == 2 and isinstance(split, tuple):
            split_x, split_y = split
            noised_split = (add_background_binary_noise(split_x, theta, invert, rand_gen, int_type),
                            split_y)
        else:
            noised_split = add_background_binary_noise(split, theta, invert, rand_gen, int_type)

        noised_dataset.append(noised_split)

    return tuple(noised_dataset)


def get_split(fold, name, split_names=SPLIT_NAMES):
    for i, s in enumerate(split_names):
        if s == name:
            logging.info('Extracting split {} ({})'.format(i, SPLIT_NAMES[i]))
            break
    return fold[i]

DIST_TYPES = ['continuous', 'discrete', 'categorical']
RAND_SEED = 1337
FEATURE_FAMILIES = {'continuous': ['normal', 'beta', 'gamma', 'exponential', 'gumbel'],
                    'discrete': ['geometric', 'poisson', 'binomial'],
                    'categorical': ['bernoulli', 'categorical']}
FAMILY_PARAMETER_RANGES = {'normal': {'loc': [-100, 100], 'scale': [0, 3]},
                           'beta': {'a': [0, 20], 'b': [0, 20]},
                           'gamma': {'shape': [0, 20], 'scale': [0, 20]},
                           'exponential': {'scale': [0, 10]},
                           'gumbel': {'loc': [-10, 10], 'scale': [0, 3]},
                           'geometric': {'p': [0, 1]},
                           'poisson': {'lam': [0, 100]},
                           'binomial': {'n': [0, 100], 'p': [0, 1]},
                           'bernoulli': {'p': [0, 1]},
                           'categorical': {'k': [2, 100]}}


def sample_feature_param(distribution, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    params = None
    if distribution == 'binomial':
        params = {'n': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['binomial']['n'])).astype(int))}
        params['p'] = rand_gen.rand()
    elif distribution == 'poisson':
        params = {'lam': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['poisson']['lam'])).astype(int))}
    elif distribution == 'categorical':
        params = {'k': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['categorical']['k'])).astype(int))}
        params['p'] = rand_gen.rand(params['k'])
        params['p'] = params['p'] / sum(params['p'])
    else:
        #
        # sample uniformly
        params = {}
        for p_name, p_range in FAMILY_PARAMETER_RANGES[distribution].items():
            assert len(p_range) == 2
            params[p_name] = rand_gen.uniform(p_range[0], p_range[1])

    return params


def sample_distribution(distribution, params, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    if distribution == 'bernoulli':
        return rand_gen.binomial(n=1, p=params['p'])
    elif distribution == 'categorical':
        return rand_gen.choice(params['k'], p=params['p'])
    else:
        try:
            return getattr(rand_gen, distribution)(**params)
        except:
            raise ValueError('Unrecognized distribution {}'.format(distribution))


def generate_random_instance_indep(feature_families, params, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_features = len(feature_families)
    rand_instance = numpy.array([sample_distribution(feature_families[j],
                                                     params[j],
                                                     rand_gen)
                                 for j in range(n_features)])
    return rand_instance


def generate_random_instance_indep_mixture(feature_families,
                                           params,
                                           cluster_priors,
                                           rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_features = len(feature_families)
    assert len(cluster_priors) == n_features

    rand_instance = []
    instance_families = []
    for j in range(n_features):
        f = rand_gen.choice(numpy.arange(len(cluster_priors[j])),
                            p=cluster_priors[j])
        x_j = sample_distribution(feature_families[j][f],
                                  params[j][f],
                                  rand_gen)
        instance_families.append(feature_families[j][f])
        rand_instance.append(x_j)
    rand_instance = numpy.array(rand_instance)

    return rand_instance, instance_families


def generate_indep_synthetic_hybrid_random_dataset(n_instances,
                                                   n_features=10,
                                                   type_priors=None,
                                                   family_priors=None,
                                                   rand_gen=None,
                                                   dtype=numpy.float64):
    """
    Synthesize a dataset of M instances and N features such that each feature
    type is drawn from a prior distribution over types,
    then a family for that feature is drawn from a prior distribution
    according to the previously selected type.
    Each instance is sampled i.i.d from the joint probability distribution
    of all features considered independent one from the other
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    #
    # generating priors
    n_types = len(DIST_TYPES)
    if type_priors is None:
        type_priors = numpy.ones(n_types) / n_types
        print('Uniform priors for types {}'.format(type_priors))

    if family_priors is None:
        family_priors = {type: (numpy.ones(len(families)) / len(families))
                         for type, families in FEATURE_FAMILIES.items()}
        print('Uniform priors for families {}'.format(family_priors))

    #
    # sampling a RV type
    feature_types = [rand_gen.choice(DIST_TYPES, p=type_priors)
                     for i in range(n_features)]

    #
    # each feature comes from a single RV family
    feature_families = [rand_gen.choice(FEATURE_FAMILIES[t], p=family_priors[t])
                        for t in feature_types]

    assert len(feature_types) == len(feature_families)
    print('Sampled feature types {}'.format(feature_types))
    print('Sampled feature families {}'.format(feature_families))

    #
    # sampling parameters for distributions
    feature_params = [sample_feature_param(f, rand_gen)
                      for f in feature_families]
    print('Sampled feature params {}'.format([(f, p)
                                              for f, p in zip(feature_families,
                                                              feature_params)]))

    rand_data = numpy.zeros((n_instances, n_features), dtype=dtype)
    for i in range(n_instances):
        rand_data[i] = generate_random_instance_indep(feature_families,
                                                      feature_params,
                                                      rand_gen)

    print(rand_data)
    return rand_data


def generate_indep_mixture_synthetic_hybrid_random_dataset(n_instances,
                                                           n_features=10,
                                                           n_clusters=None,
                                                           type_priors=None,
                                                           cluster_priors=None,
                                                           family_priors=None,
                                                           rand_gen=None,
                                                           dtype=numpy.float64):
    """
    Synthesize a dataset of M instances and N features such that each feature
    type is drawn from a prior distribution over types.
    Then for each feature one determines the number of different families (clusters) for that feature.
    After that, a family for each cluster, for a feature, is drawn from a prior distribution
    according to the previously selected type.
    Each instance is sampled i.i.d from the joint probability distribution
    of all features considered independent one from the other.
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    #
    # assuming 2 clusters per features if not specified
    if n_clusters is None:
        n_clusters = numpy.zeros(n_features).astype(int)
        n_clusters[:] = 2
    print('Feature clusters {}'.format(n_clusters))

    #
    # generating priors
    n_types = len(DIST_TYPES)
    if type_priors is None:
        type_priors = numpy.ones(n_types) / n_types
        print('Uniform priors for types {}'.format(type_priors))

    if cluster_priors is None:
        cluster_priors = []
        for nc in n_clusters:
            c_prior = rand_gen.rand(nc)
            cluster_priors.append(c_prior / c_prior.sum())
        print('Uniform priors for feature clusters {}'.format(cluster_priors))

    if family_priors is None:
        family_priors = {type: (numpy.ones(len(families)) / len(families))
                         for type, families in FEATURE_FAMILIES.items()}
        print('Uniform priors for families {}'.format(family_priors))

    #
    # sampling cluster numbers
    # cluster_numbers = [rand_gen.choice(nc, p=nc_prior)
    #                    for nc, nc_prior in zip(n_clusters,
    #                                            cluster_priors)]

    #
    # sampling a RV type
    feature_types = [rand_gen.choice(DIST_TYPES, p=type_priors)
                     for i in range(n_features)]
    print('Sampled feature types {}'.format(feature_types))

    #
    # each feature can have different feature families
    feature_families = [rand_gen.choice(FEATURE_FAMILIES[t],
                                        p=family_priors[t],
                                        size=nc,
                                        replace=False)
                        for t, nc in zip(feature_types, n_clusters)]

    assert len(feature_types) == len(feature_families)
    print('Sampled feature families {}'.format(feature_families))

    #
    # sampling parameters for distributions
    feature_params = [[sample_feature_param(f, rand_gen) for f in ff]
                      for ff in feature_families]
    print('Sampled feature params {}'.format([(f, p)
                                              for f, p in zip(feature_families,
                                                              feature_params)]))

    rand_data = numpy.zeros((n_instances, n_features), dtype=dtype)
    for i in range(n_instances):
        rand_data[i], rand_families = generate_random_instance_indep_mixture(feature_families,
                                                                             feature_params,
                                                                             cluster_priors,
                                                                             rand_gen)
        print('{}\t{}'.format(i, '\t'.join(f for f in rand_families)))

    print(rand_data)
    return rand_data

def load_feature_scheme(scheme_path):
    feature_scheme = numpy.loadtxt(scheme_path, dtype=int).astype(int)
    return feature_scheme
