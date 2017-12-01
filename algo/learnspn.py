
import numpy

import numba

from scipy.misc import logsumexp

import sys

import itertools

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from spn import MARG_IND
from spn import LOG_ZERO
from spn import RND_SEED

from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CLTreeNode

from spn.factory import SpnFactory

from collections import deque

import math

import logging

import sklearn.mixture

from algo.dataslice import DataSlice

import dataset

# import tests

NEG_INF = -sys.float_info.max


@numba.njit
def count_nonzero(array):
    nonzero_coords, = numpy.nonzero(array)
    return len(nonzero_coords)


@numba.njit
def g_test(feature_id_1,
           feature_id_2,
           instance_ids,
           data,
           feature_vals,
           g_factor):
    """
    Applying a G-test on the two features (represented by ids) on the data
    """
    # print(feature_id_1, feature_id_2, instance_ids)

    #
    # swap to preserve order, is this needed?
    if feature_id_1 > feature_id_2:
        #
        # damn numba this cannot be done lol
        # feature_id_1, feature_id_2 = feature_id_2, feature_id_1
        tmp = feature_id_1
        feature_id_1 = feature_id_2
        feature_id_2 = tmp

    # print(feature_id_1, feature_id_2, instance_ids)

    n_instances = len(instance_ids)

    feature_size_1 = feature_vals[feature_id_1]
    feature_size_2 = feature_vals[feature_id_2]

    #
    # support vectors for counting the occurrences
    feature_tot_1 = numpy.zeros(feature_size_1, dtype=numpy.uint32)
    feature_tot_2 = numpy.zeros(feature_size_2, dtype=numpy.uint32)
    co_occ_matrix = numpy.zeros((feature_size_1, feature_size_2),
                                dtype=numpy.uint32)

    #
    # counting for the current instances
    for i in instance_ids:
        co_occ_matrix[data[i, feature_id_1], data[i, feature_id_2]] += 1

    # print('Co occurrences', co_occ_matrix)
    #
    # getting the sum for each feature
    for i in range(feature_size_1):
        for j in range(feature_size_2):
            count = co_occ_matrix[i, j]
            feature_tot_1[i] += count
            feature_tot_2[j] += count

    # print('Feature tots', feature_tot_1, feature_tot_2)

    #
    # counputing the number of zero total co-occurrences for the degree of
    # freedom
    feature_nonzero_1 = count_nonzero(feature_tot_1)
    feature_nonzero_2 = count_nonzero(feature_tot_2)

    dof = (feature_nonzero_1 - 1) * (feature_nonzero_2 - 1)

    g_val = numpy.float64(0.0)

    for i, tot_1 in enumerate(feature_tot_1):
        for j, tot_2 in enumerate(feature_tot_2):
            count = co_occ_matrix[i, j]
            if count != 0:
                exp_count = tot_1 * tot_2 / n_instances
                g_val += count * math.log(count / exp_count)

    # g_val *= 2

    # testing against p value
    # dep_val = 2 * dof * g_factor + 0.001
    dep_val = numpy.float64(2 * dof * g_factor + 0.001)

    # logging.info('\t[G: %f dep-val: %f]', g_val, dep_val)
    # print("(", feature_id_1, feature_id_2, ") G:", g_val, "dep_val:", dep_val)
    # return g_val < dep_val
    return (2 * g_val) < dep_val


@numba.jit
def greedy_feature_split(data,
                         data_slice,
                         feature_vals,
                         g_factor,
                         rand_gen):
    """
    WRITEME
    """
    n_features = data_slice.n_features()

    feature_ids_mask = numpy.ones(n_features, dtype=bool)

    #
    # extracting one feature at random
    rand_feature_id = rand_gen.randint(0, n_features)
    feature_ids_mask[rand_feature_id] = False

    dependent_features = numpy.zeros(n_features, dtype=bool)
    dependent_features[rand_feature_id] = True

    # greedy bfs searching
    features_to_process = deque()
    features_to_process.append(rand_feature_id)

    while features_to_process:
        # get one
        current_feature_id = features_to_process.popleft()
        feature_id_1 = data_slice.feature_ids[current_feature_id]
        # print('curr FT', current_feature_id)

        # features to remove later
        features_to_remove = numpy.zeros(n_features, dtype=bool)

        for other_feature_id in feature_ids_mask.nonzero()[0]:

            #
            # print('considering other features', other_feature_id)
            feature_id_2 = data_slice.feature_ids[other_feature_id]
            #
            # apply a G-test
            if not g_test(feature_id_1,
                          feature_id_2,
                          data_slice.instance_ids,
                          data,
                          feature_vals,
                          g_factor):

                #
                # updating 'sets'
                features_to_remove[other_feature_id] = True
                dependent_features[other_feature_id] = True
                features_to_process.append(other_feature_id)

        # now removing from future considerations
        feature_ids_mask[features_to_remove] = False

    # translating remaining features
    first_component = data_slice.feature_ids[dependent_features]
    second_component = data_slice.feature_ids[~ dependent_features]

    return first_component, second_component


def retrieve_clustering(assignment, indexes=None):
    """
    from [2, 3, 8, 3, 1]
    to [{0}, {1, 3}, {2}, {3}]

    or

    from [2, 3, 8, 3, 1] and [21, 1, 4, 18, 11]
    to [{21}, {1, 18}, {4}, {11}]

    """

    clustering = []
    seen_clusters = dict()

    if indexes is None:
        indexes = [i for i in range(len(assignment))]

    for index, label in zip(indexes, assignment):
        if label not in seen_clusters:
            seen_clusters[label] = len(clustering)
            clustering.append([])
        clustering[seen_clusters[label]].append(index)

    # if len(clustering) < 2:
    #     print('\n\n\n\n\n\nLess than 2 clusters\n\n\n\n\n\n\n')
    # assert len(clustering) > 1
    return clustering


def balanced_random_binary_split(data, rand_gen):
    n_instances = data.shape[0]
    #
    # placing half instances in a cluster
    clustering = numpy.zeros(n_instances, dtype=int)
    clustering[:-(n_instances // 2)] = 1
    rand_gen.shuffle(clustering)

    ones = clustering.sum()
    logging.info('Random binary split #clus.0/#clus.1: {}/{}'.format(data.shape[0] - ones,
                                                                     ones))
    return clustering


def random_binary_or_split(data, rand_gen):
    n_features = data.shape[1]
    rand_feature = rand_gen.choice(n_features)
    clustering = data[:, rand_feature]
    ones = clustering.sum()
    logging.info('Random binary split OR on feature {} #0/#1: {}/{}'.format(rand_feature,
                                                                            data.shape[0] - ones,
                                                                            ones))
    return clustering


def random_binary_xor_split(data, rand_gen, n_rand_features=None, max_trials=10):
    n_features = data.shape[1]
    if n_rand_features is None:
        n_rand_features = max(int(numpy.sqrt(n_features)), 2)

    clustering = None
    i = 0
    while clustering is None:
        rand_features = rand_gen.choice(n_features, n_rand_features, replace=False)

        clustering = (data[:, rand_features].sum(axis=1)) % 2
        ones = clustering.sum()
        i += 1
        if ones == 0 or ones == data.shape[0]:
            if i > max_trials:
                logging.info('taking another random shot')
                clustering = random_binary_or_split(data, rand_gen)
            else:
                logging.info('taking another random shot')
                clustering = None

    logging.info('Random binary XOR split on features {} #0/#1: {}/{}'.format(rand_features,
                                                                              data.shape[0] - ones,
                                                                              ones))
    return clustering


# @numba.jit
def gmm_clustering(data, n_clusters=2, n_iters=100, n_restarts=3, cov_type='diag',  rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    #
    # creating the cluster from sklearn
    gmm_c = sklearn.mixture.GaussianMixture(n_components=n_clusters,
                                            covariance_type=cov_type,
                                            random_state=rand_gen,
                                            max_iter=n_iters,
                                            n_init=n_restarts)

    #
    # fitting to training set
    fit_start_t = perf_counter()
    gmm_c.fit(data)
    fit_end_t = perf_counter()
    logging.debug('GMM clustering fitted in {}'.format(fit_end_t - fit_start_t))

    #
    # getting the cluster assignment
    pred_start_t = perf_counter()
    clustering = gmm_c.predict(data)
    pred_end_t = perf_counter()
    logging.debug('Cluster assignment predicted in {}'.format(pred_end_t - pred_start_t))

    if n_clusters == 2:
        ones = clustering.sum()
        logging.info('Binary GMM split #0/#1: {}/{}'.format(data.shape[0] - ones,
                                                            ones))

    return clustering


def uncertain_gmm_clustering(data, n_clusters=2, threshold=0.7, n_iters=100, n_restarts=3, cov_type='diag',  rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    #
    # first perform GMM clustering
    gmm_c = sklearn.mixture.GaussianMixture(n_components=n_clusters,
                                            covariance_type=cov_type,
                                            random_state=rand_gen,
                                            max_iter=n_iters,
                                            n_init=n_restarts)

    #
    # fitting to training set
    fit_start_t = perf_counter()
    gmm_c.fit(data)
    fit_end_t = perf_counter()
    logging.debug('UGMM clustering fitted in {}'.format(fit_end_t - fit_start_t))

    #
    # getting the cluster probabilities
    n_instances = data.shape[0]
    pred_start_t = perf_counter()
    clustering_probs = gmm_c.predict_proba(data)
    hard_clustering = numpy.argmax(clustering_probs, axis=1)
    assert hard_clustering.shape[0] == n_instances
    max_probs = clustering_probs[numpy.arange(n_instances), hard_clustering]
    print('max probs', max_probs)
    unc_clustering = max_probs < threshold

    clustering = gmm_c.predict(data)
    unc_cluster_id = max(clustering) + 1
    print('# cluster', unc_cluster_id)
    #
    # assigning to uncertain cluster
    clustering[unc_clustering] = unc_cluster_id
    print(clustering)

    pred_end_t = perf_counter()
    logging.debug('Cluster assignment predicted in {}'.format(pred_end_t - pred_start_t))

    #
    # then, make

    if n_clusters == 2:
        ones = clustering.sum()
        logging.info('Binary UGMM split #0/#1: {}/{}'.format(data.shape[0] - ones,
                                                             ones))

    return clustering


def gmm_clustering_xor_projection(data, n_proj_components=None,
                                  n_rand_features=None,
                                  n_clusters=2, n_iters=100, n_restarts=3, cov_type='diag',
                                  rand_gen=None):
    """
    First project the original data (n_instances x n_features) into a smaller space
    (n_instances, n_proj_components) by performing n_proj_components XOR operations
    each one on n_rand_features original features
    """
    n_instances = data.shape[0]
    n_features = data.shape[1]

    if n_proj_components is None:
        n_proj_components = max(int(numpy.sqrt(n_features)), 2)

    if n_rand_features is None:
        n_rand_features = max(int(numpy.sqrt(n_features)), 2)

    clustering = None

    proj_data = numpy.zeros((n_instances, n_proj_components), dtype=data.dtype)

    for j in range(n_proj_components):

        n_r = rand_gen.choice(numpy.arange(2, n_rand_features + 1))
        rand_features = rand_gen.choice(n_features, n_r, replace=False)

        proj_data[:, j] = (data[:, rand_features].sum(axis=1)) % 2

    #
    # clustering proj data
    clustering = gmm_clustering(proj_data, n_clusters=n_clusters,
                                n_iters=n_iters, n_restarts=n_restarts)
    if n_clusters == 2:
        ones = clustering.sum()
        logging.info('XOR-projected Binary GMM split #0/#1: {}/{}'.format(n_instances - ones,
                                                                          ones))

        #
        # FIXME
        if ones == 0 or ones == data.shape[0]:
            clustering = gmm_clustering(data, n_clusters=n_clusters,
                                        n_iters=n_iters, n_restarts=n_restarts)
    return clustering


def best_density_or_split(data):
    densities = data.sum(axis=0) / data.shape[0]
    best_split = numpy.argmin(numpy.abs(densities - 0.5))
    clustering = data[:, best_split]
    ones = clustering.sum()
    logging.info('Best dense binary split on feature {} #0/#1: {}/{}'.format(best_split,
                                                                             data.shape[0] - ones,
                                                                             ones))
    return clustering


def random_density_or_split(data, rand_gen):
    n_features = data.shape[1]
    densities = data.sum(axis=0) / data.shape[0]
    rand_probs = numpy.abs(densities - 0.5)
    rand_probs = rand_probs / rand_probs.sum()
    best_split = rand_gen.choice(numpy.arange(n_features), p=rand_probs)
    clustering = data[:, best_split]
    ones = clustering.sum()
    logging.info('Rand dense binary split on feature {} #0/#1: {}/{}'.format(best_split,
                                                                             data.shape[0] - ones,
                                                                             ones))
    return clustering


def cluster_rows(data,
                 data_slice,
                 n_clusters=2,
                 cluster_method='GMM',
                 n_iters=100,
                 n_restarts=3,
                 cluster_penalty=1.0,
                 rand_gen=None,
                 sklearn_args=None):
    """
    A wrapper to abstract from the implemented clustering method

    cluster_method = GMM | DPGMM | HOEM
    """

    clustering = None

    #
    # slicing the data
    sliced_data = data[data_slice.instance_ids, :][:, data_slice.feature_ids]

    if cluster_method == 'GMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        # #
        # # creating the cluster from sklearn
        # gmm_c = sklearn.mixture.GaussianMixture(n_components=n_clusters,
        #                                         covariance_type=cov_type,
        #                                         random_state=rand_gen,
        #                                         max_iter=n_iters,
        #                                         n_init=n_restarts)

        # #
        # # fitting to training set
        # fit_start_t = perf_counter()
        # gmm_c.fit(sliced_data)
        # fit_end_t = perf_counter()

        # #
        # # getting the cluster assignment
        # pred_start_t = perf_counter()
        # clustering = gmm_c.predict(sliced_data)
        # pred_end_t = perf_counter()
        fit_start_t = perf_counter()
        clustering = gmm_clustering(sliced_data, n_iters=n_iters, n_restarts=n_restarts,
                                    cov_type=cov_type, rand_gen=rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'UGMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'

        fit_start_t = perf_counter()
        clustering = uncertain_gmm_clustering(sliced_data,
                                              threshold=0.9,
                                              n_iters=n_iters, n_restarts=n_restarts,
                                              cov_type=cov_type, rand_gen=rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'XORGMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'

        fit_start_t = perf_counter()
        clustering = gmm_clustering_xor_projection(sliced_data,
                                                   n_proj_components=None,
                                                   n_rand_features=None,
                                                   n_iters=n_iters,
                                                   n_restarts=n_restarts,
                                                   cov_type=cov_type,
                                                   rand_gen=rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'DPGMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        verbose = sklearn_args['verbose']\
            if 'verbose' in sklearn_args else False

        dpgmm_c = sklearn.mixture.DPGMM(n_components=n_clusters,
                                        covariance_type=cov_type,
                                        random_state=rand_gen,
                                        n_iter=n_iters,
                                        alpha=cluster_penalty,
                                        verbose=verbose)

        #
        # fitting to training set
        fit_start_t = perf_counter()
        dpgmm_c.fit(sliced_data)
        fit_end_t = perf_counter()

        #
        # getting the cluster assignment
        pred_start_t = perf_counter()
        clustering = dpgmm_c.predict(sliced_data)
        pred_end_t = perf_counter()

    elif cluster_method == 'rand':
        fit_start_t = perf_counter()
        clustering = balanced_random_binary_split(sliced_data, rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'randOR':
        fit_start_t = perf_counter()
        clustering = random_binary_or_split(sliced_data, rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'randXOR':
        n_rand_features = sklearn_args['n_rand_features'] \
            if 'n_rand_features' in sklearn_args else None
        fit_start_t = perf_counter()
        clustering = random_binary_xor_split(sliced_data, rand_gen,
                                             n_rand_features=n_rand_features)
        fit_end_t = perf_counter()

    elif cluster_method == 'bestDEN':
        fit_start_t = perf_counter()
        clustering = best_density_or_split(sliced_data)
        fit_end_t = perf_counter()

    elif cluster_method == 'randDEN':
        fit_start_t = perf_counter()
        clustering = random_density_or_split(sliced_data, rand_gen)
        fit_end_t = perf_counter()

    elif cluster_method == 'HOEM':
        raise NotImplementedError('Hard Online EM is not implemented yet')
    else:
        raise Exception('Clustering method {} not valid'.format(cluster_method))

    logging.info('Clustering done in %f secs', (fit_end_t - fit_start_t))

    #
    # translating the cluster assignment to
    # a list of clusters (set of instances)

    return retrieve_clustering(clustering, data_slice.instance_ids)


def cache_data_slice(data_slice, cache):
    """
    WRITEME
    """
    #
    # getting ids
    instance_ids = data_slice.instance_ids
    feature_ids = data_slice.feature_ids
    #
    # ordering
    instance_ids.sort()
    feature_ids.sort()
    #
    # making unmutable
    instances_tuple = tuple(instance_ids)
    features_tuple = tuple(feature_ids)
    hashed_slice = (instances_tuple, features_tuple)
    #
    #
    cached_slice = None
    try:
        cached_slice = cache[hashed_slice]
    except:
        cache[hashed_slice] = data_slice

    return cached_slice


def estimate_kernel_density_spn(data_slice,
                                feature_sizes,
                                data,
                                alpha,
                                node_id_assoc,
                                building_stack,
                                slices_to_process):
    """
    A mixture with one component for each instance
    """

    instance_ids = data_slice.instance_ids
    feature_ids = data_slice.feature_ids
    current_id = data_slice.id

    n_instances = len(instance_ids)
    n_features = len(feature_ids)

    logging.info('Adding a kernel density estimation ' +
                 'over a slice {0} X {1}'.format(n_instances,
                                                 n_features))

    #
    # create sum node
    root_sum_node = SumNode(var_scope=frozenset(feature_ids))

    data_slice.type = SumNode
    building_stack.append(data_slice)

    root_sum_node.id = current_id
    node_id_assoc[current_id] = root_sum_node

    #
    # for each instance
    for i in instance_ids:
        #
        # create a slice
        instance_slice = DataSlice(numpy.array([i]), feature_ids)
        slices_to_process.append(instance_slice)
        #
        # linking with appropriate weight
        data_slice.add_child(instance_slice, 1.0 / n_instances)

    return root_sum_node, node_id_assoc, building_stack, slices_to_process


from collections import Counter
SCOPES_DICT = Counter()


class LearnSPN(object):

    """
    Implementing Gens' and Domingos' LearnSPN
    Plus variants on SPN-B/T/B
    """

    def __init__(self,
                 g_factor=1.0,
                 min_instances_slice=100,
                 min_features_slice=0,
                 alpha=0.1,
                 row_cluster_method='GMM',
                 cluster_penalty=2.0,
                 n_cluster_splits=2,
                 n_iters=100,
                 n_restarts=3,
                 sklearn_args={},
                 cltree_leaves=False,
                 kde_leaves=False,
                 rand_gen=None):
        """
        WRITEME
        """
        self._g_factor = g_factor
        self._min_instances_slice = min_instances_slice
        self._min_features_slice = min_features_slice
        self._alpha = alpha
        self._row_cluster_method = row_cluster_method
        self._cluster_penalty = cluster_penalty
        self._n_cluster_splits = n_cluster_splits
        self._n_iters = n_iters
        self._n_restarts = n_restarts
        self._sklearn_args = sklearn_args
        self._cltree_leaves = cltree_leaves
        self._kde = kde_leaves
        self._rand_gen = rand_gen if rand_gen is not None \
            else numpy.random.RandomState(RND_SEED)

        logging.info('LearnSPN:\n\tg factor:%f\n\tmin inst:%d\n' +
                     '\tmin feat:%d\n' +
                     '\talpha:%f\n\tcluster pen:%f\n\tn clusters:%d\n' +
                     '\tcluster method=%s\n\tn iters: %d\n' +
                     '\tn restarts: %d\n\tcltree leaves:%s\n' +
                     '\tsklearn args: %s\n',
                     self._g_factor,
                     self._min_instances_slice,
                     self._min_features_slice,
                     self._alpha,
                     self._cluster_penalty,
                     self._n_cluster_splits,
                     self._row_cluster_method,
                     self._n_iters,
                     self._n_restarts,
                     self._cltree_leaves,
                     self._sklearn_args)

    def make_naive_factorization(self,
                                 current_slice,
                                 slices_to_process,
                                 building_stack,
                                 node_id_assoc):

        logging.info('into a naive factorization')

        #
        # retrieving info from current slice
        current_instances = current_slice.instance_ids
        current_features = current_slice.feature_ids
        current_id = current_slice.id

        #
        # putting them in queue
        child_slices = [DataSlice(current_instances, [feature_id])
                        for feature_id in current_features]
        slices_to_process.extend(child_slices)

        children_ids = [child.id for child in child_slices]

        #
        # storing the children links
        for child_slice in child_slices:
            current_slice.add_child(child_slice)
        current_slice.type = ProductNode
        building_stack.append(current_slice)

        #
        # creating the product node
        prod_node = ProductNode(
            var_scope=frozenset(current_features))
        prod_node.id = current_id

        node_id_assoc[current_id] = prod_node
        logging.debug('\tCreated Prod Node %s (with children %s)',
                      prod_node,
                      children_ids)

        return current_slice, slices_to_process, building_stack, node_id_assoc

    def fit_structure(self,
                      data,
                      feature_sizes):
        """
        data is a numpy array of size {n_instances X n_features}
        feature_sizes is an array of integers representing feature ranges
        """

        #
        # resetting the data slice ids (just in case)
        DataSlice.reset_id_counter()

        tot_n_instances = data.shape[0]
        tot_n_features = data.shape[1]

        logging.info('Learning SPN structure on a (%d X %d) dataset',
                     tot_n_instances, tot_n_features)
        learn_start_t = perf_counter()

        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # creating the first slice
        whole_slice = DataSlice.whole_slice(tot_n_instances,
                                            tot_n_features)
        slices_to_process.append(whole_slice)

        first_run = True

        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            # pointers to the current data slice
            current_instances = current_slice.instance_ids
            current_features = current_slice.feature_ids
            current_id = current_slice.id

            n_instances = len(current_instances)
            n_features = len(current_features)

            logging.info('\n*** Processing slice %d (%d X %d)',
                         current_id,
                         n_instances, n_features)
            logging.debug('\tinstances:%s\n\tfeatures:%s',
                          current_instances,
                          current_features)

            #
            # is this a leaf node or we can split?
            if n_features == 1:
                logging.info('---> Adding a leaf (just one feature)')

                (feature_id, ) = current_features
                feature_size = feature_sizes[feature_id]

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # create the node
                leaf_node = CategoricalSmoothedNode(var=feature_id,
                                                    var_values=feature_size,
                                                    data=current_slice_data,
                                                    instances=current_instances,
                                                    alpha=self._alpha)
                # print('lnvf', leaf_node._var_freqs)
                # storing links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node

                logging.debug('\tCreated Smooth Node %s', leaf_node)

            elif (n_instances <= self._min_instances_slice and n_features > 1):
                #
                # splitting the slice on each feature
                logging.info('---> Few instances (%d), decompose all features',
                             n_instances)
                #
                # shall put a cltree or
                if self._cltree_leaves:
                    logging.info('into a Chow-Liu tree')
                    #
                    # slicing data
                    slice_data_rows = data[current_instances, :]
                    current_slice_data = slice_data_rows[:, current_features]

                    current_feature_sizes = [feature_sizes[i]
                                             for i in current_features]
                    #
                    # creating a Chow-Liu tree as leaf
                    leaf_node = CLTreeNode(vars=current_features,
                                           var_values=current_feature_sizes,
                                           data=current_slice_data,
                                           alpha=self._alpha)
                    #
                    # storing links
                    leaf_node.id = current_id
                    node_id_assoc[current_id] = leaf_node

                    logging.debug('\tCreated Chow-Liu Tree Node %s', leaf_node)

                elif self._kde and n_instances > 1:
                    estimate_kernel_density_spn(current_slice,
                                                feature_sizes,
                                                data,
                                                self._alpha,
                                                node_id_assoc,
                                                building_stack,
                                                slices_to_process)

                # elif n_instances == 1:  # FIXME: there is a bug here
                else:
                    current_slice, slices_to_process, building_stack, node_id_assoc = \
                        self.make_naive_factorization(current_slice,
                                                      slices_to_process,
                                                      building_stack,
                                                      node_id_assoc)
            else:

                #
                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                split_on_features = False
                #
                # first run is a split on rows
                if first_run:
                    logging.info('-- FIRST RUN --')
                    first_run = False
                else:
                    #
                    # try clustering on cols
                    # logging.debug('...trying to split on columns')
                    split_start_t = perf_counter()
                    print(data.shape)
                    dependent_features, other_features = greedy_feature_split(data,
                                                                              current_slice,
                                                                              feature_sizes,
                                                                              self._g_factor,
                                                                              self._rand_gen)
                    split_end_t = perf_counter()
                    logging.info('...tried to split on columns in {}'.format(split_end_t -
                                                                             split_start_t))
                    if len(other_features) > 0:
                        split_on_features = True
                #
                # have dependent components been found?
                if split_on_features:
                    #
                    # splitting on columns
                    logging.info('---> Splitting on features' +
                                 ' {} -> ({}, {})'.format(len(current_features),
                                                          len(dependent_features),
                                                          len(other_features)))

                    #
                    # creating two new data slices and putting them on queue
                    first_slice = DataSlice(current_instances,
                                            dependent_features)
                    second_slice = DataSlice(current_instances,
                                             other_features)
                    slices_to_process.append(first_slice)
                    slices_to_process.append(second_slice)

                    children_ids = [first_slice.id, second_slice.id]

                    #
                    # storing link parent children
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)
                    current_slice.add_child(first_slice)
                    current_slice.add_child(second_slice)

                    #
                    # creating product node
                    prod_node = ProductNode(var_scope=frozenset(current_features))
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

                else:
                    #
                    # clustering on rows
                    logging.info('---> Splitting on rows')

                    #
                    # at most n_rows clusters, for sklearn
                    k_row_clusters = min(self._n_cluster_splits,
                                         n_instances - 1)

                    clustering = cluster_rows(data,
                                              current_slice,
                                              n_clusters=k_row_clusters,
                                              cluster_method=self._row_cluster_method,
                                              n_iters=self._n_iters,
                                              n_restarts=self._n_restarts,
                                              cluster_penalty=self._cluster_penalty,
                                              rand_gen=self._rand_gen,
                                              sklearn_args=self._sklearn_args)

                    if len(clustering) < 2:
                        logging.info('\n\n\nLess than 2 clusters\n\n (%d)',
                                     len(clustering))

                        logging.info('forcing a naive factorization')
                        current_slice, slices_to_process, building_stack, node_id_assoc = \
                            self.make_naive_factorization(current_slice,
                                                          slices_to_process,
                                                          building_stack,
                                                          node_id_assoc)

                    else:
                        # logging.debug('obtained clustering %s', clustering)
                        logging.info('clustered into %d parts (min %d)',
                                     len(clustering), k_row_clusters)
                        # splitting
                        cluster_slices = [DataSlice(cluster, current_features)
                                          for cluster in clustering]
                        cluster_slices_ids = [slice.id
                                              for slice in cluster_slices]

                        # cluster_prior = 5.0
                        # cluster_weights = [(slice.n_instances() + cluster_prior) /
                        #                    (n_instances + cluster_prior * len(cluster_slices))
                        #                    for slice in cluster_slices]
                        cluster_weights = [slice.n_instances() / n_instances
                                           for slice in cluster_slices]

                        #
                        # appending for processing
                        slices_to_process.extend(cluster_slices)

                        #
                        # storing links
                        # current_slice.children = cluster_slices_ids
                        # current_slice.weights = cluster_weights
                        current_slice.type = SumNode
                        building_stack.append(current_slice)
                        for child_slice, child_weight in zip(cluster_slices,
                                                             cluster_weights):
                            current_slice.add_child(child_slice, child_weight)

                        #
                        # building a sum node
                        SCOPES_DICT[frozenset(current_features)] += 1
                        sum_node = SumNode(var_scope=frozenset(current_features))
                        sum_node.id = current_id
                        node_id_assoc[current_id] = sum_node
                        logging.debug('\tCreated Sum Node %s (with children %s)',
                                      sum_node,
                                      cluster_slices_ids)

        learn_end_t = perf_counter()

        logging.info('\n\n\tStructure learned in %f secs',
                     (learn_end_t - learn_start_t))

        #
        # linking the spn graph (parent -> children)
        #
        logging.info('===> Building tree')

        link_start_t = perf_counter()
        root_build_node = building_stack[0]
        root_node = node_id_assoc[root_build_node.id]
        logging.debug('root node: %s', root_node)

        root_node = SpnFactory.pruned_spn_from_slices(node_id_assoc,
                                                      building_stack)
        link_end_t = perf_counter()
        logging.info('\tLinked the spn in %f secs (root_node %s)',
                     (link_end_t - link_start_t),
                     root_node)

        #
        # building layers
        #
        logging.info('===> Layering spn')
        layer_start_t = perf_counter()
        spn = SpnFactory.layered_linked_spn(root_node)
        layer_end_t = perf_counter()
        logging.info('\tLayered the spn in %f secs',
                     (layer_end_t - layer_start_t))

        logging.info('\nLearned SPN\n\n%s', spn.stats())
        #logging.info('%s', SCOPES_DICT.most_common(30))

        return spn
