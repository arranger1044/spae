from .nodes import SumNode
from .nodes import ProductNode
from .nodes import CategoricalIndicatorNode
from .nodes import CategoricalSmoothedNode
from .nodes import mpe_states_from_leaf
from .nodes import eval_max_node
from .nodes import eval_prod_node

from spn import RND_SEED
from spn import MARG_IND
from spn import NEG_INF
from spn.linked.spn import evaluate_on_dataset
from spn.theanok.spn import evaluate_on_dataset_batch

from dataset import dataset_to_instances_set

from collections import deque
from collections import defaultdict
from collections import Counter
from collections import namedtuple

import numpy

import scipy
# import numba

import logging

import itertools

import os

import subprocess

import operator

from time import perf_counter

from numpy.testing import assert_array_equal


def node_in_path_feature(node, data_repr, node_feature_assoc, instance_id, in_path=True):
    """
    Feature is 1 if the node is in the path
    """
    feature_id = node_feature_assoc[node]

    if in_path:
        data_repr[instance_id, feature_id] = 1
    else:
        data_repr[instance_id, feature_id] = 0

    return data_repr


def acc_node_in_path_feature(node, data_repr, node_feature_assoc, instance_id, in_path=True):
    """
    Sets feature val to 1 plus previous val if it was on path
    """
    feature_id = node_feature_assoc[node]

    if in_path:
        data_repr[instance_id, feature_id] += 1

    return data_repr


def max_weight_feature(node, data_repr, node_feature_assoc, instance_id):

    feature_id = node_feature_assoc[node]
    max_weight = 0.0
    for i, child in enumerate(node.children):
        if numpy.isclose(child.log_val + node.log_weights[i],
                         node.log_val):
            max_weight = node.weights[i]
            break

    data_repr[instance_id, feature_id] = max_weight

    return data_repr


def max_child_id_feature(node, data_repr, node_feature_assoc, instance_id):

    max_child = None
    for i, child in enumerate(node.children):
        if numpy.isclose(child.log_val + node.log_weights[i],
                         node.log_val):
            max_child = child
            break

    try:
        feature_id = node_feature_assoc[max_child]
        data_repr[instance_id, feature_id] += 1
    except KeyError:
        pass

    return data_repr


def max_child_id_feature_cat(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):

        for i, child in enumerate(node.children):
            if numpy.isclose(child.log_val + node.log_weights[i],
                             node.log_val):
                break

        try:
            feature_id = node_feature_assoc[node]
            data_repr[instance_id, feature_id] = i
        except KeyError:
            raise ValueError('node {}'.format(node))

    # else:
    #     raise ValueError('Not sum node')
    return data_repr


def max_hidden_var_feature(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        max_child = None
        for i, child in enumerate(node.children):
            if numpy.isclose(child.log_val + node.log_weights[i],
                             node.log_val):
                max_child = child
                break

        try:
            feature_id = node_feature_assoc[(node, max_child)]
            data_repr[instance_id, feature_id] += 1
        except KeyError:
            pass

    return data_repr


def max_hidden_var_log_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        max_child = None
        max_val = None
        for i, child in enumerate(node.children):
            if numpy.isclose(child.log_val + node.log_weights[i],
                             node.log_val):
                max_child = child
                max_val = node.log_val
                break

        try:
            feature_id = node_feature_assoc[(node, max_child)]
            data_repr[instance_id, feature_id] = max_val
        except KeyError:
            pass

    return data_repr


def max_hidden_var_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        max_child = None
        max_val = None
        for i, child in enumerate(node.children):
            if numpy.isclose(child.log_val + node.log_weights[i],
                             node.log_val):
                max_child = child
                max_val = numpy.exp(node.log_val)
                break

        try:
            feature_id = node_feature_assoc[(node, max_child)]
            data_repr[instance_id, feature_id] = max_val
        except KeyError:
            pass

    return data_repr


def hidden_var_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        for i, child in enumerate(node.children):
            val = numpy.exp(child.log_val + node.log_weights[i])

            try:
                feature_id = node_feature_assoc[(node, child)]
                data_repr[instance_id, feature_id] = val
            except KeyError:
                pass

    return data_repr


def hidden_var_log_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        for i, child in enumerate(node.children):
            val = child.log_val + node.log_weights[i]

            try:
                feature_id = node_feature_assoc[(node, child)]
                data_repr[instance_id, feature_id] = val
            except KeyError:
                pass

    return data_repr


def child_var_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        for i, child in enumerate(node.children):
            val = numpy.exp(child.log_val)

            try:
                feature_id = node_feature_assoc[(node, child)]
                data_repr[instance_id, feature_id] = val
            except KeyError:
                pass

    return data_repr


def child_var_log_val(node, data_repr, node_feature_assoc, instance_id):

    if isinstance(node, SumNode):
        for i, child in enumerate(node.children):
            val = child.log_val

            try:
                feature_id = node_feature_assoc[(node, child)]
                data_repr[instance_id, feature_id] = val
            except KeyError:
                pass

    return data_repr


def var_val(node, data_repr, node_feature_assoc, instance_id):

    val = numpy.exp(node.log_val)

    try:
        feature_id = node_feature_assoc[node]
        data_repr[instance_id, feature_id] = val
    except KeyError:
        pass

    return data_repr


def var_log_val(node, data_repr, node_feature_assoc, instance_id):

    val = node.log_val

    try:
        feature_id = node_feature_assoc[node]
        data_repr[instance_id, feature_id] = val
    except KeyError:
        pass

    return data_repr


def log_output_feature(node, data_repr, node_feature_assoc, instance_id):

    try:
        feature_id = node_feature_assoc[node]
        data_repr[instance_id, feature_id] = node.log_val
    except KeyError:
        pass

    return data_repr


def filter_sum_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes() if isinstance(node, SumNode)]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_product_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes() if isinstance(node, ProductNode)]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_non_sum_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes() if not isinstance(node, SumNode)]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_non_prod_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes() if not isinstance(node, ProductNode)]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_non_leaf_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes() if isinstance(node, SumNode) or
                     isinstance(node, ProductNode)]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_all_nodes(spn):
    feature_nodes = [node for node in spn.top_down_nodes()]
    return {node: i for i, node in enumerate(feature_nodes)}


def filter_hidden_var_nodes(spn):
    feature_nodes = [(node, child) for node in spn.top_down_nodes()
                     if isinstance(node, SumNode)
                     for child in node.children]
    return {(node, child): i for i, (node, child) in enumerate(feature_nodes)}


def filter_hidden_var_cat_nodes(spn):
    # feature_nodes = [(node, child) for node in spn.top_down_nodes()
    #                  if isinstance(node, SumNode)
    #                  for child in node.children]
    # i = 0
    # nodes_id_assoc = {}
    # filtered_features = {}
    # for node, child in feature_nodes:

    #     if node not in nodes_id_assoc:
    #         nodes_id_assoc[node] = i
    #         i += 1

    #     feature_id = nodes_id_assoc[node]
    #     filtered_features[(node, child)] = feature_id
    # return filtered_features
    filtered_sum_nodes = filter_sum_nodes(spn)
    return {(node, child): i for node, i in filtered_sum_nodes.items()
            for child in node.children}


def save_feature_info(spn, node_feature_assoc, output_file):
    """
    Storing to file info about the extracted features for later reuse
    id, node id, layer id, node type, scope
    """
    header = 'id,node,layer,type,scope\n'
    with open(output_file, 'w') as info_file:
        info_file.write(header)
        node_layer_map = {node: layer for layer in spn.bottom_up_layers()
                          for node in layer.nodes()}
        sorted_features = sorted(node_feature_assoc.items(), key=operator.itemgetter(1))
        for node, feature_id in sorted_features:
            layer_id = node_layer_map[node].id
            node_type = node.__class__.__name__
            node_scope = ''
            if hasattr(node, 'var_scope'):
                node_scope = ' '.join(str(s) for s in sorted(node.var_scope))
            elif hasattr(node, 'var'):
                node_scope = str(node.var)
            info_str = '{},{},{},{},{}\n'.format(feature_id,
                                                 node.id,
                                                 layer_id,
                                                 node_type,
                                                 node_scope)
            info_file.write(info_str)


FeatureInfo = namedtuple('FeatureInfo', ['feature_id',
                                         'node_id',
                                         'layer_id',
                                         'node_type',
                                         'node_scope'])


def store_feature_info(feature_info, info_file_path):
    """
    Saving features info to file
    """
    header = 'id,node,layer,type,scope\n'
    with open(info_file_path, 'w') as info_file:
        info_file.write(header)
        #
        # ordering by feature id
        for info in sorted(feature_info, key=lambda x: x.feature_id):
            node_scope_str = ' '.join(str(s) for s in sorted(info.node_scope))
            info_str = '{},{},{},{},{}\n'.format(info.feature_id,
                                                 info.node_id,
                                                 info.layer_id,
                                                 info.node_type,
                                                 node_scope_str)
            info_file.write(info_str)


def load_feature_info(info_file_path):
    """
    Retrieving the feature info back from a text file
    """
    with open(info_file_path, 'r') as info_file:

        feature_info = []

        lines = info_file.readlines()
        #
        # discarding the header
        lines = lines[1:]
        for l in lines:
            feature_id, node_id, layer_id, node_type, node_scope = l.split(',')
            feature_id = int(feature_id)
            node_id = int(node_id)
            layer_id = int(layer_id)
            node_scope = set([int(s) for s in node_scope.rstrip().split(' ')])
            feature_info.append(FeatureInfo(feature_id, node_id, layer_id, node_type, node_scope))

        return feature_info


def filter_features_by_layer(feature_info, layer_id):
    """
    From a list of FeatureInfo filter belonging to a certain layer
    """
    filtered_info = [info for info in feature_info if info.layer_id == layer_id]
    return filtered_info


def filter_features_by_scope_length(feature_info, scope_length):
    """
    From a list of FeatureInfo filter those having a certain scope
    """
    filtered_info = [info for info in feature_info if len(info.node_scope) == scope_length]
    return filtered_info


def filter_features_by_node_type(feature_info, node_type_str):
    """
    From a list of FeatureInfo filter those having a certain node type
    """
    filtered_info = [info for info in feature_info if info.node_type == node_type_str]
    return filtered_info


def feature_mask_from_info(feature_info, n_features):
    """
    From a list of FeatureInfo extract a boolean mask
    for the features in them
    """

    feature_mask = numpy.zeros(n_features, dtype=bool)
    for info in feature_info:
        feature_mask[info.feature_id] = True

    return feature_mask


def extract_features_nodes_mpe(spn,
                               data,
                               nodes_id_assoc=None,
                               filter_node_func=filter_sum_nodes,
                               retrieve_func=node_in_path_feature,
                               remove_zero_features=True,
                               output_feature_info=None,
                               dtype=None,
                               verbose=False,
                               sparsify_mpe=None):
    """
    Representing a dataset (n_instances x n_features)
    in a new space (n_instances x n_spn_sum_nodes) where
    the new features are built according to a retrieve function
    (e.g. sum node id, its output signal, etc)
    and an spn on the sum nodes along the mpe path for each instance
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    #
    # storing assoc: sum node -> new feature id
    if nodes_id_assoc is None:
        logging.info('Filtering nodes with {}'.format(filter_node_func))
        nodes_id_assoc = filter_node_func(spn)
    n_spn_features = len(set(nodes_id_assoc.values()))

    #
    # save scopes and other info?
    # if output_feature_info is not None:

    #     node_id_assoc_n = None
    #     if filter_node_func == filter_hidden_var_nodes:
    #         node_id_assoc_n = {child: i for (node, child), i in nodes_id_assoc.items()}
    #     else:
    #         node_id_assoc_n = nodes_id_assoc

    #     save_feature_info(spn, node_id_assoc_n, output_feature_info)

    if dtype is None:
        dtype = data.dtype

    repr_data = numpy.zeros((n_instances, n_spn_features), dtype=dtype)

    logging.info('Old data ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                              n_features,
                                                              n_spn_features))
    if verbose:
        id_nodes_assoc = {v: k for k, v in nodes_id_assoc.items()}
        feature_nodes = []
        for v in sorted(id_nodes_assoc.keys()):
            try:
                node_id = id_nodes_assoc[v].id
            except:
                node_id = (id_nodes_assoc[v][0].id,
                           id_nodes_assoc[v][1].id)
            feature_nodes.append((v, node_id))
        print(feature_nodes)

    sparse_node_id_assoc = None
    if sparsify_mpe is not None:
        sparse_node_id_assoc = collect_mpe_paths(spn, data)
        # repr_data[:] = -1
    #
    # evaluate MPE circuit for each instance
    for i in range(n_instances):

        ext_start_t = perf_counter()
        #
        # bottom up evaluation
        spn.single_mpe_eval(data[i])

        #
        # "top down"" retrieval
        nodes_to_process = deque()

        for node in spn.root_layer().nodes():
            nodes_to_process.append(node)

        while nodes_to_process:

            curr_node = nodes_to_process.popleft()
            children_to_process = None

            if isinstance(curr_node, SumNode):
                #
                # retrieve the represented value for the feature
                repr_data = retrieve_func(curr_node, repr_data, nodes_id_assoc, i)

                #
                # following the max children
                children_to_process = [child for j, child in enumerate(curr_node.children)
                                       if numpy.isclose(child.log_val + curr_node.log_weights[j],
                                                        curr_node.log_val)]

            elif isinstance(curr_node, ProductNode):
                #
                # following all children
                children_to_process = [child for child in curr_node.children]

            if children_to_process:
                nodes_to_process.extend(children_to_process)

        ext_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_end_t - ext_start_t),
              end='           \r')

    # if remove_zero_features:

    #     old_n_features = repr_data.shape[1]
    #     zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
    #     features_to_keep = [i for i in range(n_spn_features)
    #                         if not numpy.allclose(zero_feature, repr_data[:, i])]
    #     repr_data = repr_data[:, numpy.array(features_to_keep)]

    #     logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
    #                                                                       old_n_features,
    #                                                                       repr_data.shape[1]))

    if sparsify_mpe is not None:
        logging.info('Sparsifying the repr data according to the MPE path')

        for node, feature_id in nodes_id_assoc.items():
            for i in range(n_instances):
                if node not in sparse_node_id_assoc[i]:
                    repr_data[i, feature_id] = sparsify_mpe

    if remove_zero_features is not None:

        logging.info('Removing non informative features (value: {})'.format(remove_zero_features))

        inv_nodes_id_assoc = {i: node for node, i in nodes_id_assoc.items()}
        nodes_id_assoc = {}

        old_n_features = repr_data.shape[1]
        zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
        zero_feature[:] = remove_zero_features
        # features_to_keep = [i for i in range(n_spn_features)
        #                     if not numpy.allclose(zero_feature, repr_data[:, i])]
        features_to_keep = []
        j = 0
        for i in range(n_spn_features):
            if not numpy.allclose(zero_feature, repr_data[:, i]):
                features_to_keep.append(i)
                nodes_id_assoc[inv_nodes_id_assoc[i]] = j
                # node = inv_nodes_id_assoc.pop(i)
                # inv_nodes_id_assoc[j] = node
                j += 1

        repr_data = repr_data[:, numpy.array(features_to_keep)]

        # nodes_id_assoc = {node: i for i, node in inv_nodes_id_assoc.items()}

        logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                                          old_n_features,
                                                                          repr_data.shape[1]))

    #
    # save scopes and other info?
    if output_feature_info is not None:

        node_id_assoc_n = None
        if filter_node_func == filter_hidden_var_nodes:
            node_id_assoc_n = {child: i for (node, child), i in nodes_id_assoc.items()}
        elif filter_node_func == filter_hidden_var_cat_nodes:
            node_id_assoc_n = {node: i for (node, child), i in nodes_id_assoc.items()}
        else:
            node_id_assoc_n = nodes_id_assoc

        save_feature_info(spn, node_id_assoc_n, output_feature_info)

    return repr_data, nodes_id_assoc


def extract_features_nodes(spn,
                           data,
                           nodes_id_assoc=None,
                           filter_node_func=filter_sum_nodes,
                           retrieve_func=node_in_path_feature,
                           remove_zero_features=True,
                           output_feature_info=None,
                           dtype=None,
                           verbose=False):
    """
    Representing a dataset (n_instances x n_features)
    in a new space (n_instances x n_spn_sum_nodes) where
    the new features are built according to a retrieve function
    (e.g. sum node id, its output signal, etc)
    given an spn and its evaluation bottom up
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    #
    # storing assoc: sum node -> new feature id
    if nodes_id_assoc is None:
        logging.info('Filtering nodes with {}'.format(filter_node_func))
        nodes_id_assoc = filter_node_func(spn)
    n_spn_features = len(nodes_id_assoc)

    #
    # save scopes and other info?
    if output_feature_info is not None:
        save_feature_info(spn, nodes_id_assoc, output_feature_info)

    if dtype is None:
        dtype = data.dtype

    repr_data = numpy.zeros((n_instances, n_spn_features), dtype=dtype)

    logging.info('Old data ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                              n_features,
                                                              n_spn_features))
    if verbose:
        id_nodes_assoc = {v: k for k, v in nodes_id_assoc.items()}
        feature_nodes = []
        for v in sorted(id_nodes_assoc.keys()):
            try:
                node_id = id_nodes_assoc[v].id
            except:
                node_id = (id_nodes_assoc[v][0].id,
                           id_nodes_assoc[v][1].id)
            feature_nodes.append((v, node_id))
        print(feature_nodes)
    #
    # evaluate MPE circuit for each instance
    ext_all_start_t = perf_counter()
    for i in range(n_instances):

        ext_start_t = perf_counter()
        #
        # bottom up evaluation
        spn.single_eval(data[i])

        #
        # visiting all nodes

        for node in spn.top_down_nodes():

            #
            # retrieve the represented value for the feature
            repr_data = retrieve_func(node, repr_data, nodes_id_assoc, i)
        ext_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_end_t - ext_start_t),
              end='           \r')

    ext_all_end_t = perf_counter()
    logging.info('Extraction done in {} secs'.format(ext_all_end_t - ext_all_start_t))

    if remove_zero_features:

        old_n_features = repr_data.shape[1]
        zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
        features_to_keep = [i for i in range(n_spn_features)
                            if not numpy.allclose(zero_feature, repr_data[:, i])]
        repr_data = repr_data[:, numpy.array(features_to_keep)]

        logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                                          old_n_features,
                                                                          repr_data.shape[1]))
    return repr_data, nodes_id_assoc


def collect_mpe_paths(mpn, data, verbose=False):
    """
    Return a sequence of sets of nodes, one for each instance in data,
    each set contains a node of the mpn if it appears in the descending path
    for the corresponding instance
    """

    logging.info('Collecting mpe paths')
    n_instances = data.shape[0]
    paths = []

    for i in range(n_instances):
        ext_start_t = perf_counter()

        #
        # evaluate the network
        mpn.single_mpe_eval(data[i])

        #
        # top down traversal
        path = set()
        nodes_to_process = deque()

        for node in mpn.root_layer().nodes():
            nodes_to_process.append(node)

        while nodes_to_process:
            curr_node = nodes_to_process.popleft()
            path.add(curr_node)

            if isinstance(curr_node, SumNode):
                max_child_val = NEG_INF
                max_child = None
                for j, child in enumerate(curr_node.children):
                    # path.add(child)
                    child_output = curr_node.log_weights[j] + child.log_val
                    if child_output > max_child_val:
                        max_child_val = child_output
                        # print(emb_val)
                        max_child = child
                nodes_to_process.append(max_child)
                if verbose:
                    print('max node, getting max child {0} [{1}]'.format(max_child,
                                                                         len(nodes_to_process)))
            #
            # adding them all if the current node is a product node
            elif isinstance(curr_node, ProductNode):
                nodes_to_process.extend(curr_node.children)
                if verbose:
                    print('prod node, add {0} children [{1}]'.format(len(curr_node.children),
                                                                     len(nodes_to_process)))
            else:
                if verbose:
                    print('reaching a leaf {}'.format(curr_node))

        paths.append(path)
        ext_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_end_t - ext_start_t),
              end='           \r')

    return paths


def extract_features_nodes_mpn(mpn,
                               data,
                               nodes_id_assoc=None,
                               filter_node_func=filter_sum_nodes,
                               retrieve_func=node_in_path_feature,
                               remove_zero_features=None,
                               output_feature_info=None,
                               dtype=None,
                               verbose=False,
                               sparsify_mpe=None):
    """
    Representing a dataset (n_instances x n_features)
    in a new space (n_instances x n_mpn_filtered_nodes) where
    the new features are built according to a retrieve function
    (e.g. sum node id, its output signal, etc)
    given an mpn and its evaluation bottom up

    NOTE: an mps is still a linked spn on which mpe eval is called
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    #
    # storing assoc: sum node -> new feature id
    if nodes_id_assoc is None:
        logging.info('Filtering nodes with {}'.format(filter_node_func))
        nodes_id_assoc = filter_node_func(mpn)
    n_spn_features = len(nodes_id_assoc)

    if dtype is None:
        dtype = data.dtype

    repr_data = numpy.zeros((n_instances, n_spn_features), dtype=dtype)

    sparse_node_id_assoc = None
    if sparsify_mpe is not None:
        sparse_node_id_assoc = collect_mpe_paths(mpn, data)
        # repr_data[:] = -1

    logging.info('Old data ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                              n_features,
                                                              n_spn_features))
    if verbose:
        id_nodes_assoc = {v: k for k, v in nodes_id_assoc.items()}
        feature_nodes = []
        for v in sorted(id_nodes_assoc.keys()):
            try:
                node_id = id_nodes_assoc[v].id
            except:
                node_id = (id_nodes_assoc[v][0].id,
                           id_nodes_assoc[v][1].id)
            feature_nodes.append((v, node_id))
        print(feature_nodes)
    #
    # evaluate MPE circuit for each instance
    ext_all_start_t = perf_counter()
    for i in range(n_instances):
        ext_start_t = perf_counter()
        #
        # bottom up evaluation
        mpn.single_mpe_eval(data[i])

        #
        # visiting all nodes
        for node in mpn.top_down_nodes():
            #
            # retrieve the represented value for the node if filtered
            # if not sparsify_mpe or node in sparse_node_id_assoc[i]:
            repr_data = retrieve_func(node, repr_data, nodes_id_assoc, i)
        ext_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_end_t - ext_start_t),
              end='           \r')

    ext_all_end_t = perf_counter()
    logging.info('Extraction done in {} secs'.format(ext_all_end_t - ext_all_start_t))

    if sparsify_mpe is not None:
        logging.info('Sparsifying the repr data according to the MPE path')

        for node, feature_id in nodes_id_assoc.items():
            for i in range(n_instances):
                if node not in sparse_node_id_assoc[i]:
                    repr_data[i, feature_id] = sparsify_mpe

    if remove_zero_features is not None:

        logging.info('Removing non informative features (value: {})'.format(remove_zero_features))

        inv_nodes_id_assoc = {i: node for node, i in nodes_id_assoc.items()}
        nodes_id_assoc = {}

        old_n_features = repr_data.shape[1]
        zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
        zero_feature[:] = remove_zero_features
        # features_to_keep = [i for i in range(n_spn_features)
        #                     if not numpy.allclose(zero_feature, repr_data[:, i])]
        features_to_keep = []
        j = 0
        for i in range(n_spn_features):
            if not numpy.allclose(zero_feature, repr_data[:, i]):
                features_to_keep.append(i)
                nodes_id_assoc[inv_nodes_id_assoc[i]] = j
                # node = inv_nodes_id_assoc.pop(i)
                # inv_nodes_id_assoc[j] = node
                j += 1

        repr_data = repr_data[:, numpy.array(features_to_keep)]

        # nodes_id_assoc = {node: i for i, node in inv_nodes_id_assoc.items()}

        logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                                          old_n_features,
                                                                          repr_data.shape[1]))

    #
    # save scopes and other info?
    if output_feature_info is not None:

        node_id_assoc_n = None
        if filter_node_func == filter_hidden_var_nodes:
            node_id_assoc_n = {child: i for (node, child), i in nodes_id_assoc.items()}
        elif filter_node_func == filter_hidden_var_cat_nodes:
            node_id_assoc_n = {node: i for (node, child), i in nodes_id_assoc.items()}
        else:
            node_id_assoc_n = nodes_id_assoc

        save_feature_info(mpn, node_id_assoc_n, output_feature_info)

    return repr_data, nodes_id_assoc


def extract_features_nodes_der(spn,
                               data,
                               nodes_id_assoc=None,
                               filter_node_func=filter_sum_nodes,
                               retrieve_func=None,
                               remove_zero_features=True,
                               output_feature_info=None,
                               dtype=None,
                               verbose=False):
    """
    Filtering and collecting embedding based on derivatives
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    #
    # storing assoc: sum node -> new feature id
    if nodes_id_assoc is None:
        logging.info('Filtering nodes with {}'.format(filter_node_func))
        nodes_id_assoc = filter_node_func(spn)
    n_spn_features = len(nodes_id_assoc)

    #
    # save scopes and other info?
    if output_feature_info is not None:
        save_feature_info(spn, nodes_id_assoc, output_feature_info)

    if dtype is None:
        dtype = data.dtype

    repr_data = numpy.zeros((n_instances, n_spn_features), dtype=dtype)

    logging.info('Old data ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                              n_features,
                                                              n_spn_features))
    if verbose:
        id_nodes_assoc = {v: k for k, v in nodes_id_assoc.items()}
        feature_nodes = []
        for v in sorted(id_nodes_assoc.keys()):
            try:
                node_id = id_nodes_assoc[v].id
            except:
                node_id = (id_nodes_assoc[v][0].id,
                           id_nodes_assoc[v][1].id)
            feature_nodes.append((v, node_id))
        print(feature_nodes)
    #
    # evaluate MPE circuit for each instance
    ext_all_start_t = perf_counter()
    for i in range(n_instances):

        ext_start_t = perf_counter()
        #
        # bottom up evaluation
        spn.single_eval(data[i])
        spn.backprop()

        #
        # visiting all nodes
        for node in spn.top_down_nodes():

            #
            # retrieve the represented value for the feature
            # repr_data = retrieve_func(node, repr_data, nodes_id_assoc, i)
            try:
                repr_data[i, nodes_id_assoc[node]] = node.log_der
            except KeyError:
                pass
        ext_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_end_t - ext_start_t),
              end='           \r')

    ext_all_end_t = perf_counter()
    logging.info('Extraction done in {} secs'.format(ext_all_end_t - ext_all_start_t))

    if remove_zero_features:

        old_n_features = repr_data.shape[1]
        zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
        features_to_keep = [i for i in range(n_spn_features)
                            if not numpy.allclose(zero_feature, repr_data[:, i])]
        repr_data = repr_data[:, numpy.array(features_to_keep)]

        logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                                          old_n_features,
                                                                          repr_data.shape[1]))
    return repr_data


def filter_2darray_2dmask(array, mask):
    print('filter shapes', array.shape, mask.shape)
    n_instances = array.shape[0]
    # n_features = array.shape[1]

    filtered = []
    for i in range(n_instances):
        assert mask[i].sum() == mask[0].sum()
        filtered.append(array[i, mask[i]])

    return numpy.array(filtered)


def generate_missing_masks(n_components, n_instances,
                           node_feature_assoc,
                           perc, one_mask=False, rand_gen=None):
    """
    From a map node.id -> embedding component id generate a bit mask
    (n_instances x n_components) where n_components is the length of the embedding
    and containing L bits set to 1 when L = (1 - perc) * n_features_in_assoc
    """
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_features = len(node_feature_assoc) if node_feature_assoc is not None else n_components
    n_to_remove = int(perc * n_features)

    missing_mask = numpy.ones((n_instances, n_components), dtype=bool)

    #
    # considering only the feature ids in node_feature assoc
    feature_ids = [v for v in node_feature_assoc.values()] \
        if node_feature_assoc is not None else [i for i in range(n_components)]
    feature_id_set = set(feature_ids)
    for j in range(n_components):
        if j not in feature_id_set:
            missing_mask[:, j] = 0

    # print('fid', len(feature_ids))

    if one_mask:
        missing_mask[:, rand_gen.choice(feature_ids, size=n_to_remove, replace=False)] = 0
        assert_array_equal(missing_mask.sum(axis=1), numpy.array([n_features - n_to_remove
                                                                  for i in range(n_instances)]))
    else:
        for i in range(n_instances):
            # print(missing_mask[i].sum())
            missing_mask[i, rand_gen.choice(feature_ids, size=n_to_remove, replace=False)] = 0

            # print(missing_mask[i].sum(), n_features, n_to_remove, (n_features - n_to_remove))
            assert missing_mask[i].sum() == (n_features - n_to_remove)

    return missing_mask


def remove_embedding_components(node_feature_assoc, perc, rand_gen=None):
    """
    From a map node.id -> feature id of length L remove ~ perc*L components
    """
    n_components = len(node_feature_assoc)
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_to_remove = max(int(perc * n_components), 1)
    to_remove = set(rand_gen.choice(n_components, n_to_remove, replace=False))

    rem_assoc = {k: v for i, (k, v) in enumerate(node_feature_assoc.items())
                 if i in to_remove}
    return rem_assoc


def remove_embedding_components_from_mask(node_feature_assoc, mask):
    """
    From a map node.id -> feature id of length L remove
    all the components that are not present in a bit mask of length L
    """
    n_components = len(node_feature_assoc)
    assert len(mask) == n_components

    rem_assoc = {k: v for i, (k, v) in enumerate(node_feature_assoc.items())
                 if mask[v]}

    assert len(rem_assoc) == mask.sum()
    return rem_assoc


def fill_missing_embeddings_mpe(embedding,
                                mpe_embedding,
                                feature_ids):
    """
    Computed an embedding by replacing all missing components in the input one
    (i.e. belonging to nodes not appearing in the feature ids set)
    with the corresponding ones in the mpe embedding
    """

    n_features = embedding.shape[0]
    assert mpe_embedding.shape[0] == n_features

    filled_embedding = numpy.array(embedding, copy=True)

    for i in range(n_features):
        if i not in feature_ids:
            filled_embedding[i] = mpe_embedding[i]

    return filled_embedding


def evaluate_mixed_mpe_embeddings(mpn,
                                  mixed_embedding,
                                  node_feature_assoc,
                                  full_node_feature_assoc):
    """
    Evaluate an MPN bottom-up, computing node activations only if that value
    is not already present into an embedding
    """
    # mixed_node_feature_assoc = dict(node_feature_assoc)
    for node in mpn.bottom_up_nodes():
        if node.id not in node_feature_assoc:
            node_f_id = full_node_feature_assoc[node.id]
            #
            # we have to compute the new value
            children_log_vals = None
            if isinstance(node, SumNode) or isinstance(node, ProductNode):
                children_log_vals = numpy.array([child.log_val
                                                 for child in node.children])
                for i, child in enumerate(node.children):
                    # if child not in node_feature_assoc:
                    children_log_vals[i] = mixed_embedding[full_node_feature_assoc[child.id]]

                # mixed_node_feature_assoc[node.id] = node_f_id
            if isinstance(node, SumNode):

                mixed_embedding[node_f_id] = eval_max_node(children_log_vals,
                                                           numpy.array([w for w in node.log_weights]))
            elif isinstance(node, ProductNode):
                mixed_embedding[node_f_id] = eval_prod_node(children_log_vals)
            #
            # it is a leaf, it is already set

    # return mixed_node_feature_assoc, mixed_embedding
    return mixed_embedding


def evaluate_mixed_mpe_categorical_embeddings(mpn,
                                              mixed_embedding,
                                              node_feature_assoc,
                                              full_node_feature_assoc):
    """
    Evaluate an MPN bottom-up, computing node activations only if that value
    is not already present into an embedding
    """
    # mixed_node_feature_assoc = dict(node_feature_assoc)
    for node in mpn.bottom_up_nodes():
        if node.id not in node_feature_assoc:
            node_f_id = full_node_feature_assoc[node.id]
            #
            # we have to compute the new value
            children_log_vals = None
            if isinstance(node, SumNode) or isinstance(node, ProductNode):
                children_log_vals = numpy.array([child.log_val
                                                 for child in node.children])
                for i, child in enumerate(node.children):
                    # if child not in node_feature_assoc:
                    children_log_vals[i] = mixed_embedding[full_node_feature_assoc[child.id]]

                # mixed_node_feature_assoc[node.id] = node_f_id
            if isinstance(node, SumNode):

                mixed_embedding[node_f_id] = eval_max_node(children_log_vals,
                                                           numpy.array([w for w in node.log_weights]))
            elif isinstance(node, ProductNode):
                mixed_embedding[node_f_id] = eval_prod_node(children_log_vals)
            #
            # it is a leaf, it is already set

    # return mixed_node_feature_assoc, mixed_embedding
    return mixed_embedding


def decode_embeddings_mpn(mpn,
                          embeddings,
                          node_feature_assoc,
                          n_features,
                          embedding_type='activations',
                          missing_masks=None,
                          root=None,
                          re_evaluate_mpe=False,
                          dtype=numpy.int32,
                          missing_emb_val=-1,
                          verbose=False):
    """
    Decode each instance embedding reconstructing an array in the instance feature space
    embedding_type in {'activations', 'latent_categorical', 'latent_binary'}
    """

    assert node_feature_assoc is not None
    assert embedding_type in {'activations', 'latent_categorical', 'latent_binary'}
    logging.info('Decoding embeddings of type: {}'.format(embedding_type))

    latent_var_sizes = None

    #
    # collecting all MPN embeddings from MPE inference,
    # to be used when a component from the embedding to be decoded is
    # missing
    marg_all_features_instance = numpy.zeros(n_features, dtype=int)
    marg_all_features_instance.fill(MARG_IND)
    mpn.mpe_eval(marg_all_features_instance)

    mpe_node_feature_assoc = None
    mpe_embedding = None
    if embedding_type == 'activations':
        mpe_node_feature_assoc = {node.id: i for i, node in enumerate(mpn.top_down_nodes())}
        mpe_embedding = numpy.zeros(len(mpe_node_feature_assoc))
        mpe_embedding[:] = NEG_INF
        for node in mpn.top_down_nodes():
            mpe_embedding[mpe_node_feature_assoc[node.id]] = node.log_val
        assert (mpe_embedding != NEG_INF).all()
    elif embedding_type == 'latent_categorical':
        # mpe_node_feature_assoc = {node.id: i for i, node in enumerate(mpn.top_down_nodes())
        #                           if isinstance(node, SumNode)}
        mpe_node_feature_assoc = {}
        j = 0
        for i, node in enumerate(mpn.top_down_nodes()):
            if isinstance(node, SumNode):
                if node.id not in mpe_node_feature_assoc:
                    mpe_node_feature_assoc[node.id] = j
                    j += 1
        mpe_embedding = numpy.zeros(len(mpe_node_feature_assoc)).astype(numpy.int32)
        mpe_embedding[:] = -1
        for node in mpn.top_down_nodes():
            if isinstance(node, SumNode):
                for i, child in enumerate(node.children):
                    if numpy.isclose(child.log_val + node.log_weights[i],
                                     node.log_val):
                        break
                mpe_embedding[mpe_node_feature_assoc[node.id]] = i
        assert (mpe_embedding != -1).all()
    elif embedding_type == 'latent_binary':
        raise NotImplementedError('Binary version not implemented')

    n_instances = embeddings.shape[0]

    decoded_instances = numpy.zeros((n_instances, n_features), dtype=dtype)

    # node_feature_assoc = {info.node_id: info.feature_id for info in feature_info}

    decode_all_start_t = perf_counter()
    for i in range(n_instances):

        embedding = embeddings[i]
        #
        # remove instances from embedding at random?
        if missing_masks is not None:
            node_feature_assoc_i = remove_embedding_components_from_mask(node_feature_assoc,
                                                                         missing_masks[i])
            if re_evaluate_mpe:
                mixed_embedding = fill_missing_embeddings_mpe(embedding,
                                                              mpe_embedding,
                                                              set(node_feature_assoc_i.values()))
                embedding = evaluate_mixed_mpe_embeddings(mpn,
                                                          mixed_embedding,
                                                          node_feature_assoc_i,
                                                          node_feature_assoc)
                node_feature_assoc_i = node_feature_assoc
        else:
            node_feature_assoc_i = node_feature_assoc

        decode_start_t = perf_counter()
        if embedding_type == 'activations':
            decode_instance_embedding_mpn(mpn,
                                          node_feature_assoc_i,
                                          embedding,
                                          decoded_instances[i],
                                          mpe_node_feature_assoc=mpe_node_feature_assoc,
                                          mpe_embedding=mpe_embedding,
                                          root=root,
                                          missing_emb_val=missing_emb_val,
                                          verbose=verbose)
        elif embedding_type == 'latent_categorical':
            decode_instance_latent_embedding_mpn(mpn,
                                                 node_feature_assoc_i,
                                                 embedding,
                                                 decoded_instances[i],
                                                 mpe_node_feature_assoc=mpe_node_feature_assoc,
                                                 mpe_embedding=mpe_embedding,
                                                 binary=False,
                                                 latent_var_sizes=None,
                                                 root=root,
                                                 missing_emb_val=missing_emb_val,
                                                 verbose=verbose)
        elif embedding_type == 'latent_binary':
            decode_instance_latent_embedding_mpn(mpn,
                                                 node_feature_assoc_i,
                                                 embedding,
                                                 decoded_instances[i],
                                                 mpe_node_feature_assoc=mpe_node_feature_assoc,
                                                 mpe_embedding=mpe_embedding,
                                                 binary=True,
                                                 latent_var_sizes=latent_var_sizes,
                                                 root=root,
                                                 missing_emb_val=missing_emb_val,
                                                 verbose=verbose)
        decode_end_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          decode_end_t - decode_start_t),
              end='           \r')

    decode_all_end_t = perf_counter()
    logging.info('\nDecoding done in {} secs'.format(decode_all_end_t - decode_all_start_t))

    return decoded_instances


def decode_leaf_node(node,
                     embedding,
                     node_feature_assoc,
                     decoded_repr,
                     verbose=False):

    n_features = len(embedding)

    if isinstance(node, CategoricalIndicatorNode):
        decoded_repr[node.var] = node.var_val
    else:
        present = True
        try:
            node_feature_id = node_feature_assoc[node.id]
        except:
            present = False
        if present and node_feature_id < n_features:
            #
            # if it is a univariate distribution, then
            emb_val = embedding[node_feature_id]
            if isinstance(node, CategoricalSmoothedNode):
                min_dist = numpy.inf
                closest_val = None
                for i, val in enumerate(node._var_probs):
                    dist = abs(emb_val - val)

                    if verbose:
                        print(i, 'dist', dist, emb_val, val)

                    if dist < min_dist:
                        min_dist = dist
                        closest_val = i

                decoded_repr[node.var] = closest_val
            else:
                raise ValueError('unrecognized leaf type', node)
        else:
            mpe_states = mpe_states_from_leaf(node, only_first_max=True)
            if verbose:
                print('Reached a leaf with MPE: {0}'.format(mpe_states))

            # assert len(mpe_states) == 1
            #
            # updating assignment
            for var, values in mpe_states.items():
                assert len(values) == 1
                decoded_repr[var] = values[0]

    return decoded_repr


def decode_instance_embedding_mpn_noleaf(mpn,
                                         node_feature_assoc,
                                         embedding,
                                         decoded_repr,
                                         root=None,
                                         verbose=False):

    n_features = len(embedding)

    nodes_to_process = deque()
    # instance_vals = {}

    #
    # if no root specified, start from the mpn root
    if root is None:
        for node in mpn.root_layer().nodes():
            nodes_to_process.append(node)
    else:
        nodes_to_process.append(root)

    while nodes_to_process:
        curr_node = nodes_to_process.popleft()

        if isinstance(curr_node, SumNode):
            max_child_val = NEG_INF
            max_child = None
            for i, child in enumerate(curr_node.children):
                #
                # is children in the features?
                present = True
                try:
                    child_feature_id = node_feature_assoc[child.id]
                except:
                    present = False
                if present and child_feature_id < n_features:
                    emb_val = embedding[child_feature_id] + curr_node.log_weights[i]
                else:
                    #
                    # if there is no embedding value for a leaf we are doing MPE for the leaf
                    # and we are considering them to be marginalized in the upward pass
                    emb_val = curr_node.log_weights[i]
                if emb_val > max_child_val:
                    max_child_val = emb_val
                    # print(emb_val)
                    max_child = child
            nodes_to_process.append(max_child)
            if verbose:
                print('max node, getting max child {0} [{1}]'.format(max_child,
                                                                     len(nodes_to_process)))
        #
        # adding them all if the current node is a product node
        elif isinstance(curr_node, ProductNode):
            nodes_to_process.extend(curr_node.children)
            if verbose:
                print('prod node, add {0} children [{1}]'.format(len(curr_node.children),
                                                                 len(nodes_to_process)))
        else:
            #
            # it is assumed to be a leaf, but do we have its value?

            if isinstance(curr_node, CategoricalIndicatorNode):
                decoded_repr[curr_node.var] = curr_node.var_val
            else:
                present = True
                try:
                    node_feature_id = node_feature_assoc[curr_node.id]
                except:
                    present = False
                if present and node_feature_id < n_features:
                    #
                    # if it is a univariate distribution, then
                    emb_val = embedding[node_feature_id]
                    if isinstance(curr_node, CategoricalSmoothedNode):
                        min_dist = numpy.inf
                        closest_val = None
                        for i, val in enumerate(curr_node._var_probs):
                            dist = abs(emb_val - val)
                            if verbose:
                                print(i, 'dist', dist, emb_val, val)
                            if dist < min_dist:
                                min_dist = dist
                                closest_val = i
                        decoded_repr[curr_node.var] = closest_val
                    else:
                        raise ValueError('unrecognized leaf type', curr_node)
                else:
                    mpe_states = mpe_states_from_leaf(curr_node, only_first_max=True)
                    if verbose:
                        print('Reached a leaf: {0}'.format(mpe_states))

                    # assert len(mpe_states) == 1
                    #
                    # updating assignment
                    for var, values in mpe_states.items():
                        assert len(values) == 1
                        decoded_repr[var] = values[0]

    return decoded_repr


def decode_instance_embedding_mpn(mpn,
                                  node_feature_assoc,
                                  embedding,
                                  decoded_repr,
                                  mpe_node_feature_assoc,
                                  mpe_embedding,
                                  root=None,
                                  missing_emb_val=-1000,
                                  missing_emb_threshold=300,
                                  verbose=False):

    n_features = len(embedding)

    nodes_to_process = deque()
    # instance_vals = {}

    #
    # if no root specified, start from the mpn root
    if root is None:
        for node in mpn.root_layer().nodes():
            nodes_to_process.append(node)
    else:
        nodes_to_process.append(root)

    while nodes_to_process:
        curr_node = nodes_to_process.popleft()

        if isinstance(curr_node, SumNode):

            max_child_val = NEG_INF
            max_child = None
            for i, child in enumerate(curr_node.children):
                #
                # is children in the features?
                present = True
                try:

                    child_feature_id = node_feature_assoc[child.id]
                except:
                    present = False

                child_emb_val = embedding[child_feature_id]
                if present and child_feature_id < n_features and not numpy.allclose(child_emb_val,
                                                                                    missing_emb_val):
                    # if present and child_feature_id < n_features and child_emb_val < 0.0:

                    # if present and child_feature_id < n_features and child_emb_val >
                    # (missing_emb_val + missing_emb_threshold):
                    emb_val = child_emb_val + curr_node.log_weights[i]
                else:
                    #
                    # if there is no embedding value for a node we are doing MPE
                    mpe_child_feature_id = mpe_node_feature_assoc[child.id]
                    emb_val = curr_node.log_weights[i] + mpe_embedding[mpe_child_feature_id]
                if emb_val > max_child_val:
                    max_child_val = emb_val
                    # print(emb_val)
                    max_child = child
            nodes_to_process.append(max_child)
            if verbose:
                print('max node, getting max child {0} [{1}]'.format(max_child,
                                                                     len(nodes_to_process)))
        #
        # adding them all if the current node is a product node
        elif isinstance(curr_node, ProductNode):
            nodes_to_process.extend(curr_node.children)
            if verbose:
                print('prod node, add {0} children [{1}]'.format(len(curr_node.children),
                                                                 len(nodes_to_process)))
        else:
            #
            # it is assumed to be a leaf, but do we have its value?
            decoded_repr = decode_leaf_node(node=curr_node,
                                            embedding=embedding,
                                            node_feature_assoc=node_feature_assoc,
                                            decoded_repr=decoded_repr,
                                            verbose=verbose)

    return decoded_repr


def decode_instance_latent_embedding_mpn(mpn,
                                         node_feature_assoc,
                                         embedding,
                                         decoded_repr,
                                         mpe_node_feature_assoc,
                                         mpe_embedding,
                                         binary=False,
                                         latent_var_sizes=None,
                                         root=None,
                                         missing_emb_val=-1,
                                         verbose=False):

    n_features = len(embedding)

    nodes_to_process = deque()
    # instance_vals = {}

    #
    # if no root specified, start from the mpn root
    if root is None:
        for node in mpn.root_layer().nodes():
            nodes_to_process.append(node)
    else:
        nodes_to_process.append(root)

    while nodes_to_process:
        curr_node = nodes_to_process.popleft()

        if isinstance(curr_node, SumNode):

            #
            # do we have its associated latent information?
            present = True
            try:
                feature_id = node_feature_assoc[curr_node.id]
            except:
                present = False

            if present and feature_id < n_features:
                #
                # the latent embedding is made by categorical RVs
                if not binary:
                    #
                    # is the embedding value missing? (e.g. -1)
                    # in that case we shall use the mpe estimate
                    emb_val = embedding[feature_id]
                    if emb_val == missing_emb_val:
                        emb_val = mpe_embedding[mpe_node_feature_assoc[curr_node.id]]

                    max_child = curr_node.children[emb_val]
                elif latent_var_sizes:
                    raise NotImplementedError('Still to implement the binary case')
                else:
                    raise ValueError(
                        'Binary latent array specified but no latent RV size vector given')

            else:
                #
                # if there is no embedding value for a node we are doing MPE
                # assuming mpe_node_feature_assoc is categorical
                max_child_id = mpe_embedding[mpe_node_feature_assoc[curr_node.id]]
                max_child = curr_node.children[max_child_id]

            nodes_to_process.append(max_child)
            if verbose:
                print('max node, getting max child {0} [{1}]'.format(max_child,
                                                                     len(nodes_to_process)))
        #
        # adding them all if the current node is a product node
        elif isinstance(curr_node, ProductNode):
            nodes_to_process.extend(curr_node.children)
            if verbose:
                print('prod node, add {0} children [{1}]'.format(len(curr_node.children),
                                                                 len(nodes_to_process)))
        else:
            #
            # since we do not have an activation value we will go with MPE
            decoded_repr = decode_leaf_node(node=curr_node,
                                            embedding=embedding,
                                            node_feature_assoc=node_feature_assoc,
                                            decoded_repr=decoded_repr,
                                            verbose=verbose)

    return decoded_repr


def aggr_scopes_by_func(scope_values, aggr_func):
    n_features = len(scope_values)
    aggr_values = numpy.zeros(n_features)
    for i in range(n_features):
        aggr_values[i] = aggr_func(scope_values[i])
    return aggr_values


def aggr_scopes_by_sum(scope_values):
    return aggr_scopes_by_func(scope_values, numpy.sum)


def aggr_scopes_by_mean(scope_values):
    return aggr_scopes_by_func(scope_values, numpy.mean)


def aggr_scopes_by_logsumexp(scope_values):
    return aggr_scopes_by_func(scope_values, scipy.misc.logsumexp)


def aggr_scopes_by_uni_mixture(scope_values):
    n_features = len(scope_values)
    aggr_values = numpy.zeros(n_features)
    uniform_weight = 1. / n_features
    for i in range(n_features):
        aggr_values[i] = scipy.misc.logsumexp(a=scope_values[i], b=uniform_weight)
    return aggr_values


def extract_features_nodes_by_scope(spn,
                                    data,
                                    filter_node_func=filter_sum_nodes,
                                    # retrieve_func=node_in_path_feature,
                                    remove_zero_features=True,
                                    output_feature_info=None,
                                    normalizing=True,
                                    aggr_func=None,
                                    dtype=None,
                                    verbose=False):
    """
    Representing a dataset like in extract_features_nodes but
    aggregating by scope
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    #
    # storing assoc: sum node -> new feature id
    nodes_id_assoc = filter_node_func(spn)

    #
    # computing scopes
    scope_assoc = defaultdict(set)
    for node_elem in nodes_id_assoc:
        #
        # is this a compound assoc?
        try:
            node, rest = node_elem
        except:
            node = node_elem

        if hasattr(node, 'var_scope'):
            scope_assoc[node.var_scope].add(node)
        elif hasattr(node, 'var'):
            scope_assoc[frozenset(node.var)].add(node)

    # print(scope_assoc)
    n_spn_features = len(scope_assoc)
    logging.info('There are {} different scopes'.format(n_spn_features))

    #
    # creating a new node assoc
    node_id_assoc = {}
    for i, (scope, nodes_in_scope) in enumerate(scope_assoc.items()):
        for n in nodes_in_scope:
            node_id_assoc[n] = i

    # print('\n\nnode_id_assoc len {}\n\n'.format(len(node_id_assoc)))

    if dtype is None:
        dtype = data.dtype

    repr_data = numpy.zeros((n_instances, n_spn_features), dtype=dtype)

    logging.info('Old data ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                              n_features,
                                                              n_spn_features))

    scope_num_nodes = numpy.zeros(n_spn_features)
    for node, id in node_id_assoc.items():
        scope_num_nodes[id] += 1

    # print(scope_num_nodes)
    ext_all_start_t = perf_counter()
    #
    # evaluate MPE circuit for each instance
    for i in range(n_instances):

        ext_s_t = perf_counter()
        #
        # bottom up evaluation
        spn.single_eval(data[i])

        #
        #
        feature_values = [deque() for _ in range(n_spn_features)]

        #
        # visiting all nodes
        for node in spn.top_down_nodes():

            #
            # retrieve the represented value for the feature
            # repr_data = retrieve_func(node, repr_data, nodes_id_assoc, i)
            if node in node_id_assoc:
                # print(node.id, node_id_assoc[node])
                # repr_data[i, node_id_assoc[node]] += node.log_val
                feature_values[node_id_assoc[node]].append(node.log_val)

        repr_data[i] = aggr_func(feature_values)
        ext_e_t = perf_counter()
        print('\t\tprocessed instance {}/{} in {}'.format(i + 1,
                                                          n_instances,
                                                          ext_e_t - ext_s_t),
              end='           \r')

    ext_all_end_t = perf_counter()
    logging.info('Extraction done in {} secs'.format(ext_all_end_t - ext_all_start_t))

    if remove_zero_features:

        old_n_features = repr_data.shape[1]
        zero_feature = numpy.zeros(n_instances, dtype=data.dtype)
        features_to_keep = [i for i in range(n_spn_features)
                            if not numpy.allclose(zero_feature, repr_data[:, i])]
        repr_data = repr_data[:, numpy.array(features_to_keep)]

        logging.info('Removed features ({0} x {1}) -> ({0} x {2})'.format(n_instances,
                                                                          old_n_features,
                                                                          repr_data.shape[1]))

    #
    # save scopes and other info?
    if output_feature_info is not None:

        node_id_assoc_n = {}

        for f, (scope, nodes) in enumerate(scope_assoc.items()):

            node_id_assoc_n[nodes.pop()] = f

        save_feature_info(spn, node_id_assoc_n, output_feature_info)

    return repr_data


def scope_stats(spn, filter_node_func=filter_non_leaf_nodes, top_n_scopes=20):

    scope_counter = Counter()
    var_counter = Counter()

    nodes_id_assoc = filter_node_func(spn)

    for node in nodes_id_assoc:

        if hasattr(node, 'var_scope'):
            # check_node = True
            # if no_leaf:
            #     check_node = hasattr(node, 'children')
            # if check_node:
            scope_counter[node.var_scope] += 1
            for var in node.var_scope:
                var_counter[var] += 1

    if top_n_scopes > 1:
        print('Most common scopes\n{}'.format(scope_counter.most_common(top_n_scopes)))
        print('Most common vars\n{}'.format(var_counter.most_common(top_n_scopes)))

    return scope_counter, var_counter


def scope_stats_marg(spn, marg_vars):

    vars_to_marginalize = set(marg_vars)

    overlap_perc_dict = {}

    n_nodes = 0
    for node in spn.top_down_nodes():

        if hasattr(node, 'var_scope'):

            perc = len(node.var_scope & vars_to_marginalize) / len(node.var_scope)
            overlap_perc_dict[node] = perc

            n_nodes += 1

    product_children_percs_dict = {}
    for node in overlap_perc_dict:
        if isinstance(node, ProductNode):
            child_percs = [overlap_perc_dict[child] for child in node.children]
            # child_n_marg_vars = [p * len(child.var_scope)
            #                      for p, child in zip(child_percs, node.children)]
            product_children_percs_dict[node] = child_percs

    #
    # printing node stats
    all_marg_scope_nodes = [(p, n) for n, p in overlap_perc_dict.items() if p > 0.999]
    all_orig_scope_nodes = [(p, n) for n, p in overlap_perc_dict.items() if p < 0.001]

    sorted_perc = sorted([(p, n.id) for n, p in overlap_perc_dict.items()], key=lambda x: x[0])
    print('Nodes with all H: {}/{}'.format(len(all_marg_scope_nodes), n_nodes))
    print('Nodes with all X: {}/{}'.format(len(all_orig_scope_nodes), n_nodes))
    print('Nodes with mixed: scope {}/{}'.format(n_nodes -
                                                 len(all_orig_scope_nodes) -
                                                 len(all_marg_scope_nodes),
                                                 n_nodes))
    print('Sorted perc {}'.format(
        sorted_perc[len(all_orig_scope_nodes):len(all_orig_scope_nodes) + 50]))

    fully_sep_prod_nodes = []
    sep_prod_nodes = []
    not_sep_prod_nodes = []
    for node, perc_list in product_children_percs_dict.items():
        if all([p > 0.999 or p < 0.001 for p in perc_list]):
            fully_sep_prod_nodes.append((node.id, perc_list))
        elif any([p > 0.999 for p in perc_list]) and any([p < 0.001 for p in perc_list]):
            sep_prod_nodes.append((node.id, perc_list))
        else:
            not_sep_prod_nodes.append((node.id, perc_list))

    print('Fully separating product nodes: {}/{}'.format(len(fully_sep_prod_nodes),
                                                         len(product_children_percs_dict)))
    print('Separating product nodes: {}/{}'.format(len(sep_prod_nodes),
                                                   len(product_children_percs_dict)))
    print('Not separating nodes: {}'.format(not_sep_prod_nodes))


def node_mpe_instantiation(node,
                           n_features,
                           dtype=numpy.int32,
                           verbose=False,
                           dont_care_val=MARG_IND,
                           only_first_max=False):
    """
    Getting the mpe instantiation for a node (probability distribution over a scope)
    Assuming an mpe bottom-up step has already been done
    """

    instance_vals = defaultdict(set)

    #
    # traversing the spn top down
    nodes_to_process = deque()
    nodes_to_process.append(node)

    while nodes_to_process:
        curr_node = nodes_to_process.popleft()

        #
        # if it is a sum node, follow the max child
        if isinstance(curr_node, SumNode):
            max_children = []
            for i, child in enumerate(curr_node.children):

                if numpy.isclose(curr_node.log_weights[i] + child.log_val, curr_node.log_val):
                    max_children.append(child)
                    #
                    # following all nodes wtih equal value?
                    if only_first_max:
                        break
            assert len(max_children) > 0
            nodes_to_process.extend(max_children)
            if verbose:
                print('sum node, getting children {0} [{1}]'.format(len(max_children),
                                                                    len(nodes_to_process)))
        #
        # adding them all if the current node is a product node
        elif isinstance(curr_node, ProductNode):
            nodes_to_process.extend(curr_node.children)
            if verbose:
                print('prod node, add {0} children [{1}]'.format(len(curr_node.children),
                                                                 len(nodes_to_process)))
        else:
            #
            # it is assumed to be a leaf
            mpe_states = mpe_states_from_leaf(curr_node, only_first_max)

            if verbose:
                print('Reached a leaf: {0}'.format(mpe_states))

            #
            # updating assignment
            for var, values in mpe_states.items():
                for v in values:
                    instance_vals[var].add(v)
    #
    # now we need to create the combinations
    instance_states = []
    feature_ids = [k for k in instance_vals]
    feature_vals = [instance_vals[k] for k in feature_ids]
    for comb_vals in itertools.product(*feature_vals):
        #
        # create an instance, don't care values are set to MARG_IND by default
        instance = numpy.zeros(n_features, dtype=dtype)
        instance.fill(dont_care_val)
        for i, val in enumerate(comb_vals):
            instance[feature_ids[i]] = val
        instance_states.append(instance)

    instances_vec = numpy.array(instance_states)

    if instances_vec.shape[0] == 1:
        instances_vec = instances_vec.flatten()

    return instances_vec


def retrieve_all_nodes_mpe_instantiations(spn,
                                          n_features,
                                          dtype=numpy.int32,
                                          verbose=False,
                                          dont_care_val=MARG_IND,
                                          only_first_max=False):

    #
    # doing an mpe bottom up evaluation
    marg_all_features_instance = numpy.zeros(n_features, dtype=int)
    marg_all_features_instance.fill(MARG_IND)
    spn.mpe_eval(marg_all_features_instance)

    NodeInfo = namedtuple('NodeInfo', ['layer', 'mpes', 'scope'])

    node_stats = {}
    for i, layer in enumerate(spn.top_down_layers()):
        print(layer.__class__.__name__)
        for j, node in enumerate(layer.nodes()):
            mpe_instantiations = node_mpe_instantiation(node,
                                                        n_features,
                                                        dtype=dtype,
                                                        verbose=verbose,
                                                        dont_care_val=dont_care_val,
                                                        only_first_max=only_first_max)
            scope = []
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = frozenset(node.var)
            node_stats[node] = NodeInfo(layer, mpe_instantiations, scope)

    return node_stats


def random_feature_mask(feature_mask, n_rand_features, p=None, rand_gen=None):
    """
    Sets in a boolean vector as mask n_rand_features to be True
    if p is None, samples from a uniform distribution
    """
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_features = len(feature_mask)
    rand_feature_ids = rand_gen.choice(n_features, n_rand_features, replace=False, p=p)

    for f_id in rand_feature_ids:
        feature_mask[f_id] = True

    return feature_mask


def random_rectangular_feature_mask(feature_mask,
                                    n_rows, n_cols,
                                    n_min_rows=2, n_min_cols=2,
                                    n_max_rows=3, n_max_cols=3,
                                    rand_gen=None):
    """
    Sets in a boolean vector as mask a number of features to True such that
    in a n_rows x n_cols reshaping, the mask forms a rectangle
    """
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_features = len(feature_mask)
    #
    # extract the origin randomly
    possible_features = numpy.arange(n_features).reshape(n_rows, n_cols)
    possible_features = possible_features[:n_rows - n_min_rows, :n_cols - n_min_cols].flatten()

    origin_id = rand_gen.choice(possible_features, replace=False)
    origin_x = origin_id // n_cols
    origin_y = origin_id - (origin_x * n_cols)
    # print(possible_features, origin_id)

    #
    # and then length and width
    max_length = min(n_max_rows + 1, n_rows - origin_x + 1)
    max_width = min(n_max_cols + 1, n_cols - origin_y + 1)
    length = rand_gen.choice(numpy.arange(n_min_cols, max_length), replace=False)
    width = rand_gen.choice(numpy.arange(n_min_cols, max_width), replace=False)
    # print(origin_id, origin_x, origin_y, length, width)

    rand_feature_ids = []
    for i in range(origin_x, origin_x + length):
        for j in range(origin_y, origin_y + width):
            rand_feature_ids.append(i * n_cols + j)

    for f_id in rand_feature_ids:
        feature_mask[f_id] = True

    return feature_mask


def mask_dataset_marginalization(data, feature_mask, marg_value=MARG_IND, copy=True):
    #
    # make a copy?
    if copy:
        data = numpy.array(data, copy=True)

    #
    # for each feature not in feature_mask, set a marginalization
    data[:, 1 - feature_mask] = marg_value

    return data


def extract_feature_marginalization_from_masks(spn,
                                               data,
                                               feature_masks,
                                               marg_value=MARG_IND,
                                               rand_gen=None,
                                               dtype=float):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    n_gen_features = len(feature_masks)

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)
    marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)

    feat_i_t = perf_counter()
    for i, mask in enumerate(feature_masks):
        feat_s_t = perf_counter()
        marg_data.fill(marg_value)
        #
        # copy only the right features
        marg_data[:, mask] = data[:, mask]
        # print('{}\n{}'.format(i, marg_data))

        #
        # evaluate the spn to get a new feature
        preds = evaluate_on_dataset(spn, marg_data)
        repr_data[:, i] = preds
        feat_e_t = perf_counter()

        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')
        if i % 100 == 0:
            logging.info('\tProcessed {}/{} features (elapsed {})'.format(i + 1,
                                                                          len(feature_masks),
                                                                          feat_e_t - feat_i_t))

    print('')
    return repr_data


def extract_feature_marginalization_from_masks_opt_unique(spn,
                                                          data,
                                                          feature_masks,
                                                          marg_value=MARG_IND,
                                                          dtype=float):
    """
    Same as extract_feature_marginalization_from_masks but
    doing less queries to an spn, considering only the unique ones
    according to the data
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    n_gen_features = len(feature_masks)

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)
    # marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)
    marg_instance = numpy.zeros(n_features, dtype=data.dtype)

    feat_i_t = perf_counter()
    for i, mask in enumerate(feature_masks):
        feat_s_t = perf_counter()
        marg_instance.fill(marg_value)
        # marg_data.fill(marg_value)
        #
        # extracting only the right features
        masked_data = data[:, mask]
        #
        # getting the unique instances
        feature_patches_dict = defaultdict(lambda: numpy.zeros(n_instances, dtype=bool))
        for k, instance in enumerate(masked_data):
            #
            # enumerating patches
            patch = tuple(instance)

            feature_patches_dict[patch][k] = True

        #
        # retrieving feature values for the patches
        for patch, instance_mask in feature_patches_dict.items():
            marg_instance[mask] = numpy.array(patch, dtype=data.dtype)
            repr_val, = spn.single_eval(marg_instance)
            repr_data[instance_mask, i] = repr_val

        feat_e_t = perf_counter()

        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')
        if i % 100 == 0:
            logging.info('\tProcessed {}/{} features (elapsed {})'.format(i + 1,
                                                                          len(feature_masks),
                                                                          feat_e_t - feat_i_t))

    print('')
    return repr_data


# @numba.njit
def feature_mask_to_marg(feature_mask, n_ohe_features, feature_vals):
    """
    Converts a feature mask for a categorical dataset for one for a one hot encoded dataset
    Es (False, True, True, False) with feature values [2, 2, 2, 2] becomes
    (False, False, True, True, True, True, False, False)
    """
    n_features = len(feature_mask)
    ohe_feature_mask = numpy.zeros(n_ohe_features, dtype=bool)
    for j in range(n_features):
        if feature_mask[j]:
            f_id = int(numpy.sum(feature_vals[:j]))
            ohe_feature_mask[numpy.arange(f_id, f_id + feature_vals[j])] = True
    return ohe_feature_mask


def extract_feature_marginalization_from_masks_theanok(spn,
                                                       data,
                                                       feature_masks,
                                                       feature_vals=None,
                                                       marg_value=MARG_IND,
                                                       rand_gen=None,
                                                       batch_size=None,
                                                       dtype=float):
    """
    data is one hot encoded for theanok spns, masks are not
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    n_gen_features = len(feature_masks)

    if feature_vals is None:
        #
        # assuming all binary variables
        assert n_features % 2 == 0
        feature_vals = numpy.array([2 for i in range(n_features // 2)])

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)
    marg_data = numpy.ones((n_instances, n_features), dtype=data.dtype)

    feat_i_t = perf_counter()
    for i, mask in enumerate(feature_masks):
        feat_s_t = perf_counter()
        marg_data.fill(1)
        #
        # copy only the right features
        ohe_mask = feature_mask_to_marg(mask, n_features, feature_vals)
        marg_data[:, ohe_mask] = data[:, ohe_mask]
        # print('{}\n{}'.format(i, marg_data))

        #
        # evaluate the spn to get a new feature
        preds = evaluate_on_dataset_batch(spn, marg_data, batch_size)
        repr_data[:, i] = preds
        feat_e_t = perf_counter()

        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')
        if i % 100 == 0:
            logging.info('\tProcessed {}/{} features (elapsed {})'.format(i + 1,
                                                                          len(feature_masks),
                                                                          feat_e_t - feat_i_t))

    print('')
    return repr_data


def extract_feature_marginalization_from_masks_theanok_opt_unique(spn,
                                                                  data,
                                                                  feature_masks,
                                                                  feature_vals=None,
                                                                  marg_value=MARG_IND,
                                                                  batch_size=None,
                                                                  dtype=float):
    """
    Same as extract_feature_marginalization_from_masks_theanok but
    doing less queries to an spn, considering only the unique ones
    according to the data
    """

    n_instances = data.shape[0]
    n_features = data.shape[1]
    n_gen_features = len(feature_masks)

    if feature_vals is None:
        #
        # assuming all binary variables
        assert n_features % 2 == 0
        feature_vals = numpy.array([2 for i in range(n_features // 2)])

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)

    feat_i_t = perf_counter()
    for i, mask in enumerate(feature_masks):
        feat_s_t = perf_counter()

        #
        # copy only the right features
        ohe_mask = feature_mask_to_marg(mask, n_features, feature_vals)
        masked_data = data[:, ohe_mask]

        #
        # getting the unique instances
        feature_patches_dict = defaultdict(lambda: numpy.zeros(n_instances, dtype=bool))

        for k, instance in enumerate(masked_data):
            #
            # enumerating patches
            patch = tuple(instance)
            feature_patches_dict[patch][k] = True

        #
        # recreating a dataset
        n_patches = len(feature_patches_dict)
        # print('\t\tthere are {} patches'.format(n_patches))

        marg_data = numpy.ones((n_patches, n_features), dtype=data.dtype)
        ordered_patches = {}
        for k, patch in enumerate(feature_patches_dict):
            ordered_patches[patch] = k
            marg_data[k, ohe_mask] = numpy.array(patch)

        # print(marg_data.shape)
        #
        # evaluate the spn to get a new feature
        preds = evaluate_on_dataset_batch(spn, marg_data, batch_size)
        # print(preds.shape, preds[0])
        #
        # populating the dataset back
        for patch, instance_mask in feature_patches_dict.items():
            repr_val = preds[ordered_patches[patch]]
            repr_data[instance_mask, i] = repr_val

        feat_e_t = perf_counter()

        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')
        if i % 100 == 0:
            logging.info('\tProcessed {}/{} features (elapsed {})'.format(i + 1,
                                                                          len(feature_masks),
                                                                          feat_e_t - feat_i_t))

    print('')
    return repr_data


def save_features_to_file(feature_masks, output_path, delimiter=','):
    """
    Saving a seq of boolean feature masks to file  as rows of ints
    Eg. [[True, False, False], [False, True, True]]
    """
    feature_masks_array = numpy.array(feature_masks, dtype=bool)
    numpy.savetxt(output_path, feature_masks_array, fmt='%d', delimiter=delimiter)


def load_features_from_file(feature_file_path, delimiter=','):
    return numpy.loadtxt(feature_file_path, dtype=bool, delimiter=delimiter)


def feature_mask_scope(feature_mask):
    """
    Given a feature mask (a boolean ndarray), getting the scope
    of the features set to true (an int ndarray)
    """
    n_features = len(feature_mask)
    return numpy.arange(n_features, dtype=int)[feature_mask]


def extract_features_marginalization_grid(n_rows, n_cols,
                                          n_cell_rows, n_cell_cols,
                                          feature_file_path=None):

    n_features = n_rows * n_cols

    feature_masks = []

    for i in range(0, n_rows, n_cell_rows):
        for j in range(0, n_cols, n_cell_cols):
            mask = numpy.zeros((n_rows, n_cols), dtype=bool)
            mask[i:i + n_cell_rows, j:j + n_cell_cols] = True
            mask = mask.reshape(n_features)
            feature_masks.append(mask)

    if feature_file_path:
        save_features_to_file(feature_masks, feature_file_path)

    return feature_masks


def extract_features_marginalization_grid_strides(n_features,
                                                  n_rows, n_cols,
                                                  n_patch_rows, n_patch_cols,
                                                  h_stride,
                                                  v_stride,
                                                  rand_gen=None,
                                                  feature_file_path=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    n_features = n_rows * n_cols

    feature_masks = []

    for i in range(0, n_rows, v_stride):
        for j in range(0, n_cols, h_stride):
            if i + n_patch_rows <= n_rows and j + n_patch_cols <= n_cols:
                mask = numpy.zeros((n_rows, n_cols), dtype=bool)
                mask[i:i + n_patch_rows, j:j + n_patch_cols] = True
                mask = mask.reshape(n_features)
                feature_masks.append(mask)

    #
    # resizing
    feature_masks = numpy.array(feature_masks[:n_features])

    #
    # shuffling
    rand_gen.shuffle(feature_masks)

    if feature_file_path:
        save_features_to_file(feature_masks, feature_file_path)

    return feature_masks


def instance_from_disjoint_feature_masks(instance,
                                         feature_masks,
                                         feature_values,
                                         dtype=float):
    if instance is None:
        n_features = feature_masks[0].shape[0]
        instance = numpy.zeros(n_features, dtype=dtype)

    for mask, feature_val in zip(feature_masks, feature_values):
        instance[mask] = feature_val

    return instance


def extract_features_marginalization_rectangles(n_features,
                                                n_rows, n_cols,
                                                feature_batch_sizes,
                                                rect_min_sizes,
                                                rect_max_sizes,
                                                feature_file_path=None,
                                                marg_value=MARG_IND,
                                                rand_gen=None,
                                                dtype=float):
    """
    feature_batch_sizes = [10, 20, 30]
    rect_min_sizes = [(2, 2), (2, 2), (4, 4)]
    rect_max_sizes = [(3, 3), (4, 4), (2, 2)]
    """
    assert len(feature_batch_sizes) == len(rect_min_sizes)
    assert len(rect_min_sizes) == len(rect_max_sizes)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    #
    # generating the feature masks according to the parameters
    feature_masks = []
    for n_masks, (n_min_rows, n_min_cols), (n_max_rows, n_max_cols) in zip(feature_batch_sizes,
                                                                           rect_min_sizes,
                                                                           rect_max_sizes):
        assert n_min_rows <= n_max_rows
        assert n_min_cols <= n_max_cols

        for i in range(n_masks):
            mask = numpy.zeros(n_features, dtype=bool)
            mask = random_rectangular_feature_mask(mask,
                                                   n_rows,
                                                   n_cols,
                                                   n_min_rows=n_min_rows,
                                                   n_min_cols=n_min_cols,
                                                   n_max_rows=n_max_rows,
                                                   n_max_cols=n_max_cols,
                                                   rand_gen=rand_gen)
            feature_masks.append(mask)

    #
    # saving them to file=
    if feature_file_path:
        save_features_to_file(feature_masks, feature_file_path)

    # return repr_data
    return feature_masks


def extract_features_marginalization_rand(n_features,
                                          feature_batch_sizes,
                                          n_feature_sizes,
                                          feature_file_path=None,
                                          marg_value=MARG_IND,
                                          rand_gen=None,
                                          dtype=float):
    """
    feature_batch_sizes = [10, 20, 30]
    rect_min_sizes = [(2, 2), (2, 2), (4, 4)]
    rect_max_sizes = [(3, 3), (4, 4), (2, 2)]
    """
    assert len(feature_batch_sizes) == len(n_feature_sizes)

    # n_instances = data.shape[0]
    # n_features = data.shape[1]

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    #
    # generating the feature masks according to the parameters
    feature_masks = []
    for n_masks, n_rand_features in zip(feature_batch_sizes,
                                        n_feature_sizes):
        assert n_rand_features <= n_features
        print('Processing # masks {} # rand features{}'.format(n_masks,
                                                               n_rand_features))
        for i in range(n_masks):

            mask = numpy.zeros(n_features, dtype=bool)
            mask = random_feature_mask(mask,
                                       n_rand_features,
                                       rand_gen=rand_gen)
            feature_masks.append(mask)

    # #
    # # using the masks to evaluate the marginalizations
    # repr_data = extract_feature_marginalization_from_masks(spn,
    #                                                        data,
    #                                                        feature_masks,
    #                                                        marg_value=marg_value,
    #                                                        rand_gen=rand_gen,
    #                                                        dtype=dtype)

    #
    # saving them to file=
    if feature_file_path:
        save_features_to_file(feature_masks, feature_file_path)

    # return repr_data
    return feature_masks


def extract_instances_groups(data, group_func=dataset_to_instances_set, dtype=numpy.int8):
    """
    From a dataset represented by a n_instances x n_features matrix
    it extractes the gorups of identical instances and assign them to a class number
    returns the mapping matrix n_instances x n_groups
    """
    n_instances = data.shape[0]
    n_features = data.shape[1]

    #
    # grouping by a certain criterion
    logging.info('Grouping by function {}'.format(group_func))
    groups = group_func(data)
    n_groups = len(groups)
    logging.info('There are {} groups'.format(n_groups))

    repr_data = numpy.zeros((n_instances, n_groups), dtype=dtype)

    group_feature_mapping = {centroid: i for centroid, i in zip(groups, range(n_groups))}

    #
    # assign groups to instances
    for i in range(n_instances):
        repr_data[i, group_feature_mapping[tuple(data[i])]] = 1

    return repr_data


LIBRA_MARG_SYM = '*'
ACQUERY_EXEC = './acquery'
QUERY_EXT = 'q'
FEATURE_PREFIX = 'features'


def format_val(val,
               dtype=int,
               marg_value=MARG_IND,
               marg_sym=LIBRA_MARG_SYM):
    if val == marg_value:
        return marg_sym
    else:
        return str(val)


def data_through_feature_mask(data,
                              feature_mask,
                              output_path,
                              delimiter=',',
                              marg_value=MARG_IND,
                              marg_sym=LIBRA_MARG_SYM):
    """
    Masking a dataset according to a feature map, substituting all marg indices with
    a char (default to Libra's don't care char), then serialize to file
    """
    n_instances = data.shape[0]
    n_features = data.shape[1]

    marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)
    marg_data.fill(marg_value)

    #
    # storing only the request values
    marg_data[:, feature_mask] = data[:, feature_mask]

    #
    # serializing
    ser_s_t = perf_counter()
    with open(output_path, 'w') as query_file:
        for i in range(n_instances):
            instance_str = delimiter.join(format_val(f) for f in marg_data[i])
            query_file.write('{}\n'.format(instance_str))
    ser_e_t = perf_counter()
    logging.debug('Serialized feature query data to {} in {}'.format(output_path,
                                                                     ser_e_t - ser_s_t))

    return marg_data


def data_through_feature_mask_opt_unique(data,
                                         feature_mask,
                                         output_path,
                                         delimiter=',',
                                         marg_value=MARG_IND,
                                         marg_sym=LIBRA_MARG_SYM):
    """
    Same as data_through_feature_mask, but writing only the unique feature patches
    Returning the composed marg data, the feature patch to instances dict and
    the feature patch to feature id dict
    """
    n_instances = data.shape[0]
    n_features = data.shape[1]

    # marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)
    # marg_data.fill(marg_value)

    #
    # storing only the request values
    masked_data = data[:, feature_mask]

    #
    # getting the unique instances
    feature_patches_dict = defaultdict(lambda: numpy.zeros(n_instances, dtype=bool))
    for k, instance in enumerate(masked_data):
        #
        # enumerating patches
        patch = tuple(instance)

        feature_patches_dict[patch][k] = True

    n_patches = len(feature_patches_dict)

    marg_data = numpy.zeros((n_patches, n_features), dtype=data.dtype)
    marg_data.fill(marg_value)

    ordered_patches = {}
    for k, patch in enumerate(feature_patches_dict):
        ordered_patches[patch] = k
        marg_data[k, feature_mask] = numpy.array(patch)

    #
    # serializing
    ser_s_t = perf_counter()
    with open(output_path, 'w') as query_file:
        for i in range(n_patches):
            instance_str = delimiter.join(format_val(f) for f in marg_data[i])
            query_file.write('{}\n'.format(instance_str))
    ser_e_t = perf_counter()
    logging.debug('Serialized feature query data to {} in {}'.format(output_path,
                                                                     ser_e_t - ser_s_t))

    return marg_data, feature_patches_dict, ordered_patches


def ll_array_from_model_score(score_output):
    """
    Quick and dirty parsing
    """
    #
    # split strings by newlines
    lines = score_output.split('\n')
    #
    # remove all the lines that are not numbers
    lls = []
    for ll in lines:
        try:
            lls.append(float(ll))
        except ValueError:
            pass
    #
    # convert to numpy array
    return numpy.array(lls)


def acquery(model, query_file,  exec_path=ACQUERY_EXEC):
    """
    Computing the likelihood for some queries given an instance
    """

    process = None
    process = subprocess.Popen([exec_path,
                                '-m', model,
                                '-q', query_file],
                               stdout=subprocess.PIPE)

    proc_out, proc_err = process.communicate()
    #
    # TODO manage errors
    # print(proc_out)
    if proc_err is not None:
        logging.error('acquery errors: {}'.format(proc_err))

    scores = ll_array_from_model_score(proc_out.decode("utf-8"))

    return scores


def extract_features_marginalization_acquery(data,
                                             model_path,
                                             feature_masks,
                                             output_path,
                                             dtype=float,
                                             prefix=FEATURE_PREFIX,
                                             overwrite_feature_file=True,
                                             exec_path=ACQUERY_EXEC):

    n_gen_features = len(feature_masks)
    n_instances = data.shape[0]

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)

    for i, mask in enumerate(feature_masks):
        #
        # translating the mask into a query set
        query_file_name = None
        if overwrite_feature_file:
            query_file_name = '{}.{}'.format(prefix, QUERY_EXT)
        else:
            query_file_name = '{}.{}.{}'.format(prefix, i,  QUERY_EXT)
        query_file_path = os.path.join(output_path, query_file_name)

        feat_s_t = perf_counter()
        #
        # computing the queries (side effect: writing to file query_file_path)
        queries = data_through_feature_mask(data, mask, query_file_path)

        #
        # getting the scores
        feature_scores = acquery(model_path, query_file_path, exec_path=exec_path)

        assert len(queries) == len(feature_scores)

        #
        # storing them
        repr_data[:, i] = feature_scores
        feat_e_t = perf_counter()
        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')

    return repr_data


def extract_features_marginalization_acquery_opt_unique(data,
                                                        model_path,
                                                        feature_masks,
                                                        output_path,
                                                        dtype=float,
                                                        prefix=FEATURE_PREFIX,
                                                        overwrite_feature_file=True,
                                                        exec_path=ACQUERY_EXEC):

    n_gen_features = len(feature_masks)
    n_instances = data.shape[0]

    repr_data = numpy.zeros((n_instances, n_gen_features), dtype=dtype)

    feat_i_t = perf_counter()
    for i, mask in enumerate(feature_masks):
        #
        # translating the mask into a query set
        query_file_name = None
        if overwrite_feature_file:
            query_file_name = '{}.{}'.format(prefix, QUERY_EXT)
        else:
            query_file_name = '{}.{}.{}'.format(prefix, i,  QUERY_EXT)
        query_file_path = os.path.join(output_path, query_file_name)

        feat_s_t = perf_counter()
        #
        # computing the queries (side effect: writing to file query_file_path)
        queries, feature_dict, feature_ids = data_through_feature_mask_opt_unique(data,
                                                                                  mask,
                                                                                  query_file_path)

        #
        # getting the scores
        feature_scores = acquery(model_path, query_file_path, exec_path=exec_path)

        assert len(queries) == len(feature_scores)

        #
        # storing them
        for patch, instance_mask in feature_dict.items():
            repr_val = feature_scores[feature_ids[patch]]
            repr_data[instance_mask, i] = repr_val

        # repr_data[:, i] = feature_scores
        feat_e_t = perf_counter()
        print('\tProcessed feature {}/{} (done in {})'.format(i + 1,
                                                              len(feature_masks),
                                                              feat_e_t - feat_s_t),
              end='        \r')
        if i % 100 == 0:
            logging.info('\tProcessed {}/{} features (elapsed {})'.format(i + 1,
                                                                          len(feature_masks),
                                                                          feat_e_t - feat_i_t))

    return repr_data


def node_activations_for_instance(spn,
                                  nodes,
                                  instance,
                                  marg_mask=None,
                                  mean=False,
                                  log=False,
                                  hard=False,
                                  dtype=float):
    """
    Given an SPN and an instance, return a same shape instance
    containing the activations of all nodes, summed by scopes
    """

    assert instance.ndim == 1

    n_features = len(instance)
    activations = numpy.zeros(n_features, dtype=dtype)
    var_counter = Counter()

    #
    # marginalizing?
    if marg_mask is not None:
        instance = numpy.array(instance, copy=True)
        instance[numpy.logical_not(marg_mask)] = MARG_IND

    #
    # evaluate it bottom, up
    res, = spn.single_eval(instance)

    node_set = set(nodes)
    #
    # then gather the node activation vals
    for node in nodes:

        if log:
            val = node.log_val
        else:
            val = numpy.exp(node.log_val)

        scope = None
        if hasattr(node, 'var_scope'):
            scope = node.var_scope
        elif hasattr(node, 'var'):
            scope = [node.var]

        #
        # accumulating scope
        for var in scope:
            var_counter[var] += 1
            if hard:
                activations[var] += 1
            else:
                # activations[var] += (val / len(scope))
                activations[var] += val
                # activations[var] = max(val / len(scope), activations[var])
                # if instance[var] == 1:
                #     activations[var] += val
                # else:
                #     activations[var] += (1 - val)

    if mean:
        for i in range(n_features):
            activations[i] /= var_counter[i]

    return activations


def extract_features_node_activations(spn,
                                      nodes,
                                      data,
                                      marg_mask=None,
                                      mean=False,
                                      log=False,
                                      hard=False,
                                      dtype=float):
    n_instances = data.shape[0]
    n_features = data.shape[1]

    repr_data = numpy.zeros((n_instances, n_features), dtype=dtype)

    for i in range(n_instances):
        ext_s_t = perf_counter()
        repr_data[i, :] = node_activations_for_instance(spn,
                                                        nodes,
                                                        data[i],
                                                        marg_mask=marg_mask,
                                                        mean=mean,
                                                        log=log,
                                                        hard=hard,
                                                        dtype=dtype)
        ext_e_t = perf_counter()
        print('\tProcessed instance {}/{} (done in {})'.format(i + 1,
                                                               n_instances,
                                                               ext_e_t - ext_s_t),
              end='        \r')

    return repr_data


def all_single_marginals_spn(spn,
                             feature_vals,
                             dtype=numpy.int32):

    n_features = len(feature_vals)
    n_instantiations = numpy.sum(feature_vals)

    feat_s_t = perf_counter()
    marg_data = numpy.zeros((n_instantiations, n_features), dtype=dtype)
    marg_data.fill(MARG_IND)

    instance_id = 0
    for i in range(n_features):
        for j in range(feature_vals[i]):
            marg_data[instance_id, i] = j
            instance_id += 1

    marginals = evaluate_on_dataset(spn, marg_data)
    feat_e_t = perf_counter()
    logging.info('Marginals extracted in {}'.format(feat_e_t - feat_s_t))

    return marginals


def all_single_marginals_ml(data,
                            feature_vals,
                            alpha=0.0):

    n_instances = data.shape[0]
    n_features = data.shape[1]
    n_instantiations = numpy.sum(feature_vals)

    feat_s_t = perf_counter()
    marginals = numpy.zeros(n_instantiations)
    feature_vals_rep = numpy.zeros(n_instantiations)

    feature_cum_sum = numpy.cumsum(feature_vals)
    for j in range(n_features):
        prev_id = feature_cum_sum[j - 1] if j > 0 else 0
        feature_vals_rep[prev_id:prev_id + feature_vals[j]] = feature_vals[j]
        for i in range(n_instances):
            obs = data[i, j]
            feature_val_id = prev_id + obs
            marginals[feature_val_id] += 1

    marginals = (marginals + alpha) / (n_instances + alpha * numpy.array(feature_vals_rep))

    feat_e_t = perf_counter()
    logging.info('Marginals extracted in {}'.format(feat_e_t - feat_s_t))

    return marginals


def extract_features_all_marginals_spn(spn,
                                       data,
                                       feature_vals,
                                       all_marginals=None,
                                       dtype=numpy.int32):

    n_instances = data.shape[0]
    n_features = data.shape[1]

    if all_marginals is None:
        all_marginals = all_single_marginals_spn(spn, feature_vals, dtype=dtype)

    repr_data = numpy.zeros((n_instances, n_features))

    feature_cum_sum = numpy.cumsum(feature_vals)
    for i in range(n_instances):
        for j in range(n_features):
            obs = data[i, j]
            prev_id = feature_cum_sum[j - 1] if j > 0 else 0
            feature_val_id = prev_id + obs
            repr_data[i, j] = all_marginals[feature_val_id]

    return repr_data


def extract_features_all_marginals_ml(train_data,
                                      test_data,
                                      feature_vals,
                                      alpha=0.0,
                                      all_marginals=None,
                                      dtype=numpy.int32):

    n_instances = test_data.shape[0]
    n_features = test_data.shape[1]

    if all_marginals is None and train_data is not None:
        all_marginals = all_single_marginals_ml(train_data,
                                                feature_vals,
                                                alpha=alpha)

    repr_data = numpy.zeros((n_instances, n_features))

    feature_cum_sum = numpy.cumsum(feature_vals)
    for i in range(n_instances):
        for j in range(n_features):
            obs = test_data[i, j]
            prev_id = feature_cum_sum[j - 1] if j > 0 else 0
            feature_val_id = prev_id + obs
            repr_data[i, j] = all_marginals[feature_val_id]

    return repr_data


def marginalizations_for_instance(spn,
                                  instance,
                                  feature_vals,
                                  exp=False,
                                  dtype=int):
    """
    Given an SPN and an instance, return a same shape instance
    containing the activations of all nodes, summed by scopes
    """

    assert instance.ndim == 1

    n_features = len(instance)
    marg_data = numpy.zeros(n_features, dtype=instance.dtype)

    marginalizations = numpy.zeros(n_features, dtype=dtype)

    for i in range(n_features):
        marg_data.fill(MARG_IND)
        # if all_ones:
        #     marg_data[i] = 1
        # else:
        marg_data[i] = instance[i]
        #
        # evaluate it bottom, up
        res, = spn.single_eval(marg_data)

        if exp:
            res = numpy.exp(res)

        marginalizations[i] = res

    return marginalizations


import theano


def get_nearest_neighbours_theano_func():
    """
    Returns the id of the nearest instance to sample and its value,
    in the euclidean distance sense
    """

    sample = theano.tensor.vector(dtype=theano.config.floatX)
    data = theano.tensor.matrix(dtype=theano.config.floatX)

    distance_vec = theano.tensor.sum((data - sample) ** 2, axis=1)
    nn_id = theano.tensor.argmin(distance_vec)

    find_nearest_neighbour = theano.function(inputs=[sample, data],
                                             outputs=[nn_id, data[nn_id]])
    return find_nearest_neighbour


def get_nearest_neighbour(samples, data, masked=False, nn_func=None):

    if nn_func is None:
        nn_func = get_nearest_neighbours_theano_func()
        data = data.astype(theano.config.floatX)
        samples = [s.astype(theano.config.floatX) for s in samples]

    neighbours = []

    for instance in samples:
        nn_s_t = perf_counter()

        if masked:
            feature_mask = instance == MARG_IND
            # print(data.shape, instance.shape, feature_mask.shape)
            # instance = instance[feature_mask]
            # masked_data = data[:, feature_mask]
            # print(masked_data.shape, instance.shape, feature_mask.shape)
            # nn_id, instance_nn = nn_func(instance, masked_data)
            #
            # putting everything dont' care as background
            print(instance.reshape(28, 28))
            instance[feature_mask] = 1
            print(instance.reshape(28, 28))

        # else:
        nn_id, instance_nn = nn_func(instance, data)
        nn_e_t = perf_counter()
        print(data.shape)
        neighbours.append((nn_id, data[nn_id]))
        logging.info('Got nn {} in {} secs'.format(nn_id, nn_e_t - nn_s_t))
    return neighbours
