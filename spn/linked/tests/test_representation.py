import dataset

from spn import MARG_IND
from ..spn import Spn
from ..spn import evaluate_on_dataset

from ..layers import SumLayer
from ..layers import ProductLayer
from ..layers import CategoricalIndicatorLayer
from ..layers import CategoricalSmoothedLayer

from ..nodes import SumNode
from ..nodes import ProductNode
from ..nodes import CategoricalIndicatorNode
from ..nodes import CategoricalSmoothedNode

from ..representation import extract_features_nodes_mpe
from ..representation import extract_features_nodes_mpn
from ..representation import node_in_path_feature
from ..representation import acc_node_in_path_feature
from ..representation import filter_non_sum_nodes
from ..representation import max_child_id_feature
from ..representation import max_hidden_var_feature, filter_hidden_var_nodes
from ..representation import extract_features_nodes
from ..representation import max_hidden_var_val, max_hidden_var_log_val
from ..representation import hidden_var_val, hidden_var_log_val
from ..representation import filter_all_nodes
from ..representation import filter_non_leaf_nodes
from ..representation import var_log_val

from ..representation import decode_embeddings_mpn
from ..representation import node_mpe_instantiation

from ..representation import random_feature_mask
from ..representation import random_rectangular_feature_mask
from ..representation import extract_feature_marginalization_from_masks
from ..representation import extract_features_marginalization_rand
from ..representation import extract_features_marginalization_rectangles
from ..representation import extract_feature_marginalization_from_masks_opt_unique

from ..representation import extract_instances_groups

from ..representation import load_feature_info
from ..representation import store_feature_info
from ..representation import filter_features_by_layer
from ..representation import filter_features_by_scope_length
from ..representation import feature_mask_from_info
from ..representation import filter_features_by_node_type

from ..representation import feature_mask_to_marg
from ..representation import extract_feature_marginalization_from_masks_theanok
from ..representation import extract_feature_marginalization_from_masks_theanok_opt_unique

from ..representation import save_features_to_file
from ..representation import load_features_from_file
from ..representation import feature_mask_scope

from ..representation import all_single_marginals_spn
from ..representation import all_single_marginals_ml
from ..representation import extract_features_all_marginals_ml
from ..representation import extract_features_all_marginals_spn

from ..representation import remove_embedding_components
from ..representation import generate_missing_masks
from ..representation import remove_embedding_components_from_mask
from ..representation import filter_2darray_2dmask

import numpy
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def test_extract_features_sum_nodes():
    #
    # creating an SPN
    ind_x_00 = CategoricalIndicatorNode(0, 0)
    ind_x_01 = CategoricalIndicatorNode(0, 1)
    ind_x_10 = CategoricalIndicatorNode(1, 0)
    ind_x_11 = CategoricalIndicatorNode(1, 1)
    ind_x_20 = CategoricalIndicatorNode(2, 0)
    ind_x_21 = CategoricalIndicatorNode(2, 1)

    input_layer = CategoricalIndicatorLayer([ind_x_00, ind_x_01,
                                             ind_x_10, ind_x_11,
                                             ind_x_20, ind_x_21])

    #
    # sum layer
    #
    sum_node_1 = SumNode()
    sum_node_1.add_child(ind_x_00, 0.1)
    sum_node_1.add_child(ind_x_01, 0.9)

    sum_node_2 = SumNode()
    sum_node_2.add_child(ind_x_00, 0.4)
    sum_node_2.add_child(ind_x_01, 0.6)

    sum_node_3 = SumNode()
    sum_node_3.add_child(ind_x_10, 0.3)
    sum_node_3.add_child(ind_x_11, 0.7)

    sum_node_4 = SumNode()
    sum_node_4.add_child(ind_x_10, 0.6)
    sum_node_4.add_child(ind_x_11, 0.4)

    sum_node_5 = SumNode()
    sum_node_5.add_child(ind_x_20, 0.5)
    sum_node_5.add_child(ind_x_21, 0.5)

    sum_node_6 = SumNode()
    sum_node_6.add_child(ind_x_20, 0.2)
    sum_node_6.add_child(ind_x_21, 0.8)

    sum_layer = SumLayer([sum_node_1, sum_node_2,
                          sum_node_3, sum_node_4,
                          sum_node_5, sum_node_6])

    #
    # prod layer
    #
    prod_node_1 = ProductNode()
    prod_node_1.add_child(sum_node_1)
    prod_node_1.add_child(sum_node_3)
    prod_node_1.add_child(sum_node_5)

    prod_node_2 = ProductNode()
    prod_node_2.add_child(sum_node_2)
    prod_node_2.add_child(sum_node_4)
    prod_node_2.add_child(sum_node_6)

    prod_node_3 = ProductNode()
    prod_node_3.add_child(sum_node_1)
    prod_node_3.add_child(sum_node_4)
    prod_node_3.add_child(sum_node_6)

    prod_node_4 = ProductNode()
    prod_node_4.add_child(sum_node_2)
    prod_node_4.add_child(sum_node_3)
    prod_node_4.add_child(sum_node_5)

    prod_layer = ProductLayer([prod_node_1, prod_node_2,
                               prod_node_3, prod_node_4])

    root = SumNode()
    root.add_child(prod_node_1, 0.1)
    root.add_child(prod_node_2, 0.2)
    root.add_child(prod_node_3, 0.25)
    root.add_child(prod_node_4, 0.45)

    root_layer = SumLayer([root])

    spn = Spn(input_layer, [sum_layer,
                            prod_layer,
                            root_layer])

    print(spn)

    ind_data = numpy.array([[0, 1, 0, 1, 1, 0],
                            [1, 0, 0, 1, 0, 1],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 1, 0, 0, 1],
                            [1, 0, 1, 0, 1, 0]])

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    for instance in data:
        res = spn.single_eval(instance)
        print([node.log_val for node in input_layer.nodes()])
        print(numpy.exp([node.log_val for node in input_layer.nodes()]))
        print([node.log_val for node in sum_layer.nodes()])
        print(numpy.exp([node.log_val for node in sum_layer.nodes()]))
        print([node.log_val for node in prod_layer.nodes()])
        print(numpy.exp([node.log_val for node in prod_layer.nodes()]))
        print(res)
        print(numpy.exp(res))
        print('\n')

    # ret_func = node_in_path_feature
    print('MPE, max Hidden var only, id')
    ret_func = max_hidden_var_feature
    filter_func = filter_hidden_var_nodes
    new_data = extract_features_nodes_mpe(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          verbose=True)
    print(new_data)

    print('Without empty features')
    new_data = extract_features_nodes_mpe(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=True,
                                          verbose=True)
    print(new_data)

    print('MPE, max Hidden var only, val')
    ret_func = max_hidden_var_val
    filter_func = filter_hidden_var_nodes
    new_data = extract_features_nodes_mpe(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          verbose=True)
    print(new_data)

    print('Hidden max var only, val')
    ret_func = max_hidden_var_val
    filter_func = filter_hidden_var_nodes
    new_data = extract_features_nodes(spn,
                                      data,
                                      filter_node_func=filter_func,
                                      retrieve_func=ret_func,
                                      remove_zero_features=False,
                                      dtype=numpy.float,
                                      verbose=True)
    print(new_data)

    print('Hidden var only, val')
    ret_func = hidden_var_val
    filter_func = filter_hidden_var_nodes
    new_data = extract_features_nodes(spn,
                                      data,
                                      filter_node_func=filter_func,
                                      retrieve_func=ret_func,
                                      remove_zero_features=False,
                                      dtype=numpy.float,
                                      verbose=True)
    print(new_data)


def build_test_mini_spn():
    #
    # creating an SPN
    ind_x_00 = CategoricalIndicatorNode(0, 0)
    ind_x_01 = CategoricalIndicatorNode(0, 1)
    ind_x_10 = CategoricalIndicatorNode(1, 0)
    ind_x_11 = CategoricalIndicatorNode(1, 1)
    ind_x_20 = CategoricalIndicatorNode(2, 0)
    ind_x_21 = CategoricalIndicatorNode(2, 1)

    input_layer = CategoricalIndicatorLayer([ind_x_00, ind_x_01,
                                             ind_x_10, ind_x_11,
                                             ind_x_20, ind_x_21])

    #
    # sum layer
    #
    sum_node_1 = SumNode(var_scope=frozenset([0]))
    sum_node_1.add_child(ind_x_00, 0.1)
    sum_node_1.add_child(ind_x_01, 0.9)

    sum_node_2 = SumNode(var_scope=frozenset([0]))
    sum_node_2.add_child(ind_x_00, 0.4)
    sum_node_2.add_child(ind_x_01, 0.6)

    sum_node_3 = SumNode(var_scope=frozenset([1]))
    sum_node_3.add_child(ind_x_10, 0.3)
    sum_node_3.add_child(ind_x_11, 0.7)

    sum_node_4 = SumNode(var_scope=frozenset([1]))
    sum_node_4.add_child(ind_x_10, 0.6)
    sum_node_4.add_child(ind_x_11, 0.4)

    sum_node_5 = SumNode(var_scope=frozenset([2]))
    sum_node_5.add_child(ind_x_20, 0.5)
    sum_node_5.add_child(ind_x_21, 0.5)

    sum_node_6 = SumNode(var_scope=frozenset([2]))
    sum_node_6.add_child(ind_x_20, 0.2)
    sum_node_6.add_child(ind_x_21, 0.8)

    sum_layer = SumLayer([sum_node_1, sum_node_2,
                          sum_node_3, sum_node_4,
                          sum_node_5, sum_node_6])

    #
    # prod layer
    #
    prod_node_1 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_1.add_child(sum_node_1)
    prod_node_1.add_child(sum_node_3)
    prod_node_1.add_child(sum_node_5)

    prod_node_2 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_2.add_child(sum_node_2)
    prod_node_2.add_child(sum_node_4)
    prod_node_2.add_child(sum_node_6)

    prod_node_3 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_3.add_child(sum_node_1)
    prod_node_3.add_child(sum_node_4)
    prod_node_3.add_child(sum_node_6)

    prod_node_4 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_4.add_child(sum_node_2)
    prod_node_4.add_child(sum_node_3)
    prod_node_4.add_child(sum_node_5)

    prod_layer = ProductLayer([prod_node_1, prod_node_2,
                               prod_node_3, prod_node_4])

    root = SumNode(var_scope=frozenset([0, 1, 2]))
    root.add_child(prod_node_1, 0.1)
    root.add_child(prod_node_2, 0.2)
    root.add_child(prod_node_3, 0.25)
    root.add_child(prod_node_4, 0.45)

    root_layer = SumLayer([root])

    spn = Spn(input_layer, [sum_layer,
                            prod_layer,
                            root_layer])

    print(spn)

    layers = [input_layer, sum_layer, prod_layer, root_layer]
    nodes = [node for layer in layers for node in layer.nodes()]
    return spn, layers, nodes


def build_test_mini_spn_cat():

    leaf_1 = CategoricalSmoothedNode(0, 2, freqs=[0.1, 0.9], alpha=0.0)
    leaf_2 = CategoricalSmoothedNode(0, 2, freqs=[0.4, 0.6], alpha=0.0)
    leaf_3 = CategoricalSmoothedNode(1, 2, freqs=[0.3, 0.7], alpha=0.0)
    leaf_4 = CategoricalSmoothedNode(1, 2, freqs=[0.6, 0.4], alpha=0.0)
    leaf_5 = CategoricalSmoothedNode(2, 2, freqs=[0.5, 0.5], alpha=0.0)
    leaf_6 = CategoricalSmoothedNode(2, 2, freqs=[0.2, 0.8], alpha=0.0)

    input_layer = CategoricalSmoothedLayer([leaf_1, leaf_2,
                                            leaf_3, leaf_4,
                                            leaf_5, leaf_6])

    #
    # prod layer
    #
    prod_node_1 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_1.add_child(leaf_1)
    prod_node_1.add_child(leaf_3)
    prod_node_1.add_child(leaf_5)

    prod_node_2 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_2.add_child(leaf_2)
    prod_node_2.add_child(leaf_4)
    prod_node_2.add_child(leaf_6)

    prod_node_3 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_3.add_child(leaf_1)
    prod_node_3.add_child(leaf_4)
    prod_node_3.add_child(leaf_6)

    prod_node_4 = ProductNode(var_scope=frozenset([0, 1, 2]))
    prod_node_4.add_child(leaf_2)
    prod_node_4.add_child(leaf_3)
    prod_node_4.add_child(leaf_5)

    prod_layer = ProductLayer([prod_node_1, prod_node_2,
                               prod_node_3, prod_node_4])

    root = SumNode(var_scope=frozenset([0, 1, 2]))
    root.add_child(prod_node_1, 0.1)
    root.add_child(prod_node_2, 0.2)
    root.add_child(prod_node_3, 0.25)
    root.add_child(prod_node_4, 0.45)

    root_layer = SumLayer([root])

    spn = Spn(input_layer, [prod_layer,
                            root_layer])

    print(spn)

    layers = [input_layer, prod_layer, root_layer]
    nodes = [node for layer in layers for node in layer.nodes()]
    return spn, layers, nodes


def test_extract_features_all():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    spn, layers, nodes = build_test_mini_spn()

    feature_info_file = 'test_extract_features_all.feature.info'

    ret_func = var_log_val
    filter_func = filter_all_nodes
    new_data = extract_features_nodes(spn,
                                      data,
                                      filter_node_func=filter_func,
                                      retrieve_func=ret_func,
                                      remove_zero_features=False,
                                      dtype=numpy.float,
                                      output_feature_info=feature_info_file,
                                      verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)


def test_extract_features_all_mpn():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    spn, layers, nodes = build_test_mini_spn()

    feature_info_file = 'test_extract_features_all_mpn.feature.info'

    ret_func = var_log_val
    filter_func = filter_all_nodes
    new_data = extract_features_nodes_mpn(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          output_feature_info=feature_info_file,
                                          verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)


def test_encode_decode_features_mpn():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    spn, layers, nodes = build_test_mini_spn()

    feature_info_file = 'test_extract_features_all_mpn.feature.info'

    ret_func = var_log_val
    filter_func = filter_all_nodes
    new_data = extract_features_nodes_mpn(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          output_feature_info=feature_info_file,
                                          verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)

    #
    # now decoding
    feature_info = load_feature_info(feature_info_file)
    dec_data = decode_embeddings_mpn(spn,
                                     new_data,
                                     feature_info,
                                     n_features=3,
                                     verbose=True)
    print('Decoded data:\n', dec_data)


def test_encode_decode_non_leaf_features_mpn():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    spn, layers, nodes = build_test_mini_spn()

    feature_info_file = 'test_extract_features_non_leaf_mpn.feature.info'

    ret_func = var_log_val
    filter_func = filter_non_leaf_nodes
    new_data = extract_features_nodes_mpn(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          output_feature_info=feature_info_file,
                                          verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)

    #
    # now decoding
    feature_info = load_feature_info(feature_info_file)
    dec_data = decode_embeddings_mpn(spn,
                                     new_data,
                                     feature_info,
                                     n_features=3,
                                     verbose=True)
    print('Decoded data:\n', dec_data)


def test_encode_decode_all_features_mpn_cat():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    feature_info_file = 'test_extract_features_all_mpn_cat.feature.info'

    spn, layers, nodes = build_test_mini_spn_cat()

    ret_func = var_log_val
    filter_func = filter_all_nodes
    new_data = extract_features_nodes_mpn(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          output_feature_info=feature_info_file,
                                          verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)

    #
    # now decoding
    feature_info = load_feature_info(feature_info_file)
    dec_data = decode_embeddings_mpn(spn,
                                     new_data,
                                     feature_info,
                                     n_features=3,
                                     verbose=True)
    print('Decoded data:\n', dec_data)


def test_encode_decode_non_leaf_features_mpn_cat():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    feature_info_file = 'test_extract_features_non_leaf_mpn_cat.feature.info'

    spn, layers, nodes = build_test_mini_spn_cat()

    ret_func = var_log_val
    filter_func = filter_non_leaf_nodes
    new_data = extract_features_nodes_mpn(spn,
                                          data,
                                          filter_node_func=filter_func,
                                          retrieve_func=ret_func,
                                          remove_zero_features=False,
                                          dtype=numpy.float,
                                          output_feature_info=feature_info_file,
                                          verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)

    #
    # now decoding
    feature_info = load_feature_info(feature_info_file)
    dec_data = decode_embeddings_mpn(spn,
                                     new_data,
                                     feature_info,
                                     n_features=3,
                                     verbose=True)
    print('Decoded data:\n', dec_data)


def test_extract_features_all_extract_info():

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    spn, layers, nodes = build_test_mini_spn()

    feature_info_file = 'test_extract_features_all.feature.info'

    ret_func = var_log_val
    filter_func = filter_all_nodes
    new_data = extract_features_nodes(spn,
                                      data,
                                      filter_node_func=filter_func,
                                      retrieve_func=ret_func,
                                      remove_zero_features=False,
                                      dtype=numpy.float,
                                      output_feature_info=feature_info_file,
                                      verbose=True)

    print('Repre shape {}'.format(new_data.shape))
    print(new_data)

    print('Loading feature info back')
    feature_info = load_feature_info(feature_info_file)
    print(feature_info)

    n_features = len(feature_info)

    all_feature_mask = feature_mask_from_info(feature_info, n_features)
    assert_array_almost_equal(all_feature_mask, numpy.ones(n_features, dtype=bool))

    #
    # saving again
    feature_info_file_2 = 'test_extract_features_all_2.feature.info'
    store_feature_info(feature_info, feature_info_file_2)
    feature_info_2 = load_feature_info(feature_info_file_2)
    print(feature_info_2)
    for i_1, i_2 in zip(feature_info, feature_info_2):
        assert i_1 == i_2

    print('Extracting all features from different levels')
    n_layers = spn.n_layers()
    for i in range(n_layers):
        print('\tlayer: {}'.format(i))
        filtered_feature_info = filter_features_by_layer(feature_info, i)
        feature_mask = feature_mask_from_info(filtered_feature_info, n_features)
        print(feature_mask)

    print('Extracting all features from different scopes')
    scope_lengths = set()
    for node in nodes:
        scope = None
        if hasattr(node, 'var_scope'):
            scope = node.var_scope
        elif hasattr(node, 'var'):
            scope = node.var
        scope_lengths.add(scope)
    n_scope_lengths = len(scope_lengths)
    for i in range(1, n_scope_lengths):
        print('\tscope length {}'.format(i))
        filtered_feature_info = filter_features_by_scope_length(feature_info, i)

        feature_mask = feature_mask_from_info(filtered_feature_info, n_features)
        if i == 2:
            assert not any(feature_mask)

        print(feature_mask)

    print('Extracting all features from different node types')
    node_types = ('SumNode', 'ProductNode')
    for type in node_types:
        print('\tnode type {}'.format(type))
        filtered_feature_info = filter_features_by_node_type(feature_info, type)
        feature_mask = feature_mask_from_info(filtered_feature_info, n_features)
        print(feature_mask)


def test_node_mpe_instantiation():

    spn, layers, nodes = build_test_mini_spn()
    input_layer, sum_layer, prod_layer, root_layer = layers

    ind_data = numpy.array([[0, 1, 0, 1, 1, 0],
                            [1, 0, 0, 1, 0, 1],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 1, 0, 0, 1],
                            [1, 0, 1, 0, 1, 0]])

    data = numpy.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

    for instance in data:
        res = spn.single_eval(instance)
        print([node.log_val for node in input_layer.nodes()])
        print(numpy.exp([node.log_val for node in input_layer.nodes()]))
        print([node.log_val for node in sum_layer.nodes()])
        print(numpy.exp([node.log_val for node in sum_layer.nodes()]))
        print([node.log_val for node in prod_layer.nodes()])
        print(numpy.exp([node.log_val for node in prod_layer.nodes()]))
        print(res)
        print(numpy.exp(res))
        print('\n')

    #
    # mpe bottom up pass
    for instance in data:
        res = spn.single_mpe_eval(instance)
        instances = node_mpe_instantiation(spn.root(), 3)
        print('MPE instances:\n {}'.format(instances))


def test_random_feature_mask():
    n_features = 20
    feature_mask = numpy.zeros(n_features, dtype=bool)
    assert sum(feature_mask) == 0
    n_rand_features = 5
    feature_mask = random_feature_mask(feature_mask, n_rand_features)
    assert sum(feature_mask) == n_rand_features


def test_random_rectangular_feature_mask():
    n_trials = 20
    rand_gen = numpy.random.RandomState(1337)
    for i in range(n_trials):
        n_features = 49
        feature_mask = numpy.zeros(n_features, dtype=bool)
        assert sum(feature_mask) == 0
        n_rows = int(numpy.sqrt(n_features))
        n_cols = int(numpy.sqrt(n_features))

        n_min_rows = 3
        n_min_cols = 3
        n_max_rows = 4
        n_max_cols = 4
        feature_mask = random_rectangular_feature_mask(feature_mask,
                                                       n_rows, n_cols,
                                                       n_min_rows, n_min_cols,
                                                       n_max_rows, n_max_cols,
                                                       rand_gen=rand_gen)
        #
        # reshaping
        feature_mask_rect = feature_mask.reshape(n_rows, n_cols)
        print(feature_mask_rect)
        assert sum(feature_mask) >= n_min_rows * n_min_cols
        assert sum(feature_mask) <= n_max_rows * n_max_cols


def test_extract_feature_marginalization():
    n_instances = 10
    n_features = 3
    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))

    rand_gen = numpy.random.RandomState(1337)
    print(data)

    #
    # generating a set of feature masks randomly
    n_masks = 10
    masks = []
    for i in range(n_masks):
        feature_mask = numpy.zeros(n_features, dtype=bool)
        assert sum(feature_mask) == 0
        n_rand_features = 2
        feature_mask = random_feature_mask(feature_mask, n_rand_features, rand_gen=rand_gen)
        assert sum(feature_mask) == n_rand_features
        print(feature_mask)
        masks.append(feature_mask)

    spn, _layers, _nodes = build_test_mini_spn()

    repr_data = extract_feature_marginalization_from_masks(spn, data, masks, rand_gen=rand_gen)
    assert repr_data.shape[0] == n_instances
    assert repr_data.shape[1] == len(masks)
    print(repr_data)

    #
    # evaluating the spn by hand
    for i, mask in enumerate(masks):
        masked_data = numpy.array(data, copy=True)
        inv_mask = numpy.logical_not(mask)
        print('inv mask', inv_mask)
        masked_data[:, inv_mask] = MARG_IND
        # print('{}\n{}'.format(i, masked_data))
        preds = evaluate_on_dataset(spn, masked_data)
        assert_array_almost_equal(preds, repr_data[:, i])

    #
    # with the complete method
    print('Calling extract_features_marginalization_rand')
    rand_gen = numpy.random.RandomState(1337)
    feature_sizes = [10]
    n_rand_sizes = [2]
    feature_masks = extract_features_marginalization_rand(n_features,
                                                          feature_sizes,
                                                          n_rand_sizes,
                                                          rand_gen=rand_gen)

    repr_data_2 = extract_feature_marginalization_from_masks(spn,
                                                             data,
                                                             feature_masks,
                                                             marg_value=MARG_IND,
                                                             rand_gen=rand_gen,
                                                             dtype=float)

    assert_array_almost_equal(repr_data, repr_data_2)
    print('Results are reproducible by setting the seeds')


def test_extract_feature_marginalization_opt():
    n_instances = 100
    n_features = 3
    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))

    rand_gen = numpy.random.RandomState(1337)
    print(data)

    #
    # generating a set of feature masks randomly
    n_masks = 10
    masks = []
    for i in range(n_masks):
        feature_mask = numpy.zeros(n_features, dtype=bool)
        assert sum(feature_mask) == 0
        n_rand_features = 2
        feature_mask = random_feature_mask(feature_mask, n_rand_features, rand_gen=rand_gen)
        assert sum(feature_mask) == n_rand_features
        print(feature_mask)
        masks.append(feature_mask)

    spn, _layers, _nodes = build_test_mini_spn()

    repr_data = extract_feature_marginalization_from_masks(spn, data, masks)
    assert repr_data.shape[0] == n_instances
    assert repr_data.shape[1] == len(masks)
    print(repr_data)

    #
    # evaluating the spn by hand
    for i, mask in enumerate(masks):
        masked_data = numpy.array(data, copy=True)
        inv_mask = numpy.logical_not(mask)
        print('inv mask', inv_mask)
        masked_data[:, inv_mask] = MARG_IND
        # print('{}\n{}'.format(i, masked_data))
        preds = evaluate_on_dataset(spn, masked_data)
        assert_array_almost_equal(preds, repr_data[:, i])

    #
    # now with the optimization on unique values
    repr_data_2 = extract_feature_marginalization_from_masks_opt_unique(spn,
                                                                        data,
                                                                        masks)
    assert repr_data_2.shape[0] == n_instances
    assert repr_data_2.shape[1] == len(masks)
    print(repr_data_2)

    assert_array_almost_equal(repr_data, repr_data_2)


def test_extract_instances_groups():
    data = numpy.array([[1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 0, 1],
                        [0, 0, 0, 1],
                        [1, 0, 1, 0],
                        [0, 0, 0, 0],
                        [1, 0, 1, 0],
                        [0, 0, 0, 0]])

    n_instances = data.shape[0]

    repr_data = extract_instances_groups(data)
    print(repr_data)

    assert numpy.sum(repr_data) == n_instances
    assert numpy.allclose(repr_data[0], repr_data[1])
    assert numpy.allclose(repr_data[5], repr_data[7])
    assert numpy.allclose(repr_data[4], repr_data[6])


def test_feature_mask_to_marg():
    feature_vals = [2, 2, 2, 2]
    feature_mask = numpy.array([False, True, True, False], dtype=bool)
    n_ohe_features = numpy.sum(feature_vals)
    ohe_feature_mask = feature_mask_to_marg(feature_mask, n_ohe_features, feature_vals)
    print(ohe_feature_mask)

    true_mask = numpy.array([False, False,  True, True, True, True, False, False], dtype=bool)
    assert_array_almost_equal(ohe_feature_mask, true_mask)

    feature_vals = [3, 2, 2, 2, 4]
    feature_mask = numpy.array([True, False, True, True, False, False], dtype=bool)
    n_ohe_features = numpy.sum(feature_vals)
    ohe_feature_mask = feature_mask_to_marg(feature_mask, n_ohe_features, feature_vals)
    print(ohe_feature_mask)

    true_mask = numpy.array([True, True, True,
                             False, False,
                             True, True,
                             True, True,
                             False, False, False, False], dtype=bool)
    assert_array_almost_equal(ohe_feature_mask, true_mask)


from spn.factory import build_theanok_spn_from_block_linked


def test_extract_feature_marginalization_from_masks_theanok():
    n_features = 3
    n_instances = 20
    feature_vals = [2, 2, 2]

    #
    # generate some masks
    print('Calling extract_features_marginalization_rand')

    rand_gen = numpy.random.RandomState(1337)
    feature_sizes = [10]
    n_rand_sizes = [2]
    feature_masks = extract_features_marginalization_rand(n_features,
                                                          feature_sizes,
                                                          n_rand_sizes,
                                                          rand_gen=rand_gen)

    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))
    ind_data = dataset.one_hot_encoding(data, feature_vals)

    rand_gen = numpy.random.RandomState(1337)
    print(data)

    spn, _layers, _nodes = build_test_mini_spn()

    repr_data = extract_feature_marginalization_from_masks(spn,
                                                           data,
                                                           feature_masks,
                                                           rand_gen=rand_gen)

    #
    # now doing the same for theano
    theano_spn = build_theanok_spn_from_block_linked(spn, ind_data.shape[1], feature_vals)
    print(theano_spn)

    theano_repr_data = extract_feature_marginalization_from_masks_theanok(theano_spn,
                                                                          ind_data,
                                                                          feature_masks,
                                                                          feature_vals,
                                                                          rand_gen=rand_gen)
    print(theano_repr_data)

    assert_array_almost_equal(repr_data, theano_repr_data)

    #
    # now doing that by "hand"
    for i, mask in enumerate(feature_masks):
        marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)
        marg_data.fill(MARG_IND)
        marg_data[:, mask] = data[:, mask]

        ind_marg_data = dataset.one_hot_encoding(marg_data, feature_vals)
        preds = theano_spn.evaluate(ind_marg_data.astype(numpy.float32))
        assert_array_almost_equal(theano_repr_data[:, i], preds.flatten())


def test_extract_feature_marginalization_from_masks_theanok_opt():
    n_features = 3
    n_instances = 100
    feature_vals = [2, 2, 2]

    #
    # generate some masks
    print('Calling extract_features_marginalization_rand')

    rand_gen = numpy.random.RandomState(1337)
    feature_sizes = [10]
    n_rand_sizes = [2]
    feature_masks = extract_features_marginalization_rand(n_features,
                                                          feature_sizes,
                                                          n_rand_sizes,
                                                          rand_gen=rand_gen)

    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))
    ind_data = dataset.one_hot_encoding(data, feature_vals)

    rand_gen = numpy.random.RandomState(1337)
    print(data)

    spn, _layers, _nodes = build_test_mini_spn()

    repr_data = extract_feature_marginalization_from_masks(spn,
                                                           data,
                                                           feature_masks)

    #
    # now doing the same for theano
    theano_spn = build_theanok_spn_from_block_linked(spn, ind_data.shape[1], feature_vals)
    print(theano_spn)

    theano_repr_data = extract_feature_marginalization_from_masks_theanok(theano_spn,
                                                                          ind_data,
                                                                          feature_masks,
                                                                          feature_vals)
    print(theano_repr_data)

    assert_array_almost_equal(repr_data, theano_repr_data)

    #
    # now doing that by "hand"
    for i, mask in enumerate(feature_masks):
        marg_data = numpy.zeros((n_instances, n_features), dtype=data.dtype)
        marg_data.fill(MARG_IND)
        marg_data[:, mask] = data[:, mask]

        ind_marg_data = dataset.one_hot_encoding(marg_data, feature_vals)
        preds = theano_spn.evaluate(ind_marg_data.astype(numpy.float32))
        assert_array_almost_equal(theano_repr_data[:, i], preds.flatten())

    #
    # and now with optimization
    theano_repr_data_2 = extract_feature_marginalization_from_masks_theanok_opt_unique(theano_spn,
                                                                                       ind_data,
                                                                                       feature_masks,
                                                                                       feature_vals)
    print(theano_repr_data_2)

    assert_array_almost_equal(theano_repr_data, theano_repr_data_2)


def test_load_save_features_masks_file():
    #
    # creating some feature_masks
    n_masks = 100
    n_features = 10
    n_rand_features = 4
    feature_masks = []
    rand_gen = numpy.random.RandomState(1337)
    print('Creating features')
    for i in range(n_masks):
        feature_mask = numpy.zeros(n_features, dtype=bool)
        feature_mask = random_feature_mask(feature_mask, n_rand_features, rand_gen=rand_gen)
        print(feature_mask)
        feature_masks.append(feature_mask)

    #
    # saving them to file
    file_path = 'test.features'
    save_features_to_file(feature_masks, file_path)

    #
    # now loading them back
    print('Loading them back')
    rec_feature_masks = load_features_from_file(file_path)
    for i in range(rec_feature_masks.shape[0]):
        print(rec_feature_masks[i])

    #
    # asserting equality
    assert_array_almost_equal(numpy.array(feature_masks), rec_feature_masks)


def test_feature_mask_scope():

    n_rand_features = 4
    n_features = 10
    rand_gen = numpy.random.RandomState(1337)
    feature_mask = numpy.zeros(n_features, dtype=bool)
    feature_mask = random_feature_mask(feature_mask, n_rand_features, rand_gen=rand_gen)

    n_true_features = numpy.sum(feature_mask)
    print(feature_mask)
    scope = feature_mask_scope(feature_mask)
    print(scope)

    assert n_true_features == len(scope)
    scope_list = []
    for i, f in enumerate(feature_mask):
        if f:
            scope_list.append(i)
    assert_array_almost_equal(scope, numpy.array(scope_list))


def test_all_single_marginals():
    n_features = 3
    n_instances = 100
    feature_vals = [2, 2, 2]

    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))

    spn, _layers, _nodes = build_test_mini_spn()

    all_marginals = all_single_marginals_spn(spn, feature_vals)

    #
    # computing them by hand
    marginals = []
    for i in range(n_features):
        for j in range(feature_vals[i]):
            marg_instance = numpy.zeros(n_features, dtype=data.dtype)
            marg_instance.fill(MARG_IND)
            marg_instance[i] = j
            print(marg_instance)
            marg_res,  = spn.single_eval(marg_instance)
            marginals.append(marg_res)

    print('Computed', all_marginals)
    print('Expected', marginals)
    assert_array_almost_equal(numpy.array(marginals), all_marginals)


def test_extract_features_all_marginals():
    n_features = 3
    n_instances = 100
    feature_vals = [2, 2, 2]

    #
    # creating some data
    rand_gen = numpy.random.RandomState(1337)
    data = rand_gen.binomial(1, 0.5, size=(n_instances, n_features))

    spn, _layers, _nodes = build_test_mini_spn()

    repr_data = extract_features_all_marginals_spn(spn,
                                                   data,
                                                   feature_vals)

    #
    # doing by hand
#
    # computing them by hand
    marginals = []
    for i in range(n_features):
        for j in range(feature_vals[i]):
            marg_instance = numpy.zeros(n_features, dtype=data.dtype)
            marg_instance.fill(MARG_IND)
            marg_instance[i] = j
            print(marg_instance)
            marg_res,  = spn.single_eval(marg_instance)
            marginals.append(marg_res)

    marginals = numpy.array(marginals)

    repr_data_2 = numpy.zeros(data.shape)
    for i in range(n_instances):
        for j in range(n_features):
            f_id = numpy.sum(feature_vals[:j]) + data[i, j]
            repr_data_2[i, j] = marginals[f_id]

    print('Computed', repr_data)
    print('Expected', repr_data_2)
    assert_array_almost_equal(repr_data, repr_data_2)


def test_remove_embdeding_components():
    #
    # creating random assoc
    n_components = 100
    components = numpy.arange(n_components)
    numpy.random.shuffle(components)
    nodes = numpy.arange(n_components)
    numpy.random.shuffle(nodes)
    node_feature_assoc = {k: v for k, v in zip(nodes, components)}

    print(node_feature_assoc)

    percs = [0.9, .8, .7, .6, .5, .4, .3, .2, .1]
    for perc in percs:
        rem_assoc = remove_embedding_components(node_feature_assoc, perc)

        print(rem_assoc, 'len', len(rem_assoc))

        assert len(node_feature_assoc) == n_components

        assert abs(len(rem_assoc) - int(perc * n_components)) <= 1

        for k, v in rem_assoc.items():
            assert k in node_feature_assoc
            assert rem_assoc[k] == node_feature_assoc[k]


def test_generate_missing_masks():
    n_components = 100
    n_instances = 1000

    components = numpy.arange(n_components)
    numpy.random.shuffle(components)
    nodes = numpy.arange(n_components)
    numpy.random.shuffle(nodes)
    #
    # not all features can be in the assoc
    node_feature_assoc = {k: v for k, v in zip(nodes, components) if
                          numpy.random.rand() > 0.15}
    n_features = len(node_feature_assoc)
    print('n features in assoc', n_features)

    one_mask = False
    percs = [0.9, .8, .7, .6, .5, .4, .3, .2, .1]
    for perc in percs:
        print('perc', perc)
        n_to_remove = max(int(perc * n_features), 1)
        print('# features to remove', n_to_remove)
        masks = generate_missing_masks(n_components, n_instances, node_feature_assoc,
                                       perc, one_mask)
        assert_array_almost_equal(masks.sum(axis=1), numpy.array([n_features - n_to_remove
                                                                  for i in range(n_instances)]))

    one_mask = True
    percs = [0.9, .8, .7, .6, .5, .4, .3, .2, .1]
    for perc in percs:
        masks = generate_missing_masks(n_components, n_instances, node_feature_assoc,
                                       perc, one_mask)
        assert_array_almost_equal(masks.sum(axis=1) / n_features, numpy.array([1. - perc
                                                                               for i in range(n_instances)]),
                                  decimal=2)
        for i in range(n_instances):
            assert_array_equal(masks[0], masks[i])

    one_mask = True


def test_remove_embdeding_components_from_mask():
    n_instances = 1000
    n_components = 100
    components = numpy.arange(n_components)
    numpy.random.shuffle(components)
    nodes = numpy.arange(n_components)
    numpy.random.shuffle(nodes)
    node_feature_assoc = {k: v for k, v in zip(nodes, components)}

    print(node_feature_assoc)

    one_mask = False
    percs = [0.9, .8, .7, .6, .5, .4, .3, .2, .1]
    for perc in percs:

        masks = generate_missing_masks(n_components, n_instances, perc, one_mask)

        for i in range(n_instances):
            rem_assoc = remove_embedding_components_from_mask(node_feature_assoc, masks[i])

            # print(rem_assoc, 'len', len(rem_assoc))

            assert len(node_feature_assoc) == n_components

            assert abs(len(rem_assoc) - int((1 - perc) * n_components)) <= 1

            for k, v in rem_assoc.items():
                assert k in node_feature_assoc
                assert rem_assoc[k] == node_feature_assoc[k]
                assert masks[i, v]


def test_filter_2darray_2dmask():
    n_features = 100
    n_instances = 100
    n_components = n_features * n_instances
    r = 50
    a = numpy.arange(n_components).reshape(n_instances, n_features)

    rand_mask = numpy.zeros((n_instances, n_features), dtype=bool)
    for i in range(n_instances):
        rand_mask[i, numpy.random.choice(n_features, r, replace=False)] = 1

    a_f = filter_2darray_2dmask(a, rand_mask)

    for i in range(n_instances):
        k = 0
        for j in range(n_features):
            if rand_mask[i, j]:
                assert a_f[i, k] == a[i, j]
                k += 1
