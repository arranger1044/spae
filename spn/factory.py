from spn.linked.spn import Spn as SpnLinked

from spn.linked.layers import Layer as LayerLinked
from spn.linked.layers import SumLayer as SumLayerLinked
from spn.linked.layers import ProductLayer as ProductLayerLinked
from spn.linked.layers import CategoricalInputLayer
from spn.linked.layers import CategoricalSmoothedLayer \
    as CategoricalSmoothedLayerLinked
from spn.linked.layers import CategoricalIndicatorLayer \
    as CategoricalIndicatorLayerLinked
from spn.linked.layers import CategoricalCLInputLayer \
    as CategoricalCLInputLayerLinked

from spn.linked.nodes import Node
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CLTreeNode
from spn.linked.nodes import ConstantNode

from spn.theanok.layers import SumLayer_logspace as SumLayerTheanok
from spn.theanok.layers import ProductLayer_logspace as ProductLayerTheanok
from spn.theanok.layers import TheanokLayer as LayerTheanok
from spn.theanok.layers import InputLayer_logspace as InputLayerTheanok
from spn.theanok.spn import BlockLayeredSpn

from spn.utils import pairwise

from spn import INT_TYPE

import numpy

from math import ceil

from theano import config

import scipy.sparse

import sklearn.preprocessing

import random

import itertools

from collections import deque
from collections import defaultdict

import dataset

import logging


class SpnFactory(object):

    """
    WRITEME
    """

    #####################################################
    #
    #####################################################
    @classmethod
    def linked_kernel_density_estimation(cls,
                                         n_instances,
                                         features,
                                         node_dict=None,
                                         alpha=0.1
                                         # ,batch_size=1,
                                         # sparse=False
                                         ):
        """
        WRITEME
        """

        n_features = len(features)

        # the top one is a sum layer with a single node
        root_node = SumNode()
        root_layer = SumLayerLinked([root_node])

        # second one is a product layer with n_instances nodes
        product_nodes = [ProductNode() for i in range(n_instances)]
        product_layer = ProductLayerLinked(product_nodes)
        # linking them to the root node
        for prod_node in product_nodes:
            root_node.add_child(prod_node, 1. / n_instances)

        # last layer can be a categorical smoothed input
        # or sum_layer + categorical indicator input

        input_layer = None
        layers = None
        n_leaf_nodes = n_features * n_instances

        if node_dict is None:
            # creating a sum_layer with n_leaf_nodes
            sum_nodes = [SumNode() for i in range(n_leaf_nodes)]
            # store them into a layer
            sum_layer = SumLayerLinked(sum_nodes)
            # linking them to the products above
            for i, prod_node in enumerate(product_nodes):
                for j in range(n_features):
                    # getting the next n_features nodes
                    prod_node.add_child(sum_nodes[i * n_features + j])
            # now creating the indicator nodes
            input_layer = \
                CategoricalIndicatorLayerLinked(vars=features)
            # linking the sum nodes to the indicator vars
            for i, sum_node in enumerate(sum_nodes):
                # getting the feature id
                j = i % n_features
                # and thus its number of values
                n_values = features[j]
                # getting the indices of indicators
                start_index = sum(features[:j])
                end_index = start_index + n_values
                indicators = [node for node
                              in input_layer.nodes()][start_index:end_index]
                for ind_node in indicators:
                    sum_node.add_child(ind_node, 1. / n_values)

            # storing levels
            layers = [sum_layer, product_layer,
                      root_layer]
        else:
            # create a categorical smoothed layer
            input_layer = \
                CategoricalSmoothedLayerLinked(vars=features,
                                               node_dicts=node_dict,
                                               alpha=alpha)
            # it shall contain n_leaf_nodes nodes
            smooth_nodes = list(input_layer.nodes())
            assert len(smooth_nodes) == n_leaf_nodes

            # linking it
            for i, prod_node in enumerate(product_nodes):
                for j in range(n_features):
                    # getting the next n_features nodes
                    prod_node.add_child(smooth_nodes[i * n_features + j])
            # setting the used levels
            layers = [product_layer, root_layer]

        # create the spn from levels
        kern_spn = SpnLinked(input_layer, layers)
        return kern_spn

    @classmethod
    def linked_naive_factorization(cls,
                                   features,
                                   node_dict=None,
                                   alpha=0.1):
        """
        WRITEME
        """
        n_features = len(features)

        # create an input layer
        input_layer = None
        layers = None

        # first layer is a product layer with n_feature children
        root_node = ProductNode()
        root_layer = ProductLayerLinked([root_node])

        # second is a sum node on an indicator layer
        if node_dict is None:
            # creating sum nodes
            sum_nodes = [SumNode() for i in range(n_features)]
            # linking to the root
            for node in sum_nodes:
                root_node.add_child(node)
            # store into a level
            sum_layer = SumLayerLinked(sum_nodes)
            # now create an indicator layer
            input_layer = CategoricalIndicatorLayerLinked(vars=features)
            # and linking it
            # TODO make this a function
            for i, sum_node in enumerate(sum_nodes):
                # getting the feature id
                j = i % n_features
                # and thus its number of values
                n_values = features[j]
                # getting the indices of indicators
                start_index = sum(features[:j])
                end_index = start_index + n_values
                indicators = [node for node
                              in input_layer.nodes()][start_index:end_index]
                for ind_node in indicators:
                    sum_node.add_child(ind_node, 1. / n_values)

            # collecting layers
            layers = [sum_layer, root_layer]

        # or a categorical smoothed layer
        else:
            input_layer = CategoricalSmoothedLayerLinked(vars=features,
                                                         node_dicts=node_dict,
                                                         alpha=alpha)
            # it shall contain n_features nodes
            smooth_nodes = list(input_layer.nodes())
            assert len(smooth_nodes) == n_features
            for node in smooth_nodes:
                root_node.add_child(node)

            # set layers accordingly
            layers = [root_layer]

        # build the spn
        naive_fact_spn = SpnLinked(input_layer, layers)

        return naive_fact_spn

    @classmethod
    def linked_random_spn_top_down(cls,
                                   vars,
                                   n_layers,
                                   n_max_children,
                                   n_scope_children,
                                   max_scope_split,
                                   merge_prob=0.5,
                                   rand_gen=None):
        """
        WRITEME
        """

        def cluster_scopes(scope_list):
            cluster_dict = {}

            for i, var in enumerate(scope_list):
                cluster_dict[var] += {i}
            return cluster_dict

        def cluster_set_scope(scope_list):
            return {scope for scope in scope_list}

        def link_leaf_to_input_layer(sum_leaf,
                                     scope_var,
                                     input_layer,
                                     rand_gen):
            for indicator_node in input_layer.nodes():
                if indicator_node.var == scope_var:
                    rand_weight = rand_gen.random()
                    sum_leaf.add_child(indicator_node, rand_weight)
                    # print(sum_leaf, indicator_node, rand_weight)
            # normalizing
            sum_leaf.normalize()
        #
        # creating a product layer
        #

        def build_product_layer(parent_layer,
                                parent_scope_list,
                                n_max_children,
                                n_scope_children,
                                input_layer,
                                rand_gen):

            # grouping the scopes of the parents
            scope_clusters = cluster_set_scope(parent_scope_list)
            # for each scope add a fixed number of children
            children_lists = {scope: [ProductNode(var_scope=scope)
                                      for i in range(n_scope_children)]
                              for scope in scope_clusters}
            # counting which node is used
            children_counts = {scope: [0 for i in range(n_scope_children)]
                               for scope in scope_clusters}
            # now link those randomly to their parent
            for parent, scope in zip(parent_layer.nodes(), parent_scope_list):
                # only for nodes not becoming leaves
                if len(scope) > 1:
                    # sampling at most n_max_children from those in the same
                    # scope
                    children_scope_list = children_lists[scope]
                    sample_length = min(
                        len(children_scope_list), n_max_children)
                    sampled_ids = rand_gen.sample(range(n_scope_children),
                                                  sample_length)
                    sampled_children = [None for i in range(sample_length)]
                    for i, id in enumerate(sampled_ids):
                        # getting the sampled child
                        sampled_children[i] = children_scope_list[id]
                        # updating its counter
                        children_counts[scope][id] += 1

                    for child in sampled_children:
                        # parent is a sum layer, we must set a random weight
                        rand_weight = rand_gen.random()
                        parent.add_child(child, rand_weight)

                    # we can now normalize it
                    parent.normalize()
                else:
                    # binding the node to the input layer
                    (scope_var,) = scope
                    link_leaf_to_input_layer(parent,
                                             scope_var,
                                             input_layer,
                                             rand_gen)

            # pruning those children never used
            for scope in children_lists.keys():
                children_scope_list = children_lists[scope]
                scope_counts = children_counts[scope]
                used_children = [child
                                 for count, child in zip(scope_counts,
                                                         children_scope_list)
                                 if count > 0]
                children_lists[scope] = used_children

            # creating the layer and new scopelist
            # print('children list val', children_lists.values())
            children_list = [child
                             for child in
                             itertools.chain.from_iterable(
                                 children_lists.values())]
            scope_list = [key
                          for key, child_list in children_lists.items()
                          for elem in child_list]
            # print('children list', children_list)
            # print('scope list', scope_list)
            prod_layer = ProductLayerLinked(children_list)

            return prod_layer, scope_list

        def build_sum_layer(parent_layer,
                            parent_scope_list,
                            rand_gen,
                            max_scope_split=-1,
                            merge_prob=0.5):

            # keeping track of leaves
            # leaf_props = []
            scope_clusters = cluster_set_scope(parent_scope_list)

            # looping through all the parent nodes and their scopes
            # in order to decompose their scope
            dec_scope_list = []
            for scope in parent_scope_list:
                # decomposing their scope into k random pieces
                k = len(scope)
                if 1 < max_scope_split <= len(scope):
                    k = rand_gen.randint(2, max_scope_split)
                shuffled_scope = list(scope)
                rand_gen.shuffle(shuffled_scope)
                dec_scopes = [frozenset(shuffled_scope[i::k])
                              for i in range(k)]
                dec_scope_list.append(dec_scopes)
                # if a decomposed scope consists of only one var, generate a
                # leaf
                # leaves = [(parent, (dec_scope,))
                #           for dec_scope in dec_scopes if len(dec_scope) == 1]
                # leaf_props.extend(leaves)

            # generating a unique decomposition
            used_decs = {}
            children_list = []
            scope_list = []
            for parent, decs in zip(parent_layer.nodes(),
                                    dec_scope_list):
                merge_count = 0
                for scope in decs:
                    sum_node = None
                    try:
                        rand_perc = rand_gen.random()
                        if (merge_count < len(decs) - 1 and
                                rand_perc > merge_prob):
                            sum_node = used_decs[scope]
                            merge_count += 1

                        else:
                            raise Exception()
                    except:
                        # create a node for it
                        sum_node = SumNode(var_scope=scope)
                        children_list.append(sum_node)
                        scope_list.append(scope)
                        used_decs[scope] = sum_node

                    parent.add_child(sum_node)

            # unique_dec = {frozenset(dec) for dec in
            #               itertools.chain.from_iterable(dec_scope_list)}
            # print('unique dec', unique_dec)
            # building a dict scope->child
            # children_dict = {scope: SumNode() for scope in unique_dec}
            # now linking parents to their children
            # for parent, scope in zip(parent_layer.nodes(),
            #                          parent_scope_list):
            #     dec_scopes = dec_scope_list[scope]
            #     for dec in dec_scopes:
            # retrieving children
            # adding it
            #         parent.add_child(children_dict[dec])

            # we already have the nodes and their scopes
            # children_list = [child for child in children_dict.values()]
            # scope_list = [scope for scope in children_dict.keys()]

            sum_layer = SumLayerLinked(nodes=children_list)

            return sum_layer, scope_list

        # if no generator is provided, create a new one
        if rand_gen is None:
            rand_gen = random.Random()

        # create input layer
        # _vars = [2, 3, 2, 2, 4]
        input_layer = CategoricalIndicatorLayerLinked(vars=vars)

        # create root layer
        full_scope = frozenset({i for i in range(len(vars))})
        root = SumNode(var_scope=full_scope)
        root_layer = SumLayerLinked(nodes=[root])
        last_layer = root_layer

        # create top scope list
        last_scope_list = [full_scope]

        layers = [root_layer]
        layer_count = 0
        stop_building = False
        while not stop_building:
            # checking for early termination
            # this one leads to split product nodes into leaves
            if layer_count >= n_layers:
                print('Max level reached, trying to stop')
                max_scope_split = -1

            # build a new layer alternating types
            if isinstance(last_layer, SumLayerLinked):
                print('Building product layer')
                last_layer, last_scope_list = \
                    build_product_layer(last_layer,
                                        last_scope_list,
                                        n_max_children,
                                        n_scope_children,
                                        input_layer,
                                        rand_gen)
            elif isinstance(last_layer, ProductLayerLinked):
                print('Building sum layer')
                last_layer, last_scope_list = \
                    build_sum_layer(last_layer,
                                    last_scope_list,
                                    rand_gen,
                                    max_scope_split,
                                    merge_prob)

            # testing for more nodes to expand
            if last_layer.n_nodes() == 0:
                print('Stop building')
                stop_building = True
            else:
                layers.append(last_layer)
                layer_count += 1

        # checking for early termination
        # if not stop_building:
        #     if isinstance(last_layer, ProductLayerLinked):
        # building a sum layer splitting everything into one
        # length scopes
        #         last_sum_layer, last_scope_list = \
        #             build_sum_layer(last_layer,
        #                             last_scope_list,
        #                             rand_gen,
        #                             max_scope_split=-1)
        # then linking each node to the input layer
        #         for sum_leaf, scope in zip(last_sum_layer.nodes(),
        #                                    last_scope_list):
        #             (scope_var,) = scope
        #             link_leaf_to_input_layer(sum_leaf,
        #                                      scope_var,
        #                                      input_layer,
        #                                      rand_gen)
        #     elif isinstance(last_layer, SumLayerLinked):
        #         pass

        # print('LAYERS ', len(layers), '\n')
        # for i, layer in enumerate(layers):
        #     print('LAYER ', i)
        #     print(layer)
        # print('\n')
        spn = SpnLinked(input_layer=input_layer,
                        layers=layers[::-1])
        # testing
        # scope_list = [
        #     frozenset({1, 3, 4}), frozenset({2, 0}), frozenset({1, 3, 4})]
        # sum_layer = SumLayerLinked(nodes=[SumNode(), SumNode(), SumNode()])

        # prod_layer, scope_list = build_product_layer(
        #     sum_layer, scope_list, 2, 3, input_layer, rand_gen)

        # sum_layer1, scope_list_2 = build_sum_layer(prod_layer,
        #                                            scope_list,
        #                                            rand_gen,
        #                                            max_scope_split=2
        #                                            )
        # prod_layer_2, scope_list_3 = build_product_layer(sum_layer1,
        #                                                  scope_list_2,
        #                                                  2,
        #                                                  3,
        #                                                  input_layer,
        #                                                  rand_gen)
        # create spn from layers
        # spn = SpnLinked(input_layer=input_layer,
        #                 layers=[prod_layer_2, sum_layer1,
        #                         prod_layer, sum_layer, root_layer])
        return spn

    @classmethod
    def layered_linked_spn(cls, root_node):
        """
        Given a simple linked version (parent->children),
        returns a layered one (linked + layers)
        """
        layers = []
        root_layer = None
        input_nodes = []
        layer_nodes = []
        input_layer = None

        # layers.append(root_layer)
        previous_level = None

        # collecting nodes to visit
        open = deque()
        next_open = deque()
        closed = set()

        open.append(root_node)

        while open:
            # getting a node
            current_node = open.popleft()
            current_id = current_node.id

            # has this already been seen?
            if current_id not in closed:
                closed.add(current_id)
                layer_nodes.append(current_node)
                # print('CURRENT NODE')
                # print(current_node)

                # expand it
                for child in current_node.children:
                    # only for non leaf nodes
                    if (isinstance(child, SumNode) or
                            isinstance(child, ProductNode)):
                        next_open.append(child)
                    else:
                        # it must be an input node
                        if child.id not in closed:
                            input_nodes.append(child)
                            closed.add(child.id)

            # open is now empty, but new open not
            if (not open):
                # swap them
                open = next_open
                next_open = deque()

                # and create a new level alternating type
                if previous_level is None:
                    # it is the first level
                    if isinstance(root_node, SumNode):
                        previous_level = SumLayerLinked([root_node])
                    elif isinstance(root_node, ProductNode):
                        previous_level = ProductLayerLinked([root_node])
                elif isinstance(previous_level, SumLayerLinked):
                    previous_level = ProductLayerLinked(layer_nodes)
                elif isinstance(previous_level, ProductLayerLinked):
                    previous_level = SumLayerLinked(layer_nodes)

                layer_nodes = []

                layers.append(previous_level)

        #
        # finishing layers
        #

        #
        # checking for CLTreeNodes
        cltree_leaves = False
        for node in input_nodes:
            if isinstance(node, CLTreeNode):
                cltree_leaves = True
                break

        if cltree_leaves:
            input_layer = CategoricalCLInputLayerLinked(input_nodes)
        else:
            # otherwiise assuming all input nodes are homogeneous
            if isinstance(input_nodes[0], CategoricalSmoothedNode):
                # print('SMOOTH LAYER')
                input_layer = CategoricalSmoothedLayerLinked(input_nodes)
            elif isinstance(input_nodes[0], CategoricalIndicatorNode):
                input_layer = CategoricalIndicatorLayerLinked(input_nodes)

        spn = SpnLinked(input_layer=input_layer,
                        layers=layers[::-1])
        return spn

    @classmethod
    def pruned_spn_from_slices(cls, node_assoc, building_stack, logger=None):
        """
        WRITEME
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        # traversing the building stack
        # to link and prune nodes
        for build_node in reversed(building_stack):

            # current node
            current_id = build_node.id
            # print('+ Current node: %d', current_id)
            current_children_slices = build_node.children
            # print('\tchildren: %r', current_children_slices)
            current_children_weights = build_node.weights
            # print('\tweights: %r', current_children_weights)

            # retrieving corresponding node
            node = node_assoc[current_id]
            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):
                logging.debug('it is a sum node %d', current_id)
                # getting children
                for child_slice, child_weight in zip(current_children_slices,
                                                     current_children_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]
                    # print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            node.add_child(grand_child,
                                           grand_child_weight *
                                           child_weight)

                    else:
                        logging.debug('+++ Adding it as child: %d',
                                      child_node.id)
                        node.add_child(child_node, child_weight)
                        # print('children added')

            elif isinstance(node, ProductNode):
                logging.debug('it is a product node %d', current_id)
                # linking children
                for child_slice in current_children_slices:
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child in child_node.children:
                            node.add_child(grand_child)
                    else:
                        node.add_child(child_node)
                        # print('+++ Linking child %d', child_node.id)

        # this is superfluous, returning a pointer to the root
        root_build_node = building_stack[0]
        return node_assoc[root_build_node.id]

    @classmethod
    def pruned_spn_from_scopes(cls, scope_assoc, building_stack, logger=None):
        """
        WRITEME
        """
        build_node = None
        node = None
        node_assoc = None
        #
        # FIXME: this is just a stub, it does not even compile
        #
        if logger is None:
            logger = logging.getLogger(__name__)
        # traversing the building stack
        # to link and prune nodes
        for build_scope in reversed(building_stack):

            # current node
            current_id = build_scope.id
            # print('+ Current node: %d', current_id)
            current_children_slices = build_node.children
            # print('\tchildren: %r', current_children_slices)
            current_children_weights = build_node.weights
            # print('\tweights: %r', current_children_weights)

            # retrieving corresponding node
            scope_node = scope_assoc[current_id]
            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):
                logging.debug('it is a sum node %d', current_id)
                # getting children
                for child_slice, child_weight in zip(current_children_slices,
                                                     current_children_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]
                    # print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            node.add_child(grand_child,
                                           grand_child_weight *
                                           child_weight)

                    else:
                        logging.debug('+++ Adding it as child: %d',
                                      child_node.id)
                        node.add_child(child_node, child_weight)
                        # print('children added')

            elif isinstance(node, ProductNode):
                logging.debug('it is a product node %d', current_id)
                # linking children
                for child_slice in current_children_slices:
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child in child_node.children:
                            node.add_child(grand_child)
                    else:
                        node.add_child(child_node)
                        # print('+++ Linking child %d', child_node.id)

        # this is superfluous, returning a pointer to the root
        root_build_node = building_stack[0]
        return node_assoc[root_build_node.id]

    @classmethod
    def layered_pruned_linked_spn(cls, root_node):
        """
        WRITEME
        """
        #
        # first traverse the spn top down  to collect a bottom up traversal order
        # it could be done in a single pass I suppose, btw...
        building_queue = deque()
        traversal_stack = deque()

        building_queue.append(root_node)

        while building_queue:
            #
            # getting current node
            curr_node = building_queue.popleft()
            #
            # appending it to the stack
            traversal_stack.append(curr_node)
            #
            # considering children
            try:
                for child in curr_node.children:
                    building_queue.append(child)
            except:
                pass
        #
        # now using the inverse traversal order
        for node in reversed(traversal_stack):

            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):

                logging.debug('it is a sum node %d', node.id)
                current_children = node.children[:]
                current_weights = node.weights[:]

                # getting children
                children_to_add = deque()
                children_weights_to_add = deque()
                for child_node, child_weight in zip(current_children,
                                                    current_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)

                    print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        # this shall be prune
                        logging.debug('++ pruning node: %d', child_node.id)
                        # del node.children[i]
                        # del node.weights[i]

                        # adding subchildren
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            children_to_add.append(grand_child)
                            children_weights_to_add.append(grand_child_weight *
                                                           child_weight)
                            # node.add_child(grand_child,
                            #                grand_child_weight *
                            #                child_weight)

                        # print(
                        #     'remaining  children', [c.id for c in node.children])
                    else:
                        children_to_add.append(child_node)
                        children_weights_to_add.append(child_weight)

                #
                # adding all the children (ex grand children)
                node.children.clear()
                node.weights.clear()
                for child_to_add, weight_to_add in zip(children_to_add, children_weights_to_add):
                    node.add_child(child_to_add, weight_to_add)

                    # else:
                    #     print('+++ Adding it as child: %d', child_node.id)
                    #     node.add_child(child_node, child_weight)
                    #     print('children added')

            elif isinstance(node, ProductNode):

                logging.debug('it is a product node %d', node.id)
                current_children = node.children[:]

                children_to_add = deque()
                # linking children
                for i, child_node in enumerate(current_children):

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):

                        # this shall be pruned
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this must now be useless
                        # del node.children[i]

                        # adding children
                        for grand_child in child_node.children:
                            children_to_add.append(grand_child)
                            # node.add_child(grand_child)
                    else:
                        children_to_add.append(child_node)
                    #     node.add_child(child_node)
                    #     print('+++ Linking child %d', child_node.id)
                #
                # adding grand children
                node.children.clear()
                for child_to_add in children_to_add:
                    node.add_child(child_to_add)
        """
        #
        # printing
        print(\"TRAVERSAL\")
        building_queue = deque()
        building_queue.append(root_node)

        while building_queue:
            #
            # getting current node
            curr_node = building_queue.popleft()
            #
            # appending it to the stack
            print(curr_node)
            #
            # considering children
            try:
                for child in curr_node.children:
                    building_queue.append(child)
            except:
                pass
        """

        #
        # now transforming it layer wise
        # spn = SpnFactory.layered_linked_spn(root_node)
        return root_node


def merge_block_layers(layer_1, layer_2):

    #
    # check for type
    assert type(layer_1) is type(layer_2)

    #
    # merging nodes
    merged_nodes = [node for node in layer_1.nodes()]
    merged_nodes += [node for node in layer_2.nodes()]

    #
    # merging
    merged_layer = type(layer_1)(merged_nodes)

    #
    # merging i/o
    merged_inputs = layer_1.input_layers | layer_2.input_layers
    merged_outputs = layer_1.output_layers | layer_2.output_layers

    #
    # relinking
    merged_layer.input_layers = merged_inputs
    merged_layer.output_layers = merged_outputs
    for i in merged_inputs:
        i.add_output_layer(merged_layer)
    for o in merged_outputs:
        o.add_input_layer(merged_layer)

    return merged_layer


from scopes import topological_layer_sort


def compute_block_layer_depths(spn):
    #
    # sort layers topologically
    topo_sorted_layers = topological_layer_sort(list(spn.top_down_layers()))

    #
    # traverse them in this order and associate them by depth
    depth_dict = {}
    depth_dict[spn.input_layer()] = 0
    for layer in topo_sorted_layers:
        child_layer_depths = [depth_dict[p] for p in layer.input_layers]
        depth_dict[layer] = max(child_layer_depths) + 1

    return depth_dict


def edge_density_after_merge(layer_1, layer_2):
    n_input_nodes_1 = sum([l.n_nodes() for l in layer_1.input_layers])
    n_input_nodes_2 = sum([l.n_nodes() for l in layer_2.input_layers])
    n_input_nodes = n_input_nodes_1 + n_input_nodes_2
    n_output_nodes = layer_1.n_nodes() + layer_2.n_nodes()
    n_max_edges = n_input_nodes * n_output_nodes
    n_edges = layer_1.n_edges() + layer_2.n_edges()
    return n_edges / n_max_edges


def merge_block_layers_spn(spn, threshold, compute_heuristics=edge_density_after_merge):
    """
    Given an alternated layer linked SPN made by many block layers, try to
    aggregate them into macro blocks
    """
    #
    # lebeling each block with its depth level
    layer_depth_dict = compute_block_layer_depths(spn)

    #
    # create an inverse dict with depth level -> blocks
    depth_layer_dict = defaultdict(set)
    for layer, depth in layer_depth_dict.items():
        depth_layer_dict[depth].add(layer)

    #
    # here we are storing the new levels, we are assuming the input layer always to be alone
    mod_layers = []
    #
    # from each level, starting from the bottom, excluding the input layer
    for k in sorted(depth_layer_dict.keys())[1:]:

        print('Considering depth {}'.format(k))

        mergeable = True

        k_depth_layers = depth_layer_dict[k]

        while mergeable:
            #
            # retrieve layers at that depth

            #
            # for each possible pair compute an heuristic score
            best_score = -numpy.inf
            best_pair = None
            layer_pairs = itertools.combinations(k_depth_layers, 2)
            can_merge = False

            for layer_1, layer_2 in layer_pairs:
                print('\tConsidering layers: {0} {1}'.format(layer_1.id,
                                                             layer_2.id))
                score = compute_heuristics(layer_1, layer_2)
                if score > best_score and score > threshold:
                    can_merge = True
                    best_score = score
                    best_pair = (layer_1, layer_2)

            if can_merge:
                print('merging', best_pair[0].id, best_pair[1].id)
                #
                # merging the best pair
                merged_layer = merge_block_layers(*best_pair)

                #
                # disconnecting the previous ones
                best_pair[0].disconnect_layer()
                best_pair[1].disconnect_layer()

                #
                # storing them back
                k_depth_layers = [l for l in k_depth_layers
                                  if l != best_pair[0] and l != best_pair[1]]
                k_depth_layers.append(merged_layer)

            else:
                mergeable = False

        #
        # finally storing them
        mod_layers.extend(k_depth_layers)

    #
    # creating an SPN out of it:
    mod_spn = SpnLinked(input_layer=spn.input_layer(),
                        layers=mod_layers)
    return mod_spn


def retrieve_children_parent_assoc(spn, root=None):
    """
    Builds a map children node -> parent from a linked spn
    """
    if root is None:
        root = spn.root()

    parent_assoc = defaultdict(set)
    #
    # traversing it
    for node in spn.top_down_nodes():
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                parent_assoc[child].add(node)

    return parent_assoc


def linked_categorical_input_to_indicators(spn, input_layer=None):
    """
    Convertes a linked spn categorical input layer into an indicator one
    """

    #
    # get child, parent relations for node relinking
    child_assoc = retrieve_children_parent_assoc(spn)

    #
    # get input layer
    cat_input_layer = spn.input_layer()
    assert isinstance(cat_input_layer, CategoricalSmoothedLayerLinked)

    #
    # one indicator node for each var value
    vars = cat_input_layer.vars()
    if not vars:
        vars = list(sorted({node.var for node in cat_input_layer.nodes()}))

    feature_values = cat_input_layer.feature_vals()
    # print('vars', vars)
    # print('feature values', feature_values)

    indicator_nodes = [CategoricalIndicatorNode(var, val)
                       for i, var in enumerate(vars) for val in range(feature_values[i])]
    # for node in indicator_nodes:
    #     print(node)

    indicator_map = defaultdict(set)
    for ind_node in indicator_nodes:
        indicator_map[ind_node.var].add(ind_node)

    sum_nodes = []
    #
    # as many sum nodes as cat nodes
    for node in cat_input_layer.nodes():

        sum_node = SumNode(var_scope=frozenset([node.var]))
        sum_nodes.append(sum_node)

        for ind_node in sorted(indicator_map[node.var], key=lambda x: x.var_val):
            sum_node.add_child(ind_node, numpy.exp(node._var_probs[ind_node.var_val]))

        #
        # removing links to parents
        parents = child_assoc[node]
        for p_node in parents:
            #
            # assume it to be a product node
            # TODO: generalize
            assert isinstance(p_node, ProductNode)
            p_node.children.remove(node)
            p_node.add_child(sum_node)

    #
    # creating layer
    sum_layer = SumLayerLinked(sum_nodes)

    indicator_layer = CategoricalIndicatorLayerLinked(indicator_nodes)

    cat_input_layer.disconnect_layer()
    spn.set_input_layer(indicator_layer)
    spn.insert_layer(sum_layer, 0)

    return spn


def make_marginalized_network_constant(spn, vars_to_marginalize):
    """
    Replacing sub networks whose scope has to be marginalized over
    with constant nodes
    """

    #
    # get child, parent relations for node relinking
    child_assoc = retrieve_children_parent_assoc(spn)

    const_nodes_to_add = []

    scope_to_marginalize = frozenset(vars_to_marginalize)
    #
    # bottom up traversal
    for layer in spn.bottom_up_layers():

        layer_nodes_to_remove = []

        for node in layer.nodes():
            #
            # is this a non-leaf node? or a leaf whose scope is to marginalize?
            to_remove = False
            if hasattr(node, 'children'):
                #
                # if all his children are constants, we can remove it
                if all([isinstance(child, ConstantNode) for child in node.children]):
                    to_remove = True
            else:
                if node.var in scope_to_marginalize:
                    to_remove = True

            if to_remove:
                const_node = ConstantNode(node.var)
                const_nodes_to_add.append(const_node)

                parents = child_assoc[node]
                #
                # unlink it from parents and relink constant node
                for p_node in parents:
                    if isinstance(p_node, SumNode):
                        w = p_node.remove_child(node)
                        p_node.add_child(const_node, w)
                    else:
                        p_node.remove_child(node)
                        p_node.add_child(const_node)

                #
                # remove it from the layer as well
                layer_nodes_to_remove.append(node)

        for node in layer_nodes_to_remove:
            layer.remove_node(node)

        #
        # is the layer now empty?
        if not layer._nodes:
            raise ValueError('Layer is empty, unhandled case')

    #
    # adding all constant nodes to the previous nodes in a new input layer
    input_nodes = [node for node in spn.input_layer().nodes()] + const_nodes_to_add
    new_input_layer = CategoricalInputLayer(nodes=input_nodes)

    spn.set_input_layer(new_input_layer)


def split_layer_by_node_scopes(layer, node_layer_assoc, group_by=10):
    """
    Splits a layer according to its nodes scopes. It may be useful
    for indicator layers with many nodes
    """
    scopes_to_nodes = defaultdict(set)

    n_nodes = len(list(layer.nodes()))

    for node in layer.nodes():
        if hasattr(node, 'var_scope') and node.var_scope:
            scopes_to_nodes[frozenset(node.var_scope)].add(node)
        elif hasattr(node, 'var') and node.var:
            scopes_to_nodes[frozenset([node.var])].add(node)
        else:
            raise ValueError('Node without scope {}'.format(node))

    #
    # aggregating together=?
    sub_layers = None
    n_scopes = len(scopes_to_nodes)
    if group_by:  # and group_by < n_scopes:

        n_groups = n_scopes // group_by if n_scopes % group_by == 0 else n_scopes // group_by + 1
        print(n_groups)
        node_groups = [[] for j in range(n_groups)]

        for i, (_scope, nodes) in enumerate(scopes_to_nodes.items()):
            node_groups[i % n_groups].extend(nodes)

        sub_layers = [layer.__class__(nodes=nodes)
                      for nodes in node_groups if nodes]
    else:
        sub_layers = [layer.__class__(nodes=nodes)
                      for _output, nodes in scopes_to_nodes.items()]

    #
    # we have to update the node layer assoc map
    for s in sub_layers:
        for node in s.nodes():
            node_layer_assoc[node] = s

    print('[S] Layer: {} ({}) into {} layers {} ({})'.format(layer.id,
                                                             layer.__class__.__name__,
                                                             len(sub_layers),
                                                             [l.id for l in sub_layers],
                                                             [len(list(l.nodes()))
                                                                 for l in sub_layers]))

    return sub_layers


def split_layer_by_outputs(layer,
                           child_parent_assoc,
                           node_layer_assoc,
                           max_n_nodes=None):
    """
    Splits a layer into different sublayers whose nodes have outputs in the same
    layers, if possible.
    Note, they cannot be directly reused in a linked spn otherwise stats are
    getting messed up
    TODO: fix this
    """

    if max_n_nodes is None:
        max_n_nodes = numpy.inf

    output_to_nodes = defaultdict(set)

    for node in layer.nodes():
        output_layers = set()
        for parent in child_parent_assoc[node]:
            output_layers.add(node_layer_assoc[parent])
        output_to_nodes[frozenset(output_layers)].add(node)

    # sub_layers = [layer.__class__(nodes=nodes)
    #               for _output, nodes in output_to_nodes.items()]
    sub_layers = []
    for _output, nodes in output_to_nodes.items():
        #
        # do we need to break the layer even more?
        if len(nodes) < max_n_nodes:
            sub_layers.append(layer.__class__(nodes=nodes))
        else:
            for i in range(0, len(nodes), max_n_nodes):
                node_list = list(nodes)
                sub_layers.append(layer.__class__(nodes=node_list[i:i + max_n_nodes]))

    #
    # we have to update the node layer assoc map
    for s in sub_layers:
        for node in s.nodes():
            node_layer_assoc[node] = s

    print('[O] Layer: {} ({}) into {} layers {} ({})'.format(layer.id,
                                                             layer.__class__.__name__,
                                                             len(sub_layers),
                                                             [l.id for l in sub_layers],
                                                             [len(list(l.nodes()))
                                                              for l in sub_layers]))
    # if mixed_output_nodes:
    #     sub_layers.append(layer.__class__(nodes=list(mixed_output_nodes)))

    return sub_layers


def build_theanok_input_layer(input_layer, n_features, feature_vals):
    input_dim = n_features
    output_dim = len(list(input_layer.nodes()))

    mask = []
    for node in input_layer.nodes():
        mask.append(sum(feature_vals[:node.var]) + node.var_val)
    mask = numpy.array(mask)
    # print('mask', mask)
    return InputLayerTheanok(input_dim, output_dim, mask, layer_id=input_layer.id)


def build_theanok_layer(output_layer, input_layers, theano_inputs, dtype=float):
    """
    Creating a theanok layer representing the linked output_layer
    and considering its (linked) input layers already built.
    """
    output_nodes = list(output_layer.nodes())
    input_nodes = []
    for l in sorted(input_layers):
        input_nodes.extend(list(l.nodes()))

    # print('input nodes {}'.format([n.id for n in input_nodes]))

    output_dim = len(output_nodes)
    input_dim = len(input_nodes)

    output_nodes_assoc = {node: i for i, node in enumerate(output_nodes)}
    input_nodes_assoc = {node: i for i, node in enumerate(input_nodes)}

    #
    # creating the weight matrix
    W = numpy.zeros((input_dim, output_dim), dtype=dtype)
    if isinstance(output_layer, SumLayerLinked):
        for node in output_nodes:
            for j, child in enumerate(node.children):
                # print('{}->{} ({}, {})'.format(node.id,
                #                                child.id,
                #                                input_nodes_assoc[child],
                #                                output_nodes_assoc[node]))
                W[input_nodes_assoc[child], output_nodes_assoc[node]] = node.weights[j]

    elif isinstance(output_layer, ProductLayerLinked):
        for node in output_nodes:
            for child in node.children:
                # print('{}->{} ({}, {})'.format(node.id,
                #                                child.id,
                #                                input_nodes_assoc[child],
                #                                output_nodes_assoc[node]))
                W[input_nodes_assoc[child], output_nodes_assoc[node]] = 1
    else:
        raise ValueError('Unrecognized layer type: {}'.format(output_layer.__class__.__name__))

    #
    # creating scope matrix
    # TODO: creating the scope matrix
    scope = None

    #
    # creating layer
    layer = None
    if isinstance(output_layer, SumLayerLinked):
        layer = SumLayerTheanok(input_dim=input_dim,
                                output_dim=output_dim,
                                layer_id=output_layer.id,
                                weights=W)
    elif isinstance(output_layer, ProductLayerLinked):
        layer = ProductLayerTheanok(input_dim=input_dim,
                                    output_dim=output_dim,
                                    layer_id=output_layer.id,
                                    weights=W)
    else:
        raise ValueError('Unrecognized layer type: {}'.format(output_layer.__class__.__name__))

    #
    # double linking it
    for input_layer in theano_inputs:
        layer.add_input_layer(input_layer)
        input_layer.add_output_layer(layer)

    return layer


def build_theanok_spn_from_block_linked(spn,
                                        n_features,
                                        feature_vals,
                                        group_by=0,
                                        max_n_nodes_layer=None):
    """
    Translating a block linked spn into a block theano-keras-like
    """
    #
    # setting counter to the current max
    max_node_count = max([node.id for node in spn.top_down_nodes()]) + 1
    max_layer_count = max([layer.id for layer in spn.top_down_layers()]) + 1
    Node.set_id_counter(max_node_count)
    LayerLinked.set_id_counter(max_layer_count)

    #
    # transforming the categorical input layer into a layer of indicator nodes
    if isinstance(spn.input_layer(), CategoricalSmoothedLayerLinked):
        logging.info('Transforming input layer from categorical to indicators...')
        spn = linked_categorical_input_to_indicators(spn)

    node_layer_map = {node: layer for layer in spn.bottom_up_layers() for node in layer.nodes()}
    child_parent_map = retrieve_children_parent_assoc(spn)

    #
    # top down layers traversal, discarding input layer and splitting

    top_down_layers = []
    for l in list(spn.top_down_layers()):
        split_layers = split_layer_by_outputs(l, child_parent_map, node_layer_map,
                                              max_n_nodes=max_n_nodes_layer)
        # #
        # # we can split input layers even further
        # if isinstance(l, CategoricalIndicatorLayerLinked):

        #     for s in split_layers:
        #         top_down_layers.extend(split_layer_by_node_scopes(s, group_by))
        # else:

        # if l.id == max_layer_count + 1:
        #     pass
        #     for s in split_layers:
        #         top_down_layers.extend(split_layer_by_node_scopes(s,
        #                                                           node_layer_map,
        #                                                           group_by))
        # else:
        #     top_down_layers.extend(split_layers)

        top_down_layers.extend(split_layers)

    #
    # recomputing the node layer map
    node_layer_map = {node: layer for layer in top_down_layers for node in layer.nodes()}
    # for node in spn.input_layer().nodes():
    #     node_layer_map[node] = spn.input_layer()

    #
    # linked layer -> theano layer
    layer_to_layer_map = {}

    #
    # ordering input layer nodes

    #
    # proceeding bottom up
    for layer in reversed(top_down_layers):
        logging.debug('{}'.format(layer))
        #
        # retrieve the input layers
        input_layers = set()

        #
        # indicator node
        if isinstance(layer, CategoricalIndicatorLayerLinked):
            theano_layer = build_theanok_input_layer(layer, n_features, feature_vals)
        else:

            for node in layer.nodes():
                if hasattr(node, 'children'):
                    for child in node.children:
                        # if child in node_layer_map:
                        input_layers.add(node_layer_map[child])

            theano_layers = [layer_to_layer_map[l]
                             for l in sorted(input_layers) if l in layer_to_layer_map]
            #
            # building a theano layer
            dtype = None
            if isinstance(layer, SumLayerLinked):
                dtype = float
            elif isinstance(layer, ProductLayerLinked):
                dtype = int

            theano_layer = build_theanok_layer(layer, input_layers, theano_layers, dtype=dtype)

        #
        # adding it into the mapping
        layer_to_layer_map[layer] = theano_layer

    #
    # ordering the nodes
    theano_layers = [layer for layer in layer_to_layer_map.values()]
    ordered_theano_layers = topological_layer_sort(theano_layers)
    theano_layers_seq = [(layer, layer.input_layers) for layer in ordered_theano_layers]

    #
    # build and compile
    theano_spn = BlockLayeredSpn(layers=theano_layers_seq)
    #
    # printing layer stats
    logging.info(theano_spn.layer_stats())
    #
    # compiling theano functions
    theano_spn.compile()

    return theano_spn


NODE_LAYER_TYPE_ASSOC = {
    SumNode: SumLayerLinked,
    ProductNode: ProductLayerLinked,
    CategoricalIndicatorNode: CategoricalIndicatorLayerLinked}


def build_linked_layer_from_nodes(nodes):

    return NODE_LAYER_TYPE_ASSOC[nodes[0].__class__](nodes)


def build_theanok_spn_from_block_linked_top(spn,
                                            n_features,
                                            feature_vals,
                                            max_n_edges_layer=None):
    """
    """
    #
    # transforming the categorical input layer into a layer of indicator nodes
    if isinstance(spn.input_layer(), CategoricalSmoothedLayerLinked):
        logging.info('Transforming input layer from categorical to indicators...')
        spn = linked_categorical_input_to_indicators(spn)

    top_down_layers = []
    node_layer_assoc = {}
    for node in spn.top_down_nodes():
        #
        # already seen node?
        if node in node_layer_assoc:
            layer = node_layer_assoc[node]
        else:
            #    layer_assoc = {}

            # creating a new layer
            logging.info('Creating new layer')
            layer = build_linked_layer_from_nodes([node])
            node_layer_assoc[node] = layer
            top_down_layers.append(layer)
        #
        # gather all children into another layer
        if hasattr(node, 'children'):
            children_input_layer = build_linked_layer_from_nodes(node.children)
            top_down_layers.append(children_input_layer)

            for child in node.children:
                node_layer_assoc[child] = children_input_layer
            #
            # linking
            layer.add_input_layer(children_input_layer)
            children_input_layer.add_output_layer(layer)

    #
    # linked layer -> theano layer
    layer_to_layer_map = {}

    #
    # proceeding bottom up
    for layer in reversed(top_down_layers):
        logging.debug('{}'.format(layer))
        #
        # retrieve the input layers
        input_layers = set()

        #
        # indicator node
        if isinstance(layer, CategoricalIndicatorLayerLinked):
            theano_layer = build_theanok_input_layer(layer, n_features, feature_vals)
        else:

            for node in layer.nodes():
                if hasattr(node, 'children'):
                    for child in node.children:
                        # if child in node_layer_map:
                        input_layers.add(node_layer_assoc[child])

            theano_layers = [layer_to_layer_map[l]
                             for l in sorted(input_layers) if l in layer_to_layer_map]
            #
            # building a theano layer
            dtype = None
            if isinstance(layer, SumLayerLinked):
                dtype = float
            elif isinstance(layer, ProductLayerLinked):
                dtype = int

            theano_layer = build_theanok_layer(layer, input_layers, theano_layers, dtype=dtype)

        #
        # adding it into the mapping
        layer_to_layer_map[layer] = theano_layer

    #
    # ordering the nodes
    theano_layers = [layer for layer in layer_to_layer_map.values()]
    ordered_theano_layers = topological_layer_sort(theano_layers)
    theano_layers_seq = [(layer, layer.input_layers) for layer in ordered_theano_layers]

    #
    # build and compile
    theano_spn = BlockLayeredSpn(layers=theano_layers_seq)
    #
    # printing layer stats
    logging.info(theano_spn.layer_stats())
    #
    # compiling theano functions
    theano_spn.compile()

    return theano_spn
