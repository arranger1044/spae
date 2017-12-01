from .layers import Layer
from .layers import SumLayer
from .layers import ProductLayer
from .layers import compute_feature_vals

from spn import AbstractSpn, AbstractLayeredSpn
from spn import LOG_ZERO
from spn import RND_SEED

from .nodes import SumNode
from .nodes import ProductNode
from .nodes import sample_from_leaf

from collections import deque

import math
# from math import exp

from keras import backend as K

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import numpy

import sys

import logging


class Spn(AbstractLayeredSpn):

    """
    Spn layer-wise linked implementation using pure python

    WRITEME
    """

    def __init__(self,
                 input_layer=None,
                 layers=[]):
        """
        layers is a collection of layer-blocks ordered bottom up for evaluation
        """
        # meaningful layers
        self._input_layer = input_layer
        if input_layer is not None:
            self._feature_vals = self._input_layer.feature_vals()

        self._layers = None
        self._root_layer = None

        self.set_layers(layers)

    def set_input_layer(self, layer):
        """
        WRITEME
        """
        self._input_layer = layer
        self._feature_vals = self._input_layer.feature_vals()

    def set_layers(self, layers):
        """
        WRITEME
        """
        self._layers = layers
        # the root is the last level, if present
        if layers:
            self._root_layer = layers[-1]

    def add_layer(self, layer, pos=None):
        """
        WRITEME
        """
        if pos is None:
            self._layers.append(layer)
            pos = len(self._layers) - 1
            # pos = max(0, len(self._layers) - 1)
            # self._root_layer = layer
        else:
            self._layers.insert(pos, layer)
        # updating the pointer to the root
        self._root_layer = layer

    def insert_layer(self, layer, pos=None):
        """
        WRITEME
        """
        self._layers.insert(pos, layer)

    def root_layer(self):
        return self._root_layer

    def root(self):
        root_layer = self.root_layer()
        assert len(list(root_layer.nodes())) == 1
        return root_layer._nodes[0]

    def input_layer(self):
        return self._input_layer

    def top_down_nodes(self):
        for layer in self.top_down_layers():
            for node in layer.nodes():
                yield node

    def bottom_up_nodes(self):
        for layer in self.bottom_up_layers():
            for node in layer.nodes():
                yield node

    def n_sum_nodes(self):
        sum_nodes = [node for node in self.top_down_nodes()
                     if isinstance(node, SumNode)]
        return len(sum_nodes)

    def n_product_nodes(self):
        prod_nodes = [node for node in self.top_down_nodes()
                      if isinstance(node, ProductNode)]
        return len(prod_nodes)

    def unique_scopes(self):
        scopes = set()
        for node in self.top_down_nodes():
            if hasattr(node, 'var_scope'):
                scopes.add(node.var_scope)
            elif hasattr(node, 'var'):
                scopes.add(frozenset([node.var]))
        return scopes

    def n_unique_scopes(self):
        return len(self.unique_scopes())

    def is_decomposable(self):
        """
        WRITEME
        """
        return all([layer.is_decomposable()
                    for layer in self._layers
                    if isinstance(layer, ProductLayer)])

    def is_complete(self):
        """
        WRITEME
        """
        return all([layer.is_complete()
                    for layer in self._layers
                    if isinstance(layer, SumLayer)])

    def is_valid(self):
        """
        Here is checked a stricter condition for validity:
        completeness AND decomposability => validity
        """
        return self.is_complete() and self.is_decomposable()

    def eval(self, input):
        """
        WRITEME
        """
        lls = None
        # batch evaluation
        if input.ndim > 1:
            # returning a matrix of values
            # TODO clean this up and make a numpy array
            lls = []
            # for instance in input.T:
            for instance in input:
                ll = self.single_eval(instance)
                lls.append(ll)
            lls = numpy.array(lls)
        else:
            # returning an array (list)
            lls = self.single_eval(input)

        return lls

    def eval_marg(self, input, marg_node_set):
        """
        WRITEME
        """

        # retuning a matrix of values
        # TODO clean this up and make a numpy array
        lls = []
        # for instance in input.T:
        for instance in input:
            ll = self.single_eval_marg(instance, marg_node_set)
            lls.append(ll)
        lls = numpy.array(lls)

        return lls

    def single_eval(self, input):
        """
        WRITEME
        """
        # evaluating the input layer first
        self._input_layer.eval(input)
        # print('input log vals')
        # print('{0}'.format([node.id for node in self._input_layer.nodes()]))
        # print(self._input_layer.node_values())

        # then propagate upwards by evaluating other layers
        for layer in self._layers:
            layer.eval()
            # print('intermediate layer')
            # print(layer.node_values())

        # result is returned as a list of log-values
        return self._root_layer.node_values()

    def single_eval_marg(self, input, marg_node_set):
        """
        WRITEME
        """
        # evaluating the input layer first
        self._input_layer.eval_marg(input, marg_node_set)
        # print('input log vals')
        # print('{0}'.format([node.id for node in self._input_layer.nodes()]))
        # print(self._input_layer.node_values())

        # then propagate upwards by evaluating other layers
        for layer in self._layers:
            layer.eval_marg(marg_node_set)
            # print('intermediate layer')
            # print(layer.node_values())

        # result is returned as a list of log-values
        return self._root_layer.node_values()

    def test_eval(self):
        """
        this is done for testing purposes only,
        bypasses the input layer (assuming it is already
        evaluated)
        """
        for layer in self._layers:
            layer.eval()
            # print('intermediate layer')
            # print(layer.node_values())

        # result is returned as a list of log-values
        return self._root_layer.node_values()

    def mpe_eval(self, input):
        """
        WRITEME
        """
        lls = None
        # batch evaluation
        if input.ndim > 1:
            # returning a matrix of values
            # TODO clean this up and make a numpy array
            lls = []
            for instance in input.T:
                ll = self.single_mpe_eval(instance)
                lls.append(ll)
        else:
            # returning an array (list)
            lls = self.single_mpe_eval(input)

        return lls

    def single_mpe_eval(self, input):
        """
        WRITEME
        """
        # evaluating the input layer first
        # smoothed input layers are evaluated soft as well
        # (is this correct?)
        # self._input_layer.eval(input)
        self._input_layer.mpe_eval(input)

        # then propagate upwards by evaluating other layers
        # with MPE inference
        for layer in self._layers:
            layer.mpe_eval()

        # result is returned as a list of log-values
        return self._root_layer.node_values()

    def test_mpe_eval(self):
        """
        this is done for testing purposes only,
        MPE evaluation (see test_eval)
        """
        for layer in self._layers:
            layer.mpe_eval()
            # print('intermediate layer')
            # print(layer.node_values())

        # result is returned as a list of log-values
        return self._root_layer.node_values()

    def build_k(self, input_dtype='int32'):
        """
        Build keras graph by dispatching its creation to  layer
        """
        #
        # creating placeholder for input
        self.input_k = K.placeholder(ndim=2, dtype=input_dtype)

        build_start_t = perf_counter()
        #
        # building input layer first
        self._input_layer.build_k(self.input_k)

        #
        # then hidden layers
        for layer in self._layers:
            layer.build_k()

        root_node = self.root()
        # print(type(root_node))
        # print(type(root_node.log_vals))
        self.eval_func_keras = K.function(inputs=[self.input_k],
                                          outputs=[root_node.log_vals])

        build_end_t = perf_counter()
        logging.info('Spn build to keras in {} secs'.format(build_end_t - build_start_t))

    def score(self, data, _keras=True):
        """
        Evaluates the log-likelihood of the data
        """
        if _keras:
            return self.eval_func_keras([data])[0]
        else:
            return self.eval(data)

    def get_features(self):
        """
        """
        return compute_feature_vals([n for n in self._input_layer.nodes()])

    def sample(self,
               n_instances=1,
               feature_values=None,
               one_hot_encoding=False,
               starting_node=None,
               dtype=numpy.int32,
               rand_gen=None,
               verbose=False):
        """
        Sampling an SPN generating n_instances vectors
        """

        if rand_gen is None:
            rand_gen = numpy.random.RandomState(RND_SEED)

        if feature_values is None:
            feature_values = self.get_features()

        if verbose:
            print('Feature values {0}'.format(feature_values))

        if one_hot_encoding:
            n_features = numpy.sum(feature_values)
        else:
            n_features = len(feature_values)

        if verbose:
            print('Sampling {0} instances over {1} features'.format(n_instances,
                                                                    n_features))

        instances_vec = numpy.zeros((n_instances, n_features), dtype=dtype)
        #
        # setting all values to an unwanted val
        instances_vec.fill(-1)

        if starting_node is None:
            starting_node = self.root()

        for i in range(n_instances):
            #
            # traversing the spn top down
            nodes_to_process = deque()
            nodes_to_process.append(starting_node)

            while nodes_to_process:
                curr_node = nodes_to_process.popleft()
                #
                # if it is a sum node, sample just one child
                if isinstance(curr_node, SumNode):
                    n_children = len(curr_node.children)
                    sampled_children_id = rand_gen.choice(n_children, p=curr_node.weights)
                    nodes_to_process.append(curr_node.children[sampled_children_id])
                    if verbose:
                        print('sum node, getting child {0} [{1}]'.format(sampled_children_id,
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
                    instances_vec = sample_from_leaf(curr_node,
                                                     instances_vec,
                                                     i,
                                                     rand_gen,
                                                     feature_values)
                    if verbose:
                        print('Reached a leaf: {0}'.format(instances_vec[i]))

        #
        # are all values filled?
        assert numpy.sum(instances_vec == -1) == 0

        #
        # if just one instance, flattening the ndarray
        if n_instances == 1:
            instances_vec = instances_vec.flatten()

        return instances_vec

    def to_text(self, filename):
        """
        Serialization routine to text format
        """

        LAYER_GLYPH = '-'

        n_layers = self.n_layers()

        with open(filename, 'w') as out_stream:

            #
            # writing the first line
            out_stream.write("spn\n\n")
            #
            # write the features
            features_str = " ".join(list(map(str, self._feature_vals)))
            out_stream.write(features_str + '\n\n')

            #
            # exploring other layers
            for i, layer in enumerate(self.top_down_layers()):
                out_stream.write(LAYER_GLYPH + ' ' +
                                 str(n_layers - i) + '\n')
                #
                # printing nodes
                for node in layer.nodes():
                    node_str = node.node_short_str()
                    out_stream.write(node_str + '\n')
                out_stream.write('\n')

            #
            # dumping input layer
            out_stream.write(LAYER_GLYPH + str(1) + '\n')
            for node in self._input_layer.nodes():
                node_str = node.node_short_str()
                out_stream.write(node_str + '\n')
            out_stream.write('\n')

    def backprop(self):
        """
        WRITEME
        """
        # set top layer derivative to one
        self._layers[-1].set_log_derivative(0.0)
        # backpropagate to the leaves
        for layer in self.top_down_layers():
            layer.backprop()

    def test_weight_update(_l_id,
                           _n_id,
                           _w_id,
                           old_weight,
                           grad):
        eta = 0.1
        return old_weight + eta * grad

    def backprop_and_update(self, weight_update_rule):
        """
        WRITEME
        """
        # set top layer derivative to one
        self._layers[-1].set_log_derivative(0.0)
        # backpropagate to the leaves
        layer_id = 0
        for layer in self.top_down_layers():
            layer.backprop()
            # for sum layers
            if isinstance(layer, SumLayer):
                # updating weights according to a simple rule
                layer.update_weights(weight_update_rule, layer_id)
                layer_id += 1

    def mpe_backprop(self):
        """
        WRITEME
        """
        # set top layer derivative to one
        self._layers[-1].set_log_derivative(0.0)
        # backpropagate to the leaves
        for layer in self.top_down_layers():
            layer.mpe_backprop()

    def get_weights(self, empty=False):
        """
        Returning the weights of the network in a multi dimensional
        array (lists of lists of lists, (sum)layers x nodes x weights)
        or an empty structure
        """
        # creates a multi dim array for storing weights
        # [layer_id][node_id][weight_id] all positional integers
        weights_ds = None
        # filling it with nodes
        if not empty:
            weights_ds = [[[weight for weight in node.weights]
                           for node in layer.nodes()]
                          for layer in self.top_down_layers()
                          if isinstance(layer, SumLayer)]
        # filling it with zeros
        else:
            weights_ds = [[[0.0 for child in node.children]
                           for node in layer.nodes()]
                          for layer in self.top_down_layers()
                          if isinstance(layer, SumLayer)]
        return weights_ds

    def set_weights(self, weights_ds):
        """
        Setting the network weights from a data structure
        """
        layer_id = 0
        for layer in self.top_down_layers():
            if isinstance(layer, SumLayer):
                for node_id, node in enumerate(layer.nodes()):
                    node.set_weights(weights_ds[layer_id][node_id])
                layer_id += 1

    def mpe_traversal(self):
        """
        WRITEME
        this shall be a generator for traversing the spn top down,
        halting only in proximity of weights to be updated according
        to MPE inference

        - assuming a mpe_eval() bottom-up pass has been done (?)
        - according to Poon, one can do a sum eval step and then a
          max backprop step...
        """
        # creating a queue
        nodes_to_process = deque()
        # adding the root nodes
        for i, node in enumerate(self._layers[-1].nodes()):
            nodes_to_process.append((0, node.id, node))
        # print('roots', len(nodes_to_process))
        # bfs search
        child_nodes = deque()
        while nodes_to_process:
            # pop the first one
            level, par_id, curr_node = nodes_to_process.popleft()
            # print('now examining', level, id, curr_node)
            # searching for the max valued child
            # max_val = LOG_ZERO
            # clearing the deque
            child_nodes.clear()
            for i, child in enumerate(curr_node.children):
                # this is done by peharz
                # posterior = child.log_val + child.log_der

                # this, instead shall be the 'classic one'
                # posterior = child.log_val + log_weight
                # if posterior > max_val:
                #     max_val = posterior
                #     child_nodes.clear()
                # if posterior == max_val:
                #     child_nodes.append(level, id, i, child)

                # print('children', child.log_val +
                #       curr_node.log_weights[i], curr_node.log_val)

                # compute the value, in theory the max_child has the
                # same values as the parent
                if numpy.isclose(child.log_val + curr_node.log_weights[i],
                                 curr_node.log_val):
                    child_nodes.append((i, child))
            # now for each prod child
            for child_pos, child_node in child_nodes:
                # print(node)
                # yielding the node
                yield (level, par_id, child_pos)
                # for each child they have, add it to be processed
                # checking for non leaf nodes
                try:
                    for j, sum_node in enumerate(child_node.children):
                        nodes_to_process.append(
                            (level + 1, sum_node.id, sum_node))
                except:
                    pass

    def fit(self, train, valid, test, algo='sgd', options=None):
        """
        WRITEME
        """

    def fit_sgd(self,
                train, valid, test,
                n_epochs=50,
                batch_size=1,
                hard=False,
                learning_rate=0.1,
                grad_method=0,  # 0=SGD, 1=ADAGRAD, 2=ADADELTA
                validation_frequency=None,
                early_stopping=30,
                rand_gen=None,
                epsilon=1e-7):
        """
        Basic SGD
        """

        # def simple_grad(weight, grad):
        #     return weight + learning_rate * grad

        #
        # ADAGRAD & ADADELTA
        #
        ada_grad_history = None
        ada_grad_updates = None

        if grad_method == 1 or grad_method == 2:
            # initialize the previous gradients history
            # used both for ADAGRAD and ADADELTA update rules
            ada_grad_history = self.get_weights(empty=True)

        if grad_method == 2:
            # for ADADELTA, storing the previous updates as well
            ada_grad_updates = self.get_weights(empty=True)

        def compute_grad(layer_id,
                         node_id,
                         weight_id,
                         weight,
                         grad):

            weight_update = weight
            if grad_method == 0:  # SGD NAIVE
                weight_update = weight + learning_rate * grad

            elif grad_method == 1:  # ADAGRAD
                # getting the previous gradient history
                h_grad = ada_grad_history[layer_id][node_id][weight_id]
                # update it
                h_grad += grad * grad
                # save it back
                ada_grad_history[layer_id][node_id][weight_id] = h_grad
                grad = grad / (epsilon + math.sqrt(h_grad))
                weight_update = weight + learning_rate * grad

            elif grad_method == 2:  # ADADELTA
                # getting the previous gradient history
                h_grad = ada_grad_history[layer_id][node_id][weight_id]
                # not a simple squared grad
                h_grad = (learning_rate * h_grad +
                          (1.0 - learning_rate) * grad * grad)
                ada_grad_history[layer_id][node_id][weight_id] = h_grad

                h_update = ada_grad_updates[layer_id][node_id][weight_id]
                update_t = ((math.sqrt(epsilon + h_update)) /
                            (math.sqrt(epsilon + h_grad))) * grad

                h_update = (learning_rate * h_update +
                            (1.0 - learning_rate) * (update_t * update_t))
                ada_grad_updates[layer_id][node_id][weight_id] = h_update
                weight_update = weight + update_t

            return weight_update

        # keep track of ll
        epoch_cost = 0.0
        old_ll = 0.0

        epoch = 0
        done_looping = False

        best_iter = 0
        best_valid_avg_ll = -numpy.inf
        best_params = self.get_weights()
        best_train_avg_ll = -numpy.inf
        local_valid_avg_ll = -numpy.inf

        n_train_instances = train.shape[0]
        n_train_batches = (n_train_instances
                           // batch_size)

        if validation_frequency is None:
            validation_frequency = n_train_batches

        no_improvement = 0
        #
        # epochs loop
        #
        while (epoch < n_epochs) and (not done_looping):

            epoch = epoch + 1
            print('>>>>> epoch {0}/{1}'.format(epoch, n_epochs))

            epoch_start_t = perf_counter()

            # save and reset ll, I could save them all to plot them maybe
            old_ll = epoch_cost

            avg_time = 0.0
            epoch_cost = 0.0

            # shuffling the dataset
            rand_gen.shuffle(train)

            #
            # for each training example
            #
            for m, instance in enumerate(train):

                inst_start_t = perf_counter()
                # evaluate it
                sample_lls = None
                if hard:
                    sample_lls = self.mpe_eval(instance)
                else:
                    sample_lls = self.eval(instance)

                # cumulate it (assuming one radix only)
                sample_ll = sample_lls[0]
                epoch_cost += sample_ll

                eval_end_t = perf_counter()
                # print('eval time', eval_end_t - inst_start_t)

                back_start_t = perf_counter()

                # backprop
                self.backprop_and_update(compute_grad)

                back_end_t = perf_counter()
                # print('backpr time', back_end_t - back_start_t)

                avg_time += (back_end_t - inst_start_t)

                sys.stdout.write(
                    '\r-- mini batch {:d}/{:d} ({:.4f} secs avg)'
                    ' [{:.4f} ll avg]'
                    .format(m + 1, n_train_batches,
                            avg_time / (m + 1),
                            epoch_cost / (m + 1)))
                sys.stdout.flush()
                #
                # checking for validation set performance
                #
                if ((m + 1) % validation_frequency == 0 and
                        valid is not None):
                    valid_start_t = perf_counter()
                    valid_lls = self.eval(valid)
                    valid_avg_ll = numpy.mean(valid_lls)
                    valid_end_t = perf_counter()
                    print('\n\tLL on val:{ll} in {ss} secs'.
                          format(ll=valid_avg_ll,
                                 ss=(valid_end_t - valid_start_t)))
                    #
                    # now comparing with best score
                    #
                    if valid_avg_ll > best_valid_avg_ll:
                        print('\tNEW BEST VALID LL: {0}'.
                              format(valid_avg_ll))
                        best_iter = epoch * (m + 1)
                        best_valid_avg_ll = valid_avg_ll
                        best_train_avg_ll = epoch_cost / (m + 1)
                        # saving the model
                        best_params = self.get_weights()
                        #
                        # Evaluating on the test set with best params
                        #
                        if test is not None:
                            test_start_t = perf_counter()
                            test_lls = self.eval(test)
                            test_avg_ll = numpy.mean(test_lls)
                            test_end_t = perf_counter()
                            print('\tLL on TEST:{ll} in {ss} secs'.
                                  format(ll=test_avg_ll,
                                         ss=(test_end_t - test_start_t)))
                    #
                    # early stopping
                    #
                    if valid_avg_ll > local_valid_avg_ll:
                        no_improvement = 0
                        local_valid_avg_ll = valid_avg_ll
                    else:
                        no_improvement += 1

                    if no_improvement >= early_stopping:
                        print('No improvement on valid set after {0} checks'.
                              format(no_improvement))
                        done_looping = True
                        break

            epoch_end_t = perf_counter()
            print('\n elapsed {0} secs'.format(epoch_end_t -
                                               epoch_start_t))

            rel_imp = abs((epoch_cost - old_ll) / epoch_cost)
            print('relative improvement -> {0}'.format(rel_imp))
            if rel_imp < epsilon:
                done_looping = True

        #
        # Evaluating on the test set with best params
        #
        if test is not None:
            self.set_weights(best_params)
            test_start_t = perf_counter()
            test_lls = self.eval(test)
            test_avg_ll = numpy.mean(test_lls)
            test_end_t = perf_counter()
            print('\nLL on TEST:{ll} in {ss} secs'.
                  format(ll=test_avg_ll,
                         ss=(test_end_t - test_start_t)))

        return best_train_avg_ll, best_valid_avg_ll, test_avg_ll

    def fit_em(self,
               train, valid, test,
               n_epochs=50,
               batch_size=1,
               hard=True,
               epsilon=1e-7):
        """
        EM learning (see peharz)
        """

        # keeping track of sum layers only
        sum_layers_only = [i for i, layer
                           in enumerate(self._layers)
                           if isinstance(layer, SumLayer)]
        # reversing the list
        sum_layers_only.reverse()
        print('sum layers', sum_layers_only)
        # allocating a temp struct for weight updating
        # it is a dynamic 3d-tensor
        # w_updates[l][n][c] contains the new weight for the
        # node c with parent n in the layer l (all integers)
        # note that l counts layers ids top down
        # TODO pass to a numpy tensor, even if sparse
        w_updates = [i for i in range(len(sum_layers_only))]
        # create a dict for each layer: node_id -> layer pos
        w_layer_pos = []
        for i, layer_id in enumerate(sum_layers_only):
            layer = self._layers[layer_id]
            layer_updates = [[0.0 for child in node.children]
                             for node in layer.nodes()]
            w_updates[i] = layer_updates
            w_layer_pos.append({})
            # w_layer_pos[i] = {}
            for j, node in enumerate(layer.nodes()):
                w_layer_pos[i][node.id] = j

        # print('w updates', w_updates)
        # print('w layer pos', w_layer_pos)

        # keep track of ll
        epoch_cost = 0.0
        old_ll = 0.0

        epoch = 0
        done_looping = False

        n_train_instances = train.shape[0]
        n_train_batches = (n_train_instances
                           // batch_size)

        # epochs loop
        while (epoch < n_epochs) and (not done_looping):

            epoch = epoch + 1
            print('>>>>> epoch {0}/{1}'.format(epoch, n_epochs))

            epoch_start_t = perf_counter()
            # reset updates
            for l, layer in enumerate(w_updates):
                for n, node in enumerate(layer):
                    for c, child in enumerate(node):
                        w_updates[l][n][c] = 0.0

            # save and reset ll
            old_ll = epoch_cost
            epoch_cost = 0.0

            avg_time = 0.0

            # for each training example
            # TODO we could shuffle them...
            for m, instance in enumerate(train):

                inst_start_t = perf_counter()
                # evaluate it
                sample_lls = None
                if hard:
                    sample_lls = self.mpe_eval(instance)
                else:
                    sample_lls = self.eval(instance)

                # cumulate it (assuming one radix only)
                sample_ll = sample_lls[0]
                epoch_cost += sample_ll

                eval_end_t = perf_counter()
                # print('evalua time', eval_end_t - inst_start_t)

                # weight update (hard?)
                if hard:
                    hard_start_t = perf_counter()
                    # descending with MPE inference
                    # just adding a + 1 counter
                    for l_id, par_id, child_pos in self.mpe_traversal():
                        try:
                            par_pos = w_layer_pos[l_id][par_id]
                            w_updates[l_id][par_pos][child_pos] += 1.0
                        except:
                            print('error', l_id, par_id, par_pos, child_pos)
                    hard_end_t = perf_counter()
                    # print('hard time', hard_end_t -
                    #       hard_start_t)

                else:
                    back_start_t = perf_counter()
                    # backprop
                    self.backprop()
                    back_end_t = perf_counter()
                    # print('backpr time', back_end_t -
                    #       back_start_t)
                    update_start_t = perf_counter()
                    # update weights
                    for l, layer_upd in enumerate(w_updates):
                        layer = self._layers[sum_layers_only[l]]
                        for p, parent in enumerate(layer_upd):
                            sum_node = layer._nodes[p]
                            for c in range(len(parent)):
                                child = sum_node.children[c]
                                child_log_w = sum_node.log_weights[c]
                                w_updates[l][p][c] += \
                                    math.exp(child.log_val +
                                             sum_node.log_der +
                                             child_log_w -
                                             sample_ll)
                    update_end_t = perf_counter()
                    # print('update time', update_end_t -
                    #       update_start_t)
                inst_end_t = perf_counter()
                # print('instan time', inst_end_t -
                #       inst_start_t)

                avg_time += (inst_end_t - inst_start_t)

                sys.stdout.write(
                    '\r-- mini batch {:d}/{:d} ({:.4f} secs avg)'
                    ' [{:.4f} ll avg]'
                    .format(m + 1, n_train_batches,
                            avg_time / (m + 1),
                            epoch_cost / (m + 1)))
                sys.stdout.flush()

            # normalizing weight updates
            for l, layer_upd in enumerate(w_updates):
                layer = self._layers[sum_layers_only[l]]
                for p, parent in enumerate(layer_upd):
                    sum_node = layer._nodes[p]
                    sum_node_tot = 0.0
                    num_children = len(parent)
                    for c in range(num_children):
                        sum_node_tot += w_updates[l][p][c]

                    # if no update occurred , all weights are normalized to
                    # 1/num_children
                    if sum_node_tot > 0.0:
                        for c in range(num_children):
                            w_updates[l][p][c] /= sum_node_tot
                    else:
                        for c in range(num_children):
                            w_updates[l][p][c] = 1.0 / float(num_children)
                    # setting the weights
                    sum_node.set_weights(w_updates[l][p])

            # checking for convergence

                    print('\ttrain ll', epoch_cost / train.shape[0])
            # computing the log-likelihood on the validation set, if any
            # rel_imp = abs((current_ll - old_ll) / current_ll)
            # print('relative improvement -> {0}'.format(rel_imp))
            # if rel_imp < epsilon:
            #     break

            epoch_end_t = perf_counter()
            print('elapsed {0} secs'.format(epoch_end_t -
                                            epoch_start_t))

            epoch_end_t = perf_counter()
            print('elapsed {0} secs'.format(epoch_end_t -
                                            epoch_start_t))
            if valid is not None:
                valid_start_t = perf_counter()
                valid_lls = self.eval(valid)
                valid_avg_ll = numpy.mean(valid_lls)
                valid_end_t = perf_counter()
                print('\tLL on val:{ll} in {ss} secs'.
                      format(ll=valid_avg_ll,
                             ss=(valid_end_t - valid_start_t)))

            # rel_imp = abs((current_ll - old_ll) / current_ll)
            rel_imp = abs((epoch_cost - old_ll) / epoch_cost)
            print('relative improvement -> {0}'.format(rel_imp))
            if rel_imp < epsilon:
                break


def evaluate_on_dataset(spn, data):

    n_instances = data.shape[0]
    pred_lls = numpy.zeros(n_instances)

    for i, instance in enumerate(data):
        (pred_ll, ) = spn.single_eval(instance)
        pred_lls[i] = pred_ll

    return pred_lls
