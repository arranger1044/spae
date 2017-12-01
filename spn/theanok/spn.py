import numpy
import theano
import theano.tensor as T

from .initializations import ndim_tensor

import sys
# from .layers import TheanokLayer
# from .layers import SumLayer_logspace
# from .layers import ProductLayer_logspace
# from .layers import InputLayer_logspace

import theano.misc.pkl_utils

import pickle

from collections import defaultdict
#
# TODO: inherit from AbstractSpn


class SequentialSpn(object):

    def __init__(self,
                 layers=[]):
        """
        WRITEME
        """
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        """
        WRITEME
        """
        self.layers.append(layer)
        #
        # for each layer after the first one
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
            # if not hasattr(self.layers[0], 'input'):
            #     self.set_input()

    def get_output(self, train=False):
        """
        WRITEME
        """
        return self.layers[-1].get_output(train)

    # def set_input(self):
    #     for l in self.layers:
    #         if hasattr(l, 'input'):
    #             ndim = l.input.ndim
    #             self.layers[0].input = ndim_tensor(ndim)
    #             break

    def get_input(self, train=False):
        # if not hasattr(self.layers[0], 'input'):
        #     self.set_input()
        return self.layers[0].get_input(train)

    def compile(self, ):
        """
        WRITEME
        """

    def fit(self, ):
        """
        WRITEME
        """

    def predict(self, ):
        """
        WRITEME
        """

    def evaluate(self, ):
        """
        WRITEME
        """


class BlockLayeredSpn(object):

    def __init__(self,
                 layers=[],
                 output_layers=[]):
        """
        Layers is a sequence of pairs (layer, input_layers_seq)
        where input_layers_seq is the sequence containing the inputs to layer
        """

        #
        # adding and BUILDING layers
        self.layers = []

        # input_layers = []
        self.input_layers = []

        #
        # setting one single input to all
        self.input = theano.tensor.matrix()

        #
        # filtering the layers with not direct input
        for layer, prevs in layers:
            # if not prevs:
            #     input_layers.append(layer)

            self.add(layer, prevs)

        # for layer in input_layers:
        #     layer.input = self.input

        self.output_layers = output_layers

    def add(self, layer, input_layers=[]):
        """
        WRITEME
        """
        self.layers.append(layer)
        #
        # for each layer after the first one
        # if len(self.layers) > 1:

        if not input_layers:
            self.add_input_layer(layer)
        else:
            for input_layer in input_layers:
                layer.add_input_layer(input_layer)
                input_layer.add_output_layer(layer)

            # self.layers[-1].set_previous(self.layers[-2])

            # if not hasattr(self.layers[0], 'input'):
            #     self.set_input()

    def add_input_layer(self, layer):
        layer.input = self.input
        self.input_layers.append(layer)

    def add_output_layer(self, layer):
        self.output_layers.append(layer)

    def get_output(self, train=False):
        """
        WRITEME
        """
        #
        # TODO: generalize
        # assuming just one last level as output
        return self.layers[-1].get_output(train)

    # def set_input(self):
    #     for l in self.layers:
    #         if hasattr(l, 'input'):
    #             ndim = l.input.ndim
    #             self.layers[0].input = ndim_tensor(ndim)
    #             break

    def get_input(self, train=False):

        # if not hasattr(self.layers[0], 'input'):
        #     self.set_input()

        # return self.layers[0].get_input(train)
        # return self.input_layer.get_input(train)
        return self.input

    def compile(self, ):
        """
        Building functions:
          - to evaluate the network (pointwise and marginal evidence)
          - MPE evidence
          - to predict
        """
        network_input = self.get_input()
        network_output = self.get_output()

        self.evaluate_func = theano.function([network_input], network_output)

    def fit(self, ):
        """
        WRITEME
        """

    def predict(self, ):
        """
        WRITEME
        """

    def evaluate(self, instances, flatten=False):
        """
        Evaluates the network bottom up after seeing the evidences in instances
        """
        res = self.evaluate_func(instances)
        if flatten:
            res = res.flatten()
        return res

    def evaluate_mpe(self, instances):
        """
        Evaluates the network bottom up for MPE
        after seeing the evidences in instances
        """
        #
        # bottom-up step
        #
        signal_map = {}
        for layer in self.layers:

            if not hasattr(layer, 'evaluate_layer_func'):
                layer.compile()

            #
            # retrieve input signal
            inputs = [signal_map[in_layer] for in_layer in sorted(layer.input_layers)]
            if not inputs:
                inputs = [instances]
            #
            # concatenating
            input_signals = numpy.concatenate(inputs, axis=1)

            output_signal = layer.evaluate(input_signals)

            signal_map[layer] = output_signal

        #
        # top-down step
        #

    def __repr__(self):
        layer_strings = [msg for msg in map(str, self.layers)]
        layer_strings.reverse()
        stats = '\n'.join(layer_strings)
        return stats

    def layer_stats(self):
        layer_strings = ['[{}] {}'.format(l.id, l.stats()) for l in self.layers]
        layer_strings.reverse()
        stats = '\n'.join(layer_strings)
        return stats

    def remove_double_links_layers(self):
        """
        Removing the output_layers pointers
        """
        for layer in self.layers:
            layer.output_layers = set()

    def double_linking_layers(self):
        """
        Setting references to one layer
        """
        inv_layer_assoc = defaultdict(set)
        for layer in self.layers:
            for input_layer in layer.input_layers:
                inv_layer_assoc[input_layer].add(layer)

        for input_layer, output_layers in inv_layer_assoc.items():
            for o in output_layers:
                input_layer.add_output_layer(o)

    def dump(self, file):
        #
        # removing circular links before
        self.remove_double_links_layers()

        sys.setrecursionlimit(1000000000)
        # theano.misc.pkl_utils.dump(self, file)
        pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        # spn = theano.misc.pkl_utils.load(file)
        spn = pickle.load(file)
        #
        # putting back circular links
        spn.double_linking_layers()
        return spn


def evaluate_on_dataset_batch(spn, data, batch_size=None):

    n_instances = data.shape[0]
    pred_lls = numpy.zeros(n_instances)

    if batch_size is None:
        batch_size = n_instances

    n_batches = max(n_instances // batch_size, 1)
    for i in range(n_batches):
        preds = spn.evaluate(data[i * batch_size: (i + 1) * batch_size])
        pred_lls[i * batch_size: (i + 1) * batch_size] = preds.flatten()

    #
    # some instances remaining?
    rem_instances = n_instances - n_batches * batch_size
    if rem_instances > 0:
        preds = spn.evaluate(data[rem_instances:])
        pred_lls[rem_instances:] = preds.flatten()

    return pred_lls
