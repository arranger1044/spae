import sys

import numpy

# marginalize indicator
MARG_IND = -1

# log of zero const, to avoid -inf
# numpy.exp(LOG_ZERO) = 0
LOG_ZERO = -1e3


def IS_LOG_ZERO(log_val):
    """
    checks for a value to represent the logarithm of 0.
    The identity to be verified is that:
    IS_LOG_ZERO(x) && exp(x) == 0
    according to the constant LOG_ZERO
    """
    return (log_val <= LOG_ZERO)


# defining a numerical correction for 0
EPSILON = sys.float_info.min

# size for integers
INT_TYPE = 'int8'

# seed for random generators
RND_SEED = 31

# negative infinity for worst log-likelihood
NEG_INF = -sys.float_info.max

# abstract class definition
from abc import ABCMeta
from abc import abstractmethod


class AbstractSpn(metaclass=ABCMeta):

    """
    WRITEME
    """
    # __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, input_layer=None):
        """
        WRITEME
        """

    @abstractmethod
    def eval(self, input):
        """
        WRITEME
        """
    #
    # we just have one input layer
    #
    @abstractmethod
    def set_input_layer(self, layer):
        """
        WRITEME
        """

    @abstractmethod
    def top_down_nodes(self):
        """
        Top down node traversal
        """

    @abstractmethod
    def fit(self, train, valid, test, algo, options):
        """
        WRITEME
        """

    def __repr__(self):
        """
        Printing an SPN summary
        """
        layer_strings = [msg for msg in map(str, self._layers)]
        layer_strings.reverse()
        layer_strings.append(str(self._input_layer))
        stats = '\n'.join(layer_strings)
        return stats

    def input_layer(self):
        """
        WRITEME
        """
        return self._input_layer

    def smooth_leaves(self, alpha):
        """
        Laplacian smoothing of the probability values
        of the leaf nodes (if the leaf represents a univariate distribution)
        """
        self._input_layer.smooth_probs(alpha)

    def n_nodes(self):
        """
        WRITEME
        """
        # nodes = self._input_layer.n_nodes()
        nodes = len(list(self.top_down_nodes()))

        return nodes

    def n_edges(self):
        """
        WRITEME
        """
        #
        # adding input layer too, it may contain cltrees
        # edges = self._input_layer.n_edges()
        edges = [len(node.children) for node in self.top_down_nodes()
                 if hasattr(node, 'children')]

        return sum(edges)

    def n_leaves(self):
        """
        WRITEME
        """
        return self._input_layer.n_nodes()

    def n_weights(self):
        """
        WRITEME
        """
        weights = [len(node.weights) for node in self.top_down_nodes()
                   if hasattr(node, 'weights')]
        return weights

    @abstractmethod
    def n_sum_nodes(self):
        """
        Return the numbers of sum nodes in the network
        """

    @abstractmethod
    def n_product_nodes(self):
        """
        Return the numbers of prod nodes in the network
        """

    def stats(self):
        """
        WRITEME
        """
        # total stats
        stats = '*************************\n'\
            '* nodes:\t{0}\t*\n'\
            '* edges:\t{1}\t*\n'\
            '* weights:\t{2}\t*\n'\
            '*************************'.format(self.n_nodes(),
                                               self.n_edges(),
                                               self.n_weights())
        return stats


class AbstractLayeredSpn(AbstractSpn, metaclass=ABCMeta):

    """
    WRITEME
    """
    # __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, input_layer=None, layers=[]):
        """
        WRITEME
        """

    @abstractmethod
    def eval(self, input):
        """
        WRITEME
        """
    #
    # layer setting routines
    #
    @abstractmethod
    def set_input_layer(self, layer):
        """
        WRITEME
        """

    @abstractmethod
    def set_layers(self, layers):
        """
        WRITEME
        """

    @abstractmethod
    def add_layer(self, layer, pos=None):
        """
        WRITEME
        """

    @abstractmethod
    def fit(self, train, valid, test, algo, options):
        """
        WRITEME
        """

    def __repr__(self):
        """
        Printing an SPN summary
        WRITEME
        """
        layer_strings = [msg for msg in map(str, self._layers)]
        layer_strings.reverse()
        layer_strings.append(str(self._input_layer))
        stats = '\n'.join(layer_strings)
        return stats

    def top_down_layers(self):
        """
        Traversing layers top down
        """
        for layer in reversed(self._layers):
            yield layer
        yield self._input_layer

    def bottom_up_layers(self):
        """
        Traversing laeyrs bottom up
        """
        yield self._input_layer
        for layer in self._layers:
            yield layer

    def input_layer(self):
        """
        WRITEME
        """
        return self._input_layer

    def smooth_leaves(self, alpha):
        """
        Laplacian smoothing of the probability values
        of the leaf nodes (if the leaf represents a univariate distribution)
        """
        self._input_layer.smooth_probs(alpha)

    def n_layers(self):
        """
        WRITEME
        """
        return len(self._layers) + 1

    def n_nodes(self):
        """
        WRITEME
        """
        nodes = self._input_layer.n_nodes()
        for layer in self._layers:
            nodes += layer.n_nodes()
        return nodes

    def n_edges(self):
        """
        WRITEME
        """
        #
        # adding input layer too, it may contain cltrees
        edges = self._input_layer.n_edges()
        for layer in self._layers:
            edges += layer.n_edges()

        return edges

    def n_leaves(self):
        """
        WRITEME
        """
        return self._input_layer.n_nodes()

    def n_weights(self):
        """
        WRITEME
        """
        weights = 0
        for layer in self._layers:
            weights += layer.n_weights()
        return weights

    def n_sum_nodes(self):
        """
        Return the numbers of sum nodes in the network
        """
        return None

    def n_product_nodes(self):
        """
        Return the numbers of prod nodes in the network
        """
        return None

    def n_unique_scopes(self):
        """
        Return the number of different scopes in the network
        """

    def stats(self):
        """
        WRITEME
        """
        # total stats
        stats = '*************************\n'\
            '* depth:\t{0}\t*\n'\
            '* nodes:\t{1}\t*\n'\
            '    - sum:\t{4}\t*\n'\
            '    - prod:\t{5}\t*\n'\
            '    - leaves:\t{6}\t*\n'\
            '* edges:\t{2}\t*\n'\
            '* weights:\t{3}\t*\n'\
            '* scopes:\t{7}\t*\n'\
            '*************************'.format(self.n_layers(),
                                               self.n_nodes(),
                                               self.n_edges(),
                                               self.n_weights(),
                                               self.n_sum_nodes(),
                                               self.n_product_nodes(),
                                               self.n_leaves(),
                                               self.n_unique_scopes())
        return stats


def evaluate_on_dataset_batch(spn, data, batch_size=None, _keras=False):

    n_instances = data.shape[0]
    pred_lls = numpy.zeros(n_instances)

    if batch_size is None:
        batch_size = n_instances

    n_batches = max(n_instances // batch_size, 1)
    for i in range(n_batches):
        preds = spn.score(data[i * batch_size: (i + 1) * batch_size], _keras=_keras)
        pred_lls[i * batch_size: (i + 1) * batch_size] = preds.flatten()

    #
    # some instances remaining?
    rem_instances = n_instances - n_batches * batch_size
    if rem_instances > 0:
        preds = spn.score(data[rem_instances:], _keras=_keras)
        pred_lls[rem_instances:] = preds.flatten()

    return pred_lls
