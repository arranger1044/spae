import numpy
import theano
import theano.tensor as T

from spn import LOG_ZERO
from .initializations import Initialization, sharedX, ndim_tensor
import os

#
# inspired by Keras
#


def exp_activation(x):
    return T.exp(x)


def log_activation(x):

    return T.log(x).clip(LOG_ZERO, 0.)


def log_sum_exp_activation(x, axis=1):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


class TheanokLayer(object):

    """
    WRITEME
    """
    __module__ = os.path.splitext(os.path.basename(__file__))[0]
    _counter_id = 0

    def __init__(self,
                 output_dim,
                 weights,
                 input_dim=None,
                 input_layers=None,
                 layer_id=None,
                 init='uniform',
                 activation=None,
                 constraints=None,
                 scope=None,
                 batch_size=None):
        """
        initing
        """
        #
        # set id
        if layer_id:
            self.id = layer_id
        else:
            self.id = TheanokLayer._counter_id
            TheanokLayer._counter_id += 1

        #
        # storing input/output layers refs
        self.input_layers = set()
        if input_layers:
            input_dim = sum([l.output_dim for l in input_layers])
            for i in input_layers:
                self.input_layers.add(i)
        elif not input_dim:
            raise ValueError('Input dim not specified')

        self.output_layers = set()

        if weights is not None:
            assert weights.shape[0] == input_dim
            assert weights.shape[1] == output_dim
            self.initial_weights = weights
        else:
            self.initial_weights = None

        self.batch_size = batch_size
        #
        # setting scope matrix
        self.scope = scope

        #
        # setting dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.set_input_shape((self.input_dim,))

        #
        # setting activation
        self.activation = activation

        #
        # setting optimization constraints
        self.constrains = constraints

        #
        # parameters initialization
        self.init = Initialization.get(init)

    def build(self):
        """
        WRITEME
        """
        #
        # nb_samples X n_input_units
        self.input = T.matrix()

        #
        # n_input_units X n_output_units
        if self.initial_weights is not None:
            self.W = sharedX(self.initial_weights, name='W_{}'.format(self.id))
            self.weights = [self.W]

        #
        # n_output_units X n_vars
        if self.scope:
            self.C = sharedX(self.scope, name='C_{}'.format(self.id))

    #
    # TODO: clean the superfluous parts here from keras
    def set_input_shape(self, input_shape):
        if type(input_shape) not in [tuple, list]:
            raise Exception('Invalid input shape - input_shape should be a tuple of int.')
        input_shape = (None,) + tuple(input_shape)
        if hasattr(self, 'input_ndim') and self.input_ndim:
            if self.input_ndim != len(input_shape):
                raise Exception('Invalid input shape - Layer expects input ndim=' +
                                str(self.input_ndim) + ', was provided with input shape '
                                + str(input_shape))
        self._input_shape = input_shape
        self.input = ndim_tensor(len(self._input_shape))
        self.build()

    def add_input_layer(self, layer):
        self.input_layers.add(layer)

    def add_output_layer(self, layer):
        self.output_layers.add(layer)

    def set_previous(self, previous_layers):
        """
        WRITEME
        """
        previous_output_dim = sum([l.output_dim for l in previous_layers])
        assert self.input_dim == previous_output_dim

        # self.previous = previous_layer
        self.input_layers = previous_layers
        self.build()

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W))
        return output

    def get_input(self, train=False):
        # if hasattr(self, 'previous'):
        #     return self.previous.get_output(train=train)
        if hasattr(self, 'input_layers') and self.input_layers:
            previous_outputs = [l.get_output(train=train) for l in sorted(self.input_layers)]
            return theano.tensor.concatenate(previous_outputs, axis=1)
        elif hasattr(self, 'input'):
            return self.input
        else:
            raise Exception('Layer is not connected\
            and is not an input layer.')

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

    def n_nodes(self):
        return self.output_dim

    def n_edges(self):
        if hasattr(self, 'W'):
            return numpy.sum(self.W.get_value() > 0.0)
        else:
            return 0

    def __eq__(self, layer):
        return self.id == layer.id

    def __lt__(self, layer):
        return self.id < layer.id

    def __hash__(self):
        # print('has')
        # from pprint import pprint
        # pprint(vars(self))
        return hash(self.id)

    def compile(self,):
        """
        Creating a theano function to retrieve the layer output
        """
        self.evaluate_layer_func = theano.function([self.input], self.get_output())
        # output = self.get_output()
        # self.evaluate_layer_func = theano.function([self.get_input()], output)

    def evaluate(self, input_signal, flatten=False):
        res = self.evaluate_layer_func(input_signal)
        if flatten:
            res = res.flatten()
        return res

    def stats(self):
        n_edges = self.n_edges()
        stats_str = '{1}\tx\t{0},\t{2}\t({3})'.format(self.n_nodes(),
                                                      self.input_dim,
                                                      n_edges,
                                                      n_edges / (self.n_nodes() * self.input_dim))
        return stats_str

    def __repr__(self):
        layer_str = 'id:{0} [{1}]->[{2}]\n'.format(self.id,
                                                   ','.join([str(l.id)
                                                             for l in sorted(self.input_layers)]),
                                                   ','.join([str(l.id)
                                                             for l in sorted(self.output_layers)]))
        weights_str = ""
        if hasattr(self, 'W'):
            weights_str = '\n{}\n'.format(self.W.get_value())
        div = '\n**********************************************************\n'
        stats_str = self.stats()
        return layer_str + weights_str + stats_str + div


class SumLayer(TheanokLayer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 weights,
                 init='uniform'):
        """
        Properly calling basic layer
        """

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=weights,
                         init=init,
                         activation=log_activation,
                         constraints=None)

    def build(self):
        #
        # building the base layer
        super().build()

        #
        # I should have saved

        #
        # then storing the weights as parameters
        self.params = [self.W]

    def __repr__(self):
        return '[sum layer:]\n' + TheanokLayer.__repr__(self)


class InputLayer_logspace(TheanokLayer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 mask,
                 layer_id=None):
        """
        Just doing the logarithm of the input
        """
        assert len(mask) == output_dim
        self.mask = mask

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=None,
                         layer_id=layer_id,
                         # activation=log_sum_exp_activation,
                         constraints=None)

    def build(self):
        #
        # building the base layer
        super().build()

        self.M = sharedX(self.mask, name='mask_{}'.format(self.id), dtype=int)

        self.params = []

    def get_output(self, train=False):
        X = self.get_input(train)

        return T.clip(T.log(X[:, self.M]), LOG_ZERO, 0)

    def __repr__(self):
        return '[input layer log:]\n' + TheanokLayer.__repr__(self)


class SumLayer_logspace(TheanokLayer):

    # __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self,
                 input_dim,
                 output_dim,
                 weights,
                 layer_id=None,
                 init='uniform'):
        """
        The activation function is the logsumexp,
        for numerical stability here we are assuming the product layer to be linear layer
        """

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=weights,
                         init=init,
                         layer_id=layer_id,
                         # activation=log_sum_exp_activation,
                         constraints=None)

    def build(self):
        #
        # building the base layer
        super().build()

        #
        # then storing the weights as parameters
        self.params = [self.W]

    def get_output(self, train=False):
        X = self.get_input(train)

        X = T.log(self.W) + X.dimshuffle(0, 1, 'x')
        x_max = T.max(X, axis=1, keepdims=True)
        return (T.log(T.sum(T.exp(X - x_max), axis=1, keepdims=True)) + x_max).reshape((X.shape[0],
                                                                                        self.W.shape[1]))

    def __repr__(self):
        return '[sum layer log:]\n' + TheanokLayer.__repr__(self)


class MaxLayer_logspace(TheanokLayer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 weights,
                 init='uniform',
                 layer_id=None,
                 batch_size=None):
        """
        The activation function is a max (still in the log space)
        """

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=weights,
                         init=init,
                         layer_id=layer_id,
                         # activation=log_sum_exp_activation,
                         constraints=None,
                         batch_size=batch_size)

    def build(self):
        #
        # building the base layer
        super().build()

        # # FIXME: this shall cope with a variable batch size
        # # storing a tensor for the max position values
        # weight_shape = self.W.shape.eval()
        # m_values = numpy.zeros((self.batch_size, weight_shape[0], weight_shape[1]))
        # self.M = sharedX(m_values, name='M_{}'.format(self.id))

        #
        # then storing the weights as parameters
        self.params = [self.W]

    def get_output(self, train=False):
        X = self.get_input(train)

        X = T.log(self.W) + X.dimshuffle(0, 1, 'x')
        X_max = T.max(X, axis=1)

        return X_max

    def compile(self,):

        #
        # and adding a function to retrieve the max map
        # a binary mask that has 1 when there was the max connection
        X = self.input
        X = T.log(self.W) + X.dimshuffle(0, 1, 'x')
        #
        # TODO: mask only one value (argmax)
        M = T.switch(T.eq(T.max(X, axis=1, keepdims=True), X), 1, 0)

        self.evaluate_layer_func = theano.function([self.input], [self.get_output(), M])

    def __repr__(self):
        return '[max layer log:]\n' + TheanokLayer.__repr__(self)


class ProductLayer(TheanokLayer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 weights,
                 layer_id=None,
                 batch_size=None):
        """
        Properly calling basic layer
        """

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=weights,
                         activation=exp_activation,
                         layer_id=layer_id,
                         batch_size=batch_size)

    def build(self):
        #
        # building the base layer
        super().build()

        #
        # Shall we have to store the parameters for product layers?
        self.params = [self.W]

    def __repr__(self):
        return '[prod layer:]\n' + TheanokLayer.__repr__(self)


class ProductLayer_logspace(TheanokLayer):

    # __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self,
                 input_dim,
                 output_dim,
                 weights,
                 layer_id=None,
                 batch_size=None):
        """
        No activation function, the output is in the log domain
        """

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         weights=weights,
                         layer_id=layer_id,
                         batch_size=batch_size)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = T.dot(X, self.W)
        return output

    def build(self):
        #
        # building the base layer
        super().build()

        #
        # Shall we have to store the parameters for product layers?
        self.params = [self.W]

    def __repr__(self):
        return '[prod layer log:]\n' + TheanokLayer.__repr__(self)
