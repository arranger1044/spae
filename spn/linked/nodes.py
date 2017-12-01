
from spn import utils
from spn import LOG_ZERO
from spn import MARG_IND
from spn import IS_LOG_ZERO
from spn import RND_SEED

import numpy

from math import log
from math import exp

from cltree.cltree import CLTree

import dataset

import numba

from keras import backend as K

import tensorflow as tf

from collections import defaultdict

NODE_SYM = 'u'  # unknown type
SUM_NODE_SYM = '+'
PROD_NODE_SYM = '*'
INDICATOR_NODE_SYM = 'i'
DISCRETE_VAR_NODE_SYM = 'd'
CHOW_LIU_TREE_NODE_SYM = 'c'
CONSTANT_NODE_SYM = 'k'


node_specs = [('log_val', numba.float64)]


# @numba.jitclass(node_specs)
class Node(object):

    """
    WRITEME
    """
    # class id counter
    id_counter = 0

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        # default val is 0.
        self.log_val = LOG_ZERO

        # setting id and incrementing
        self.id = Node.id_counter
        Node.id_counter += 1

        # derivative computation
        self.log_der = LOG_ZERO

        self.var_scope = var_scope

        # self.children_log_vals = numpy.array([])

    def __repr__(self):
        return 'id: {id} scope: {scope}'.format(id=self.id,
                                                scope=self.var_scope)

    # this is probably useless, using it for test purposes
    def set_val(self, val):
        """
        WRITEME
        """
        if numpy.allclose(val, 0, 1e-10):
            self.log_val = LOG_ZERO
        else:
            self.log_val = log(val)

    def __hash__(self):
        """
        A node has a unique id
        """
        return hash(self.id)

    def __eq__(self, other):
        """
        WRITEME
        """
        return self.id == other.id

    def node_type_str(self):
        return NODE_SYM

    def node_short_str(self):
        return "{0} {1}\n".format(self.node_type_str(),
                                  self.id)

    @classmethod
    def reset_id_counter(cls):
        """
        WRITEME
        """
        Node.id_counter = 0

    @classmethod
    def set_id_counter(cls, val):
        """
        WRITEME
        """
        Node.id_counter = val


@numba.njit
def eval_sum_node(children_log_vals, log_weights):
    """
    numba version
    """

    max_log = LOG_ZERO

    n_children = children_log_vals.shape[0]

    # getting the max
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        w_sum = ch_log_val + log_weight
        if w_sum > max_log:
            max_log = w_sum

    # log_unnorm = LOG_ZERO
    # max_child_log = LOG_ZERO

    sum_val = 0.
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        # for node, log_weight in zip(children, log_weights):
        # if node.log_val is False:
        ww_sum = ch_log_val + log_weight
        sum_val += exp(ww_sum - max_log)

    # is this bad code?
    log_val = LOG_ZERO
    if sum_val > 0.:
        log_val = log(sum_val) + max_log

    return log_val
    # log_unnorm = log(sum_val) + max_log
    # self.log_val = log_unnorm - numpy.log(self.weights_sum)
    # return self.log_val


@numba.njit
def eval_max_node(children_log_vals, log_weights):
    """
    numba version
    """

    max_log = LOG_ZERO

    n_children = children_log_vals.shape[0]

    # getting the max
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        w_sum = ch_log_val + log_weight
        if w_sum > max_log:
            max_log = w_sum

    # log_unnorm = LOG_ZERO

    return max_log
    # log_unnorm = log(sum_val) + max_log
    # self.log_val = log_unnorm - numpy.log(self.weights_sum)
    # return self.log_val


@numba.jit
def eval_sum_node_py(sum_node):
    """
    WRITEME
    """

    max_log = LOG_ZERO

    n_children = len(sum_node.children)

    # getting the max
    for i in range(n_children):
        ch_log_val = sum_node.children[i].log_val
        log_weight = sum_node.log_weights[i]
        w_sum = ch_log_val + log_weight
        if w_sum > max_log:
            max_log = w_sum

    # log_unnorm = LOG_ZERO
    # max_child_log = LOG_ZERO

    sum_val = 0.
    for i in range(n_children):
        ch_log_val = sum_node.children[i].log_val
        log_weight = sum_node.log_weights[i]
        ww_sum = ch_log_val + log_weight
        sum_val += exp(ww_sum - max_log)

    # is this bad code?
    log_val = LOG_ZERO
    if sum_val > 0.:
        log_val = log(sum_val) + max_log

    return log_val


class SumNode(Node):

    """
    WRITEME
    """

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        Node.__init__(self, var_scope)
        self.children = []
        self.weights = []
        self.log_weights = []
        self.log_weights_vals = numpy.array([])
        self.weights_sum = 0

    def add_child(self, child, weight):
        """
        WRITEME
        """
        self.children.append(child)
        self.weights.append(weight)
        weight_log = log(weight) if weight > 0.0 else LOG_ZERO
        self.log_weights.append(weight_log)
        # self.children_log_vals = numpy.zeros(len(self.children))
        # self.log_weights_vals = numpy.array(self.log_weights)
        self.weights_sum += weight

    def remove_child(self, child):
        child_pos = self.children.index(child)
        child_weight = self.weights[child_pos]
        self.weights_sum -= child_weight

        self.children.pop(child_pos)
        self.weights.pop(child_pos)
        self.log_weights.pop(child_pos)

        # self.children_log_vals = numpy.zeros(len(self.children))
        # self.log_weights_vals = numpy.array(self.log_weights)

        return child_weight

    def set_weights(self, weights):
        """
        WRITEME
        """

        self.weights = weights

        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # updating log weights
        for i, weight in enumerate(weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

        # and also the sum
        self.weights_sum = sum(weights)

        self.log_weights_vals = numpy.array(self.log_weights)

    # @numba.jit
    def eval(self):
        """
        WRITEME
        """

        # self.log_val = eval_sum_node_py(self)
        # return

        # resetting the log derivative
        self.log_der = LOG_ZERO

        max_log = LOG_ZERO

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > max_log:
                max_log = w_sum

        # log_unnorm = LOG_ZERO
        # max_child_log = LOG_ZERO

        sum_val = 0.
        for node, log_weight in zip(self.children, self.log_weights):
            # if node.log_val is False:
            ww_sum = node.log_val + log_weight
            sum_val += exp(ww_sum - max_log)

        # is this bad code?
        if sum_val > 0.:
            self.log_val = log(sum_val) + max_log

        else:
            self.log_val = LOG_ZERO

        #
        # # up to now numba

        # log_unnorm = log(sum_val) + max_log
        # self.log_val = log_unnorm - numpy.log(self.weights_sum)
        # return self.log_val

        # self.log_val = eval_sum_node(numpy.array([child.log_val
        #                                           for child in self.children]),
        #                              numpy.array(self.log_weights))
        # for i, child in enumerate(self.children):
        #     self.children_log_vals[i] = child.log_val
        # self.log_val = eval_sum_node(self.children_log_vals,
        #                              self.log_weights_vals)

    def mpe_eval(self):
        """
        WRITEME
        """
        # resetting the log derivative
        self.log_der = LOG_ZERO

        # log_val is used as an accumulator, one less var
        # self.log_val = LOG_ZERO
        self.log_val = -numpy.inf

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > self.log_val:
                self.log_val = w_sum

    def backprop(self):
        """
        WRITE
        """
        # if it is not zero we can pass
        if self.log_der > LOG_ZERO:
            # dS/dS_n = sum_{p}: dS/dS_p * dS_p/dS_n
            # per un nodo somma p
            #
            for child, log_weight in zip(self.children, self.log_weights):
                # print('child before', child.log_der)
                # if child.log_der == LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    child.log_der = self.log_der + log_weight
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    self.log_der + log_weight)
                # print('child after', child.log_der)
        # update weight log der too ?

    def mpe_backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:
            # the child der is the max der among parents
            for child in self.children:
                child.log_der = max(child.log_der, self.log_der)

    def normalize(self):
        """
        WRITEME
        """
        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # computing log(self.weights)
        for i, weight in enumerate(self.weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

    def is_complete(self):

        _complete = True
        # all children scopes shall be equal
        children_scopes = [child.var_scope
                           for child in self.children]

        # adding this node scope
        children_scopes.append(self.var_scope)

        for scope1, scope2 in utils.pairwise(children_scopes):
            if scope1 != scope2:
                _complete = False
                break

        return _complete

    def n_children(self):
        """
        WRITEME
        """
        return len(self.children)

    def build_k(self):
        """
        Building the computational graph for the node in keras
        Note: children log_vals are (batch_size x 1) matrixes
        and the node output is a (batch_size x 1) matrix as well
        """
        self.W = K.variable(value=numpy.array(self.log_weights))
        children_outputs = K.concatenate([c.log_vals for c in self.children],
                                         axis=1) + self.W

        self.log_vals = K.log(K.sum(K.exp(children_outputs), axis=1, keepdims=True))

    def node_type_str(self):
        return SUM_NODE_SYM

    def node_short_str(self):
        children_str = " ".join(["{id}:{weight}".format(id=node.id,
                                                        weight=weight)
                                 for node, weight in zip(self.children,
                                                         self.weights)])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [(node.id, weight)
                         for node, weight in zip(self.children,
                                                 self.weights)]
        msg = ''
        for id, weight in children_info:
            msg += ' ({id} {weight})'.format(id=id,
                                             weight=weight)
        return 'Sum Node {line1}\n{line2}'.format(line1=base,
                                                  line2=msg)


@numba.njit
def eval_prod_node(children_log_vals):
    """
    WRITEME
    """

    n_children = children_log_vals.shape[0]

    # and the zero children counter
    # zero_children = 0

    # computing the log value
    log_val = 0.0
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        # if ch_log_val <= LOG_ZERO:
        #     zero_children += 1

        log_val += ch_log_val

    return log_val  # , zero_children


class ProductNode(Node):

    """
    WRITEME
    """

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        Node.__init__(self, var_scope)
        self.children = []
        # bit for zero children, see Darwiche
        self.zero_children = 0

    def add_child(self, child):
        """
        WRITEME
        """
        self.children.append(child)

        # self.children_log_vals = numpy.zeros(len(self.children))

    def remove_child(self, child):
        child_pos = self.children.index(child)
        self.children.pop(child_pos)

        # self.children_log_vals = numpy.zeros(len(self.children))

    def eval(self):
        """
        WRITEME
        """
        # resetting the log derivative
        self.log_der = LOG_ZERO

        # and the zero children counter
        self.zero_children = 0

        # computing the log value
        self.log_val = 0.0
        for node in self.children:
            if node.log_val <= LOG_ZERO:
                self.zero_children += 1

            self.log_val += node.log_val

        #
        # numba
        # self.log_val = \
        #     eval_prod_node(numpy.array([child.log_val
        #                                 for child in self.children]))
        # return self.log_val
        # for i, child in enumerate(self.children):
        #     self.children_log_vals[i] = child.log_val
        # self.log_val = eval_prod_node(self.children_log_vals)
        # return self.log_val

    def mpe_eval(self):
        """
        Just redirecting normal evaluation
        """
        self.eval()

    def backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:

            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                    # log_der = 0.0
                    # for node in self.children:
                    #     if node != child:
                    #         log_der += node.log_val
                # adding this parent value
                log_der += self.log_der
                # if child.log_der <= LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    # first assignment
                    child.log_der = log_der
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    log_der)

    def mpe_backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:
            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                # adding this parent value
                log_der += self.log_der
                # updating child log der with the max instead of sum
                child.log_der = max(child.log_der, log_der)

    def backprop2(self):
        """
        WRITEME
        """
        # if more than one child has a zero value, cannot propagate
        if self.log_val <= LOG_ZERO:
            count = 0
            for child in self.children:
                if child.log_val <= LOG_ZERO:
                    count += 1
                    if count > 1:
                        return

        # only when needed
        if self.log_der > LOG_ZERO:
            for child in self.children:
                # print('b child val', child.log_val, child.log_der)
                if child.log_val <= LOG_ZERO:
                    # print('child log zero')
                    # shall loop on other children
                    # maybe this is memory consuming, but shall be faster
                    # going to numpy array shall be faster
                    log_der = sum([node.log_val for node in self.children
                                   if node.log_val > LOG_ZERO]) + \
                        self.log_der
                    if child.log_der <= LOG_ZERO:
                        # print('first log, add', log_der)
                        child.log_der = log_der
                    else:

                        child.log_der = numpy.logaddexp(child.log_der,
                                                        log_der)
                        # print('not first log, added', child.log_der)
                # if it is 0 there is no point updating children
                elif self.log_val > LOG_ZERO:
                    # print('par val not zero')
                    if child.log_der <= LOG_ZERO:
                        child.log_der = self.log_der + \
                            self.log_val - \
                            child.log_val
                        # print('child val not zero', child.log_der)
                    else:
                        child.log_der = numpy.logaddexp(child.log_der,
                                                        self.log_der +
                                                        self.log_val -
                                                        child.log_val)
                        # print('child log der not first', child.log_der)

    def is_decomposable(self):

        decomposable = True
        whole = set()
        for child in self.children:
            child_scope = child.var_scope
            for scope_var in child_scope:
                if scope_var in whole:
                    decomposable = False
                    break
                else:
                    whole.add(scope_var)
            else:
                continue
            break

        if whole != self.var_scope:
            decomposable = False
        return decomposable

    def n_children(self):
        """
        WRITEME
        """
        return len(self.children)

    def build_k(self):
        """
        Building the computational graph for the node in keras
        Note: children log_vals are (batch_size x 1) matrixes
        and the node output is a (batch_size x 1) matrix as well
        """

        children_outputs = K.concatenate([c.log_vals for c in self.children],
                                         axis=1)

        self.log_vals = K.sum(children_outputs, axis=1, keepdims=True)

    def node_type_str(self):
        return PROD_NODE_SYM

    def node_short_str(self):
        children_str = " ".join(["{id}".format(id=node.id)
                                 for node in self.children])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [node.id
                         for node in self.children]
        msg = ''
        for id in children_info:
            msg += ' ({id})'.format(id=id)
        return 'Prod Node {line1}\n{line2}'.format(line1=base,
                                                   line2=msg)

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


def make_piecewise_linear_node(data, var,
                               disc_method="histogram",
                               n_bins="auto",
                               interval=None,
                               alpha=None,
                               isotonic=True):

    #
    # discretize in some way
    x, y = continuous_data_to_piecewise_linear_pdf(data,
                                                   disc_method=disc_method,
                                                   n_bins=n_bins,
                                                   interval=interval,
                                                   alpha=alpha)

    #
    # going isotonic unimodal on the discretizations?
    if isotonic:
        x, y = isotonic_unimodal_regression_R(x, y, normalize_data=True)

    #
    # use the piecewise line to now create a node
    leaf_node = PiecewiseLinearPDFNode(var, x, y)

    return leaf_node


def is_piecewice_linear_pdf(x, y):
    return numpy.allclose(numpy.trapz(y, x), 1.0)


def isotonic_unimodal_regression_R(x, y, normalize_data=True):
    """
    Perform unimodal isotonic regression via the Iso package in R
    """

    numpy2ri.activate()
    # n_instances = x.shape[0]
    # assert y.shape[0] == n_instances

    importr('Iso')
    z = robjects.r["ufit"](y, x=x, type='b')
    iso_x, iso_y = numpy.array(z.rx2('x')), numpy.array(z.rx2('y'))

    if normalize_data:
        auc = numpy.trapz(iso_y, iso_x)
        iso_y = iso_y / auc

    assert is_piecewice_linear_pdf(iso_x, iso_y), numpy.trapz(iso_y, iso_x)

    return iso_x, iso_y

import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def continuous_data_to_piecewise_linear_pdf(data,
                                            disc_method="histogram",
                                            n_bins="auto",
                                            interval=None,
                                            alpha=None):
    """
    First we have to discretize the data,
    then we 'fit' a piecewise linear function to represent the pdf
    Outputs two sequences X = x1, x2, ..., xn ; Y = y1, y2, ..., yn representing
    the piecewise linear
    """

    n_instances = data.shape[0]
    disc_data, bins, = None, None
    if disc_method == 'histogram':
        #
        # discretization through an histogram
        disc_data, bins = numpy.histogram(data, bins=n_bins, range=interval, density=False)
        from matplotlib import pyplot
        pyplot.hist(data, n_bins)
        assert len(disc_data) + 1 == len(bins)
    else:
        raise ValueError('{} discretization method not implemented'.format(disc_method))

    n_vals = len(bins) - 1
    #
    # apply Laplace smoothing to the histogram
    if alpha is not None:
        disc_data = (disc_data + alpha) / (n_instances + n_vals * alpha)
    else:
        disc_data = disc_data / n_instances

    # #
    # # histogram smoothing by mean shift?
    # for i in range(1, len(bins) - 1):
    #     disc_data[i] = numpy.sum(disc_data[i - 1:i + 2]) / 3

    #
    # getting the line through the points centered in the bins
    x, y = [], []
    for (b0, b1), f in zip(pairwise(bins), disc_data):
        x.append(b0 + (b1 - b0) / 2)
        y.append(f)

    return numpy.array(x), numpy.array(y)


class PiecewiseLinearPDFNode(Node):

    """
    WRITEME
    """

    def __init__(self, var,
                 x_range,
                 y_range,
                 norm=1):
        """
        x_range: the x values of the domain of the RV
        y_range: the pdf values of the (in the exp domain)
        norm: normalization constant (1 default,
                                      if None we compute it as the area)
        """
        Node.__init__(self, frozenset({var}))
        self.var = var
        self._x_pieces = x_range
        self._y_pieces = y_range
        if norm is None:
            self._Z = numpy.trapz(self._y_pieces,
                                  self._x_pieces)
        else:
            self._Z = norm

    def eval(self, input):
        """
        1. get the interval in which input falls
        2. interpolate
        both could be achieved with numpy.interp
        """
        #
        # values outside the provided interval are assumed to have zero mass
        if input < self._x_pieces[0] or input > self._x_pieces[-1]:
            self.log_val = LOG_ZERO
        else:
            norm_prob = numpy.interp(x=input,
                                     xp=self._x_pieces,
                                     yp=self._y_pieces) / self._Z
            self.log_val = numpy.log(norm_prob)


class CategoricalIndicatorNode(Node):

    """
    WRITEME
    """

    def __init__(self, var, var_val):
        """
        WRITEME
        """
        Node.__init__(self, frozenset({var}))
        self.var = var
        self.var_val = var_val

    def eval(self, input):
        """
        WRITEME
        """
        obs = input[self.var]

        self.log_der = LOG_ZERO

        if obs == MARG_IND:
            self.log_val = 0.
        elif obs == self.var_val:
            self.log_val = 0.
        else:
            self.log_val = LOG_ZERO

    def mpe_eval(self, obs):
        """
        Just redirecting normal evaluation
        """
        self.eval(obs)

    def backprop(self):
        pass

    def n_children(self):
        return 0

    def build_k(self, input):
        """
        Building the computational graph for the node in keras
        Note: children log_vals are (batch_size x 1) matrixes
        and the node output is a (batch_size x 1) matrix as well
        """
        obs = K.reshape(input[:, self.var], (input.shape[0], 1))
        log_vals = K.switch(K.not_equal(obs, self.var_val),
                            K.switch(K.not_equal(obs, MARG_IND), LOG_ZERO, 0.),
                            0.)
        self.log_vals = log_vals

    def node_type_str(self):
        return INDICATOR_NODE_SYM

    def node_short_str(self):

        return "{type} {id} <{var}> {val}".format(type=self.node_type_str(),
                                                  id=self.id,
                                                  var=self.var,
                                                  val=self.var_val)

    def __repr__(self):
        base = Node.__repr__(self)

        return """Indicator Node {line1} var: {var} val: {val}""".format(line1=base,
                                                                         var=self.var,
                                                                         val=self.var_val)


class CLTreeNode(Node):

    """
    An input node representing a Chow-Liu Tree over a set of r.v.
    """

    def __init__(self,
                 vars,
                 var_values,
                 data,
                 factors=None,
                 alpha=0.1):
        """
        vars = the sequence of feature ids
        var_values = the sequence of feature values
        alpha = smoothing parameter
        data = the data slice (2d ndarray) upon which to grow a cltree
        factors = the already computed factors (this is when the model has already been conputed)
        """
        Node.__init__(self, frozenset(vars))

        self.vars = numpy.array(vars)

        self._alpha = alpha
        #
        # assuming all variables to be homogeneous
        # TODO: generalize this
        self._n_var_vals = var_values[0]
        self.var_values = numpy.array(var_values)

        #
        # assuming data is never None
        self._data = data
        self._cltree = CLTree(data,
                              features=self.vars,
                              n_feature_vals=self._n_var_vals,
                              feature_vals=self.var_values,
                              alpha=alpha,
                              sparse=True,
                              mem_free=True)

    def smooth_probs(self, alpha, data=None):
        """
        The only waya to smooth here is to rebuild the whole tree
        """
        self._alpha = alpha

        if data is not None:
            self._data = data
        # else:
        #     raise ValueError('Cannot smooth without data')

        self._cltree = CLTree(data=self._data,
                              features=self.vars,
                              n_feature_vals=self._n_var_vals,
                              feature_vals=self.var_values,
                              alpha=alpha,
                              # copy_mi=False,
                              sparse=True,
                              mem_free=True)

    def eval(self, obs):
        """
        Dispatching inference to the cltree
        """
        #
        # TODO: do something for the derivatives
        self.log_der = LOG_ZERO

        # self.log_val = self._cltree.eval(obs)
        self.log_val = self._cltree.eval_fact(obs)

    def mpe_eval(self, obs):
        """
        WRITEME
        """
        raise NotImplementedError('MPE inference not yet implemented')

    def n_children(self):
        return len(self.vars)

    def node_type_str(self):
        return CHOW_LIU_TREE_NODE_SYM

    def node_short_str(self):
        vars_str = ','.join([var for var in self.vars])
        return "{type} {id}" +\
            " <{vars}>" +\
            " {tree} {factors}".format(type=self.node_type_str(),
                                       id=self.id,
                                       vars=vars_str,
                                       tree=self._cltree.tree_repr(),
                                       factors=self._cltree.factors_repr())

    def __repr__(self):
        """
        WRITEME
        """
        base = Node.__repr__(self)

        return ("""CLTree Smoothed Node {line1}
            vars: {vars} vals: {vals} tree:{tree}""".
                format(line1=base,
                       vars=self.vars,
                       vals=self._n_var_vals,
                       tree=self._cltree.tree_repr()))


@numba.njit
def eval_numba(obs, vars):
    if obs == MARG_IND:
        return 0.
    else:
        return vars[obs]


@numba.njit
def eval_numba_mpe(obs, vars):
    if obs == MARG_IND:
        return numpy.max(vars)
    else:
        return vars[obs]


#@numba.njit
def count_frequences(data, n_vals):
    freqs = numpy.zeros(n_vals, dtype=int)
    for d in data:
        freqs[d] += 1
    return freqs

class CategoricalSmoothedNode(Node):

    """
    WRITEME
    """

    def __init__(self, var, var_values, alpha=0.1,
                 freqs=None, data=None, instances=None):
        """
        WRITEME
        """

        Node.__init__(self, frozenset({var}))

        self.var = var
        self.var_val = var_values

        # building storing freqs
        if data is None:
            if freqs is None:
                self._var_freqs = [1 for i in range(var_values)]
            else:
                self._var_freqs = freqs[:]
        else:
            # better checking for numpy arrays shape
            assert data.shape[1] == 1
            #
            # counting

            self._var_freqs = count_frequences(data, self.var_val)
            #assert(sum(self._var_freqs) > 0)
            
            # (freqs_dict,), _features = dataset.data_2_freqs(data)
            # print('fq', freqs_dict)
            # self._var_freqs = freqs_dict['freqs']

        # computing the smoothed ll
        self._var_probs = CategoricalSmoothedNode.smooth_ll([f for f in self._var_freqs[:]],
                                                            alpha)


        # storing instance ids (it is a list)
        self._instances = instances

    
    def smooth_ll(freqs, alpha):
        """
        WRITEME
        """

        vals = len(freqs)
        freqs_sum = sum(freqs)
        # print(freqs, freqs_sum, vals, alpha, )
        #print(freqs)
        for i, freq in enumerate(freqs):
            log_freq = LOG_ZERO
            if (freq + alpha) > 0.:
                log_freq = log(freq + alpha)
                
            freqs[i] = (log_freq -
                        log(freqs_sum + vals * alpha))
            #print('lfq', freq, log_freq, freqs_sum, freqs_sum + vals * alpha, log(freqs_sum + vals * alpha), freqs[i])
        #print('s', freqs)
        # return freqs
        return numpy.array(freqs)

    def smooth_freq_from_data(data, alpha):
        """
        WRITEME
        """
        # data here shall have only one feature
        assert data.shape[1] == 1
        (freqs_dict,), _features = dataset.data_2_freqs(data)

        return CategoricalSmoothedNode.smooth_ll(freqs_dict['freqs'], alpha)

    def smooth_probs(self, alpha, data=None):
        """
        WRITEME
        """
        if data is None:
            # var_values = len(self._var_freqs)
            # print(self._var_freqs)
            smooth_probs = \
                CategoricalSmoothedNode.smooth_ll([f for f in self._var_freqs[:]],
                                                  alpha)
        else:
            # slicing in two different times to preserve the 2 dims
            data_slice_var = data[:, [self.var]]
            # checking to be sure: it shall be a list btw
            if isinstance(self._instances, list):
                data_slice = data_slice_var[self._instances]
            else:
                data_slice = data_slice_var[list(self._instances)]
            # print('SLICE', data_slice_var, data_slice)
            smooth_probs = \
                CategoricalSmoothedNode.smooth_freq_from_data(data_slice,
                                                              alpha)

        self._var_probs = smooth_probs

    def eval(self, input):
        """
        WRITEME
        """

        obs = input[self.var]
        self.log_der = LOG_ZERO

        # if obs == MARG_IND:
        #     self.log_val = 0.
        # else:
        #     self.log_val = self._var_probs[obs]
        self.log_val = eval_numba(obs, self._var_probs)

    def mpe_eval(self, input):
        """
        When not marginalizing it is the same as for eval()
        """
        # self.eval(obs)
        obs = input[self.var]
        self.log_der = LOG_ZERO

        self.log_val = eval_numba_mpe(obs, self._var_probs)

    def backprop(self):
        pass

    def n_children(self):
        return 0

    def build_k(self, input):
        """
        Building the computational graph for the node in keras
        Note: children log_vals are (batch_size x 1) matrixes
        and the node output is a (batch_size x 1) matrix as well
        """
        log_probs = K.variable(value=self._var_probs)
        self.log_probs = K.reshape(log_probs, (-1, self._var_probs.shape[0]))
        obs = input[:, self.var]

        # self.zeros = K.zeros((K.shape(input)[0], 1))

        if (K.backend() == 'tensorflow'):
            import tensorflow as tf
            self.zeros = tf.tile(K.variable(value=numpy.array([[0.0]])),
                                 tf.pack([tf.shape(obs)[0], 1]))
            self.zeros_int = tf.tile(K.variable(value=numpy.array([0]), dtype='int32'),
                                     tf.shape(obs))
            self.marg_ind = K.variable(value=numpy.array([MARG_IND], dtype=int),
                                       dtype=input.dtype)
            eq_test = K.transpose(K.equal(obs,  self.marg_ind))
            eq_test_int = K.equal(obs,  self.marg_ind)
            # print('obs', tf.rank(obs), tf.shape(obs))
            # print('eq', tf.rank(eq_test), tf.shape(eq_test))
            # print('zeros', tf.rank(self.zeros), tf.shape(self.zeros))
            # ll = K.transpose(K.gather(K.transpose(self.log_probs), obs))
            obs_valid = tf.select(eq_test_int, self.zeros_int, obs)
            ll = K.gather(K.transpose(self.log_probs), obs_valid)
            # print('log', tf.rank(ll), tf.shape(ll))
            self.log_vals = tf.select(eq_test, self.zeros, ll)
            # self.log_vals = K.reshape(self.log_vals, (K.shape(self.log_vals)[1], -1))
        else:
            self.log_vals = K.switch(K.equal(obs,  MARG_IND), 0.0,  self.log_probs[:, obs])
            self.log_vals = K.reshape(self.log_vals, (self.log_vals.shape[1], -1))

    def node_type_str(self):
        return DISCRETE_VAR_NODE_SYM

    def node_short_str(self):
        freqs_str = " ".join(self._var_freqs)
        return "{type} {id}" +\
            " <{var}>" +\
            " {freqs}".format(type=self.node_type_str(),
                              id=self.id,
                              vars=self.var,
                              freqs=freqs_str)

    def __repr__(self):
        base = Node.__repr__(self)

        return ("""Categorical Smoothed Node {line1}
            var: {var} val: {val} [{ff}] [{ll}]""".
                format(line1=base,
                       var=self.var,
                       val=len(self._var_freqs),
                       ff=[freq for freq in self._var_freqs],
                       ll=[ll for ll in self._var_probs]))

    def var_values(self):
        """
        WRITEME
        """
        return len(self._var_freqs)


class ConstantNode(Node):

    """
    A node emitting always a constant signal
    """

    def __init__(self, scope, const_value=0.0):
        """
        WRITEME
        """
        Node.__init__(self, frozenset(scope))
        self.log_val = const_value
        self.log_der = LOG_ZERO

    def n_children(self):
        return 0

    def node_type_str(self):
        return CONSTANT_NODE_SYM

    def eval(self, obs):
        """
        EMPTY
        """
        return

    def mpe_eval(self, obs):
        """
        EMPTY
        """
        return

    def node_short_str(self):

        return "{type} {id} <{var}> {val}".format(type=self.node_type_str(),
                                                  id=self.id,
                                                  var=self.scope,
                                                  val=self.log_val)

    def __repr__(self):
        base = Node.__repr__(self)

        return """Constant Node {line1} var: {var} val: {val}""".format(line1=base,
                                                                        var=self.scope,
                                                                        val=self.log_val)


def sample_from_leaf(leaf, instances_vec, instance_id, rand_gen=None, feature_values=None):
    """
    Samples an instance according to the leaf distribution and store
    those values into an array
    """
    #
    # discriminating on the leaf type:
    if isinstance(leaf, CategoricalSmoothedNode):
        if rand_gen is None:
            rand_gen = numpy.random.RandomState(RND_SEED)
        # print(leaf._var_probs)
        sampled_value = rand_gen.choice(leaf.var_val, p=numpy.exp(leaf._var_probs))
        instances_vec[instance_id, leaf.var] = sampled_value
    elif isinstance(leaf, CategoricalIndicatorNode):
        assert feature_values is not None

        vars_id = numpy.sum(feature_values[:leaf.var])
        #
        # and setting all others to 0
        instances_vec[instance_id, vars_id:vars_id + feature_values[leaf.var]] = 0
        #
        # and the leaf value to 1
        instances_vec[instance_id, vars_id + leaf.var_val] = 1

    elif isinstance(leaf, CLTreeNode):
        raise ValueError('CLT sampling not implemented yet')
    else:
        raise ValueError('Unrecognized leaf type')

    return instances_vec


def mpe_states_from_leaf(node, only_first_max=False):
    """
    Getting the mpe assignment for a leaf node. If it is an indicator node
    it returns its value, otherwise the mode of the univariate distribution it represents.
    TODO: extending it to multivariate nodes
    """

    mpe_state_dict = defaultdict(list)

    #
    # discriminating on the leaf type:
    if isinstance(node, CategoricalSmoothedNode):
        max_val = numpy.max(node._var_probs)
        for i, val in enumerate(node._var_probs):
            if numpy.isclose(max_val, val):
                mpe_state_dict[node.var].append(i)
                if only_first_max:
                    break

    elif isinstance(node, CategoricalIndicatorNode):
        mpe_state_dict[node.var].append(node.var_val)

    elif isinstance(node, CLTreeNode):
        raise ValueError('CLT sampling not implemented yet')
    else:
        raise ValueError('Unrecognized leaf type')

    # print(mpe_state_dict)
    return mpe_state_dict
