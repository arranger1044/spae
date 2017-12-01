import numpy
from numpy.testing import assert_array_almost_equal

import theano

from spn.theanok.layers import SumLayer, ProductLayer
from ..layers import SumLayer_logspace
from ..layers import ProductLayer_logspace
from ..layers import MaxLayer_logspace

from spn import LOG_ZERO


def test_theano_sum_layer():
    input_vec = numpy.array([[1., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 1., 0., 1.],
                             [1., 1., 1., 1., 1., 1.]],
                            dtype=theano.config.floatX)

    W = numpy.array([[0.6, 0.4, 0., 0., 0., 0.],
                     [0.3, 0.7, 0., 0., 0., 0.],
                     [0., 0., 0.1, 0.9, 0., 0.],
                     [0., 0., 0.7, 0.3, 0., 0.],
                     [0., 0., 0., 0., 0.5, 0.5],
                     [0., 0., 0., 0., 0.2, 0.8]],
                    dtype=theano.config.floatX).T

    layer = SumLayer(input_dim=6,
                     output_dim=6,
                     weights=W)

    layer.build()
    input = layer.get_input()
    output = layer.get_output()

    eval_func = theano.function([input], output)

    res = eval_func(input_vec)
    # print(output.shape.eval())
    print(res)

# [[ -5.10825574e-01  -1.20397282e+00  -2.30258512e+00  -3.56674969e-01
#    -6.93147182e-01  -1.60943794e+00]
#  [ -1.00000000e+03  -1.00000000e+03  -1.00000000e+03  -1.00000000e+03
#    -1.00000000e+03  -1.00000000e+03]
#  [ -5.10825574e-01  -1.20397282e+00  -1.05360545e-01  -1.20397282e+00
#    -6.93147182e-01  -2.23143533e-01]
#  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#     0.00000000e+00   0.00000000e+00]]


def test_theano_prod_layer():

    input_vec = numpy.array([[1., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 1., 0., 1.],
                             [1., 1., 1., 1., 1., 1.]],
                            dtype=theano.config.floatX)

    W = numpy.array([[1., 0., 1., 0., 1., 0.],
                     [0., 1., 0., 1., 0., 1.],
                     [1., 0., 1., 0., 0., 1.],
                     [0., 1., 0., 1., 1., 0.]]).T

    layer = ProductLayer(input_dim=6,
                         output_dim=4,
                         weights=W)

    layer.build()
    input = layer.get_input()
    output = layer.get_output()

    eval_func = theano.function([input], output)

    res = eval_func(input_vec)
    # print(output.shape.eval())
    print(res)

    # [[ 20.08553696   1.           7.38905621   2.71828175]
    #  [  1.           1.           1.           1.        ]
    # [  2.71828175   7.38905621   7.38905621   2.71828175]
    # [ 20.08553696  20.08553696  20.08553696  20.08553696]]


def test_theano_sum_layer_log():

    import pickle

    data = numpy.array([[0, 1, 0, 1, 1, 0],
                        [1, 0, 0, 1, 0, 1],
                        [0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0]]).astype(numpy.float32)

    log_data = numpy.clip(numpy.log(data), LOG_ZERO, 0)
    W_1 = numpy.array([[.1, .4, .0, .0, .0, .0],
                       [.9, .6, .0, .0, .0, .0],
                       [.0, .0, .3, .6, .0, .0],
                       [.0, .0, .7, .4, .0, .0],
                       [.0, .0, .0, .0, .5, .2],
                       [.0, .0, .0, .0, .5, .8]])

    layer = SumLayer_logspace(input_dim=6,
                              output_dim=6,
                              weights=W_1)

    layer.build()
    input = layer.get_input()
    output = layer.get_output()

    eval_func = theano.function([input], output)

    log_res = eval_func(log_data)
    res = numpy.exp(log_res)
    # print(output.shape.eval())

    expected_res = numpy.dot(data, W_1)
    expected_log_res = numpy.log(expected_res)

    print(res)
    print(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)

    #
    # now trying to compile it
    layer.compile()

    log_res = layer.evaluate(log_data)
    res = numpy.exp(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)

    model_path = 'test.theanok.sum'
    with open(model_path, 'wb') as f:
        pickle.dump(layer, f)

    #
    # deserialization
    with open(model_path, 'rb') as f:
        layer = pickle.load(f)

    log_res = layer.evaluate(log_data)
    res = numpy.exp(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)


def test_theano_prod_layer_log():

    import pickle
    data = numpy.array([[0.9,  0.6,  0.7,  0.4,  0.5,  0.2],
                        [0.1,  0.4,  0.7,  0.4,  0.5,  0.8],
                        [0.9,  0.6,  0.3,  0.6,  0.5,  0.2],
                        [0.9,  0.6,  0.3,  0.6,  0.5,  0.8],
                        [0.1,  0.4,  0.3,  0.6,  0.5,  0.2]]).astype(numpy.float32)
    log_data = numpy.clip(numpy.log(data), LOG_ZERO, 0)

    W_2 = numpy.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]]).astype(numpy.float32)

    layer = ProductLayer_logspace(input_dim=6,
                                  output_dim=4,
                                  weights=W_2)

    layer.build()
    input = layer.get_input()
    output = layer.get_output()

    expected_res = W_2 * data[:, :, numpy.newaxis]
    expected_res[expected_res == 0] = 1
    expected_res = numpy.prod(expected_res, axis=1)
    expected_log_res = numpy.log(expected_res)

    print(expected_res)

    eval_func = theano.function([input], output)

    log_res = eval_func(log_data)
    res = numpy.exp(log_res)
    # print(output.shape.eval())
    print(res)
    print(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)

    #
    # now trying to compile it
    layer.compile()

    log_res = layer.evaluate(log_data)
    res = numpy.exp(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)

    model_path = 'test.theanok.prodlayer'
    with open(model_path, 'wb') as f:
        pickle.dump(layer, f)

    #
    # deserialization
    with open(model_path, 'rb') as f:
        layer = pickle.load(f)

    log_res = layer.evaluate(log_data)
    res = numpy.exp(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)


def test_theano_max_layer_log_I():

    data = numpy.array([[0, 1, 0, 1, 1, 0],
                        [1, 0, 0, 1, 0, 1],
                        [0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0]]).astype(numpy.float32)

    log_data = numpy.clip(numpy.log(data), LOG_ZERO, 0)
    W_1 = numpy.array([[.1, .4, .0, .0, .0, .0],
                       [.9, .6, .0, .0, .0, .0],
                       [.0, .0, .3, .6, .0, .0],
                       [.0, .0, .7, .4, .0, .0],
                       [.0, .0, .0, .0, .5, .2],
                       [.0, .0, .0, .0, .5, .8]])

    layer = MaxLayer_logspace(input_dim=6,
                              output_dim=6,
                              weights=W_1,
                              batch_size=data.shape[0])

    layer.build()
    input = layer.get_input()
    output = layer.get_output()

    eval_func = theano.function([input], output)

    log_res = eval_func(log_data)
    res = numpy.exp(log_res)
    # print(output.shape.eval())

    expected_res = W_1 * data[:, :, numpy.newaxis]
    print('Expected res', expected_res)
    expected_res = numpy.max(expected_res, axis=1)
    expected_log_res = numpy.log(expected_res)

    print(res)
    print(log_res)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)

    #
    # now trying to compile it
    layer.compile()

    log_res, M = layer.evaluate(log_data)
    res = numpy.exp(log_res)

    print(M)

    assert_array_almost_equal(expected_res, res)
    assert_array_almost_equal(expected_log_res, log_res)
