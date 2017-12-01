import numpy
from numpy.testing import assert_array_almost_equal

import theano

from ..spn import SequentialSpn
from ..spn import BlockLayeredSpn
from ..layers import SumLayer, ProductLayer
from ..layers import SumLayer_logspace
from ..layers import ProductLayer_logspace
from ..layers import MaxLayer_logspace

from spn import LOG_ZERO


def test_theano_spn_build():
    #
    # initial weights
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
                     [0., 0., 0., 0., 0.2, 0.8]]).T

    W_1 = numpy.array([[1., 0., 1., 0., 1., 0.],
                       [0., 1., 0., 1., 0., 1.],
                       [1., 0., 1., 0., 0., 1.],
                       [0., 1., 0., 1., 1., 0.]]).T
    #
    # creating an architecture
    model = BlockLayeredSpn()

    sum_layer = SumLayer(output_dim=6,
                         input_dim=6,
                         weights=W)

    model.add_input_layer(sum_layer)
    model.add(sum_layer)

    model.add(ProductLayer(output_dim=4,
                           input_dim=6,
                           weights=W_1),
              [sum_layer])

    input = model.get_input()
    output = model.get_output()

    f = theano.function([input], output)

    res = f(input_vec)
    print(res)

    # First (sum) layer
    # [[ -5.10825624e-01  -1.00000000e+03  -5.10825624e-01   0.00000000e+00]
    #  [ -1.20397280e+00  -1.00000000e+03  -1.20397280e+00   0.00000000e+00]
    # [ -2.30258509e+00  -1.00000000e+03  -1.05360516e-01   0.00000000e+00]
    # [ -3.56674944e-01  -1.00000000e+03  -1.20397280e+00   0.00000000e+00]
    # [ -6.93147181e-01  -1.00000000e+03  -6.93147181e-01   0.00000000e+00]
    # [ -1.60943791e+00  -1.00000000e+03  -2.23143551e-01   0.00000000e+00]]
    # Second (prod) layer
    # [[ 0.03   0.     0.27   1.   ]
    #  [ 0.042  0.     0.072  1.   ]
    # [ 0.012  0.     0.432  1.   ]
    # [ 0.105  0.     0.045  1.   ]]
    # Third (sum) layer
    # [   -2.97887516 -1000.            -1.83979492     0.        ]
    # [ 0.05085  0.       0.15885  1.     ]


def test_spn_compile():

    data = numpy.array([[0, 1, 0, 1, 1, 0],
                        [1, 0, 0, 1, 0, 1],
                        [0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0]]).astype(numpy.float32)

    W_1 = numpy.array([[.1, .4, .0, .0, .0, .0],
                       [.9, .6, .0, .0, .0, .0],
                       [.0, .0, .3, .6, .0, .0],
                       [.0, .0, .7, .4, .0, .0],
                       [.0, .0, .0, .0, .5, .2],
                       [.0, .0, .0, .0, .5, .8]]).astype(numpy.float32)

    W_2 = numpy.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]]).astype(numpy.float32)

    W_3 = numpy.array([[0.1],
                       [0.2],
                       [0.25],
                       [0.45]]).astype(numpy.float32)

    model = BlockLayeredSpn()

    sum_layer = SumLayer(output_dim=6,
                         input_dim=6,
                         weights=W_1)

    # model.add_input_layer(sum_layer)
    model.add(sum_layer)

    prod_layer = ProductLayer(output_dim=4,
                              input_dim=6,
                              weights=W_2)
    model.add(prod_layer,
              [sum_layer])

    root_layer = SumLayer(output_dim=1,
                          input_dim=4,
                          weights=W_3)
    model.add(root_layer,
              [prod_layer])

    model.compile()

    log_res = model.evaluate(data)
    print(log_res)
    res = numpy.exp(log_res)
    print(res)

    #
    # expected res (sum layer 1)
    expected_res = numpy.dot(data, W_1)
    expected_log_res = numpy.log(expected_res)
    #
    # (prod layer)
    expected_res = W_2 * expected_res[:, :, numpy.newaxis]
    expected_res[expected_res == 0] = 1
    expected_res = numpy.prod(expected_res, axis=1)
    expected_log_res = numpy.log(expected_res)
    #
    # (root layer)
    expected_res = numpy.dot(expected_res, W_3)
    expected_log_res = numpy.log(expected_res)

    print('Expected res', expected_res)
    print('Expected log res', expected_log_res)

    assert_array_almost_equal(res, expected_res)
    assert_array_almost_equal(log_res, expected_log_res)


def test_spn_compile_logspace():

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
                       [.0, .0, .0, .0, .5, .8]]).astype(numpy.float32)

    W_2 = numpy.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]]).astype(numpy.float32)

    W_3 = numpy.array([[0.1],
                       [0.2],
                       [0.25],
                       [0.45]]).astype(numpy.float32)

    model = BlockLayeredSpn()

    sum_layer = SumLayer_logspace(output_dim=6,
                                  input_dim=6,
                                  weights=W_1)

    # model_path = 'test.theanok'
    # with open(model_path, 'wb') as f:
    #     print('dumping 1')
    #     pickle.dump(sum_layer, f)

    # #
    # # deserialization
    # with open(model_path, 'rb') as f:
    #     print('loading 1')
    #     a_model = pickle.load(f)

    # model.add_input_layer(sum_layer)
    model.add(sum_layer)

    # model_path = 'test.theanok'
    # with open(model_path, 'wb') as f:
    #     print('dumping 2')
    #     pickle.dump(sum_layer, f)

    # #
    # # deserialization
    # with open(model_path, 'rb') as f:
    #     print('loading 2')
    #     a_model = pickle.load(f)

    prod_layer = ProductLayer_logspace(output_dim=4,
                                       input_dim=6,
                                       weights=W_2)

    # model_path = 'test.theanok'

    # with open(model_path, 'wb') as f:
    #     print('dumping s1')
    #     pickle.dump(prod_layer, f)

    # #
    # # deserialization
    # with open(model_path, 'rb') as f:
    #     print('loading s1')
    #     a_model = pickle.load(f)

    model.add(prod_layer,
              [sum_layer])

    # model_path = 'test.theanok'
    # with open(model_path, 'wb') as f:
    #     print('dumping 3')
    #     pickle.dump(sum_layer, f)

    # #
    # # deserialization
    # with open(model_path, 'rb') as f:
    #     print('loading 3')
    #     a_model = pickle.load(f)

    root_layer = SumLayer_logspace(output_dim=1,
                                   input_dim=4,
                                   weights=W_3)
    model.add(root_layer,
              [prod_layer])

    # for layer in model.layers:
    #     layer.output_layers = set()

    #
    # serialization
    # model_path = 'test.theanok'
    # with open(model_path, 'wb') as f:
    #     print('dumping f')
    #     pickle.dump(model, f)

    # #
    # # deserialization
    # with open(model_path, 'rb') as f:
    #     print('loading f')
    #     model = pickle.load(f)

    model.compile()

    log_res = model.evaluate(log_data)
    print(log_res)
    res = numpy.exp(log_res)
    print(res)

    #
    # expected res (sum layer 1)
    expected_res = numpy.dot(data, W_1)
    expected_log_res = numpy.log(expected_res)
    #
    # (prod layer)
    expected_res = W_2 * expected_res[:, :, numpy.newaxis]
    expected_res[expected_res == 0] = 1
    expected_res = numpy.prod(expected_res, axis=1)
    expected_log_res = numpy.log(expected_res)
    #
    # (root layer)
    expected_res = numpy.dot(expected_res, W_3)
    expected_log_res = numpy.log(expected_res)

    print('Expected res', expected_res)
    print('Expected log res', expected_log_res)

    assert_array_almost_equal(res, expected_res)
    assert_array_almost_equal(log_res, expected_log_res)

    #
    # serialization
    model_path = 'test.theanok'
    with open(model_path, 'wb') as f:
        model.dump(f)

    #
    # deserialization
    with open(model_path, 'rb') as f:
        model = BlockLayeredSpn.load(f)

    log_res = model.evaluate(log_data)
    print(log_res)
    res = numpy.exp(log_res)
    print(res)


def test_mpn_compile_logspace():

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
                       [.0, .0, .0, .0, .5, .8]]).astype(numpy.float32)

    W_2 = numpy.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]]).astype(numpy.float32)

    W_3 = numpy.array([[0.1],
                       [0.2],
                       [0.25],
                       [0.45]]).astype(numpy.float32)

    model = BlockLayeredSpn()

    max_layer = MaxLayer_logspace(output_dim=6,
                                  input_dim=6,
                                  weights=W_1,
                                  batch_size=data.shape[0])

    # model.add_input_layer(max_layer)
    model.add(max_layer)

    prod_layer = ProductLayer_logspace(output_dim=4,
                                       input_dim=6,
                                       weights=W_2)
    model.add(prod_layer,
              [max_layer])

    root_layer = MaxLayer_logspace(output_dim=1,
                                   input_dim=4,
                                   weights=W_3,
                                   batch_size=data.shape[0])
    model.add(root_layer,
              [prod_layer])

    model.compile()

    log_res = model.evaluate(log_data)
    print(log_res)
    res = numpy.exp(log_res)
    print(res)

    #
    # expected res (max layer 1) this is the same as if
    # the layer were a sum layer
    expected_res = numpy.dot(data, W_1)
    expected_log_res = numpy.log(expected_res)
    #
    # (prod layer) unchanged
    expected_res = W_2 * expected_res[:, :, numpy.newaxis]
    expected_res[expected_res == 0] = 1
    expected_res = numpy.prod(expected_res, axis=1)
    expected_log_res = numpy.log(expected_res)
    #
    # (root layer)
    expected_res = expected_res * W_3.T
    expected_res = numpy.max(expected_res, axis=1, keepdims=True)

    expected_log_res = numpy.log(expected_res)

    print('Expected res', expected_res)
    print('Expected log res', expected_log_res)

    assert_array_almost_equal(res, expected_res)
    assert_array_almost_equal(log_res, expected_log_res)
