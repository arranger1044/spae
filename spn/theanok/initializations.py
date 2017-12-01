import numpy as np

import theano
import theano.tensor as T


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))


def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)


def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))


def ndim_tensor(ndim):
    if ndim == 1:
        return T.vector()
    elif ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()


class Initialization(object):

    _init_dict = {'uniform': uniform,
                  }

    @classmethod
    def get(cls, init):
        if init in Initialization._init_dict:
            return Initialization._init_dict[init]
        else:
            raise ValueError('Init method not present', init)
