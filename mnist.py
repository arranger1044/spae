from visualize import array_2_mat
from visualize import plot_m_by_n_images

import numpy

import matplotlib
import matplotlib.pyplot as pyplot

import pickle

import os


RANDOM_SEED = 1337


def load_mnist_data_split_from_txt(data_path):

    data = numpy.loadtxt(data_path, delimiter=' ')

    x, y = data[:, :-1], data[:, -1].astype(numpy.int32)

    print('Loaded dataset:\n\tx: {}\ty: {}'.format(x.shape, y.shape))
    assert x.shape[0] == y.shape[0]
    assert y.ndim == 1
    assert x.shape[1] == 784

    return x, y


def load_mnist_from_txt(data_dir, split_names=['mnist_train.txt',
                                               'mnist_valid.txt',
                                               'mnist_test.txt']):
    split_paths = [os.path.join(data_dir, file_name)
                   for file_name in split_names]
    data_splits = [load_mnist_data_split_from_txt(path) for path in split_paths]
    return data_splits


def save_mnist_pickle(data_splits, output_path):
    with open(output_path, 'wb') as data_file:
        pickle.dump(data_splits, data_file)


def load_mnist_pickle(data_path):
    data_splits = None
    with open(data_path, 'rb') as data_file:
        data_splits = pickle.load(data_file)
    return data_splits


def plot_mnist_digits(image_arrays,
                      m, n,
                      fig_size=(12, 12),
                      invert=True,
                      cmap=matplotlib.cm.binary,
                      save_path=None,
                      pdf=False):

    image_matrixes = None
    if invert:
        image_matrixes = [array_2_mat(1 - img, 28, 28) for img in image_arrays]
    else:
        image_matrixes = [array_2_mat(img, 28, 28) for img in image_arrays]

    plot_m_by_n_images(image_matrixes, m, n, fig_size, cmap, save_path, pdf)


def binarize_image(image_array, rand_gen, dtype=numpy.int32):

    assert image_array.ndim == 1

    bin_image_array = numpy.zeros(image_array.shape, dtype=dtype)
    n_features = image_array.shape[0]
    for i in range(n_features):
        bin_image_array[i] = rand_gen.choice(2, p=[1 - image_array[i],
                                                   image_array[i]])
    return bin_image_array


def binarize_mnist_data_split(images, rand_gen, dtype=numpy.int32):
    n_images = len(images)
    bin_images = [binarize_image(images[i], rand_gen, dtype) for i in range(n_images)]
    return numpy.array(bin_images)


def binarize_mnist(data_splits, rand_gen=None, dtype=numpy.int32):
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RANDOM_SEED)

    binarized_splits = [(binarize_mnist_data_split(x, rand_gen, dtype), y) for x, y in data_splits]

    return binarized_splits
