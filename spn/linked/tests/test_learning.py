from spn.linked.spn import Spn

from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode

import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

import dataset
from spn.factory import SpnFactory

syn_train_data = numpy.array([[0., 1., 1., 0.],
                              [0., 1., 0., 0.],
                              [1., 0., 0., 1.],
                              [0., 1., 1., 1.],
                              [1., 1., 0., 0.],
                              [1., 1., 0., 0.],
                              [1., 1., 1., 0.],
                              [0., 1., 0., 1.],
                              [1., 0., 1., 1.],
                              [1., 0., 0., 1.]])

syn_val_data = numpy.array([[1., 1., 1., 0.],
                            [1., 1., 1., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 1.],
                            [1., 0., 0., 1.]])


def test_mini_spn_fit_em():
    vars = numpy.array([2, 2, 2, 2])
    input_layer = CategoricalIndicatorLayer(vars=vars)

    print(input_layer)
    ind1 = input_layer._nodes[0]
    ind2 = input_layer._nodes[1]
    ind3 = input_layer._nodes[2]
    ind4 = input_layer._nodes[3]
    ind5 = input_layer._nodes[4]
    ind6 = input_layer._nodes[5]
    ind7 = input_layer._nodes[6]
    ind8 = input_layer._nodes[7]

    # creating a sum layer of 4 nodes
    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()
    sum4 = SumNode()

    sum1.add_child(ind1, 0.6)
    sum1.add_child(ind2, 0.4)
    sum2.add_child(ind3, 0.5)
    sum2.add_child(ind4, 0.5)
    sum3.add_child(ind5, 0.7)
    sum3.add_child(ind6, 0.3)
    sum4.add_child(ind7, 0.4)
    sum4.add_child(ind8, 0.6)

    sum_layer = SumLayer(nodes=[sum1, sum2,
                                sum3, sum4])

    # and a top layer of 3 products
    prod1 = ProductNode()
    prod2 = ProductNode()
    prod3 = ProductNode()

    prod1.add_child(sum1)
    prod1.add_child(sum2)
    prod2.add_child(sum2)
    prod2.add_child(sum3)
    prod3.add_child(sum3)
    prod3.add_child(sum4)

    prod_layer = ProductLayer(nodes=[prod1, prod2, prod3])

    # root layer
    root = SumNode()

    root.add_child(prod1, 0.4)
    root.add_child(prod2, 0.25)
    root.add_child(prod3, 0.35)

    root_layer = SumLayer(nodes=[root])

    spn = Spn(input_layer=input_layer,
              layers=[sum_layer, prod_layer, root_layer])

    print(spn)

    # training on obs
    spn.fit_em(train=syn_train_data,
               valid=syn_val_data,
               test=None,
               hard=True)


def atest_nltcs_em_fit():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Build kernel density estimation')
    spn = SpnFactory.linked_kernel_density_estimation(n_instances,
                                                      features)
    print('EM training')

    spn.fit_em(train, valid, test,
               hard=True,
               epochs=2)


def profiling():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Build kernel density estimation')
    spn = SpnFactory.linked_kernel_density_estimation(n_instances,
                                                      features)
    # print(spn)
    print('EM training')
    spn.fit_em(train, valid, test,
               hard=True,
               epochs=2)


def test_em():
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    with PyCallGraph(output=GraphvizOutput()):
        profiling()


def test_sgd():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Build kernel density estimation')
    spn = SpnFactory.linked_kernel_density_estimation(
        n_instances,
        features)

    print('Created SPN with\n' + spn.stats())

    print('Starting SGD')
    spn.fit_sgd(train, valid, test,
                learning_rate=0.1,
                n_epochs=20,
                batch_size=1,
                hard=False)

import random


def test_random_spn_sgd():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    n_layers = 1
    n_max_children = 2000
    n_scope_children = 2000
    max_scope_split = -1
    merge_prob = 0.5
    seed = 1337
    rand_gen = random.Random(seed)

    print('Build random spn')
    spn = SpnFactory.linked_random_spn_top_down(features,
                                                n_layers,
                                                n_max_children,
                                                n_scope_children,
                                                max_scope_split,
                                                merge_prob,
                                                rand_gen=rand_gen)

    assert spn.is_valid()
    print('Stats\n')
    print(spn.stats())

    np_rand_gen = numpy.random.RandomState(seed)

    spn.fit_sgd(train, valid, test,
                learning_rate=0.2,
                n_epochs=10,
                batch_size=1,
                grad_method=1,
                validation_frequency=100,
                rand_gen=np_rand_gen,
                hard=False)


def test_random_spn_em():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    n_layers = 2
    n_max_children = 4
    n_scope_children = 5
    max_scope_split = 3
    merge_prob = 0.5
    print('Build random spn')
    spn = SpnFactory.linked_random_spn_top_down(features,
                                                n_layers,
                                                n_max_children,
                                                n_scope_children,
                                                max_scope_split,
                                                merge_prob)

    assert spn.is_valid()
    print('Stats\n')
    print(spn.stats())

    spn.fit_em(train, valid, test,
               hard=False,
               n_epochs=10)


from spn.linked.learning import SpectralStructureLearner
from spn.linked.learning import CoClusteringStructureLearner


def test_spectral_structure_learner_sim():
    data = numpy.array([[1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 0, 1]], dtype='int8')
    learner = SpectralStructureLearner(sigma=2.0)
    sim_gauss_1 = learner.compute_similarity_matrix(data,
                                                    metric='gaussian')
    sim_gauss_2 = learner.compute_similarity_matrix_pair(data,
                                                         metric=learner.gaussian_kernel)
    print(sim_gauss_1)
    print(sim_gauss_2)
    sim_gtest_2 = learner.compute_similarity_matrix_pair(data,
                                                         metric=learner.g_test)
    print(sim_gtest_2)
    sim_gtest_1 = learner.compute_similarity_matrix(data,
                                                    metric='gtest')

    print(sim_gtest_1)


def test_spectral_structure_learner_diag_sum():
    # create a similarity matrix
    W = numpy.array([[1.0, 0.2, 0.3],
                     [0.2, 1.0, 0.5],
                     [0.3, 0.5, 1.0]])
    learner = SpectralStructureLearner()
    D = learner.diag_sum(W)
    print('Diagonal sum matrix:', D)
    assert_almost_equal(D[0, 0], 1.5)
    assert_almost_equal(D[1, 1], 1.7)
    assert_almost_equal(D[2, 2], 1.8)
    # must check for all other cells to be 0


def test_spectral_structure_learner_cut_val():
    # create a similarity matrix
    W = numpy.array([[1.0, 0.2, 0.3],
                     [0.2, 1.0, 0.5],
                     [0.3, 0.5, 1.0]])
    learner = SpectralStructureLearner()

    D = learner.diag_sum(W)
    print('Diagonal sum matrix:', D)

    # create a clustering [0,2][1]
    f = numpy.array([1, -1, 1])

    vol_f = learner.vol(W, f, 1)
    vol_minus_f = learner.vol(W, f, -1)
    print('vol_f', vol_f)
    print('vol_m_f', vol_minus_f)

    assert_almost_equal(D[0, 0] + D[2, 2], vol_f)
    assert_almost_equal(D[1, 1], vol_minus_f)

    f_clu = learner.f_clu(W, f)
    print('f clu', f_clu)

    cut_f = learner.cut_val_f(W, f_clu)
    print('cut val:', cut_f)
    cut_w = learner.cut_val_w(W, f)
    print('cut val w:', cut_w)

    ncut_f = learner.ncut_val(W, f_clu)
    print('ncut val:', ncut_f)
    ncut_hand = ((W[0, 1] + W[2, 1]) / vol_f +
                 (W[0, 1] + W[2, 1]) / vol_minus_f)
    print('ncut by hand', ncut_hand)

    ncut_s = learner.ncut(W, f)
    print('ncut simple', ncut_s)
    assert_almost_equal(ncut_hand, ncut_s)
    assert_almost_equal(ncut_hand, ncut_f)
    # assert_almost_equal(W[0, 1] + W[2, 1], cut_f)

    clustering = [[0, 2], [1]]
    f_s = learner.f_assignment_from_clusters(W, clustering)
    print('f from clustering', f_s)
    assert_array_almost_equal(f_clu, f_s)


def test_spectral_structure_learner_labels():
    labels = [0, 1, 1, 0, 1, 2, 1, 0, 2, 2]
    ids = [12, 11, 1, 0, 56, 107, 12, 9, 70, 8]
    learner = SpectralStructureLearner()
    clustering = learner.from_labels_to_clustering(labels, ids)
    print(clustering)
    assert clustering == [[12, 0, 9], [11, 1, 56, 12], [107, 70, 8]]

#
# found a very old bug:
# synth_data = numpy.random.binomial(100, 0.5, (200, 15))
synth_data = numpy.random.binomial(1, 0.5, (200, 15))
synth_feats = numpy.zeros(15, dtype='int8')
synth_feats.fill(2)


def test_compare_spectral_performance():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    learner = SpectralStructureLearner()
    k = 5
    ids = [i for i in range(train.shape[1])]
    labels, clusters, valid = \
        learner.spectral_clustering(train.T, ids, k,
                                    affinity_metric='gtest',
                                    pair=True)


def test_spectral_clustering():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    learner = SpectralStructureLearner(sigma=4.0)
    k = 5
    ids = [i for i in range(train.shape[0])]
    labels, clusters, valid = learner.spectral_clustering(train, ids, k)
    print('labels:{0}\nclusters:{1}'.format(labels, clusters))


def test_spectral_cluster_learner():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    print('features', synth_feats)
    learner = SpectralStructureLearner()
    k = 5
    spn = learner.fit_structure(synth_data,
                                synth_feats,
                                k_row_clusters=k,
                                min_instances_slice=2,
                                pairwise=True)
    print(spn.stats())


def test_greedy_split_features():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    learner = SpectralStructureLearner()
    k = 2
    ids = [i for i in range(train.shape[1])]
    g_factor = 9
    seed = 1337
    rand_gen = numpy.random.RandomState(seed)
    data_slice = train[:100, :]
    # splitting on the features
    clustering = learner.greedy_split_features(data_slice.T,
                                               ids,
                                               g_factor,
                                               rand_gen)
    print(clustering)

    labels, clustering, valid = \
        learner.spectral_clustering(data_slice.T,
                                    ids,
                                    k,
                                    affinity_metric='gtest',
                                    validity_check=True,
                                    threshold=0.8,
                                    rand_gen=rand_gen)
    print(clustering)

# some constants for the next tests
clusters_test = [[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                 [2, 2, 2, 1, 1, 0, 2, 3, 2, 1, 3, 1, 2, 0, 3, 2],
                 [4, 5, 5, 3, 3, 0, 5, 6, 5, 3, 7, 2, 5, 1, 8, 4],
                 [6, 9, 9, 3, 4, 0, 9, 10, 8, 5, 11, 2, 8, 1, 12, 7],
                 [6, 10, 10, 3, 4, 0, 11, 12, 8, 5, 13, 2, 9, 1, 14, 7],
                 [6, 10, 11, 3, 4, 0, 12, 13, 8, 5, 14, 2, 9, 1, 15, 7]]
n_features = 16
feature_sizes = [i for i in range(n_features)]


def test_coc_read_hierarchy_from_file():

    # specify test file
    filename = 'spn/linked/tests/coc/co_cluster.col'
    # create learner
    learner = CoClusteringStructureLearner()
    cluster_ass = learner.read_hierarchy_from_file(filename)
    print(cluster_ass)

    assert clusters_test == cluster_ass


def test_coc_build_linked_hierarchy():
    # specify test file
    filename = 'spn/linked/tests/coc/co_cluster.col'
    # create learner
    learner = CoClusteringStructureLearner()
    cluster_ass = learner.read_hierarchy_from_file(filename)

    # building the linked representation
    linked_hier = learner.build_linked_hierarchy(cluster_ass)
    print(linked_hier)


def test_coc_build_spn_from_co_clusters():
    # specify test file
    col_filename = 'spn/linked/tests/coc/cc_test.col'
    row_filename = 'spn/linked/tests/coc/cc_test.row'
    data_filename = 'spn/linked/tests/coc/data_test.csv'

    # create learner
    learner = CoClusteringStructureLearner()

    # build from file
    cluster_ass_col = learner.read_hierarchy_from_file(col_filename)
    cluster_ass_row = learner.read_hierarchy_from_file(row_filename)
    data = dataset.csv_2_numpy(data_filename, path='')

    row_h = learner.build_linked_hierarchy(cluster_ass_row)
    col_h = learner.build_linked_hierarchy(cluster_ass_col)

    print(row_h)
    print(col_h)

    spn = learner.build_spn_from_co_clusters(row_h,
                                             col_h,
                                             data,
                                             feature_sizes,
                                             min_instances_slice=1,
                                             max_depth=10)
    print(spn)
    print(spn.stats())

import cProfile
import re

if __name__ == '__main__':
    # print('ndrangheta')
    # cProfile.run('profiling()')
    profiling()
