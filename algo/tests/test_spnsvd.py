from algo.spnsvd import SpnSVD
from algo.spnsvd import select_max_norm_col
from algo.spnsvd import extract_rank_one_submatrix
from algo.spnsvd import degenerate_submatrix

import numpy
from numpy.testing import assert_array_almost_equal

import logging

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def test_select_matrix_norm_col():
    #
    # generate a random binary matrix
    perc = 0.02
    n_rows = 1000
    n_cols = 100
    rand_matrix = numpy.random.binomial(1, perc, (n_rows, n_cols))
    #
    # setting one random column to all ones
    rand_col = numpy.random.choice(numpy.arange(n_cols))
    rand_matrix[:, rand_col] = 1

    max_j = select_max_norm_col(rand_matrix)

    assert max_j == rand_col


toy_matrix = numpy.array([[40, 40, 40, 40],
                          [40, 40, 40, 40],
                          [40, 18, 18, 40],
                          [20, 18, 18, 20],
                          [20, 20, 20, 20],
                          [20, 20, 20, 20]], dtype=int)


def test_example_rank_one_matrix_extraction():
    #
    # testing rank extraction on the toy matrix
    gammas = numpy.logspace(0, 1, num=4)
    gammas += 0.01  # to ensure > 1

    r = numpy.arange(toy_matrix.shape[0])
    c = numpy.arange(toy_matrix.shape[1])

    max_iters = 1000
    epsilon = 1e-5

    for gamma in gammas:

        m, n = extract_rank_one_submatrix(toy_matrix,
                                          gamma,
                                          max_iters,
                                          epsilon)

        print('gamma: {0}:\n\tm: {1}\n\tn: {2}'.format(gamma,
                                                       m,
                                                       n))
        if degenerate_submatrix(toy_matrix, m, n):
            print('degenerate solution')
            assert_array_almost_equal(m, r)
            assert_array_almost_equal(n, c)
        else:
            assert not numpy.array_equal(m, r) or not numpy.array_equal(n, c)


def test_toy_example_spnsvd():
    gammas = numpy.logspace(0, 1, num=4)
    gammas += 0.01  # to ensure > 1

    alpha = 0.1
    max_iters = 1000
    epsilon = 1e-5

    for gamma in gammas:
        #
        # init learner
        print('\n---- gamma:{0} ----\n'.format(gamma))
        learner = SpnSVD(gamma,
                         alpha=alpha,
                         r1_n_iters=max_iters,
                         epsilon=epsilon)
        #
        # fitting structure on toy example
        feature_sizes = numpy.array([40 for i in range(toy_matrix.shape[1])],
                                    dtype=int)
        spn = learner.fit_structure(toy_matrix, feature_sizes)

        print(spn)


def test_spnsvd_eval_nltcs():

    import dataset

    logging.basicConfig(level=logging.INFO)
    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    train_feature_vals = numpy.array([2 for i in range(train.shape[1])])
    print('Loaded dataset', dataset_name)

    gamma = 2.01
    alpha = 0.1

    cltl = True
    #
    # initing the algo
    learner = SpnSVD(gamma=gamma,
                     alpha=alpha,
                     cltree_leaves=cltl)

    learn_start_t = perf_counter()
    #
    # start learning
    spn = learner.fit_structure(train,
                                train_feature_vals)
    learn_end_t = perf_counter()
    print('Network learned in', (learn_end_t - learn_start_t), 'secs')

    # print(spn)

    #
    # now checking performances

    # infer_start_t = perf_counter()
    # train_ll = 0.0
    # print('Starting inference')
    # for instance in train:
    #     (pred_ll, ) = spn.eval(instance)
    #     train_ll += pred_ll
    # train_avg_ll = train_ll / train.shape[0]
    # infer_end_t = perf_counter()
    # # n avg ll -6.0180987340354 done in 43.947853350000514 secs
    # print('train avg ll', train_avg_ll, 'done in',
    #       infer_end_t - infer_start_t, 'secs')

    infer_start_t = perf_counter()
    test_ll = 0.0
    print('Starting inference')
    for instance in test:
        (pred_ll, ) = spn.eval(instance)
        test_ll += pred_ll
    test_avg_ll = test_ll / test.shape[0]
    infer_end_t = perf_counter()
    # n avg ll -6.0180987340354 done in 43.947853350000514 secs
    print('test avg ll', test_avg_ll, 'done in',
          infer_end_t - infer_start_t, 'secs')


#
# to profile inference
if __name__ == '__main__':
    test_spnsvd_eval_nltcs()
