from spn import MARG_IND

from spn.utils import stats_format
from spn.utils import approx_scope_histo_quartiles

from spn.linked.spn import evaluate_on_dataset
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode

from spn.linked.representation import load_features_from_file
from spn.linked.representation import retrieve_all_nodes_mpe_instantiations
from spn.linked.representation import scope_stats
from spn.linked.representation import extract_features_all_marginals_spn
from spn.linked.representation import extract_features_all_marginals_ml
from spn.linked.representation import node_activations_for_instance
from spn.linked.representation import extract_features_marginalization_grid
from spn.linked.representation import extract_feature_marginalization_from_masks
from spn.linked.representation import instance_from_disjoint_feature_masks
from spn.linked.representation import get_nearest_neighbour
from spn.linked.representation import random_rectangular_feature_mask
from spn.linked.representation import extract_instances_groups

from visualize import array_2_mat
from visualize import plot_m_by_n_images, plot_m_by_n_by_p_by_q_images
from visualize import plot_m_by_n_images_list
from visualize import tiling_sizes
from visualize import binary_cmap, inv_binary_cmap
from visualize import ternary_cmap, inv_ternary_cmap
from visualize import quaternary_cmap, inv_quaternary_cmap
from visualize import quaternary_white_cmap, inv_quaternary_white_cmap
from visualize import scope_histogram, layer_scope_histogram, multiple_scope_histogram
from visualize import scope_maps, scope_map_layerwise
from visualize import plot_m_by_n_heatmaps
from visualize import plot_heatmap
from visualize import make_checkboard_bg
from visualize import make_double_checkboard_bg
# from visualize import visualize_node_activations_for_instance

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy

import random

import datetime

import os

import logging

import matplotlib
import matplotlib.pyplot as pyplot

from collections import defaultdict

import pickle
import gzip

PREDS_EXT = 'lls'

TRAIN_PREDS_EXT = 'train.{}'.format(PREDS_EXT)
VALID_PREDS_EXT = 'valid.{}'.format(PREDS_EXT)
TEST_PREDS_EXT = 'test.{}'.format(PREDS_EXT)

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)


PICKLE_SPLIT_EXT = 'pickle'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'
SAMPLE_IMGS_EXT = 'samples'
MPE_VIS_EXT = 'mpe'


def filter_sum_nodes(spn):
    return [node for node in spn.top_down_nodes() if isinstance(node, SumNode)]


def filter_product_nodes(spn):
    return [node for node in spn.top_down_nodes() if isinstance(node, ProductNode)]


def filter_leaf_nodes(spn):
    return [node for node in spn.top_down_nodes()
            if not isinstance(node, ProductNode) and not isinstance(node, SumNode)]


def filter_nodes_by_layer(spn, layer_id):
    return [node for i, layer in enumerate(spn.bottom_up_layers())
            for node in layer.nodes() if layer_id == i]


def filter_nodes_by_scope_length(spn, min_scope_len, max_scope_len):
    return [node for node in spn.top_down_nodes()
            if ((hasattr(node, 'var_scope') and
                 len(node.var_scope) >= min_scope_len and
                 len(node.var_scope) < max_scope_len)
                or
                (hasattr(node, 'var') and
                 len(node.var) >= min_scope_len and
                 len(node.var) < max_scope_len))]


def filter_nodes(spn, filter_str):

    nodes = None

    if filter_str == 'all':
        nodes = list(spn.top_down_nodes())

    elif filter_str == 'sum':
        nodes = filter_sum_nodes(spn)

    elif filter_str == 'prod':
        nodes = filter_product_nodes(spn)

    elif filter_str == 'leaves':
        nodes = filter_leaf_nodes(spn)

    elif 'layer' in filter_str:
        layer_id = int(filter_str.replace('layer', ''))
        nodes = filter_nodes_by_layer(spn, layer_id)

    elif 'scope' in filter_str:
        scope_ids = int(filter_str.replace('scope', ''))
        min_scope, max_scope = scope_ids.split(',')
        min_scope, max_scope = int(min_scope), int(max_scope)
        nodes = filter_nodes_by_scope_length(spn, min_scope, max_scope)

    return nodes


def retrieve_instance(splits, instance_id):

    cum_id = 0
    for split in splits:
        n_instances = len(split)
        cum_id += n_instances
        if instance_id < cum_id:
            return split[cum_id - n_instances + instance_id]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str,
                        help='Dataset name (es bmnist)')

    parser.add_argument('--model', type=str, nargs='+',
                        default=[],
                        help='Spn model file path')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--sample', type=int,
                        default=None,
                        help='Sampling N instances and visualize them')

    parser.add_argument('--nn', action='store_true',
                        help='Displaying the images nns in the train')

    parser.add_argument('--size', type=int, nargs='+',
                        default=(28, 28),
                        help='Image sample sizes (rows x cols) pixels')

    parser.add_argument('--fig-size', type=int, nargs='+',
                        default=(12, 8),
                        help='Figure size')

    parser.add_argument('--space', type=float, nargs='+',
                        default=(0.1, 0.1),
                        help='Space between tiles')

    parser.add_argument('--n-cols', type=int,
                        default=10,
                        help='Number of columns in image tiling')

    parser.add_argument('--n-cols-layer', type=int,
                        default=5,
                        help='Number of columns in layer mpe image tiling')

    parser.add_argument('--n-cols-scope', type=int,
                        default=5,
                        help='Number of columns in scope mpe image tiling')

    parser.add_argument('--tile-size', type=float, nargs='+',
                        default=(1.5, 1.5),
                        help='Size of a single image tile height')

    parser.add_argument('--scope-range', type=int, nargs='+',
                        default=None,
                        help='Scope range for mpe instantiations')

    parser.add_argument('--hid-groups', type=str, nargs='+',
                        default=None,
                        help='Data path and Number of instance clusters after mpe descents')

    parser.add_argument('--mask-sizes', type=int, nargs='+',
                        default=(10, 10, 10, 10),
                        help='Min max sizes for rect random mask')

    parser.add_argument('--max-n-images', type=int,
                        default=9,
                        help='Max N of images to visualize at once')

    parser.add_argument('--max-n-images-layers', type=int,
                        default=25,
                        help='Max N of mpe images to visualize at once')

    parser.add_argument('--max-n-images-scopes', type=int,
                        default=25,
                        help='Max N of mpe images to visualize at once')

    parser.add_argument('--dpi', type=int,
                        default=900,
                        help='Image dpi')

    parser.add_argument('--ylim', type=int,
                        default=10e5,
                        help='Max limit y axis')

    parser.add_argument('--xlim', type=int, nargs='+',
                        default=None,
                        help='Max limit x axis')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/repr/',
                        help='Output dir path')

    parser.add_argument('--suffix', type=str,
                        help='Dataset output suffix')

    parser.add_argument('--lines', type=str,
                        default=None,
                        help='Path to tsv containing line data')

    parser.add_argument('--scatter-emb', type=str,
                        default=None,
                        help='Path to tsv containing embedding size/accuracy data')

    parser.add_argument('--mpe', type=str, nargs='+',
                        default=None,
                        help='MPE node visualization type (sum|prod|layer|scope)')

    parser.add_argument('--invert', action='store_true',
                        help='Inverting colormaps')

    parser.add_argument('--scope', type=str, nargs='+',
                        default=None,
                        help='Showing scope length diagrams (hist|layer|map|lmap)')

    parser.add_argument('--activations', type=str, nargs='+',
                        default=None,
                        help='Visualize one instance activations'
                        'Params: instance-id [filters]')

    parser.add_argument('--marg-activations', type=str, nargs='+',
                        default=None,
                        help='Visualize one instance marginal activations'
                        'Params: instance-id [instance-id]*')

    parser.add_argument('--part-activations', action='store_true',
                        help='Visualize partition function activations')

    parser.add_argument('--sqrt-trans', action='store_true',
                        help='Transform activation values with sqrt function')

    parser.add_argument('--scope-image-map', action='store_true',
                        help='Visualize scope image map')

    parser.add_argument('--colorbar', action='store_true',
                        help='Visualize colorbars for heatmaps')

    parser.add_argument('--all-maxes', action='store_true',
                        help='Getting all max children in during MPE traversal')

    parser.add_argument('--baseline', type=float,
                        default=None,
                        help='LR baseline value for line plot')

    parser.add_argument('--no-show', action='store_true',
                        help='Non showing plots')

    parser.add_argument('--no-legend', action='store_true',
                        help='Non showing legend for scatterplot')

    parser.add_argument('--save', type=str,
                        default=None,
                        help='Saving format')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    logging.basicConfig(level=logging.INFO)

    #
    # parsing the args
    args = parser.parse_args()

    assert len(args.size) == 2
    n_rows, n_cols = args.size
    assert len(args.tile_size) == 2
    row_tile_size, col_tile_size = args.tile_size
    logging.info('Images size {}, tile size {}'.format(args.size,
                                                       args.tile_size))

    #
    # fixing a seed
    rand_gen = numpy.random.RandomState(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    #
    # setting verbosity level
    # if args.verbose == 1:
    #     logging.basicConfig(level=logging.INFO)
    # elif args.verbose == 2:
    #     logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    train = None
    valid = None
    test = None
    pickle_split_path = '{}.pickle'.format(args.dataset)
    print('Looking for {}'.format(pickle_split_path))
    picklez_split_path = '{}.pklz'.format(args.dataset)

    dataset_name = args.dataset.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name = dataset_name.replace('.pklz', '')
    dataset_name = dataset_name.replace('.pkl', '')
    dataset_name = dataset_name.replace('.pickle', '')

    if os.path.exists(pickle_split_path):
        logging.info('Loading from pickle {}'.format(pickle_split_path))
        with open(pickle_split_path, 'rb') as split_file:
            train, valid, test = pickle.load(split_file)
    elif os.path.exists(picklez_split_path):
        logging.info('Loading from pickle {}'.format(picklez_split_path))
        with gzip.open(picklez_split_path, 'rb') as split_file:
            train, valid, test = pickle.load(split_file)        
    else:
        logging.info('Loading datasets: %s', args.dataset)
        
        train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

    n_instances = train.shape[0]
    n_features = train.shape[1]
    n_test_instances = test.shape[0]
    freqs, feature_vals = dataset.data_2_freqs(train)

    logging.info('\ttrain: {}\n\tvalid: {}\n\ttest: {}'.format(train.shape,
                                                               valid.shape,
                                                               test.shape))

    #
    # loading models
    # assert len(args.model) > 0
    spns = []
    for model_path in args.model:
        logging.info('\nLoading spn model from: {}'.format(model_path))
        spn = None

        with open(model_path, 'rb') as model_file:
            load_start_t = perf_counter()
            spn = pickle.load(model_file)
            load_end_t = perf_counter()
            logging.info('done in {}'.format(load_end_t - load_start_t))

        spns.append(spn)

        logging.info('Spn stats:\n\tlayers\t{}\n\t'
                     'nodes\t{}\n\tweights\t{}\n\tleaves\t{}'.format(spn.n_layers(),
                                                                     spn.n_nodes(),
                                                                     spn.n_weights(),
                                                                     spn.n_leaves()))
    # NOTE
    # from now on only the last model will be used by some options!

    color_map = None

    w_space, h_space = float(args.space[0]), float(args.space[1])
    logging.info('Spaces w: {} h: {}'.format(w_space, h_space))
    if args.sample is not None:
        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(args.output, '{}.{}.{}'.format(args.sample,
                                                                           dataset_name,
                                                                           SAMPLE_IMGS_EXT))
            pdf = True if args.save == 'pdf' else False

        if args.invert:
            color_map = inv_binary_cmap
        else:
            color_map = binary_cmap

        logging.info('Sampling {} instances'.format(args.sample))
        sample_s_t = perf_counter()
        sampled_instances = spn.sample(args.sample, rand_gen=rand_gen)
        sample_e_t = perf_counter()
        logging.info('\tdone in {}'.format(sample_e_t - sample_s_t))

        image_matrixes = [array_2_mat(img, n_rows, n_cols) for img in sampled_instances]
        n_images = min(len(image_matrixes), args.max_n_images)
        m, n = tiling_sizes(n_images, args.n_cols)
        canvas_size = n * col_tile_size,  m * row_tile_size
        logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
        plot_m_by_n_images(image_matrixes[:n_images],
                           m, n,
                           fig_size=canvas_size,
                           cmap=color_map,
                           save_path=sample_save_path,
                           w_space=w_space,
                           h_space=h_space,
                           pdf=pdf)

        if args.nn:
            sample_save_path = None
            pdf = False
            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}.nn'.format(args.sample,
                                                                                  dataset_name,
                                                                                  SAMPLE_IMGS_EXT))
                pdf = True if args.save == 'pdf' else False
            #
            # retrieving the nn
            nns = get_nearest_neighbour(sampled_instances, train)
            image_matrixes = [array_2_mat(img, n_rows, n_cols) for nn_id, img in nns]
            logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
            plot_m_by_n_images(image_matrixes[:n_images],
                               m, n,
                               fig_size=canvas_size,
                               w_space=w_space,
                               h_space=h_space,
                               cmap=color_map,
                               save_path=sample_save_path,
                               pdf=pdf)

    if args.mpe is not None:

        only_first_max = False if args.all_maxes else True

        if args.invert:
            # color_map = inv_ternary_cmap
            color_map = inv_quaternary_cmap
        else:
            # color_map = ternary_cmap
            color_map = quaternary_cmap

        #
        # getting nodes mpe instantiation
        logging.info('Getting nodes MPE instantiations')
        mpe_s_t = perf_counter()
        node_stats = retrieve_all_nodes_mpe_instantiations(spn,
                                                           n_features,
                                                           only_first_max=only_first_max)
        mpe_e_t = perf_counter()
        logging.info('\tdone in {}'.format(mpe_e_t - mpe_s_t))

        #
        # now different kind of visualization depending on filters
        if 'sum' in args.mpe:
            logging.info('Visualizing only sum nodes')
            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}'.format('sum-mpe',
                                                                               dataset_name,
                                                                               MPE_VIS_EXT))
                pdf = True if args.save == 'pdf' else False

            sum_node_mpes = {node: info.mpes
                             for node, info in node_stats.items() if isinstance(node, SumNode)}
            #
            # retrieving images
            mpe_insts_list = []
            for node, mpe_insts in sum_node_mpes.items():
                len_mpe = mpe_insts.shape[0] if mpe_insts.ndim > 1 else 1
                if len_mpe > 1:
                    mpe_insts_list.append(mpe_insts[0])
                else:
                    mpe_insts_list.append(mpe_insts)

            image_matrixes = [array_2_mat(img, n_rows, n_cols) for img in mpe_insts_list]
            n_images = min(len(image_matrixes), args.max_n_images)
            m, n = tiling_sizes(n_images, args.n_cols)
            canvas_size = m * row_tile_size, n * col_tile_size
            logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
            plot_m_by_n_images(image_matrixes[:n_images],
                               m, n,
                               fig_size=canvas_size,
                               cmap=color_map,
                               save_path=sample_save_path,
                               pdf=pdf)

        elif 'prod' in args.mpe:
            logging.info('Visualizing only product nodes')

            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}'.format('prod-mpe',
                                                                               dataset_name,
                                                                               MPE_VIS_EXT))
                pdf = True if args.save == 'pdf' else False

            prod_node_mpes = {node: info.mpes
                              for node, info in node_stats.items() if isinstance(node, ProductNode)}
            #
            # retrieving images
            mpe_insts_list = []
            for node, mpe_insts in prod_node_mpes.items():
                len_mpe = mpe_insts.shape[0] if mpe_insts.ndim > 1 else 1
                if len_mpe > 1:
                    mpe_insts_list.append(mpe_insts[0])
                else:
                    mpe_insts_list.append(mpe_insts)

            image_matrixes = [array_2_mat(img, n_rows, n_cols) for img in mpe_insts_list]
            n_images = min(len(image_matrixes), args.max_n_images)
            m, n = tiling_sizes(n_images, args.n_cols)
            canvas_size = m * row_tile_size, n * col_tile_size
            logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
            plot_m_by_n_images(image_matrixes[:n_images],
                               m, n,
                               fig_size=canvas_size,
                               cmap=color_map,
                               save_path=sample_save_path,
                               pdf=pdf)

        elif 'layer' in args.mpe:
            logging.info('Visualizing by layers')
            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}'.format('layer-mpe',
                                                                               dataset_name,
                                                                               MPE_VIS_EXT))
                pdf = True if args.save == 'pdf' else False

            layer_node_mpes = defaultdict(list)
            for node, info in node_stats.items():
                layer_node_mpes[info.layer].append((node, info.mpes))

            n_layers = len(layer_node_mpes)
            m_layer, n_layer = tiling_sizes(n_layers, args.n_cols_layer)

            layer_mpe_insts_list = []
            for layer, node_mpes in layer_node_mpes.items():

                mpe_insts_list = []
                for node, mpe_insts in node_mpes:
                    len_mpe = mpe_insts.shape[0] if mpe_insts.ndim > 1 else 1

                    if len_mpe > 1:
                        mpe_insts_list.append(array_2_mat(mpe_insts[0], n_rows, n_cols))
                    else:
                        mpe_insts_list.append(array_2_mat(mpe_insts, n_rows, n_cols))

                n_images = min(len(mpe_insts_list), args.max_n_images_layers)
                layer_mpe_insts_list.append(mpe_insts_list[:n_images])

            m, n = args.n_cols, args.n_cols
            canvas_size = ((m * m_layer + m_layer - 1) * row_tile_size,
                           (n * n_layer + n_layer - 1) * col_tile_size)
            logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
            plot_m_by_n_by_p_by_q_images(layer_mpe_insts_list,
                                         m, n, m_layer, n_layer,
                                         fig_size=canvas_size,
                                         cmap=color_map,
                                         save_path=sample_save_path,
                                         pdf=pdf)

        elif 'scope' in args.mpe:

            print(args.mpe)
            #
            # ordering by scope
            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}'.format('scope-mpe',
                                                                               dataset_name,
                                                                               MPE_VIS_EXT))
                pdf = True if args.save == 'pdf' else False

            if args.scope_range:
                assert len(args.scope_range) == 2

                min_scope_len, max_scope_len = int(args.scope_range[0]), int(args.scope_range[1])
                logging.info('Visualizing by scopes {} -> {}'.format(min_scope_len,
                                                                     max_scope_len))

                scope_nodes = [info.mpes for node, info in node_stats.items()
                               if len(info.scope) >= min_scope_len and
                               len(info.scope) < max_scope_len]
                # scope_lengths = [info for node, info in node_stats.items()
                #                  if len(info.scope) >= min_scope_len and
                #                  len(info.scope) < max_scope_len]
                # for info in scope_lengths:
                #     print(len(info.scope))
                #     print(info.mpes)

                #
                # shuffling them
                random.shuffle(scope_nodes)

                mpe_insts_list = []
                for mpe_insts in scope_nodes:
                    len_mpe = mpe_insts.shape[0] if mpe_insts.ndim > 1 else 1

                    if len_mpe > 1:
                        mpe_insts_list.append(mpe_insts[0])
                    else:
                        # print(array_2_mat(mpe_insts, n_rows, n_cols))
                        mpe_insts_list.append(mpe_insts)

                n_images = min(len(mpe_insts_list), args.max_n_images_scopes)
                logging.info('Found {} instantiations, limiting to {}'.format(len(mpe_insts_list),
                                                                              n_images))
                m, n = tiling_sizes(n_images, args.n_cols)
                canvas_size = n * col_tile_size,  m * row_tile_size
                logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
                mpe_insts_mat = [array_2_mat(img, n_rows, n_cols)
                                 for img in mpe_insts_list[:n_images]]

                #
                # checkboard bg
                mpe_insts_mat = [make_checkboard_bg(img) for img in mpe_insts_mat]
                # mpe_insts_mat = [make_double_checkboard_bg(img) for img in mpe_insts_mat]

                color_map_list = []
                for img in mpe_insts_list[:n_images]:
                    c_map = None
                    #
                    # more than two values
                    img_unique_values = numpy.unique(img)
                    if img_unique_values.shape[0] > 2:
                        if args.invert:
                            c_map = inv_quaternary_cmap
                        else:
                            c_map = quaternary_cmap
                    else:
                        #
                        # what is the only value present?
                        if MARG_IND in img_unique_values:
                            other_val = [v for v in img_unique_values
                                         if v != MARG_IND][0]
                            if other_val == 1:
                                if args.invert:
                                    c_map = inv_quaternary_white_cmap
                                else:
                                    c_map = quaternary_white_cmap
                            elif other_val == 0:
                                if args.invert:
                                    c_map = quaternary_white_cmap
                                else:
                                    c_map = inv_quaternary_white_cmap
                        else:
                            if args.invert:
                                c_map = inv_binary_cmap
                            else:
                                c_map = binary_cmap

                    color_map_list.append(c_map)

                cmap = pyplot.cm.gray
                cmap.set_under(color='red')    
                # plt.imshow(a, interpolation='none', cmap=cmap, vmin=0.0000001)
                color_map_list = [cmap for i in range(n_images)]
                
                plot_m_by_n_images_list(mpe_insts_mat,
                                        m, n,
                                        fig_size=canvas_size,
                                        cmap=color_map_list,
                                        save_path=sample_save_path,
                                        dpi=args.dpi,
                                        w_space=w_space,
                                        h_space=h_space,
                                        pdf=pdf)

                #
                # FIXME:
                # when there are only two values the ternary color map is false
                # plot_m_by_n_images(mpe_insts_mat,
                #                    m, n,
                #                    fig_size=canvas_size,
                #                    cmap=color_map,
                #                    save_path=sample_save_path,
                #                    dpi=args.dpi,
                #                    w_space=w_space,
                #                    h_space=h_space,
                #                    pdf=pdf)

                if args.nn:

                    if args.invert:
                        color_map = inv_binary_cmap
                    else:
                        color_map = binary_cmap

                    sample_save_path = None
                    pdf = False
                    if args.save:
                        sample_save_path = os.path.join(args.output, '{}.{}.{}.nn'.format('scope-mpe',
                                                                                          dataset_name,
                                                                                          MPE_VIS_EXT))
                        pdf = True if args.save == 'pdf' else False
                    #
                    # retrieving the nn
                    nns = get_nearest_neighbour(mpe_insts_list[:n_images], train, masked=False)
                    image_matrixes = [array_2_mat(img, n_rows, n_cols) for nn_id, img in nns]
                    logging.info(
                        'Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
                    plot_m_by_n_images(image_matrixes[:n_images],
                                       m, n,
                                       fig_size=canvas_size,
                                       w_space=w_space,
                                       h_space=h_space,
                                       cmap=color_map,
                                       save_path=sample_save_path,
                                       pdf=pdf)

            else:

                logging.info('Visualizing by scopes')

                scope_node_mpes = defaultdict(list)
                for node, info in node_stats.items():
                    scope_node_mpes[len(info.scope)].append((node, info.mpes))

                n_scopes = len(scope_node_mpes)
                m_scope, n_scope = tiling_sizes(n_scopes, args.n_cols_scope)
                scope_mpe_insts_list = []
                for scope_length in sorted(scope_node_mpes):
                    logging.info(scope_length)
                    node_mpes = scope_node_mpes[scope_length]

                    mpe_insts_list = []
                    for node, mpe_insts in node_mpes:
                        len_mpe = mpe_insts.shape[0] if mpe_insts.ndim > 1 else 1

                        if len_mpe > 1:
                            mpe_insts_list.append(array_2_mat(mpe_insts[0], n_rows, n_cols))
                        else:
                            mpe_insts_list.append(array_2_mat(mpe_insts, n_rows, n_cols))

                    n_images = min(len(mpe_insts_list), args.max_n_images_scopes)
                    scope_mpe_insts_list.append(mpe_insts_list[:n_images])

                m, n = args.n_cols, args.n_cols
                canvas_size = ((m * m_scope + m_scope - 1) * row_tile_size,
                               (n * n_scope + n_scope - 1) * col_tile_size)
                logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
                plot_m_by_n_by_p_by_q_images(scope_mpe_insts_list,
                                             m, n, m_scope, n_scope,
                                             fig_size=canvas_size,
                                             cmap=color_map,
                                             save_path=sample_save_path,
                                             pdf=pdf)

    if args.scope:

        if 'hist' in args.scope:
            logging.info('Showing scope histogram')

            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}-{}'.format('scope-histogram',
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            xlims = None
            if args.xlim:
                xlims = int(args.xlim[0]), int(args.xlim[1])
            #
            # plotting histogram
            scope_list = scope_histogram(spn,
                                         # fig_size=args.size,
                                         ylim=args.ylim,
                                         xlim=xlims,
                                         dpi=args.dpi,
                                         save_path=sample_save_path,
                                         pdf=pdf,
                                         show=not(args.no_show))
            #
            # computing scope quartiles
            no_leaves = True
            if no_leaves:
                scope_list[0] = 0

            quartiles = approx_scope_histo_quartiles(scope_list)
            logging.info('Approx quartile scope lengths {}'.format(quartiles))

        elif 'map' in args.scope:
            logging.info('Showing scope maps')

            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}-{}'.format('scope-map',
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            xlim = None
            if args.xlim:
                xlim = int(args.xlim[0])
            #
            # retrieving scope lists
            scope_lists = scope_maps(spns,
                                     cmap=pyplot.get_cmap('Purples'),
                                     min_val=None,
                                     max_val=None,
                                     xlim=xlim,
                                     fig_size=args.fig_size,
                                     w_space=w_space,
                                     h_space=h_space,
                                     dpi=args.dpi,
                                     save_path=sample_save_path,
                                     pdf=pdf)

        elif 'lmap' in args.scope:
            logging.info('Showing scope maps layerwise')

            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}-{}'.format('scope-map-layerwise',
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            xlim = None
            if args.xlim:
                xlim = int(args.xlim[0])
            #
            # retrieving scope lists
            scope_lists = scope_map_layerwise(spn,
                                              # cmap=pyplot.get_cmap('jet'),
                                              cmap=pyplot.get_cmap('binary'),
                                              xlim=xlim,
                                              fig_size=args.fig_size,
                                              w_space=w_space,
                                              h_space=h_space,
                                              dpi=args.dpi,
                                              save_path=sample_save_path,
                                              pdf=pdf,
                                              show=not(args.no_show))

        elif 'comp-hist' in args.scope:
            logging.info('Showing comparative scope histograms')
            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}'.format('comp-scope-histogram',
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            multiple_scope_histogram(spns,
                                     save_path=sample_save_path,
                                     y_log=True,
                                     pdf=pdf)

        elif 'layer' in args.scope:
            logging.info('Showing scope histgram by layer')

            pdf = False
            sample_save_path = None

            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}'.format('scope-layer-histogram',
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            n_layers = spn.n_layers()
            m_layer, n_layer = tiling_sizes(n_layers, args.n_cols_layer)
            layer_scope_histogram(spn,
                                  m_layer, n_layer,
                                  save_path=sample_save_path,
                                  pdf=pdf)

    if args.activations:

        assert len(args.activations) > 0

        instance_id, filters = args.activations[0], args.activations[1:]
        if len(args.activations) > 1:

            filters_str = '-'.join(str(f) for f in filters)

            sample_save_path = None
            pdf = False
            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}.{}'.format(filters_str,
                                                                               dataset_name,
                                                                               SAMPLE_IMGS_EXT))
                pdf = True if args.save == 'pdf' else False

            if args.invert:
                color_map = inv_binary_cmap
            else:
                color_map = binary_cmap

            instance_id = int(instance_id)
            instance = retrieve_instance((train, valid, test), instance_id)

            #
            # collecting one vis for each filter
            visualizations = [array_2_mat(instance, n_rows, n_cols)]
            cmaps = [color_map]
            min_max_list = [(None, None)]
            for node_filter in filters:

                nodes = filter_nodes(spn, node_filter)
                logging.info('Considering only {} nodes {}'.format(len(nodes), node_filter))

                activations = node_activations_for_instance(spn, nodes, instance)
                visualizations.append(array_2_mat(activations, n_rows, n_cols))
                cmaps.append(matplotlib.cm.jet)
                min_max_list.append((min(activations), max(activations)))

            #
            # marginals
            # marg_instance = extract_features_all_marginals_spn(spn,
            #                                                    instance.reshape(1, instance.shape[0]),
            #                                                    feature_vals)

            # marg_instance_ml = extract_features_all_marginals_ml(train,
            #                                                      instance.reshape(1,
            #                                                                       instance.shape[0]),
            #                                                      feature_vals)

            # two_by_two_marg_masks = extract_features_marginalization_grid(n_rows, n_cols,
            #                                                               4, 4)
            # two_by_two_marg_features = extract_feature_marginalization_from_masks(spn,
            #                                                                       instance.reshape(
            #                                                                           1, instance.shape[0]),
            #                                                                       two_by_two_marg_masks)
            # two_by_two_marg_instance = instance_from_disjoint_feature_masks(None,
            #                                                                 two_by_two_marg_masks,
            #                                                                 two_by_two_marg_features[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)
            # visualizations.append(array_2_mat(two_by_two_marg_instance, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.gray)

            # marg_instance = marg_instance[0]
            # visualizations.append(array_2_mat(marg_instance, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.gray)

            # marg_instance_ml = marg_instance_ml[0]
            # visualizations.append(array_2_mat(marg_instance_ml, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.gray)

        else:
            #
            # default filters

            sample_save_path = None
            pdf = False
            if args.save:
                sample_save_path = os.path.join(args.output,
                                                '{}.all.all-mean.sum.prod.marg-1-2.{}.{}'.format(instance_id,
                                                                                                 dataset_name,
                                                                                                 SAMPLE_IMGS_EXT))
                pdf = True if args.save == 'pdf' else False

            if args.invert:
                color_map = inv_binary_cmap
            else:
                color_map = binary_cmap

            norm = True

            instance_id = int(instance_id)
            instance = retrieve_instance((train, valid, test), instance_id)

            #
            # collecting one vis for each filter
            visualizations = [array_2_mat(instance, n_rows, n_cols)]
            cmaps = [color_map]
            min_max_list = [(None, None)]

#
            # all mean
            nodes = filter_nodes(spn, 'all')
            logging.info('Considering only all nodes {} (mean)'.format(len(nodes)))

            activations = node_activations_for_instance(spn, nodes, instance, mean=True)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_all = min(activations)
            max_all = max(activations)
            min_max_list.append((min_all, max_all))
            # min_max_list.append((min(activations), max(activations)))

            # # sum
            # nodes = filter_nodes(spn, 'sum')
            # logging.info('Considering only mean sum nodes {}'.format(len(nodes)))

            # activations = node_activations_for_instance(spn, nodes, instance,  mean=True)
            # # if norm:
            # #     activations = numpy.sqrt(activations)
            # visualizations.append(array_2_mat(activations, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.jet)
            # min_sum = min(activations)
            # max_sum = max(activations)

            # #
            # # prod
            # nodes = filter_nodes(spn, 'prod')
            # logging.info('Considering only mean prod nodes {}'.format(len(nodes)))

            # activations = node_activations_for_instance(spn, nodes, instance,  mean=True)
            # visualizations.append(array_2_mat(activations, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.jet)
            # min_prod = min(activations)
            # max_prod = max(activations)

            # print(min_sum, min_prod, max_sum, max_prod)
            # #
            # # normalizing sum and prods
            # min_val = min(min_sum, min_prod, min_all)
            # max_val = max(max_prod, max_sum, max_all)

            # min_max_list.append((min_val, max_val))
            # min_max_list.append((min_val, max_val))
            # min_max_list.append((min_val, max_val))

            #
            # all
            nodes = filter_nodes(spn, 'all')
            logging.info('Considering only all nodes {}'.format(len(nodes)))

            activations = node_activations_for_instance(spn, nodes, instance)
            if norm:
                activations = numpy.sqrt(activations)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_all = min(activations)
            max_all = max(activations)
            min_max_list.append((min_all, max_all))

            #
            # sum
            nodes = filter_nodes(spn, 'sum')
            logging.info('Considering only sum nodes {}'.format(len(nodes)))

            activations = node_activations_for_instance(spn, nodes, instance)
            # if norm:
            #     activations = numpy.sqrt(activations)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_sum = min(activations)
            max_sum = max(activations)

            #
            # prod
            nodes = filter_nodes(spn, 'prod')
            logging.info('Considering only prod nodes {}'.format(len(nodes)))

            activations = node_activations_for_instance(spn, nodes, instance)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_prod = min(activations)
            max_prod = max(activations)

            #
            # normalizing sum and prods
            min_max_list.append((min(min_sum, min_prod), max(max_prod, max_sum)))
            min_max_list.append((min(min_sum, min_prod), max(max_prod, max_sum)))

            #
            # marginal with mask
            feature_mask = numpy.zeros(len(instance), dtype=bool)
            feature_mask = random_rectangular_feature_mask(feature_mask,
                                                           n_rows, n_cols,
                                                           *args.mask_sizes)
            inv_feature_mask = numpy.logical_not(feature_mask)

            nodes = filter_nodes(spn, 'all')
            logging.info('Considering only all nodes mask {}'.format(len(nodes)))

            activations = node_activations_for_instance(spn,
                                                        nodes,
                                                        instance,
                                                        marg_mask=feature_mask)
            if norm:
                activations = numpy.sqrt(activations)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            # min_mask = min(min(activations), min_mask)
            # max_mask = max(max(activations), max_mask)
            min_mask = min(activations)
            max_mask = max(activations)

            min_max_list.append((min_mask, max_mask))

            nodes = filter_nodes(spn, 'all')
            logging.info('Considering only all nodes inv mask {}'.format(len(nodes)))

            activations = node_activations_for_instance(spn,
                                                        nodes,
                                                        instance,
                                                        marg_mask=inv_feature_mask)
            if norm:
                activations = numpy.sqrt(activations)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_mask = min(min(activations), min_mask)
            max_mask = max(max(activations), max_mask)
            min_max_list.append((min_mask, max_mask))

            nodes = filter_nodes(spn, 'all')
            logging.info('Considering only all nodes all marg {}'.format(len(nodes)))

            all_instance = numpy.zeros(len(instance), dtype=instance.dtype)
            all_instance[:] = MARG_IND
            activations = node_activations_for_instance(spn,
                                                        nodes,
                                                        all_instance,
                                                        marg_mask=feature_mask)
            if norm:
                activations = numpy.sqrt(activations)
            visualizations.append(array_2_mat(activations, n_rows, n_cols))
            cmaps.append(matplotlib.cm.jet)
            min_mask = min(min(activations), min_mask)
            max_mask = max(max(activations), max_mask)
            min_max_list.append((min_mask, max_mask))

            # nodes = filter_nodes(spn, 'all')
            # logging.info('Considering only all nodes all marg {} (mean)'.format(len(nodes)))

            # all_instance = numpy.zeros(len(instance), dtype=instance.dtype)
            # all_instance[:] = MARG_IND
            # activations = node_activations_for_instance(spn,
            #                                             nodes,
            #                                             all_instance,
            #                                             marg_mask=feature_mask,
            #                                             mean=True)
            # # if norm:
            # activations = numpy.sqrt(activations)
            # visualizations.append(array_2_mat(activations, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.jet)
            # # min_mask = min(min(activations), min_mask)
            # # max_mask = max(max(activations), max_mask)
            # print(min(activations), max(activations))
            # min_max_list.append((None, None))

            min_max_list[2] = (min_mask, max_mask)

        m = 1
        n = len(visualizations)
        canvas_size = n * col_tile_size, m * row_tile_size
        print(canvas_size)
        plot_m_by_n_heatmaps(visualizations,
                             min_max_list,
                             m=m, n=n,
                             cmaps=cmaps,
                             colorbars=args.colorbar,
                             fig_size=canvas_size,
                             w_space=w_space, h_space=h_space,
                             dpi=args.dpi,
                             save_path=sample_save_path,
                             pdf=pdf)

    if args.part_activations:

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.backends.backend_pdf import PdfPages

        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(
                args.output, '{}-partition-activations'.format(args.suffix))
            pdf = True if args.save == 'pdf' else False

        nodes = filter_nodes(spn, 'all')
        logging.info(
            'Visualizing the partition function.\nConsidering only all nodes all marg {}'.format(len(nodes)))

        all_ones_instance = numpy.zeros(train.shape[1], dtype=train.dtype)
        all_ones_instance[:] = MARG_IND
        activations = node_activations_for_instance(spn,
                                                    nodes,
                                                    all_ones_instance)
        if args.sqrt_trans:
            activations = numpy.sqrt(activations)

        plot_heatmap(array_2_mat(activations, n_rows, n_cols),
                     (col_tile_size, row_tile_size),
                     colorbar=args.colorbar,
                     cmap=matplotlib.cm.jet,
                     dpi=args.dpi,
                     save_path=sample_save_path,
                     pdf=pdf,
                     show=not(args.no_show))

    if args.scope_image_map:

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.backends.backend_pdf import PdfPages

        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(
                args.output, '{}-scope-map'.format(args.suffix))
            pdf = True if args.save == 'pdf' else False

        scope_map = numpy.zeros(train.shape[1], dtype=int)
        for node in spn.top_down_nodes():
            scope = None
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = [node.var]
            for f in scope:
                scope_map[f] += 1

        canvas_size = col_tile_size, row_tile_size
        print(canvas_size)
        fig, ax = pyplot.subplots(figsize=canvas_size, dpi=args.dpi)
        img = ax.matshow(array_2_mat(scope_map, n_rows, n_cols),
                         cmap=matplotlib.cm.jet,
                         # vmin=min_max_list[id][0],
                         # vmax=min_max_list[id][1],
                         norm=None)
        pyplot.xticks(numpy.array([]))
        pyplot.yticks(numpy.array([]))
        pyplot.axis('off')
        if args.colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            pyplot.colorbar(img, cax=cax)

        pyplot.tight_layout()
        if sample_save_path:
            fig.savefig(sample_save_path + '.svg', bbox_inches='tight', pad_inches=0)
            if pdf:
                pp = PdfPages(sample_save_path + '.pdf')
                pp.savefig(fig, bbox_inches='tight', pad_inches=0)
                pp.close()
        pyplot.show()
        # plot_m_by_n_heatmaps([activations],
        #                      ((min(activations), max(activations)),),
        #                      m=1, n=1,
        #                      cmaps=[matplotlib.cm.jet],
        #                      colorbars=args.colorbar,
        #                      fig_size=canvas_size,
        #                      w_space=w_space, h_space=h_space,
        #                      dpi=args.dpi,
        #                      save_path=sample_save_path,
        #                      pdf=pdf)

    if args.marg_activations:

        # instance_id, filters = args.marg_activations[0], args.marg_activations[1:]
        instance_ids = args.marg_activations

        sample_save_path = None
        pdf = False
        if args.save:
            instance_ids_str = '-'.join(i for i in instance_ids)
            sample_save_path = os.path.join(args.output, '{}.{}.marg'.format(instance_ids_str,
                                                                             dataset_name))
            pdf = True if args.save == 'pdf' else False

        if args.invert:
            color_map = inv_binary_cmap
        else:
            color_map = binary_cmap

        visualizations = []
        cmaps = []
        min_max_list = []
        min_list = []
        max_list = []

        # for spn in spns:
        #

        spn_1 = spns[0]
        spn_2 = spns[1]
        spn_3 = spns[2]

        if 'ocr_letters' == dataset_name:
            max_patch_res = 8
        else:
            max_patch_res = 7

        for instance_id in instance_ids:
            instance_id = int(instance_id)
            instance = retrieve_instance((train, valid, test), instance_id)

            # collecting one vis for each filter
            visualizations.append(array_2_mat(instance, n_rows, n_cols))
            cmaps.append(color_map)
            min_max_list.append((None, None))

            norm = True
            # # all single marginals ml
            # marg_instance_ml = extract_features_all_marginals_ml(train,
            #                                                      instance.reshape(1,
            #                                                                       instance.shape[0]),
            #                                                      feature_vals)
            # marg_instance_ml = numpy.log(marg_instance_ml[0])
            # visualizations.append(array_2_mat(marg_instance_ml, n_rows, n_cols))
            # cmaps.append(matplotlib.cm.gray)
            # min_marg_ml = min(marg_instance_ml)
            # max_marg_ml = max(marg_instance_ml)
            # print(min_marg_ml, max_marg_ml)

            # all single marginals spn
            marg_instance_spn = extract_features_all_marginals_spn(spn_1,
                                                                   instance.reshape(
                                                                       1, instance.shape[0]),
                                                                   feature_vals)
            marg_instance_spn = marg_instance_spn[0]

            visualizations.append(array_2_mat(marg_instance_spn, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_spn = min(marg_instance_spn)
            max_marg_spn = max(marg_instance_spn)
            min_list.append(min_marg_spn)
            max_list.append(max_marg_spn)
            print(min_marg_spn, max_marg_spn)

            marg_masks_2 = extract_features_marginalization_grid(n_rows, n_cols,
                                                                 2, 2)
            marg_features_2 = extract_feature_marginalization_from_masks(spn_1,
                                                                         instance.reshape(1,
                                                                                          instance.shape[0]),
                                                                         marg_masks_2)
            marg_instance_2 = instance_from_disjoint_feature_masks(None,
                                                                   marg_masks_2,
                                                                   marg_features_2[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)

            visualizations.append(array_2_mat(marg_instance_2, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_2 = min(marg_instance_2)
            max_marg_2 = max(marg_instance_2)
            min_list.append(min_marg_2)
            max_list.append(max_marg_2)
            print(min_marg_2, max_marg_2)

            marg_masks_4 = extract_features_marginalization_grid(n_rows, n_cols,
                                                                 4, 4)
            marg_features_4 = extract_feature_marginalization_from_masks(spn_1,
                                                                         instance.reshape(1,
                                                                                          instance.shape[0]),
                                                                         marg_masks_4)
            marg_instance_4 = instance_from_disjoint_feature_masks(None,
                                                                   marg_masks_4,
                                                                   marg_features_4[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)

            visualizations.append(array_2_mat(marg_instance_4, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_4 = min(marg_instance_4)
            max_marg_4 = max(marg_instance_4)
            min_list.append(min_marg_4)
            max_list.append(max_marg_4)
            print(min_marg_4, max_marg_4)

            marg_masks_7 = extract_features_marginalization_grid(n_rows, n_cols,
                                                                 max_patch_res, max_patch_res)
            marg_features_7 = extract_feature_marginalization_from_masks(spn_1,
                                                                         instance.reshape(1,
                                                                                          instance.shape[0]),
                                                                         marg_masks_7)
            marg_instance_7 = instance_from_disjoint_feature_masks(None,
                                                                   marg_masks_7,
                                                                   marg_features_7[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)

            visualizations.append(array_2_mat(marg_instance_7, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_7 = min(marg_instance_7)
            max_marg_7 = max(marg_instance_7)
            min_list.append(min_marg_7)
            max_list.append(max_marg_7)
            print(min_marg_7, max_marg_7)

            marg_masks_7 = extract_features_marginalization_grid(n_rows, n_cols,
                                                                 max_patch_res, max_patch_res)
            marg_features_7 = extract_feature_marginalization_from_masks(spn_2,
                                                                         instance.reshape(1,
                                                                                          instance.shape[0]),
                                                                         marg_masks_7)
            marg_instance_7 = instance_from_disjoint_feature_masks(None,
                                                                   marg_masks_7,
                                                                   marg_features_7[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)

            visualizations.append(array_2_mat(marg_instance_7, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_7 = min(marg_instance_7)
            max_marg_7 = max(marg_instance_7)
            min_list.append(min_marg_7)
            max_list.append(max_marg_7)
            print(min_marg_7, max_marg_7)

            marg_masks_7 = extract_features_marginalization_grid(n_rows, n_cols,
                                                                 max_patch_res, max_patch_res)
            marg_features_7 = extract_feature_marginalization_from_masks(spn_3,
                                                                         instance.reshape(1,
                                                                                          instance.shape[0]),
                                                                         marg_masks_7)
            marg_instance_7 = instance_from_disjoint_feature_masks(None,
                                                                   marg_masks_7,
                                                                   marg_features_7[0])
            # print(two_by_two_marg_instance)
            # print(two_by_two_marg_instance.shape)

            visualizations.append(array_2_mat(marg_instance_7, n_rows, n_cols))
            cmaps.append(matplotlib.cm.gray)
            min_marg_7 = min(marg_instance_7)
            max_marg_7 = max(marg_instance_7)
            min_list.append(min_marg_7)
            max_list.append(max_marg_7)
            print(min_marg_7, max_marg_7)

        min_all = min(min_list)
        max_all = max(max_list)
        min_max_list = [(min_all, max_all)] * (len(visualizations))

        # n_models = len(spns)

        # m = n_models
        # n = len(visualizations) // n_models
        m = len(instance_ids)
        n = len(visualizations) // m
        logging.info('Printing {} x {}'.format(m, n))

        for i in range(len(instance_ids)):
            min_max_list[i * n] = (None, None)
        # for i in range(n_models):
        #     min_max_list[n * i] = (None, None)
        # min_max_list[0] = (None, None)

        canvas_size = n * col_tile_size, m * row_tile_size
        print(canvas_size)
        plot_m_by_n_heatmaps(visualizations,
                             min_max_list,
                             m=m, n=n,
                             cmaps=cmaps,
                             colorbars=False,
                             fig_size=canvas_size,
                             w_space=w_space, h_space=h_space,
                             dpi=args.dpi,
                             save_path=sample_save_path,
                             pdf=pdf)

    if args.lines:
        # import seaborn
        from matplotlib.backends.backend_pdf import PdfPages

        # pyplot.style.use('ggplot')
        import seaborn
        # seaborn.set_style('white')
        # seaborn.set_context(rc={'lines.markeredgewidth': 0.1})
        # seaborn.set_context('poster')
        seaborn.set_style('white')
        seaborn.set_context('poster', font_scale=1.8)

        logging.info('Visualizing accuracy lines in {}'.format(args.lines))
        #
        # reading them from file
        lines = numpy.loadtxt(args.lines, delimiter='\t')
        n_obs = lines.shape[0]
        x_axis = numpy.arange(100, 1001, 100)
        names = ['SPN-I', 'SPN-II', 'SPN-III', 'MT-I', 'MT-II', 'MT-III']
        colors = seaborn.color_palette("husl", 6)
        # colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        line_styles = ['dotted', '--', '-', 'dotted', '--', '-']
        markers = ['o', 'o', 'o', 'x', 'x', 'x']
        markersizes = [8., 8., 8., 15., 15., 15.]
        linewidths = [4., 2.5, 2.5, 4., 2.5, 2.5]

        lines = lines * 100

        if args.baseline is not None:
            names.append('LR')
            colors = seaborn.color_palette("husl", 7)
            line_styles.append('-')
            markers.append(None)
            markersizes.append(0.0)
            linewidths.append(4.)
            base_line = numpy.zeros((lines.shape[0], 1))
            base_line[:] = args.baseline
            print(lines.shape, base_line.shape, base_line)
            lines = numpy.hstack((lines, base_line))

        n_series = lines.shape[1]
        logging.info('There are {} series with {} obs'.format(n_series, n_obs))

        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(args.output, 'lines.{}'.format(dataset_name))
            pdf = True if args.save == 'pdf' else False

        # matplotlib.rcParams.update({'font.size': 52})
        fig_size = args.fig_size
        fig, ax = pyplot.subplots(figsize=fig_size, dpi=args.dpi)

        for i in range(n_series):
            pyplot.plot(x_axis, lines[:, i],
                        label=names[i],
                        linestyle=line_styles[i],
                        linewidth=linewidths[i],
                        markersize=markersizes[i],
                        markeredgewidth=1,
                        markeredgecolor=colors[i],
                        marker=markers[i],
                        color=colors[i])

        legend = ax.legend(names, loc='lower right')
        pyplot.xlabel('# features')
        pyplot.ylabel('test accuracy')
        pyplot.tight_layout()

        if sample_save_path:
            fig.savefig(sample_save_path + '.svg')
            if pdf:
                pp = PdfPages(sample_save_path + '.pdf')
                pp.savefig(fig)
                pp.close()

        # pyplot.subplots_adjust(bottom=1.25)
        if not args.no_show:
            pyplot.show()

    if args.scatter_emb:
        # import seaborn
        from matplotlib.backends.backend_pdf import PdfPages

        import seaborn
        # seaborn.set_context(rc={'lines.markeredgewidth': 0.1})
        seaborn.set_style('white')
        seaborn.set_context('poster', font_scale=1.8)

        logging.info('Visualizing scatter plot for embedding'
                     ' sizes/accuracies {}'.format(args.scatter_emb))
        #
        # reading them from file
        emb_data = numpy.loadtxt(args.scatter_emb, delimiter='\t')
        n_models = emb_data.shape[0]

        names = ['SPN-I full', 'SPN-II full', 'SPN-III full',
                 'RBM-5h', 'RBM-1k', 'RBM-5k',
                 'MADE-5h', 'MADE-1k',
                 'SPN-I sum', 'SPN-I prod',
                 'SPN-II sum', 'SPN-II prod',
                 'SPN-III sum', 'SPN-III prod',
                 'SPN-III S', 'SPN-III M', 'SPN-III L',
                 'SPN-I aggr', 'SPN-I aggr-leaves',
                 'SPN-II aggr', 'SPN-II aggr-leaves',
                 'SPN-III aggr', 'SPN-III aggr-leaves']

        line_styles = ['dotted', '--', '-', 'dotted', '--', '-']
        # markers = ['o', 'v', '^',
        #            '<', '>', '1',
        #            '2', '3', '4', 's', 'p', '*', 'h', 'H', '+',
        #            'x', 'D', 'd',
        #            '8',
        #            '.',
        #            '+']
        # colors = seaborn.color_palette("husl", n_models)
        markers = ['o', 'o', 'o',
                   's', 's', 's',
                   '*', '*',
                   '<', '>', '<', '>', '<', '>',
                   'D', 'D', 'D',
                   '^', 'v', '^', 'v', '^', 'v']
        markersizes = [8., 8., 8., 15., 15., 15.]
        colors = ['w', '0.65', 'k',
                  'w', '0.65', 'k',
                  'w', 'k',
                  'w', 'w', '0.65', '0.65', 'k', 'k',
                  'w', '0.65', 'k',
                  'w', 'w', '0.65', '0.65', 'k', 'k']

        logging.info('There are {} models'.format(n_models))

        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(args.output, 'scatter-emb-{}'.format(dataset_name))
            pdf = True if args.save == 'pdf' else False

        # matplotlib.rcParams.update({'font.size': 52})
        fig_size = args.fig_size
        fig, ax = pyplot.subplots(figsize=fig_size, dpi=args.dpi)

        embs = []
        for i in range(n_models):
            emb,  = ax.plot(emb_data[i, 0], emb_data[i, 1],
                            markers[i], c=colors[i],
                            mew=1,
                            ms=15)
            embs.append(emb)
        ax.set_xscale('log')

        import matplotlib.patches as mpatches

        recs = []
        for i in range(n_models):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

        # from matplotlib.font_manager import FontProperties

        # font_p = FontProperties()
        # font_p.set_size('small')
        if not(args.no_legend):
            lgd = legend = ax.legend(embs, names,
                                     # loc='lower right',
                                     loc='center left', bbox_to_anchor=(1, 0.5),
                                     # scatterpoints=1,
                                     ncol=2,
                                     # prop=font_p
                                     )

        pyplot.xlabel('embedding size')
        pyplot.ylabel('test accuracy')
        pyplot.tight_layout()

        if sample_save_path:
            if not(args.no_legend):
                fig.savefig(sample_save_path + '.svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                fig.savefig(sample_save_path + '.svg', bbox_inches='tight')
            if pdf:
                pp = PdfPages(sample_save_path + '.pdf')
                if not(args.no_legend):
                    pp.savefig(fig, bbox_extra_artists=(lgd,), bbox_inches='tight')
                else:
                    pp.savefig(fig, bbox_inches='tight')
                pp.close()

        # pyplot.subplots_adjust(bottom=1.25)
        if not args.no_show:
            pyplot.show()

    if args.hid_groups:

        sample_save_path = None
        pdf = False
        if args.save:
            sample_save_path = os.path.join(args.output, 'lines.{}'.format(dataset_name))
            pdf = True if args.save == 'pdf' else False

        if args.invert:
            color_map = inv_binary_cmap
        else:
            color_map = binary_cmap

        assert len(args.hid_groups) > 1
        repr_data_path, *group_ids = args.hid_groups
        group_ids = [int(gid) for gid in group_ids]

        logging.info('Visualizing groups from repr data: {}'.format(args.dataset))
        with open(repr_data_path, 'rb') as gfile:
            repr_train, repr_valid, repr_test = pickle.load(gfile)

        #
        # freeing memory
        repr_train = repr_train.astype(numpy.int8)
        repr_valid = None
        repr_test = None

        logging.info('Visualizing instances from groups {}'.format(group_ids))

        #
        # processing the training set
        ext_s_t = perf_counter()
        group_train = extract_instances_groups(repr_train)
        ext_e_t = perf_counter()
        n_groups = group_train.shape[1]
        logging.info('Found {} groups in {} secs'.format(n_groups,
                                                         ext_e_t - ext_s_t))

        repr_train = None

        if len(group_ids) == 1 and group_ids[0] == -1:
            group_ids = numpy.arange(n_groups)
        #
        # now extracting the groups
        for gid in group_ids:

            sample_save_path = None
            pdf = False
            if args.save:
                sample_save_path = os.path.join(args.output, '{}.{}'.format(gid,
                                                                            dataset_name))
                pdf = True if args.save == 'pdf' else False

            logging.info('Considering group {}'.format(gid))
            instance_map = group_train[:, gid].astype(bool)
            assert len(instance_map) == train.shape[0]

            logging.info('There are {} images in group'.format(sum(instance_map)))

            #
            # getting instances
            instances = train[instance_map, :]
            print(instances.shape)
            image_matrixes = [array_2_mat(img, n_rows, n_cols) for img in instances]

            #
            # shuffling?
            random.shuffle(image_matrixes)

            n_images = min(len(image_matrixes), args.max_n_images)
            m, n = tiling_sizes(n_images, args.n_cols)
            canvas_size = m * row_tile_size, n * col_tile_size
            logging.info('Displaying {} x {} images on canvas {}'.format(m, n, canvas_size))
            plot_m_by_n_images(image_matrixes[:n_images],
                               m, n,
                               fig_size=canvas_size,
                               cmap=color_map,
                               w_space=w_space, h_space=h_space,
                               dpi=args.dpi,
                               save_path=sample_save_path,
                               pdf=pdf)
