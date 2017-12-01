import logging

from spn import MARG_IND

import numpy

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker

from matplotlib.colors import LogNorm


# from spn.utils import get_best_value_from_frame

# import seaborn

from collections import defaultdict
from collections import Counter

color_bi_list = [(1., 1., 1.), (0., 0., 0.)]
binary_cmap = matplotlib.colors.ListedColormap(color_bi_list)
inv_binary_cmap = matplotlib.colors.ListedColormap(list(reversed(color_bi_list)))
color_tri_list = [(1., 1., 1.), (0., 0., 0.),  (1., 0., 0.), ]

ternary_cmap = matplotlib.colors.ListedColormap([(1., 0., 0.), (1., 1., 1.), (0., 0., 0.)])
inv_ternary_cmap = matplotlib.colors.ListedColormap([(1., 0., 0.), (0., 0., 0.), (1., 1., 1.)])

checkboard_bg_gray = (0.59, 0.597, 0.59)
checkboard_fg_gray = (0.394, 0.398, 0.39)
quaternary_cmap = matplotlib.colors.ListedColormap([checkboard_bg_gray,
                                                    checkboard_fg_gray,
                                                    (1., 1., 1.),
                                                    (0., 0., 0.)])
inv_quaternary_cmap = matplotlib.colors.ListedColormap([checkboard_bg_gray,
                                                        checkboard_fg_gray,
                                                        (0., 0., 0.),
                                                        (1., 1., 1.)])

quaternary_white_cmap = matplotlib.colors.ListedColormap([checkboard_bg_gray,
                                                          checkboard_fg_gray,
                                                          (1., 1., 1.)])
inv_quaternary_white_cmap = matplotlib.colors.ListedColormap([checkboard_bg_gray,
                                                              checkboard_fg_gray,
                                                              (0., 0., 0.)])


#
# changing font size
# seaborn.set_context("poster", font_scale=1.7, rc={'font.size': 32,
#                                                   # 'axes.labelsize': fontSize,
#                                                   # 'xtick.labelsize': fontSize,
#                                                   # 'ytick.labelsize': fontSize,
#                                                   # 'legend.fontsize': fontSize,
#                                                   'text.usetex': True
#                                                   })

# matplotlib.rcParams.update({'font.size': 22})


def beautify_with_seaborn():
    #

    import seaborn
    seaborn.set_style('white')
    seaborn.despine(trim=True)
    seaborn.set_context('poster')


def make_checkboard_bg(img,
                       bg_value=-1,
                       checkboard_pattern=[-1, -2]):
    #
    # create checkboard image
    wr = 0 if img.shape[1] % 2 == 0 else 1
    hr = 0 if img.shape[0] % 2 == 0 else 1
    w = img.shape[1] // 2 + wr
    h = img.shape[0] // 2 + hr
    re = numpy.r_[w * checkboard_pattern]
    ro = numpy.r_[w * list(reversed(checkboard_pattern))]

    checkboard_img = numpy.row_stack(h * (re, ro))

    fg_indexes = img != bg_value
    checkboard_img[fg_indexes] = img[fg_indexes]

    return checkboard_img[:img.shape[0], :img.shape[1]]


def make_double_checkboard_bg(img,
                              bg_value=-1,
                              checkboard_pattern=[-1, -1, -2, -2]):
    #
    # create checkboard image
    wr = 0 if img.shape[1] % 2 == 0 else 1
    hr = 0 if img.shape[0] % 2 == 0 else 1
    w = img.shape[1] // 2 + wr
    h = img.shape[0] // 2 + hr
    re1 = numpy.r_[w * checkboard_pattern]
    re2 = numpy.r_[w * checkboard_pattern]
    ro1 = numpy.r_[w * list(reversed(checkboard_pattern))]
    ro2 = numpy.r_[w * list(reversed(checkboard_pattern))]

    checkboard_img = numpy.row_stack(h * (re1, re2, ro1, ro2))

    fg_indexes = img != bg_value
    checkboard_img[fg_indexes] = img[fg_indexes]

    return checkboard_img[:img.shape[0], :img.shape[1]]


def visualize_curves(curves,
                     output=None,
                     labels=None,
                     lines=None,
                     linestyles=None,
                     linewidths=None,
                     palette='husl',
                     markers=None,
                     loc=None,
                     colors=None,
                     fig_size=(10, 8)):
    """
    WRITEME
    """

    import seaborn
    seaborn.set_style('white')
    seaborn.set_context(rc={'lines.markeredgewidth': 0.1})
    seaborn.set_context('poster')

    n_curves = len(curves)
    n_lines = len(lines)

    #
    # default legend location, upper right
    if loc is None:
        loc = 3

    #
    # setting the palette
    # seaborn.set_palette(palette, n_colors=(n_curves + n_lines))
    if colors is None:
        colors = [seaborn.color_palette("husl", 12)[1],
                  seaborn.color_palette("husl", 12)[4],
                  seaborn.color_palette("husl")[0],
                  seaborn.color_palette("husl", 12)[8],
                  seaborn.color_palette("husl", 12)[9]]
        print(colors)

    #
    # default linestyle
    default_linestyle = '-'
    if linestyles is None:
        linestyles = [default_linestyle for i in range(n_curves)]
    default_width = 6
    if linewidths is None:
        linewidths = [default_width for i in range(n_curves)]
    if markers is None:
        # markers = ['s', 'D', '2', '3', '1']
        markers = ['3', '1', '2', '3', 's']

    fig, ax = pyplot.subplots(figsize=fig_size)
    for i, curve in enumerate(curves):

        curve_x, curve_y = curve
        if labels is not None:
            label = labels[i]
            line = ax.plot(curve_x, curve_y,
                           label=label,
                           linestyle=linestyles[i],
                           linewidth=linewidths[i],
                           marker=markers[i],
                           mew=0.1,
                           color=colors[i],
                           markeredgecolor='none'
                           )
        else:
            line = ax.plot(curve_x, curve_y,
                           linestyle=linestyles[i],
                           linewidth=linewidths[i],
                           marker=markers[i],
                           mew=0.1,
                           color=colors[i],
                           markeredgecolor='none'
                           )

    #
    # lastly plotting straight lines, if present
    if lines is not None:
        default_linestyles = ['--', '-.', ':']
        for i, line_y in enumerate(lines):
            #
            # this feels a little bit hackish, assuming all share the same axis
            prototypical_x_axis = curves[0][0]
            start_x = prototypical_x_axis[0]
            end_x = prototypical_x_axis[-1]
            ax.plot([start_x, end_x],
                    [line_y, line_y],
                    linestyle=default_linestyles[i],
                    color=colors[i + len(curves)],
                    linewidth=default_width)  # linestyles[i + n_curves])

    #
    # setting up the legend
    if labels is not None:
        legend = ax.legend(labels, loc=loc)

    seaborn.despine()

    pyplot.xlabel('# components')
    pyplot.ylabel('test ll')

    if output is not None:
        # fig = pyplot.gcf()
        # fig_width = 18.5
        # fig_height = 10.5
        # dpi = 150
        # fig.set_size_inches(fig_width, fig_height)
        # fig.savefig(output,
        #             # additional_artists=[legend],
        #             dpi=dpi,
        #             bbox_inches='tight')
        # pyplot.close(fig)

        pp = PdfPages(output + '.pdf')
        pp.savefig(fig)
        pp.close()
    else:
        #
        # shall this be mutually exclusive with file saving?
        pyplot.show()


DATASET_LIST = ['nltcs', 'msnbc', 'kdd',
                'plants', 'baudio', 'jester', 'bnetflix',
                'accidents', 'tretail', 'pumsb_star',
                'dna', 'kosarek', 'msweb',
                'book', 'tmovie', 'cwebkb',
                'cr52', 'c20ng', 'bbc', 'ad']


def visualize_histograms(histograms,
                         output=None,
                         labels=DATASET_LIST,
                         linestyles=None,
                         rotation=90,
                         legend=None,
                         y_log=False,
                         colors=['seagreen', 'orange', 'cornflowerblue']):
    """
    Plotting histograms one near the other
    """

    import seaborn

    n_histograms = len(histograms)
    #
    # assuming homogeneous data leengths
    # TODO: better error checking
    n_ticks = len(histograms[0])

    bin_width = 1 / (n_histograms + 1)
    bins = [[i + j * bin_width for i in range(n_ticks)]
            for j in range(1, n_histograms + 1)]

    #
    # setting up seaborn
    seaborn.set_style("white")
    seaborn.set_context("poster")
    # seaborn.set_palette(palette, n_colors=n_histograms)

    fig, ax = pyplot.subplots()

    if legend is not None:
        _legend = pyplot.legend(legend)
    #
    # setting labels
    middle_histogram = n_histograms // 2 + 1  # if n_histograms > 1 else 0
    pyplot.xticks(bins[middle_histogram], DATASET_LIST)
    if rotation is not None:
        locs, labels = pyplot.xticks()
        pyplot.setp(labels, rotation=90)

    #
    # actual plotting
    print(histograms)
    for i, histogram in enumerate(histograms):
        ax.bar(bins[i], histogram, width=bin_width,
               facecolor=colors[i], edgecolor="none",
               log=y_log)

    seaborn.despine()

    if output is not None:

        pp = PdfPages(output)
        pp.savefig(fig)
        pp.close()


def jitter(arr, std=.02):
    stdev = std * (max(arr) - min(arr))
    return arr + numpy.random.randn(len(arr)) * stdev


def plot_depth_vs_size(frame_list,
                       labels,
                       depth_col_label='n_levels:',
                       size_col_label='n_edges:',
                       fig_size=(9, 8),
                       save_path=None,
                       pdf=False,
                       colors=None,
                       markers=None,
                       jitter_points=(.04, .04),
                       marker_size=80):

    import seaborn

    if not colors:
        colors = seaborn.color_palette("husl")

    if not markers:
        markers = ['o' for _frame in frame_list]

    fig = pyplot.figure(figsize=fig_size)
    ax1 = fig.add_subplot(111)

    # fig.suptitle('nltcs')
    pyplot.xlabel('depth')
    pyplot.ylabel('# edges')

    for i, frame in enumerate(frame_list):

        ax1.scatter(x=jitter(frame[depth_col_label].values, jitter_points[0]),
                    y=jitter(frame[size_col_label].values, jitter_points[1]),
                    c=colors[i],
                    edgecolor='none',
                    marker=markers[i],
                    label=labels[i],
                    s=marker_size)

    seaborn.despine()
    pyplot.legend(loc='upper right')
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()
    pyplot.show()

    return ax1


def get_best_value_from_frame(frame,
                              best_col,
                              attribute=None):
    if not attribute:
        attribute = best_col

        best_values = frame[frame[best_col] == frame[best_col].max()][attribute].values
    assert len(best_values) == 1
    return best_values[0]


def plot_comparative_histograms(frame_lists,
                                col_name,
                                best_col_name,
                                x_labels,
                                y_label,
                                save_path=None,
                                linestyles=None,
                                rotation=90,
                                legend=None,
                                fig_size=(10, 8),
                                y_log=False,
                                colors=None,
                                pdf=False):
    """
    Plotting bars one near the other
    """

    import seaborn
    if not colors:
        colors = seaborn.color_palette("husl")

    #
    # extracting histograms from frames
    histograms = [[] for _list in frame_lists]

    for i, f_list in enumerate(frame_lists):
        for frame in f_list:
            param_value = get_best_value_from_frame(frame, best_col_name, col_name)
            histograms[i].append(param_value)

    n_histograms = len(histograms)

    #
    # assuming homogeneous data lengths
    # TODO: better error checking
    n_ticks = len(histograms[0])

    bin_width = 1 / (n_histograms + 1)
    bins = [[i + j * bin_width for i in range(n_ticks)]
            for j in range(1, n_histograms + 1)]

    fig = pyplot.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # fig.suptitle('nltcs')
    pyplot.xlabel('datasets')
    pyplot.ylabel(y_label)

    if legend is not None:
        _legend = pyplot.legend(legend)
    #
    # setting labels
    middle_histogram = n_histograms // 2 + 1  # if n_histograms > 1 else 0
    pyplot.xticks(bins[middle_histogram], x_labels)
    if rotation is not None:
        locs, labels = pyplot.xticks()
        pyplot.setp(labels, rotation=90)

    #
    # actual plotting
    print(histograms)
    for i, histogram in enumerate(histograms):
        ax.bar(bins[i], histogram, width=bin_width,
               facecolor=colors[i], edgecolor="none",
               log=y_log)

    seaborn.despine()
    pyplot.legend(loc='upper right')
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()
    pyplot.show()

    return ax


def plot_m_by_n_images(images,
                       m, n,
                       fig_size=(12, 12),
                       cmap=matplotlib.cm.binary,
                       w_space=0.1,
                       h_space=0.1,
                       dpi=900,
                       save_path=None,
                       pdf=False):
    """
    Plot images in a mxn tiling
    """
    print(w_space, h_space)
    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)

    print(len(images))
    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    for x in range(m):
        for y in range(n):
            id = n * x + y
            if id < len(images):
                ax = fig.add_subplot(gs1[id])
                ax.matshow(images[id], cmap=cmap)
                pyplot.xticks(numpy.array([]))
                pyplot.yticks(numpy.array([]))

    # pyplot.tight_layout()
    pyplot.subplots_adjust(left=None, right=None, wspace=w_space, hspace=h_space)
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    pyplot.show()


def plot_m_by_n_images_list(images,
                            m, n,
                            fig_size=(12, 12),
                            cmap=[matplotlib.cm.binary],
                            w_space=0.1,
                            h_space=0.1,
                            dpi=900,
                            save_path=None,
                            pdf=False,
                            show=False):
    """
    Plot images in a mxn tiling
    """
    print(w_space, h_space)
    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)

    assert len(images) == len(cmap)

    print(len(images))
    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    for x in range(m):
        for y in range(n):
            id = n * x + y
            if id < len(images):
                ax = fig.add_subplot(gs1[id])
                ax.matshow(images[id], cmap=cmap[id], vmin=-0.00001)
                pyplot.xticks(numpy.array([]))
                pyplot.yticks(numpy.array([]))

    # pyplot.tight_layout()
    pyplot.subplots_adjust(left=None, right=None, wspace=w_space, hspace=h_space)
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    if show:
        pyplot.show()


def plot_m_by_n_by_p_by_q_images(images_lists,
                                 m, n, p, q,
                                 fig_size=(16, 16),
                                 cmap=matplotlib.cm.binary,
                                 save_path=None,
                                 pdf=False):
    """
    Plot images in a (mxp + p - 1) x (nxq + q - 1) tiling
    """
    fig = pyplot.figure(figsize=fig_size)

    tot_rows = m * p + p - 1
    tot_cols = n * q + q - 1
    for i, images in enumerate(images_lists):
        i_row = i // q
        i_col = i - q * i_row
        # print('i', i, i_row, i_col)
        for j, img in enumerate(images):
            j_row = j // n
            j_col = j - n * j_row
            if j < m * n:
                # print('j', j, j_row, j_col)
                t_row = i_row * m + i_row + j_row
                t_col = i_col * n + i_col + j_col
                id = tot_cols * t_row + t_col
                # print('id', id)
                ax = fig.add_subplot(tot_rows,
                                     tot_cols, id + 1)

                ax.matshow(img, cmap=cmap)
                pyplot.xticks(numpy.array([]))
                pyplot.yticks(numpy.array([]))

    pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    pyplot.show()


#
# axes utils
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_m_by_n_heatmaps(images,
                         min_max_list,
                         m, n,
                         cmaps,
                         fig_size=(12, 12),
                         w_space=0.1,
                         h_space=0.1,
                         colorbars=False,
                         dpi=900,
                         save_path=None,
                         pdf=False):
    """
    Plot images in a mxn tiling
    """

    import seaborn

    seaborn.set_style('white')
    seaborn.set_context('poster')

    assert len(min_max_list) == len(images)

    print(len(images))
    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)

    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    for x in range(m):
        for y in range(n):
            id = n * x + y
            if id < len(images):
                # ax = fig.add_subplot(m, n, id + 1)
                ax = fig.add_subplot(gs1[id])
                if id > 0:
                    norm = None  # LogNorm(vmin=min_act, vmax=max_act)
                    print('min max', id, min_max_list[id])
                    img = ax.matshow(images[id],
                                     cmap=cmaps[id],
                                     vmin=min_max_list[id][0],
                                     vmax=min_max_list[id][1],
                                     norm=norm)
                    pyplot.xticks(numpy.array([]))
                    pyplot.yticks(numpy.array([]))
                    pyplot.axis('off')
                    if colorbars:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        pyplot.colorbar(img, cax=cax)
                else:
                    img = ax.matshow(images[id], cmap=cmaps[id])
                    pyplot.xticks(numpy.array([]))
                    pyplot.yticks(numpy.array([]))
                    pyplot.axis('off')

    # pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()
    pyplot.show()


def array_2_mat(array, n_rows, n_cols):
    array = numpy.array(array, copy=True)
    return array.reshape(n_rows, n_cols)


def tiling_sizes(n_images, n_cols=None):

    n_rows = None

    if n_cols is None:
        n_rows = int(numpy.sqrt(n_images))
        n_cols = n_rows
    else:
        n_rows = max(n_images // n_cols, 1)

    rem_tiles = n_images - n_rows * n_cols

    if rem_tiles > 0:
        n_rem_rows, n_rem_cols = tiling_sizes(rem_tiles, n_cols)
        return n_rows + n_rem_rows, n_cols

    return n_rows, n_cols


def scope_histogram(spn,
                    fig_size=(12, 5),
                    dpi=900,
                    ylim=None,
                    xlim=None,
                    save_path=None,
                    pdf=False,
                    show=True):

    import seaborn

    seaborn.set_style('white')
    # seaborn.despine(trim=True)
    seaborn.set_context('poster', font_scale=1.8)

    scope_dict = defaultdict(list)
    for node in spn.top_down_nodes():
        scope = None
        if hasattr(node, 'var_scope'):
            scope = node.var_scope
        elif hasattr(node, 'var'):
            scope = frozenset(node.var)

        scope_dict[len(scope)].append(node)

    max_scope_len = max(scope_dict.keys())
    scope_list = [0] * max_scope_len
    for scope_len, nodes in scope_dict.items():
        scope_list[scope_len - 1] = len(nodes)

    print(scope_list)

    fig, ax = pyplot.subplots(figsize=fig_size)
    # ax.bar(numpy.arange(max_scope_len),
    #        scope_list,
    #        log=True)

    # width = 3e-3
    width = 0.1
    for i in range(0, len(scope_list)):
        # x_pos = [10 ** (numpy.log10(i) - width),
        #          10 ** (numpy.log10(i) - width),
        #          10 ** (numpy.log10(i) + width),
        #          10 ** (numpy.log10(i) + width)]
        x_pos = [i - width + 1, i - width + 1, i + width + 1, i + width + 1]
        y_pos = [0,
                 scope_list[i],
                 scope_list[i],
                 0]
        ax.fill(x_pos,
                y_pos, 'black')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    else:
        ax.set_xlim([-1, len(scope_list) + 1])

    if ylim:
        ax.set_ylim([0.1, ylim])

    pyplot.xlabel('scope length')
    pyplot.ylabel('# nodes')
    pyplot.tight_layout()

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    if show:
        pyplot.show()
    # rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

    return scope_list


def scope_maps(spns,
               height=20,
               fig_size=(12, 4),
               dpi=900,
               # cmap=matplotlib.cm.jet,
               cmap=matplotlib.cm.binary,
               min_val=None,
               max_val=None,
               xlim=None,
               w_space=0.0,
               h_space=1.1,
               save_path=None,
               pdf=False):

    import seaborn

    seaborn.set_style('white')
    seaborn.despine(trim=True)
    seaborn.set_context('poster',  font_scale=1.8)

    scope_lists = []
    for spn in spns:
        scope_dict = defaultdict(list)
        for node in spn.top_down_nodes():
            scope = None
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = frozenset(node.var)

            scope_dict[len(scope)].append(node)

        max_scope_len = max(scope_dict.keys())
        scope_list = [0] * max_scope_len
        for scope_len, nodes in scope_dict.items():
            scope_list[scope_len - 1] = len(nodes)

        print(scope_list)
        scope_lists.append(scope_list)

    m = len(spns)
    n = 1
    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)

    if xlim is None:
        xlim = len(scope_lists[0])

    height_val = xlim * (fig_size[1] / len(spns)) / fig_size[0]
    step = 99 if xlim // 10 > 10 else 29

    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    for i in range(m):
        norm = None
        ax = fig.add_subplot(gs1[i])
        matrix_map = numpy.log10(numpy.array(scope_lists[i][:xlim]) + 1).reshape(1, xlim)
        matrix_map = numpy.lib.pad(matrix_map, ((0, 0), (5, 5)), 'constant')
        matrix_map = numpy.repeat(matrix_map, height_val, axis=0)
        print(matrix_map)
        img = ax.matshow(matrix_map,
                         cmap=cmap,
                         vmin=min_val,
                         vmax=max_val,
                         norm=norm)
        pyplot.xticks(numpy.array([]))
        pyplot.yticks(numpy.array([]))
        # pyplot.axis('off')

    ax.set_xticks(numpy.arange(1, xlim, step))
    ax.set_xlabel('scope length')
    # pyplot.ylabel('# nodes')
    # pyplot.tight_layout()

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    pyplot.show()
    # rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

    return scope_list


def scope_map_layerwise(spn,
                        fig_size=(20, 1),
                        dpi=900,
                        # cmap=matplotlib.cm.jet,
                        cmap=matplotlib.cm.binary,
                        xlim=None,
                        w_space=-100.0,
                        h_space=-100.0,
                        save_path=None,
                        pdf=False,
                        show=True):

    import seaborn

    cmap.set_under('white')

    seaborn.set_style('white')
    # seaborn.despine(trim=True)
    seaborn.set_context('poster',  font_scale=1.8)

    max_scope_len = max([len(node.var_scope) for node in spn.top_down_nodes()
                         if hasattr(node, 'var_scope')])
    scope_lists = []
    for layer in spn.bottom_up_layers():
        scope_dict = defaultdict(list)
        for node in layer.nodes():
            scope = None
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = frozenset(node.var)

            scope_dict[len(scope)].append(node)

        # max_scope_len = max(scope_dict.keys())
        scope_list = [0] * max_scope_len
        for scope_len, nodes in scope_dict.items():
            scope_list[scope_len - 1] = len(nodes)

        print(scope_list)
        scope_lists.append(scope_list)

    m = len(scope_lists)
    n = 1
    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)

    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    maps = []

    min_val = numpy.Inf
    max_val = -numpy.inf

    eps = numpy.spacing(0.0)

    norm = None
    if xlim is None:
        xlim = len(scope_lists[0])
    height_val = max(1, xlim * (fig_size[1] / len(scope_lists)) / fig_size[0])
    step = 100 if xlim // 10 > 50 else 30
    print('height:', height_val, xlim, step)
    for i in range(m):
        # print(scope_lists[i])

        # matrix_map = numpy.log10(numpy.array(scope_lists[i][:xlim])).reshape(1, xlim)
        matrix_map = numpy.zeros(xlim)
        matrix_map[numpy.array(scope_lists[i][:xlim]) > 0] = 1
        matrix_map = matrix_map.reshape(1, xlim)
        matrix_map = numpy.lib.pad(matrix_map, ((0, 0), (5, 5)), 'constant')
        matrix_map = numpy.repeat(matrix_map, height_val, axis=0)
        # print(matrix_map)
        maps.append(matrix_map)
        min_val = min([matrix_map.min(), min_val])
        max_val = max([matrix_map.max(), max_val])

    print('minmax', min_val, max_val)

    show_all_y_labels = True if len(maps) < 10 else False
    n_layers = len(maps)
    for i in range(n_layers):
        ax = fig.add_subplot(gs1[i], frameon=False)
        # if i > 0:
        #     step = 100 if xlim // 10 > 10 else 30
        # matrix_map = numpy.repeat(maps[i], height_val, axis=0)
        img = ax.matshow(maps[i],
                         cmap=cmap,
                         vmin=min_val + eps,
                         vmax=max_val,
                         norm=norm)
        pyplot.xticks(numpy.array([]))
        pyplot.yticks(numpy.array([]), rotation=90)
        # ax.yaxis.set_rotate_label(False)
        if show_all_y_labels or i % 2 == 1:
            ax.set_ylabel(str(n_layers - i - 1), rotation=0, ha='right', va='center')
        # pyplot.axis('off')
        if i == 0:
            ticks = [1] + [i for i in range(step, xlim, step)]  # + [xlim]
            ax.set_xticks(ticks)

    fig.text(0.03, 0.5, 'depth', rotation=90, ha='center', va='center', fontsize=32)
    # ax.set_xticks(numpy.arange(1, xlim, step))
    ax.set_xlabel('scope length')
    # pyplot.ylabel('# nodes')
    # pyplot.tight_layout()
    # fig.subplots_adjust(wspace=0, hspace=0)
    if save_path:
        fig.savefig(save_path + '.svg', bbox_inches='tight', pad_inches=0)
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig, bbox_inches='tight', pad_inches=0)
            pp.close()

    if show:
        pyplot.show()
    # rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

    return scope_list


def multiple_scope_histogram(spns,
                             fig_size=(14, 5),
                             dpi=900,
                             save_path=None,
                             y_log=True,
                             colors=None,
                             pdf=False):

    import seaborn

    seaborn.set_style('white')
    # seaborn.despine(trim=True)
    seaborn.set_context('poster')

    if not colors:
        # colors = seaborn.color_palette("husl")
        colors = ['red', 'green', 'black']

    n_ticks = 0

    spn_scope_lists = []
    for spn in spns:
        scope_dict = defaultdict(list)
        for node in spn.top_down_nodes():
            scope = None
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = frozenset(node.var)

            scope_dict[len(scope)].append(node)

        #
        # assuming all spns to have the same scope
        max_scope_len = max(scope_dict.keys())
        n_ticks = max(max_scope_len, 0)
        scope_list = [0] * max_scope_len
        for scope_len, nodes in scope_dict.items():
            scope_list[scope_len - 1] = len(nodes)

        print(scope_list)
        spn_scope_lists.append(scope_list)

    n_histograms = len(spns)
    bin_width = 1 / (n_histograms + 1)
    bins = [[i + j * bin_width for i in range(n_ticks)]
            for j in range(1, n_histograms + 1)]

    fig, ax = pyplot.subplots(figsize=fig_size)

    width = 0.1

    for i, histogram in enumerate(spn_scope_lists):
        # ax.bar(bins[i], spn_scope_lists[i], width=bin_width,
        #        facecolor=colors[i], edgecolor="none",
        #        log=y_log)

        # seaborn.despine()
        for j in range(0, len(scope_list)):
            # x_pos = [10 ** (numpy.log10(i) - width),
            #          10 ** (numpy.log10(i) - width),
            #          10 ** (numpy.log10(i) + width),
            #          10 ** (numpy.log10(i) + width)]
            x_pos = [bins[i][j] - width, bins[i][j] - width,
                     bins[i][j] + width, bins[i][j] + width]
            y_pos = [0,
                     spn_scope_lists[i][j],
                     spn_scope_lists[i][j],
                     0]
            ax.fill(x_pos,
                    y_pos, colors[i])
        if y_log:
            ax.set_yscale('log')
            # ax.set_xscale('log')
        ax.set_xlim([0, len(scope_list) + 1])

    pyplot.xlabel('scope length')
    pyplot.ylabel('# nodes')
    pyplot.tight_layout()

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    pyplot.show()


def layer_scope_histogram(spn,
                          m, n,
                          fig_size=(16, 16),
                          save_path=None,
                          pdf=False):

    fig = pyplot.figure(figsize=fig_size)

    scope_dict = defaultdict(lambda: defaultdict(list))

    layer_list = []
    for layer in spn.bottom_up_layers():
        layer_list.append(layer)
        for node in layer.nodes():
            scope = None
            if hasattr(node, 'var_scope'):
                scope = node.var_scope
            elif hasattr(node, 'var'):
                scope = frozenset(node.var)

            scope_dict[layer][len(scope)].append(node)

    max_scope_len = max([max(scope_dict[l].keys()) for l in scope_dict])
    max_val = max([max([len(n) for n in scope_dict[l].items()]) for l in scope_dict])
    for i, layer in enumerate(layer_list):
        scope_list = [0] * max_scope_len
        for scope_len, nodes in scope_dict[layer].items():
            scope_list[scope_len - 1] = len(nodes)

        print(scope_list)

        if i < m * n:
            ax = fig.add_subplot(m, n, i + 1)
            ax.bar(numpy.arange(max_scope_len),
                   scope_list,
                   log=True)
            ax.set_ylim(top=max_val)

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()

    pyplot.show()
    # rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)


def visualize_node_activations_for_instance(spn,
                                            nodes,
                                            instance,
                                            marg_mask=None,
                                            mean=False,
                                            hard=False,
                                            fig_size=(10, 10),
                                            n_rows=28, n_cols=28,
                                            dtype=float,
                                            cmap=matplotlib.cm.spectral,
                                            save_path=None,
                                            pdf=False):
    """
    Given an SPN and an instance, return a same shape instance
    containing the activations of all nodes, summed by scopes
    """

    import seaborn

    seaborn.set_style('white')
    seaborn.despine(trim=True)

    assert instance.ndim == 1

    n_features = len(instance)
    activations = numpy.zeros(n_features, dtype=dtype)
    var_counter = Counter()

    #
    # marginalizing?
    if marg_mask is not None:
        instance = numpy.array(instance, copy=True)
        instance[numpy.logical_not(marg_mask)] = MARG_IND

    # print(instance, type(instance))
    # instance = instance.astype(numpy.int32)
    # print(instance, type(instance), instance.shape)
    #
    # evaluate it bottom, up
    res, = spn.single_eval(instance)

    #
    # then gather the node activation vals
    for node in nodes:
        val = numpy.exp(node.log_val)
        scope = None
        if hasattr(node, 'var_scope'):
            scope = node.var_scope
        elif hasattr(node, 'var'):
            scope = [node.var]

        #
        # accumulating scope
        for var in scope:
            var_counter[var] += 1
            if hard:
                activations[var] += 1
            else:
                # activations[var] += (val * len(scope))
                activations[var] += (val / len(scope))
                # if instance[var] == 1:
                #     activations[var] += val
                # else:
                #     activations[var] += (1 - val)

    if mean:
        for i in range(n_features):
            activations[i] /= var_counter[i]

    fig, ax = pyplot.subplots(figsize=fig_size)
    activation_image = array_2_mat(activations,
                                   n_rows=n_rows,
                                   n_cols=n_cols)
    cax = ax.matshow(activation_image, cmap=cmap)
    fig.colorbar(cax)

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()
    pyplot.show()

    return activations


def visualize_marginalizations_for_instance(spn,
                                            instance,
                                            all_ones=False,
                                            exp=False,
                                            fig_size=(10, 10),
                                            n_rows=28, n_cols=28,
                                            dtype=float,
                                            cmap=matplotlib.cm.spectral,
                                            save_path=None,
                                            pdf=False):
    """
    Given an SPN and an instance, return a same shape instance
    containing the activations of all nodes, summed by scopes
    """

    import seaborn

    seaborn.set_style('white')
    seaborn.despine(trim=True)

    assert instance.ndim == 1

    n_features = len(instance)
    marg_data = numpy.zeros(n_features, dtype=instance.dtype)

    marginalizations = numpy.zeros(n_features, dtype=dtype)

    for i in range(n_features):
        marg_data.fill(MARG_IND)
        if all_ones:
            marg_data[i] = 1
        else:
            marg_data[i] = instance[i]
        #
        # evaluate it bottom, up
        res, = spn.single_eval(marg_data)

        if exp:
            res = numpy.exp(res)

        marginalizations[i] = res

    fig, ax = pyplot.subplots(figsize=fig_size)
    activation_image = array_2_mat(marginalizations,
                                   n_rows=n_rows,
                                   n_cols=n_cols)
    cax = ax.matshow(activation_image, cmap=cmap)
    fig.colorbar(cax)

    if save_path:
        fig.savefig(save_path + '.svg')
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig)
            pp.close()
    pyplot.show()

    return marginalizations


def visualize_feature_importances(feature_weights,
                                  feature_info=None,
                                  mean=False,
                                  aggr_func=max,
                                  # fig_size=(10, 10),
                                  n_rows=28, n_cols=28,
                                  dtype=float,
                                  # cmap=matplotlib.cm.spectral,
                                  # save_path=None,
                                  # pdf=False,
                                  # use_seaborn=False
                                  ):
    """
    Given an SPN and an instance, return a same shape instance
    containing the activations of all nodes, summed by scopes
    """

    # if use_seaborn:
    #     import seaborn

    #     seaborn.set_style('white')
    #     seaborn.despine(trim=True)

    n_features = len(feature_weights)

    feature_scope_assoc = None
    if feature_info is None:
        feature_scope_assoc = {i: {i} for i in range(n_features)}
        logging.info('No feature info provided, building 1-to-1 assoc')
    else:
        feature_scope_assoc = {info.feature_id: info.node_scope for info in feature_info}

    n_orig_features = n_rows * n_cols
    logging.info('Looking to visualize into a {}-d space (from {} features)'.format(n_orig_features,
                                                                                    n_features))
    activations = numpy.zeros(n_orig_features)
    var_counter = Counter()
    scope_counter = Counter()

    for i, f in enumerate(feature_weights):
        scope = feature_scope_assoc[i]

        for var in scope:
            var_counter[var] += 1
            scope_counter[var] += (len(scope))
            # activations[var] += f
            activations[var] = aggr_func([f, activations[var]])

    if mean:
        print('scope count\n', [var_counter[k] for k in range(10)])
        print('scope vals\n', activations[:10])
        for i in range(n_orig_features):
            # activations[i] = (activations[i] * var_counter[i]) / scope_counter[i]
            activations[i] /= var_counter[i]

        print('norm scope vals\n', activations[:10])

    # fig, ax = pyplot.subplots(figsize=fig_size)
    activation_image = array_2_mat(activations,
                                   n_rows=n_rows,
                                   n_cols=n_cols)
    # cax = ax.matshow(activation_image, cmap=cmap)
    # fig.colorbar(cax)

    # if save_path:
    #     fig.savefig(save_path + '.svg')
    #     if pdf:
    #         pp = PdfPages(save_path + '.pdf')
    #         pp.savefig(fig)
    #         pp.close()
    # pyplot.show()

    return activation_image


def plot_heatmap(matrix,
                 canvas_size,
                 colorbar=False,
                 cmap=matplotlib.cm.jet,
                 dpi=300,
                 colorbar_label_size=5.5,
                 colorbar_n_ticks=7,
                 cbar_linewidth=0,
                 save_path=None,
                 pdf=False,
                 show=True,
                 use_seaborn=False):

    if use_seaborn:
        import seaborn

        seaborn.set_style('white')
        seaborn.despine(trim=True)

    print(canvas_size)
    fig, ax = pyplot.subplots(figsize=canvas_size, dpi=dpi)
    img = ax.matshow(matrix,
                     cmap=cmap,
                     # vmin=min_max_list[id][0],
                     # vmax=min_max_list[id][1],
                     norm=None)
    pyplot.xticks(numpy.array([]))
    pyplot.yticks(numpy.array([]))
    pyplot.axis('off')
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05, )
        cbar = pyplot.colorbar(img, cax=cax, drawedges=True)
        cbar.ax.tick_params(labelsize=colorbar_label_size, pad=2)
        # cbar.ax.yaxis.set_ticks_position('left')
        cbar.locator = ticker.MaxNLocator(nbins=colorbar_n_ticks)
        cbar.update_ticks()
        cbar.dividers.set_linewidth(cbar_linewidth)
        cbar.outline.set_linewidth(cbar_linewidth)

    pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path + '.svg', bbox_inches='tight', pad_inches=0.02)
        if pdf:
            pp = PdfPages(save_path + '.pdf')
            pp.savefig(fig, bbox_inches='tight', pad_inches=0.02)
            pp.close()

    if show:
        pyplot.show()


if __name__ == '__main__':

    labels = [i for i in range(-10, 10)]
    points = [numpy.exp(i) for i in labels]

    visualize_curves([(labels, points)], labels=['a', 'b'])
