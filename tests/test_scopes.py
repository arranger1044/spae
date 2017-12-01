from scopes import ScopeSubset
from scopes import ScopeGraph
from scopes import ScopePartition
from scopes import Region
from scopes import create_poon_region_graph

from nose.tools import raises


def test_scope_disjointness():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [0, 4, 9]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [9, 1, 5]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    vars_4 = []
    scope_4 = ScopeSubset(vars_4)
    print(scope_4)

    assert scope_1.disjoint(scope_2)
    assert scope_1.disjoint(scope_4)
    assert not scope_1.disjoint(scope_3)
    #
    # is it symmetric?
    assert scope_3.disjoint(scope_4)
    assert not scope_2.disjoint(scope_3)
    assert scope_2.disjoint(scope_1)


def test_scope_union():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [0, 4, 9]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [9, 1, 5]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    vars_4 = []
    scope_4 = ScopeSubset(vars_4)
    print(scope_4)

    assert scope_4.union(scope_3) == scope_3
    assert not scope_3.union(scope_1).disjoint(scope_2)


def test_scope_equality():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [3, 2, 2, 1]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    assert scope_1.id != scope_2.id
    assert scope_1 == scope_2
    assert hash(scope_1) == hash(scope_2)

    vars_4 = []
    scope_4 = ScopeSubset(vars_4)
    print(scope_4)

    vars_5 = []
    scope_5 = ScopeSubset(vars_5)
    print(scope_5)
    assert scope_4.id != scope_5.id
    assert scope_4 == scope_5

    #
    # what about representing vars as sets?
    vars_6 = set([3, 2, 1])
    scope_6 = ScopeSubset(vars_6)
    print(scope_6)

    assert scope_1.id != scope_6.id
    assert scope_1 == scope_6
    assert hash(scope_1) == hash(scope_6)


@raises(ValueError)
def test_partition_creation_I():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [3, 2, 2, 1]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2])
    print(partition)


@raises(ValueError)
def test_partition_creation_II():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [1, 2, 3, 4, 5]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2],
                               scope=scope_3)
    print(partition)


def test_partition_creation_III():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [0, 1, 2, 3, 4, 5]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2],
                               scope=scope_3)
    print(partition)


def test_partition_creation_IIII():
    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2])
    print(partition)


@raises(ValueError)
def test_partition_add_scope_I():

    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2])
    print(partition)

    vars_3 = [0, 1, 2, 3, 4, 5]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    partition.add_scope(scope_3)


def test_partition_add_scope_II():

    #
    # creating different scopes
    vars_1 = [1, 2, 3]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    #
    # a partition over overlapping scopes
    partition = ScopePartition(scopes=[scope_1, scope_2])
    print(partition)

    vars_3 = [6, 9]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    partition.add_scope(scope_3)
    print(partition)


def test_partition_equality():
    #
    # creating different scopes
    vars_1 = [0, 1, 2, 3, 4, 5]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [2, 1, 3]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    #
    # and a partition over them
    partition_1 = ScopePartition(scopes=[scope_3, scope_2])

    partition_2 = ScopePartition(scopes=[scope_3, scope_2], scope=scope_1)

    assert partition_1 == partition_2

    #
    # creating the same scopes
    vars_4 = [0, 1, 2, 3, 4, 5]
    scope_4 = ScopeSubset(vars_4)
    print(scope_4)

    vars_5 = [5, 0, 4]
    scope_5 = ScopeSubset(vars_5)
    print(scope_5)

    vars_6 = [2, 1, 3]
    scope_6 = ScopeSubset(vars_6)
    print(scope_6)

    partition_3 = ScopePartition(scopes=[scope_5, scope_6])

    partition_4 = ScopePartition(scopes=[scope_5, scope_6], scope=scope_4)

    assert partition_1 == partition_3
    assert partition_4 == partition_3
    assert partition_1 == partition_4

    vars_7 = [0, 4]
    scope_7 = ScopeSubset(vars_7)
    print(scope_7)

    vars_8 = [1, 2, 3, 5]
    scope_8 = ScopeSubset(vars_8)
    print(scope_8)

    #
    # now a different one (by top scope)
    partition_5 = ScopePartition(scopes=[scope_7, scope_6])

    assert partition_1 != partition_5
    assert partition_3 != partition_5

    #
    # now different by partition
    partition_6 = ScopePartition(scopes=[scope_7, scope_8])

    assert partition_1 != partition_6
    assert partition_3 != partition_6
    assert partition_5 != partition_6

    #
    #
    # now testing the hashing
    part_set = set()
    part_set.add(partition_1)
    part_set.add(partition_2)
    part_set.add(partition_3)
    part_set.add(partition_4)
    part_set.add(partition_5)
    part_set.add(partition_6)

    assert len(part_set) == 3

    unique_ids = {p.id for p in [partition_1, partition_5, partition_6]}
    for p in part_set:
        print(p)
        assert p.id in unique_ids


def test_scope_graph_creation_I():

    #
    # creating different scopes
    vars_1 = [0, 1, 2, 3, 4, 5]
    scope_1 = ScopeSubset(vars_1)
    print(scope_1)

    vars_2 = [5, 0, 4]
    scope_2 = ScopeSubset(vars_2)
    print(scope_2)

    vars_3 = [2, 1, 3]
    scope_3 = ScopeSubset(vars_3)
    print(scope_3)

    partition_1 = ScopePartition(scopes=[scope_2, scope_3],
                                 scope=scope_1)

    vars_4 = [0, 4]
    scope_4 = ScopeSubset(vars_4)
    print(scope_4)

    vars_5 = [5]
    scope_5 = ScopeSubset(vars_5)
    print(scope_5)

    partition_2 = ScopePartition(scopes=[scope_4, scope_5],
                                 scope=scope_2)

    vars_6 = [0, 2, 5]
    scope_6 = ScopeSubset(vars_6)
    print(scope_6)

    vars_7 = [1, 3, 4]
    scope_7 = ScopeSubset(vars_7)
    print(scope_7)

    partition_3 = ScopePartition(scopes=[scope_6, scope_7],
                                 scope=scope_1)

    vars_8 = [0, 2]
    scope_8 = ScopeSubset(vars_8)
    print(scope_8)

    vars_9 = [5]
    scope_9 = ScopeSubset(vars_9)
    print(scope_9)

    graph = ScopeGraph(root_scope=scope_1)
    the_scopes = [scope_1, scope_2, scope_3, scope_4, scope_5, scope_6, scope_7]
    for s in the_scopes:
        graph.add_scope(s)

    scope_8 = graph.add_scope(scope_8)
    scope_9 = graph.add_scope(scope_9)
    partition_4 = ScopePartition(scopes=[scope_8, scope_9],
                                 scope=scope_6)

    assert scope_9.id == scope_5.id

    the_partitions = [partition_1, partition_2, partition_3, partition_4]
    for p in the_partitions:
        graph.add_partition(p)

    assert graph._scopes == [
        scope_1, scope_2, scope_3, scope_4, scope_5, scope_6, scope_7, scope_8]
    assert graph._partitions == [partition_1, partition_2, partition_3, partition_4]
    print(graph)

    graph_I = ScopeGraph(root_scope=scope_1,
                         scopes=[scope_1, scope_2, scope_3, scope_4, scope_5, scope_6, scope_7],
                         partitions=[partition_1, partition_2, partition_3])

    print('second')
    print(graph_I)


def test_region_creation():
    n_cols = 16
    n_rows = 16
    #
    # create a region for the whole region
    whole_region = Region(0, n_rows, 0, n_cols, n_rows, n_cols)
    whole_scope = frozenset([i for i in range(n_cols * n_rows)])

    print(whole_scope)
    print(whole_region.vars)
    assert whole_region.vars == whole_scope

    #
    # a region of a single element
    point_region = Region(0, 1, 0, 1, n_rows, n_cols)
    assert point_region.vars == frozenset([0])

    #
    # and the last one
    point_region = Region(n_rows - 1, n_rows, n_cols - 1, n_cols, n_rows, n_cols)
    assert point_region.vars == frozenset([n_rows * n_cols - 1])

    #
    # a long edge region
    left_edge = Region(0, n_rows, 0, 1, n_rows, n_cols)
    assert left_edge.vars == frozenset([i for i in range(0, n_cols * n_rows, n_cols)])


def test_get_horizontal_partitions():
    n_cols = 4
    n_rows = 4
    #
    # create a region for the whole region
    whole_region = Region(0, n_rows, 0, n_cols, n_rows, n_cols)

    #
    # cutting into single pixel regions
    regs = Region.get_horizontal_partitions(whole_region, skip=1)

    print(regs)

    assert len(regs) == (n_rows - 1)

    #
    # cutting into two pixels regions
    skip = 2
    regs = Region.get_horizontal_partitions(whole_region, skip=skip)
    print(regs)

    assert len(regs) == (n_rows // skip - 1)

    #
    # cutting with a skip that is too large
    skip = n_rows
    regs = Region.get_horizontal_partitions(whole_region, skip=skip)
    print(regs)

    assert len(regs) == 0


def test_get_vertical_partitions():
    n_cols = 4
    n_rows = 4
    #
    # create a region for the whole region
    whole_region = Region(0, n_rows, 0, n_cols, n_rows, n_cols)

    #
    # cutting into single pixel regions
    regs = Region.get_vertical_partitions(whole_region, skip=1)

    print(regs)

    assert len(regs) == (n_cols - 1)

    #
    # cutting into two pixels regions
    skip = 2
    regs = Region.get_vertical_partitions(whole_region, skip=skip)
    print(regs)

    assert len(regs) == (n_cols // skip - 1)

    #
    # cutting with a skip that is too large
    skip = n_cols
    regs = Region.get_vertical_partitions(whole_region, skip=skip)
    print(regs)

    assert len(regs) == 0


def test_create_poon_region_graph():

    n_cols = 64
    n_rows = 64
    coarse = 4
    #
    # create initial region
    root_region = Region.create_whole_region(n_rows, n_cols)

    region_graph = create_poon_region_graph(root_region, coarse=coarse)

    # print(region_graph)
    print('# partitions', region_graph.n_partitions())
    print('# regions', region_graph.n_scopes())

from scopes import build_linked_spn_from_scope_graph
from scopes import get_scope_graph_from_linked_spn

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode

from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer

from spn.linked.spn import Spn as LinkedSpn

from spn import MARG_IND

import numpy


def test_build_linked_spn_from_scope_graph():

    #
    # creating a region graph as an input scope graph
    n_cols = 2
    n_rows = 2
    coarse = 2
    #
    # create initial region
    root_region = Region.create_whole_region(n_rows, n_cols)

    region_graph = create_poon_region_graph(root_region, coarse=coarse)

    # print(region_graph)
    print('# partitions', region_graph.n_partitions())
    print('# regions', region_graph.n_scopes())

    print(region_graph)

    #
    #
    k = 2
    spn = build_linked_spn_from_scope_graph(region_graph, k)

    print(spn)

    print(spn.stats())

    #
    # back to the scope graph
    root_layer = list(spn.root_layer().nodes())
    assert len(root_layer) == 1
    root = root_layer[0]

    scope_graph = get_scope_graph_from_linked_spn(root)
    print(scope_graph)

    assert scope_graph == region_graph

    #
    # building an spn from scratch
    #
    # building leaf nodes
    n_vars = 4
    vars = [0, 1, 2, 3]
    leaves = [CategoricalIndicatorNode(var, val) for var in range(n_vars) for val in [0, 1]]
    input_layer = CategoricalIndicatorLayer(nodes=leaves, vars=vars)

    #
    # building root
    root_node = SumNode(var_scope=frozenset(vars))
    root_layer = SumLayer([root_node])

    #
    # building product nodes
    prod_list_1 = [ProductNode(var_scope=vars) for i in range(4)]
    prod_list_2 = [ProductNode(var_scope=vars) for i in range(4)]
    prod_nodes_1 = prod_list_1 + prod_list_2
    product_layer_1 = ProductLayer(prod_nodes_1)

    for p in prod_nodes_1:
        root_node.add_child(p, 1.0 / len(prod_nodes_1))

    #
    # build sum nodes
    sum_list_1 = [SumNode() for i in range(2)]
    sum_list_2 = [SumNode() for i in range(2)]
    sum_list_3 = [SumNode() for i in range(2)]
    sum_list_4 = [SumNode() for i in range(2)]

    sum_layer_2 = SumLayer(sum_list_1 + sum_list_2 + sum_list_3 + sum_list_4)

    sum_pairs = []
    for s_1 in sum_list_1:
        for s_2 in sum_list_2:
            sum_pairs.append((s_1, s_2))

    for p, (s_1, s_2) in zip(prod_list_1, sum_pairs):
        p.add_child(s_1)
        p.add_child(s_2)

    sum_pairs = []
    for s_3 in sum_list_3:
        for s_4 in sum_list_4:
            sum_pairs.append((s_3, s_4))

    for p, (s_3, s_4) in zip(prod_list_2, sum_pairs):
        p.add_child(s_3)
        p.add_child(s_4)

    #
    # again product nodes
    prod_list_3 = [ProductNode() for i in range(4)]
    prod_list_4 = [ProductNode() for i in range(4)]
    prod_list_5 = [ProductNode() for i in range(4)]
    prod_list_6 = [ProductNode() for i in range(4)]

    product_layer_3 = ProductLayer(prod_list_3 + prod_list_4 + prod_list_5 + prod_list_6)

    for s in sum_list_1:
        for p in prod_list_3:
            s.add_child(p, 1.0 / len(prod_list_3))

    for s in sum_list_2:
        for p in prod_list_4:
            s.add_child(p, 1.0 / len(prod_list_4))

    for s in sum_list_3:
        for p in prod_list_5:
            s.add_child(p, 1.0 / len(prod_list_5))

    for s in sum_list_4:
        for p in prod_list_6:
            s.add_child(p, 1.0 / len(prod_list_6))

    #
    # build sum nodes
    sum_list_5 = [SumNode() for i in range(2)]
    sum_list_6 = [SumNode() for i in range(2)]
    sum_list_7 = [SumNode() for i in range(2)]
    sum_list_8 = [SumNode() for i in range(2)]

    sum_layer_4 = SumLayer(sum_list_5 + sum_list_6 + sum_list_7 + sum_list_8)

    sum_pairs = []
    for s_5 in sum_list_5:
        for s_7 in sum_list_7:
            sum_pairs.append((s_5, s_7))

    for p, (s_5, s_7) in zip(prod_list_3, sum_pairs):
        p.add_child(s_5)
        p.add_child(s_7)

    sum_pairs = []
    for s_6 in sum_list_6:
        for s_8 in sum_list_8:
            sum_pairs.append((s_6, s_8))

    for p, (s_6, s_8) in zip(prod_list_4, sum_pairs):
        p.add_child(s_6)
        p.add_child(s_8)

    sum_pairs = []
    for s_5 in sum_list_5:
        for s_6 in sum_list_6:
            sum_pairs.append((s_5, s_6))

    for p, (s_5, s_6) in zip(prod_list_5, sum_pairs):
        p.add_child(s_5)
        p.add_child(s_6)

    sum_pairs = []
    for s_7 in sum_list_7:
        for s_8 in sum_list_8:
            sum_pairs.append((s_7, s_8))

    for p, (s_7, s_8) in zip(prod_list_6, sum_pairs):
        p.add_child(s_7)
        p.add_child(s_8)

    #
    # linking to input layer
    for s in sum_list_5:
        for i in leaves[0:2]:
            s.add_child(i, 0.5)

    for s in sum_list_6:
        for i in leaves[2:4]:
            s.add_child(i, 0.5)

    for s in sum_list_7:
        for i in leaves[4:6]:
            s.add_child(i, 0.5)

    for s in sum_list_8:
        for i in leaves[6:]:
            s.add_child(i, 0.5)

    lspn = LinkedSpn(input_layer=input_layer, layers=[sum_layer_4,
                                                      product_layer_3,
                                                      sum_layer_2,
                                                      product_layer_1,
                                                      root_layer])
    print(lspn)
    print(lspn.stats())

    #
    # trying to evaluate them
    input_vec = numpy.array([[1., 1., 1., 0.],
                             [0., 0., 0., 0.],
                             [0., 1., 1., 0.],
                             [MARG_IND, MARG_IND, MARG_IND, MARG_IND]]).T

    res = spn.eval(input_vec)
    print('First evaluation')
    print(res)

    res = lspn.eval(input_vec)
    print('Second evaluation')
    print(res)


from spn.factory import compute_block_layer_depths


def test_compute_block_layer_depths_I():
    #
    # creating a region graph as an input scope graph
    n_cols = 2
    n_rows = 2
    coarse = 2
    #
    # create initial region
    root_region = Region.create_whole_region(n_rows, n_cols)

    region_graph = create_poon_region_graph(root_region, coarse=coarse)

    # print(region_graph)
    print('# partitions', region_graph.n_partitions())
    print('# regions', region_graph.n_scopes())

    print(region_graph)

    #
    #
    k = 2
    spn = build_linked_spn_from_scope_graph(region_graph, k)

    print(spn)

    print(spn.stats())

    depth_dict = compute_block_layer_depths(spn)

    for layer, depth in depth_dict.items():
        print(layer.id, depth)

    # 0 0
    # 1 5
    # 2 3
    # 3 3
    # 4 3
    # 5 3
    # 6 1
    # 7 1
    # 8 1
    # 9 1
    # 10 4
    # 11 4
    # 12 2
    # 13 2
    # 14 2
    # 15 2


def test_compute_block_layer_depths_II():

    input_layer = CategoricalIndicatorLayer([])

    sum_layer_1 = SumLayer([])

    prod_layer_21 = ProductLayer([])
    prod_layer_22 = ProductLayer([])

    sum_layer_3 = SumLayer([])

    prod_layer_41 = ProductLayer([])
    prod_layer_42 = ProductLayer([])

    sum_layer_5 = SumLayer([])

    #
    # linking them
    sum_layer_1.add_input_layer(input_layer)
    input_layer.add_output_layer(sum_layer_1)

    prod_layer_21.add_input_layer(sum_layer_1)
    prod_layer_22.add_input_layer(sum_layer_1)
    sum_layer_1.add_output_layer(prod_layer_21)
    sum_layer_1.add_output_layer(prod_layer_22)

    sum_layer_3.add_input_layer(prod_layer_21)
    sum_layer_3.add_input_layer(prod_layer_22)
    sum_layer_3.add_input_layer(input_layer)
    prod_layer_21.add_output_layer(sum_layer_3)
    prod_layer_22.add_output_layer(sum_layer_3)
    input_layer.add_output_layer(sum_layer_3)

    prod_layer_41.add_input_layer(sum_layer_3)
    prod_layer_41.add_input_layer(sum_layer_1)
    prod_layer_42.add_input_layer(sum_layer_3)
    prod_layer_42.add_input_layer(input_layer)
    sum_layer_3.add_output_layer(prod_layer_41)
    sum_layer_3.add_output_layer(prod_layer_42)
    sum_layer_1.add_output_layer(prod_layer_41)
    input_layer.add_output_layer(prod_layer_42)

    sum_layer_5.add_input_layer(prod_layer_41)
    sum_layer_5.add_input_layer(prod_layer_42)
    prod_layer_41.add_output_layer(sum_layer_5)
    prod_layer_42.add_output_layer(sum_layer_5)

    #
    # creating an SPN with unordered layer list
    spn = LinkedSpn(input_layer=input_layer, layers=[sum_layer_1,
                                                     sum_layer_3,
                                                     sum_layer_5,
                                                     prod_layer_21,
                                                     prod_layer_22,
                                                     prod_layer_41,
                                                     prod_layer_42])

    depth_dict = compute_block_layer_depths(spn)

    print(spn)

    for layer, depth in depth_dict.items():
        print(layer.id, depth)


from spn.factory import merge_block_layers_spn


def test_merge_block_layers_spn():

    #
    # creating a region graph as an input scope graph
    n_cols = 2
    n_rows = 2
    coarse = 2
    #
    # create initial region
    root_region = Region.create_whole_region(n_rows, n_cols)

    region_graph = create_poon_region_graph(root_region, coarse=coarse)

    # print(region_graph)
    print('# partitions', region_graph.n_partitions())
    print('# regions', region_graph.n_scopes())

    print(region_graph)

    #
    #
    k = 2
    spn = build_linked_spn_from_scope_graph(region_graph, k)

    print(spn)

    print(spn.stats())

    threshold = 0.0

    mod_spn = merge_block_layers_spn(spn, threshold)

    print(mod_spn)
