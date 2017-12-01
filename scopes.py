from collections import deque
from collections import defaultdict


class ScopeSubset(object):

    """
    The set of r.v.s over which a pdf is defined
    """

    _instance_id = 0

    def __init__(self, vars):
        """
        Representing the scope as a frozenset
        """
        self.vars = frozenset(vars)
        self.id = ScopeSubset._instance_id
        ScopeSubset._instance_id += 1

        #
        # linking to partitions
        self.partitions = []

    @classmethod
    def reset_index(cls):
        ScopeSubset._scope_index = {}
        ScopeSubset._instance_id = 0

    @classmethod
    def disjoint_scopes(cls, scope_1, scope_2):
        if scope_1.vars & scope_2.vars:
            return False
        else:
            return True

    @classmethod
    def _union(cls, scope_1, scope_2):
        return ScopeSubset(scope_1.vars.union(scope_2.vars))

    def union(self, other_scope):
        return ScopeSubset._union(self, other_scope)

    def disjoint(self, other_scope):
        return ScopeSubset.disjoint_scopes(self, other_scope)

    def add_partition(self, partition):
        self.partitions.append(partition)

    def is_atomic(self):
        return len(self.vars) == 1

    def __eq__(self, item):
        return self.vars == item.vars

    def __hash__(self):
        return hash(self.vars)

    def __repr__(self):
        var_str = ",".join([str(var) for var in self.vars])
        part_str = ",".join([str(p.id) for p in self.partitions])
        return "(s:{0}, vars:{{{1}}}, p:[{2}])\n".format(self.id, var_str, part_str)


class ScopePartition(object):

    """
    A partition over a scope, represented as a sequence of disjoint Scope(Subset)s
    Plus a global Scope(Subset) as the union of all Scope(Subset)s
    """

    _instance_id = 0

    def __init__(self, scopes=None, scope=None):
        """
        Optionally initing with a collection of disjoint scopes
        """
        if scopes:
            self.scopes = scopes
        else:
            self.scopes = []

        self.id = ScopePartition._instance_id
        ScopePartition._instance_id += 1

        #
        # computing the whole scope of the partition  as the union of the scopes
        _scope = set()
        for s in self.scopes:
            # check disjointness
            if s.vars & _scope:
                raise ValueError('Creating a partition over non-disjoint scopes',
                                 s, _scope)
            _scope = _scope.union(s.vars)

        if scope is None:
            self._scope = ScopeSubset(_scope)
        else:
            #
            # now the union shall give the whole scope
            if _scope != scope.vars:
                raise ValueError('The partition scope provided is not the union'
                                 'of the single scopes',
                                 scope)
            self._scope = scope

    def add_scope(self, scope):

        self.scopes.append(scope)

        if not scope.disjoint(self._scope):
            raise ValueError('Creating a partition over non disjoin scopes',
                             scope, self._scope)

        self._scope = self._scope.union(scope)

    def scope(self):
        return self._scope

    def __eq__(self, partition):
        return (len(self.scopes) == len(partition.scopes)
                and self.scope() == partition.scope()
                and {s for s in self.scopes} == {s for s in partition.scopes})

    def __hash__(self):
        return hash((self._scope.vars, frozenset([scope.vars for scope in self.scopes])))

    def __repr__(self):
        scopes_str = ",".join(str(s.id) for s in self.scopes)
        return "(p:{0} s:{1} <{2}>)\n".format(self.id,
                                              self._scope.id,
                                              scopes_str)


class ScopeGraph(object):

    """
    A scope graph is a bipartite graph of Scope(Subset)s and (Scope)Partitions
    Optionally rooted.

    This equals to the RegionGraph in Dennis2012
    """

    def __init__(self,
                 scopes=None,
                 partitions=None,
                 root_scope=None):
        """
        WRITEME
        """

        self._scope_id_dict = {}
        self._id_scope_dict = {}

        self._partition_id_dict = {}
        self._id_partition_dict = {}

        self._scopes = []
        if scopes:
            for s in scopes:
                #
                # resetting the partitions linked
                # TODO: is this good?
                s.partitions = []
                self.add_scope(s)

        self._partitions = []
        if partitions:
            for p in partitions:
                self.add_partition(p)

        self.root = root_scope

    def id_to_scope(self, id):
        return self._id_scope_dict[id]

    def scope_to_id(self, scope):
        return self._scope_id_dict[scope]

    def get_scope(self, scope):
        return self._id_scope_dict[self._scope_id_dict[scope]]

    def get_partition(self, partition):
        return self._id_partition_dict[self._partition_id_dict[partition]]

    def is_scope_present(self, scope):
        return scope in self._scope_id_dict

    def is_partition_present(self, partition):
        return partition in self._partition_id_dict

    def add_scope(self, scope):
        #
        # if not present we add it
        if scope not in self._scope_id_dict:
            # print('adding scope', scope)
            self._scope_id_dict[scope] = scope.id
            self._id_scope_dict[scope.id] = scope

            self._scopes.append(scope)
        #
        # else we simply return it
        # return self._id_scope_dict[self._scope_id_dict[scope]]
        return self.get_scope(scope)

    def add_partition(self, partition):
        """
        WRITEME
        """
        if partition not in self._partition_id_dict:

            self._partition_id_dict[partition] = partition.id
            self._id_partition_dict[partition.id] = partition

            scope = partition.scope()
            if scope in self._scope_id_dict:
                # scope = self._id_scope_dict[self._scope_id_dict[scope]]
                scope = self.get_scope(scope)

            # print('adding partitions {0} with scope {1}'.format(partition, scope))
            scope.add_partition(partition)

            self._partitions.append(partition)

        return self.get_partition(partition)

    def traverse_scopes(self,
                        root_scope=None,
                        yield_partitions=False,
                        order='bfs'):

        if not root_scope:
            root_scope = self.root

        scopes_to_process = deque()
        scopes_to_process.append(root_scope)

        def enqueue_scope(scope):
            scopes_to_process.append(scope)

        def stack_scope(scope):
            scopes_to_process.appendleft(scope)

        visited_scopes = set()

        add_scope = None
        if order == 'bfs':
            add_scope = enqueue_scope
        elif order == 'dfs':
            add_scope = stack_scope
        else:
            raise ValueError('Invalid traversing order', order)

        while scopes_to_process:

            current_scope = scopes_to_process.popleft()
            visited_scopes.add(current_scope)
            yield current_scope

            for current_partition in current_scope.partitions:

                if yield_partitions:
                    yield current_partition

                for part_scope in current_partition.scopes:
                    if part_scope not in visited_scopes:
                        add_scope(part_scope)
                        visited_scopes.add(part_scope)

    def n_nodes(self):
        return len(self._partitions) + len(self._scopes)

    def n_scopes(self):
        return len(self._scopes)

    def n_partitions(self):
        return len(self._partitions)

    def __eq__(self, scope_graph):

        equal = True
        for node_1, node_2 in zip(self.traverse_scopes(yield_partitions=True),
                                  scope_graph.traverse_scopes(yield_partitions=True)):
            equal = node_1 == node_2
            if not equal:
                break
        return equal

    def __repr__(self):
        trav_repr = " ".join([str(node) for node in self.traverse_scopes(order='bfs',
                                                                         yield_partitions=True)])
        return trav_repr


class Region(ScopeSubset):

    """
    A Region is a Scope(Subset) made by r.v.s that are spatially
    adjacent
    """

    @classmethod
    def id_from_coords(cls, i, j, n_cols):
        return i * n_cols + j

    @classmethod
    def from_region_to_var_ids(cls, a_1, a_2, b_1, b_2, tot_n_cols):
        """
        Translating a Region coordinate represntation
        to an id representation (set of ids)
        """

        n_cols = b_2 - b_1
        n_rows = a_2 - a_1
        starting_id = Region.id_from_coords(a_1, b_1, tot_n_cols)
        vars = set()
        for i in range(n_rows):
            for j in range(n_cols):
                vars.add(starting_id + j)
            starting_id += tot_n_cols
        return vars

    def __init__(self, x_1, x_2, y_1, y_2, image_n_rows, image_n_cols):
        """
        A region in an image (n_rows X n_cols) can be represented by 4 coordinates:
        (x_1, y_1) (x_2, y_2)
        """

        #
        # TODO: add consistency checking by assertions
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2

        self.width = y_2 - y_1
        self.height = x_2 - x_1

        self.image_n_rows = image_n_rows
        self.image_n_cols = image_n_cols

        vars = Region.from_region_to_var_ids(x_1, x_2, y_1, y_2, image_n_cols)
        super().__init__(vars)

    def is_coarse_region(self, base_res):
        return self.width <= base_res and self.height <= base_res

    @classmethod
    def get_vertical_partitions(cls, region, skip=1):

        assert skip > 0

        regions = []

        for i in range(region.y_1 + skip, region.y_2, skip):
            #
            # create two new regions
            left_region = Region(region.x_1,
                                 region.x_2,
                                 region.y_1,
                                 i,
                                 region.image_n_rows,
                                 region.image_n_cols)
            right_region = Region(region.x_1,
                                  region.x_2,
                                  i,
                                  region.y_2,
                                  region.image_n_rows,
                                  region.image_n_cols)
            regions.append((left_region, right_region))

            #
            # and a partition over them
            # partitions.add(ScopePartition(scopes=[left_region,
            #                                       right_region],
            #                               scope=region)
            #                              )

        return regions

    @classmethod
    def get_horizontal_partitions(cls, region, skip=1):

        assert skip > 0

        # partitions = []
        regions = []

        for i in range(region.x_1 + skip, region.x_2, skip):
            #
            # create two new regions
            top_region = Region(region.x_1,
                                i,
                                region.y_1,
                                region.y_2,
                                region.image_n_rows,
                                region.image_n_cols)

            bottom_region = Region(i,
                                   region.x_2,
                                   region.y_1,
                                   region.y_2,
                                   region.image_n_rows,
                                   region.image_n_cols)
            regions.append((top_region, bottom_region))

            #
            # and a partition over them
            # partitions.append(ScopePartition(scopes=[top_region,
            #                                          bottom_region],
            #                                  scope=region))

        return regions

    @classmethod
    def create_whole_region(cls, n_rows, n_cols):
        return cls(0, n_rows, 0, n_cols, n_rows, n_cols)

    def __repr__(self):
        var_str = ",".join([str(var) for var in self.vars])
        part_str = ",".join([str(p.id) for p in self.partitions])
        return "(r:{0}, [{1}, {2}, {3}, {4}], vars:{{{5}}}, p:[{6}])\n".format(self.id,
                                                                               self.x_1,
                                                                               self.y_1,
                                                                               self.x_2,
                                                                               self.y_2,
                                                                               var_str,
                                                                               part_str)


def create_poon_region_graph(region,
                             coarse):

    region_graph = ScopeGraph(root_scope=region)
    region_graph.add_scope(region)

    regions_to_process = deque()
    regions_to_process.append(region)

    while regions_to_process:

        #
        # get a region to process
        current_region = regions_to_process.popleft()

        #
        # is this a fine region?
        skip = coarse
        if current_region.is_coarse_region(coarse):
            skip = 1

        #
        # get all possible decompositions
        # horizontally and vertically
        regions_to_consider = Region.get_vertical_partitions(current_region,
                                                             skip=skip)

        regions_to_consider.extend(Region.get_horizontal_partitions(current_region,
                                                                    skip=skip))

        for region_1, region_2 in regions_to_consider:
            #
            # check wheter they have been already used in the region graph
            if not region_graph.is_scope_present(region_1):
                regions_to_process.append(region_1)

            if not region_graph.is_scope_present(region_2):
                regions_to_process.append(region_2)

            #
            # adding them to the graph, or retrieving the ones
            # already added
            # TODO: this may be made clearer
            region_1 = region_graph.add_scope(region_1)
            region_2 = region_graph.add_scope(region_2)

            #
            # now creating a partition
            partition = ScopePartition(scopes=[region_1, region_2], scope=current_region)
            region_graph.add_partition(partition)

    return region_graph

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode

from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer

from spn.linked.spn import Spn as LinkedSpn

import numpy

import itertools


def topological_layer_sort(layers):
    """
    layers is a sequence of layers
    """

    #
    #
    layers_dict = {layer: layer.input_layers for layer in layers}

    sorted_layers = []

    while layers_dict:

        acyclic = False
        temp_layers_dict = dict(layers_dict)
        for layer, descendants in temp_layers_dict.items():
            for desc_layer in descendants:
                if desc_layer in layers_dict:
                    break
            else:
                acyclic = True
                del layers_dict[layer]
                sorted_layers.append(layer)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return sorted_layers


def build_linked_spn_from_scope_graph(scope_graph, k, root_scope=None, feature_values=None):
    """
    Turning a ScopeGraph into an SPN by puttin k sum nodes for each scope
    and a combinatorial number of product nodes to wire the partition nodes

    This is the algorithm used in Poon2011 and is shown (and used) as BuildSPN in Dennis2012
    """

    if not root_scope:
        root_scope = scope_graph.root

    n_vars = len(root_scope.vars)
    if not feature_values:
        #
        # assuming binary r.v.s
        feature_values = [2 for _i in range(n_vars)]

    #
    # adding leaves
    leaves_dict = defaultdict(list)
    leaves_list = []
    for var in sorted(root_scope.vars):
        for var_val in range(feature_values[var]):
            leaf = CategoricalIndicatorNode(var, var_val)
            leaves_list.append(leaf)
            leaves_dict[var].append(leaf)

    input_layer = CategoricalIndicatorLayer(nodes=leaves_list, vars=list(sorted(root_scope.vars)))

    #
    # in a first pass we need to assign each scope/region k sum nodes
    sum_nodes_assoc = {}
    for r in scope_graph.traverse_scopes(root_scope=root_scope):

        num_sum_nodes = k

        if r == root_scope:
            num_sum_nodes = 1

        added_sum_nodes = [SumNode(var_scope=r.vars) for i in range(num_sum_nodes)]
        #
        # creating a sum layer
        sum_layer = SumLayer(added_sum_nodes)
        sum_nodes_assoc[r] = sum_layer

        #
        # if this is a univariate scope, we link it to leaves corresponding to its r.v.
        if r.is_atomic():
            single_rv = set(r.vars).pop()
            rv_leaves = leaves_dict[single_rv]
            uniform_weight = 1.0 / len(rv_leaves)
            for s in added_sum_nodes:
                for leaf in rv_leaves:
                    s.add_child(leaf, uniform_weight)
            #
            # linking to input layer
            sum_layer.add_input_layer(input_layer)
            input_layer.add_output_layer(sum_layer)

    layers = []
    #
    # looping again to add and wire product nodes
    for r in scope_graph.traverse_scopes(root_scope=root_scope):

        sum_layer = sum_nodes_assoc[r]
        layers.append(sum_layer)

        for p in r.partitions:

            sum_layer_descs = [sum_nodes_assoc[r_p] for r_p in p.scopes]
            sum_nodes_lists = [list(layer.nodes()) for layer in sum_layer_descs]
            num_prod_nodes = numpy.prod([len(r_p) for r_p in sum_nodes_lists])

            #
            # adding product nodes
            added_prod_nodes = [ProductNode(var_scope=r.vars) for i in range(num_prod_nodes)]
            #
            # adding product layer and linking
            prod_layer = ProductLayer(added_prod_nodes)
            sum_layer.add_input_layer(prod_layer)
            prod_layer.add_output_layer(sum_layer)
            for desc in sum_layer_descs:
                prod_layer.add_input_layer(desc)
                desc.add_output_layer(prod_layer)
            layers.append(prod_layer)

            #
            # linking to parents
            sum_nodes_parents = sum_layer.nodes()
            for sum_node in sum_nodes_parents:
                uniform_weight = 1.0 / (len(added_prod_nodes) * len(r.partitions))
                for prod_node in added_prod_nodes:
                    sum_node.add_child(prod_node, uniform_weight)
            #
            # linking to children
            sum_nodes_to_wire = list(itertools.product(*sum_nodes_lists))

            assert len(added_prod_nodes) == len(sum_nodes_to_wire)

            for prod_node, sum_nodes in zip(added_prod_nodes, sum_nodes_to_wire):
                for sum_node in sum_nodes:
                    prod_node.add_child(sum_node)

    #
    # toposort
    layers = topological_layer_sort(layers)

    spn = LinkedSpn(layers=layers, input_layer=input_layer)

    return spn


def get_scope_graph_from_linked_spn(root):
    """
    Build a scopegraph from a linked SPN
    """

    root_scope = ScopeSubset(root.var_scope)
    scope_graph = ScopeGraph(root_scope=root_scope)
    scope_graph.add_scope(root_scope)

    nodes_to_process = deque()
    nodes_to_process.append(root)

    while nodes_to_process:
        curr_node = nodes_to_process.popleft()

        #
        # just product nodes shall be fine
        parent_scope = None
        if isinstance(curr_node, ProductNode):
            parent_scope = ScopeSubset(curr_node.var_scope)
            # print('1', parent_scope)
            parent_scope = scope_graph.add_scope(parent_scope)
            # print('2', parent_scope)

        #
        # creating a region or scope partition
        if hasattr(curr_node, 'children'):
            child_nodes = curr_node.children
            nodes_to_process.extend(child_nodes)

            if parent_scope:
                child_scopes = []
                for scope in [ScopeSubset(node.var_scope) for node in child_nodes]:
                    child_scopes.append(scope_graph.add_scope(scope))
                #
                # create partition node
                partition = ScopePartition(scopes=child_scopes, scope=parent_scope)
                # print('3', partition)
                scope_graph.add_partition(partition)

    return scope_graph
