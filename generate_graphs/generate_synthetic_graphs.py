#Taken from https://github.com/RexYing/gnn-model-explainer

import os
import torch
import math
import random

import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

#import featgen
#import utils


def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


# Following GraphWave's representation of structural similarity


def clique(start, nb_nodes, nb_to_remove=0, role_start=0):
    """ Defines a clique (complete graph on nb_nodes nodes,
    with nb_to_remove  edges that will have to be removed),
    index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    nb_nodes    :    int correspondingraph to the nb of nodes in the clique
    role_start  :    starting index for the roles
    nb_to_remove:    int-- numb of edges to remove (unif at RDM)
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    a = np.ones((nb_nodes, nb_nodes))
    np.fill_diagonal(a, 0)
    graph = nx.from_numpy_matrix(a)
    edge_list = graph.edges().keys()
    roles = [role_start] * nb_nodes
    if nb_to_remove > 0:
        lst = np.random.choice(len(edge_list), nb_to_remove, replace=False)
        print(edge_list, lst)
        to_delete = [edge_list[e] for e in lst]
        graph.remove_edges_from(to_delete)
        for e in lst:
            print(edge_list[e][0])
            print(len(roles))
            roles[edge_list[e][0]] += 1
            roles[edge_list[e][1]] += 1
    mapping_graph = {k: (k + start) for k in range(nb_nodes)}
    graph = nx.relabel_nodes(graph, mapping_graph)
    return graph, roles


def cycle(start, len_cycle, role_start=0):
    """Builds a cycle graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    return graph, roles


def diamond(start, role_start=0):
    """Builds a diamond graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 6))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    graph.add_edges_from(
        [
            (start + 4, start),
            (start + 4, start + 1),
            (start + 4, start + 2),
            (start + 4, start + 3),
        ]
    )
    graph.add_edges_from(
        [
            (start + 5, start),
            (start + 5, start + 1),
            (start + 5, start + 2),
            (start + 5, start + 3),
        ]
    )
    roles = [role_start] * 6
    return graph, roles


def tree(start, height, r=2, role_start=0):
    """Builds a balanced r-tree of height h
    INPUT:
    -------------
    start       :    starting index for the shape
    height      :    int height of the tree 
    r           :    int number of branches per node 
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a tree shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at role_start)
    """
    graph = nx.balanced_tree(r, height)
    roles = [0] * graph.number_of_nodes()
    return graph, roles


def fan(start, nb_branches, role_start=0):
    """Builds a fan-like graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of fan branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph, roles = star(start, nb_branches, role_start=role_start)
    for k in range(1, nb_branches - 1):
        roles[k] += 1
        roles[k + 1] += 1
        graph.add_edges_from([(start + k, start + k + 1)])
    return graph, roles


def ba(start, width, role_start=0, m=5):
    """Builds a BA preferential attachment graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles


def house(start, role_start=0):
    """Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles


def grid(start, dim=2, role_start=0):
    """ Builds a 2by2 grid
    """
    grid_G = nx.grid_graph([dim, dim])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start)
    roles = [role_start for i in grid_G.nodes()]
    return grid_G, roles


def star(start, nb_branches, role_start=0):
    """Builds a star graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of star branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + nb_branches + 1))
    for k in range(1, nb_branches + 1):
        graph.add_edges_from([(start, start + k)])
    roles = [role_start + 1] * (nb_branches + 1)
    roles[0] = role_start
    return graph, roles


def path(start, width, role_start=0):
    """Builds a path graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int length of the path
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + width))
    for i in range(width - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    roles = [role_start] * width
    roles[0] = role_start + 1
    roles[-1] = role_start + 1
    return graph, roles


def build_graph(
    width_basis,
    basis_type,
    list_shapes,
    start=0,
    rdm_basis_plugins=False,
    add_random_edges=0,
    m=5,
):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """

    #Call ba function (or other type) with given parameters (eval() runs given expression)
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id, plugins




"""
Main idea is to make some basis shape, and attach other shapes to this.

basis_type: Core shape. Attach other shapes to this core shape
width_basis: Size of basis type

attached_shapes: list of shapes to attach
attached_shape_sizes: size of each nb_shape we are adding. -1 if not applicable

nb_shapes: The number of attached shapes to add (1 number per attached shape)
num_features: Number of irrelevant features to add per node

Returns:
    adj: dense adjacency of generated graph
    features: vector of 1's per each node
	labels: unique label for each possible role a node can have 
            (eg. belong to grid/ star center/ etc...)
"""

def generate_graph(basis_shape, width_basis, attached_shapes, attached_shape_sizes, 
    nb_shapes, num_features):

    #Type and size of shapes to add
    list_shapes = []

    for shape_index in range(len(attached_shapes)):

        for _ in range(nb_shapes[shape_index]):

            list_shapes.append([attached_shapes[shape_index], attached_shape_sizes[shape_index]])

    G, labels, _ = build_graph(width_basis, basis_shape, list_shapes, start=0)
    G = perturb([G], 0.1)[0] #Add random noise edges

    adj = torch.Tensor(nx.adjacency_matrix(G).todense())

    features = torch.ones(len(G.nodes), num_features) #Features are all 1's

    labels = torch.Tensor(labels)

    #Get class distribution
    class_distribution = []

    for c in range(int(labels.max().item())+1):

        class_c_percent = (labels == c).sum().item()/labels.shape[0]
        class_distribution.append(class_c_percent)

    return adj, features, labels, class_distribution


def adj_to_indicies(adj):

    num_nodes = adj.shape[0]
    num_edges = int(adj.sum().item())

    edge_indicies = torch.zeros((2, num_edges), dtype=torch.long)

    edge_index = 0 #index into edge_indicies

    for i in range(num_nodes):

        for j in range(num_nodes):

            if(adj[i,j] == 1):

                edge_indicies[0, edge_index] = i
                edge_indicies[1, edge_index] = j

                edge_index += 1

    return edge_indicies


#n is number of graphs to generate
#Generate tree/grid/star graph
# Class 0 = Tree, Class 1 = grid, Class 2 = star center, Class 3 = star
def gen_synthetic_graph1(n):

    basis_shape = "tree"
    possible_width_basis = [5,6,7]

    attached_shapes = ["grid", "star"]
    grid_size = 3
    star_size = 8
    attached_shape_sizes = [grid_size, star_size]
    possible_nb_grid_patterns = [10,11,12]
    possible_nb_star_patterns = [10,11,12]

    num_features = 3

    adj_list = []
    edge_indicies_list = []
    features_list = []
    labels_list = []

    #Generate n graphs
    for i in range(n):

        print("Generating graph", i)

        #Select slightly different hyperparameters for each graph
        width_basis = random.choice(possible_width_basis)
        nb_grid_patterns = random.choice(possible_nb_grid_patterns)
        nb_star_patterns = random.choice(possible_nb_star_patterns)
        nb_shapes = [nb_grid_patterns, nb_star_patterns]


        adj, features, labels, class_distribution = generate_graph(basis_shape, width_basis, 
            attached_shapes, attached_shape_sizes, nb_shapes, num_features)

        edge_indicies = adj_to_indicies(adj)

        adj_list.append(adj)
        edge_indicies_list.append(edge_indicies)
        features_list.append(features)
        labels_list.append(labels)

    return adj_list, edge_indicies_list, features_list, labels_list, class_distribution


#Class 0 = Tree, Class 1 = Cycle, Class 2/3 = star
def gen_synthetic_graph2(n):

    basis_shape = "tree"
    possible_width_basis = [5,6,7]

    attached_shapes = ["cycle", "star"]
    cycle_size = 4
    star_size = 8
    attached_shape_sizes = [cycle_size, star_size]
    possible_nb_cycle_patterns = [18,19,20]
    possible_nb_star_patterns = [11,12,13]

    num_features = 3

    adj_list = []
    edge_indicies_list = []
    features_list = []
    labels_list = []

    #Generate n graphs
    for i in range(n):

        print("Generating graph", i)

        #Select slightly different hyperparameters for each graph
        width_basis = random.choice(possible_width_basis)
        nb_cycle_patterns = random.choice(possible_nb_cycle_patterns)
        nb_star_patterns = random.choice(possible_nb_star_patterns)
        nb_shapes = [nb_cycle_patterns, nb_star_patterns]


        adj, features, labels, class_distribution = generate_graph(basis_shape, width_basis, 
            attached_shapes, attached_shape_sizes, nb_shapes, num_features)

        edge_indicies = adj_to_indicies(adj)

        adj_list.append(adj)
        edge_indicies_list.append(edge_indicies)
        features_list.append(features)
        labels_list.append(labels)

    return adj_list, edge_indicies_list, features_list, labels_list, class_distribution


#Class 0 = BA, Class 1 = Cycle, Class 2/3 = Star
def gen_synthetic_graph3(n):

    basis_shape = "ba"
    possible_width_basis = [100, 110, 120]

    attached_shapes = ["cycle", "star"]
    cycle_size = 5
    star_size = 6 #Tree height
    attached_shape_sizes = [cycle_size, star_size]
    possible_nb_cycle_patterns = [18, 19, 20]
    possible_nb_tree_patterns = [18, 19, 20]

    num_features = 3

    adj_list = []
    edge_indicies_list = []
    features_list = []
    labels_list = []

    #Generate n graphs
    for i in range(n):

        print("Generating graph", i)

        #Select slightly different hyperparameters for each graph
        width_basis = random.choice(possible_width_basis)
        nb_cycle_patterns = random.choice(possible_nb_cycle_patterns)
        nb_tree_patterns = random.choice(possible_nb_tree_patterns)
        nb_shapes = [nb_cycle_patterns, nb_tree_patterns]


        adj, features, labels, class_distribution = generate_graph(basis_shape, width_basis, 
            attached_shapes, attached_shape_sizes, nb_shapes, num_features)

        edge_indicies = adj_to_indicies(adj)

        adj_list.append(adj)
        edge_indicies_list.append(edge_indicies)
        features_list.append(features)
        labels_list.append(labels)

    return adj_list, edge_indicies_list, features_list, labels_list, class_distribution