# Creating similarity network from similarity matrix
# Replace each node with a pie chart with the news categories from which it belongs
# Slice size denotes CI rank for the respective news categories

import matplotlib.font_manager as font_manager
import pickle
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from community import community_louvain
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.centrality import closeness_centrality
from networkx.algorithms.cuts import normalized_cut_size
import json

from os import listdir, makedirs
from os.path import isfile, join
import pandas as pd
import random

from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

font = font_manager.FontProperties(family='Arial')

seed = 12
np.random.seed(seed)
random.seed(seed)


SIM_NETWORK_PATH = '../data/similarity/sim_network_large_2020.pkl'
RETWEET_GRAPH_JSON_PATH = lambda x: '../data/urls/{}/retweet_graphs_to_draw/retweet_graph_top_combined_topnum30.json'.format(x)
SAVE_DIR = './output/'
# create output dir
makedirs(join(SAVE_DIR), exist_ok=True)


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=2.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    pos_modifier = 2
    for node in g.nodes():
        pos[node] = pos_communities[node] + ( pos_nodes[node] * pos_modifier)

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, k=0.10, weight='weight', **kwargs)
        pos.update(pos_subgraph)


    return pos

def assign_color(tag):
   # print(tag)
    if tag == 'fake':
        return '#282828'
    elif tag == 'extreme_bias_right' or tag == 'far_right':
        return '#4F0906'
    elif tag  == 'right':
        return '#8F100B'
    elif tag == 'lean_right':
        return '#DB4742'
    elif tag == 'center':
        return '#CFDB00'
    elif tag == 'lean_left':
        return '#4495DB'
    elif tag == 'left':
        return '#0E538F'
    elif tag == 'extreme_bias_left' or tag == 'far_left':
        return '#082E4F'
    else:
        print('ERROR incorrect tag:', tag)
        sys.exit()

def find_cut_threshold(G, partition):
    num_partitions = list(set(list(partition.values())))
    
    total_outer_weights = []
    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)
        
        outer_weights = []
        for target_node in inner_nodes:
            for outer_node in outer_nodes:
                if target_node != outer_node and G.has_edge(target_node, outer_node):
                    outer_weights.append(G[target_node][outer_node]['weight'])
        total_outer_weights.append(np.mean(list(set(outer_weights))))
    
    # modifier = 0.06
    modifier = 0.06
    return 2 * np.mean(total_outer_weights) + modifier

def remove_graph_edges(G, partition):
    threshold = find_cut_threshold(G, partition)
    print(threshold)
    delete_edge = []
    ignore_nodes = []
    for i in G.nodes():
        for j in G.nodes():
            if i != j and G.has_edge(i, j):
                if i not in ignore_nodes and j not in ignore_nodes:
                    weight = G[i][j]['weight']
                    if weight < threshold and (j, i) not in delete_edge:
                        delete_edge.append((i, j))

    for edge in delete_edge:
        i, j = edge
        G.remove_edge(i, j)

    return G


def get_edge_colors_community(G, partition):
    edge_colors = {(u,v): 'white' for u,v in G.edges()}
    num_partitions = list(set(list(partition.values())))

    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)

        for i in inner_nodes:
            for j in inner_nodes:
                if (i,j) in edge_colors:
                    edge_colors[(i,j)] = '#F7AC57'
            for j in outer_nodes:
                if (i,j) in edge_colors:
                    edge_colors[(i,j)] = '#803EF7'
    
    return list(edge_colors.values())

def cap_node_degree(G, partition):    
    degree_threshold = 5
    num_partitions = list(set(list(partition.values())))

    ignore_nodes = []
    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)

        for i in inner_nodes:
            neighbors = {j: G[i][j]['weight'] for j in list(G.neighbors(i))}
            neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1])}
            neighbors = list(neighbors.keys())

            delete_neighbors = []
            if len(neighbors) > degree_threshold:
                count = 0
                while len(neighbors) - len(delete_neighbors) > degree_threshold and count <= len(neighbors) - 1:
                    j = neighbors[count]
                    if G.degree[j] > degree_threshold:
                        delete_neighbors.append(j)
                    count += 1

            for j in delete_neighbors:
                if i not in ignore_nodes and j not in ignore_nodes:
                    if G.degree[i] > degree_threshold and G.degree[j] > degree_threshold:
                        if G.has_edge(i, j):
                            G.remove_edge(i, j)

    return G
def get_edge_thickness(G, partition, method='inverse'):    
    num_partitions = list(set(list(partition.values())))

    thickness = {(u,v): G[u][v]['weight'] for u,v in G.edges()}

    if method == 'community':
        inner_modifier = 0.1
        outer_modifier = 1

        for partition_val in num_partitions:
            inner_nodes = []
            outer_nodes = []
            for node, comm in partition.items():
                if comm == partition_val:
                    inner_nodes.append(node)
                else:
                    outer_nodes.append(node)

            for i in inner_nodes:
                for j in inner_nodes:
                    if i != j and (i,j) in thickness:
                        thickness[(i,j)] = inner_modifier
            
            for i in inner_nodes:
                for j in outer_nodes:
                    if (i,j) in thickness:
                        thickness[(i,j)] = outer_modifier
    elif method == 'inverse':
        for pair in thickness:
            # thickness[pair] = 1 / thickness[pair]
            thickness[pair] = thickness[pair]

    thickness = np.array(list(thickness.values())).reshape(-1, 1)
    thickness = MinMaxScaler((0.1, 1)).fit_transform(thickness).flatten()
    print(thickness)
    return thickness


def community_based_pos(pos, partition):
    # print(list(set(list(partition.values()))))
    target_partition = 0 # 0 = left-leaning echo chamber, 2 = right-leaning echo chamber
    target_repos = 12.3
    repos = 11
    target_y_shift = -45
    y_shift = 40
    target_x_shift = 22
    x_shift = -22
    
    tmp_pos = {}
    for node, xy in pos.items():
        if partition[node] == target_partition:
            tmp_pos[node] = xy * target_repos
            tmp_pos[node][1] = tmp_pos[node][1] + target_y_shift
            tmp_pos[node][0] = tmp_pos[node][0] + target_x_shift
        else:
            tmp_pos[node] = xy * repos
            tmp_pos[node][1] = tmp_pos[node][1] + y_shift
            tmp_pos[node][0] = tmp_pos[node][0] + x_shift
    pos = tmp_pos.copy()

    return pos


def random_partitition(G, k=2):
    partition = {}
    for node in G:
        partition[node] = np.random.randint(1, k + 1)
    return partition

def community_centrality(G, nodes, partition):
    num_partitions = list(set(list(partition.values())))
    res = {}
    for partition_val in num_partitions:
        eval_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                eval_nodes.append(node)
        comm_edgelist = []
        for i, node_i in enumerate(eval_nodes):
            for j, node_j in enumerate(eval_nodes):
                if i != j and G.has_edge(node_i, node_j):
                    comm_edgelist.append((node_i, node_j, {'weight': G[node_i][node_j]['weight']}))
        sub_G = nx.Graph(comm_edgelist)
        cent = nx.closeness_centrality(sub_G, distance='weight')
        cent = dict(sorted(cent.items(), key=lambda item: item[1], reverse=True))
        cent = {nodes[k]: v for k, v in cent.items()}
        res[partition_val] = np.mean(list(cent.values()))
    return res

def community_separation_v2(G, partition):
    num_partitions = list(set(list(partition.values())))
    
    avg_cut = {}
    for inner_comm in num_partitions:
        for outer_comm in num_partitions:
            inner_nodes = []
            outer_nodes = []
            if inner_comm != outer_comm:
                for node, comm in partition.items():
                    if comm == inner_comm:
                        inner_nodes.append(node)
                    elif comm == outer_comm:
                        outer_nodes.append(node)
                val = normalized_cut_size(G, inner_nodes, outer_nodes, weight='weight')
                cut_pair = tuple(set([inner_comm, outer_comm]))
                if cut_pair not in avg_cut:
                    avg_cut[cut_pair] = val
    
    return (np.mean(list(avg_cut.values())))

def entropic_similarity(G, nodes, partition, node_dict):
    def get_label_arr(user_id, node_dict, label_arr_template):
        fracs = node_dict[user_id]['fracs']
        labels = node_dict[user_id]['labels']
        label_arr = np.zeros(len(set(label_arr_template.values())))
        for i in range(len(labels)):
            label_arr[label_arr_template[labels[i]]] += fracs[i]
        return label_arr

    label_arr_template = {'fake': 0, 'extreme_bias_right': 1, 'far_right': 1, 'right': 1, 'lean_right': 1, 'center': 2, 'lean_left': 3, 'left': 3, 'extreme_bias_left': 3, 'far_left': 3}

    num_partitions = list(set(list(partition.values())))
    res = {}
    for target_comm in num_partitions:
        H = 0
        for node_i, comm_i in partition.items():
            if comm_i == target_comm:
                label_arr_i = get_label_arr(nodes[node_i], node_dict, label_arr_template)

                for node_j, comm_j in partition.items():
                    if node_i != node_j and comm_j == target_comm:
                        label_arr_j = get_label_arr(nodes[node_j], node_dict, label_arr_template)

                        s_ij = 1 - cosine(label_arr_i, label_arr_j)
                        if s_ij != 0:
                            H_add = s_ij * np.log(s_ij) + (1 - s_ij) * np.log(1 - s_ij)
                            if not np.isnan(H_add):
                                H += H_add
        H = -H
        res[target_comm] = H

    return res
    
def get_inter_intra_edge_perc(G, partition):
    num_partitions = list(set(list(partition.values())))
    
    inners = []
    outers = []
    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)
        
        intra_weights = []
        for target_node in inner_nodes:
            for inner_node in inner_nodes:
                if inner_node != target_node and G.has_edge(target_node, inner_node):
                    intra_weights.append(G[target_node][inner_node]['weight'])

        inter_weights = []
        for inner_node in inner_nodes:
            for outer_node in outer_nodes:
                if inner_node != outer_node and G.has_edge(inner_node, outer_node):
                    inter_weights.append(G[inner_node][outer_node]['weight'])

        # intra_weights_perc = np.sum(intra_weights) / (np.sum(intra_weights) + np.sum(inter_weights))
        # inter_weights_perc = 1 - intra_weights_perc

        intra_weights_perc = np.sum(intra_weights) / np.sum(inter_weights)

        inners.append(intra_weights_perc)

    return np.mean(inners)

def get_partition_stats(G, nodes, partition, node_dict):
    print()

    mod = community_louvain.modularity(partition, G)
    print('Modularity:', mod)

    cut = community_separation_v2(G, partition)
    print('Normalized cut:', cut)

    in_comm_frac = get_inter_intra_edge_perc(G, partition)
    print('In-community edge frac:', in_comm_frac)

    # ent = entropic_similarity(G, nodes, partition, node_dict)
    # print('Entropic similarity:', ent)

    # cent = community_centrality(G, nodes, partition)
    # print('Community centrality:', cent)

    print()

if __name__ == '__main__':
    year = 2020
    ignore_extreme_left = True
    TOPNUM = 30 # the json file contains top 30 users per media category

    data = pickle.load(open(SIM_NETWORK_PATH, 'rb'))

    M = data['sim_matrix'] 
    nodes = data['nodes']
    tags = data['tags']

    cateogry_list = ['far_left', 'left', 'lean_left', 'center', 'lean_right', 'right', 'far_right', 'fake']
    if ignore_extreme_left:
        cateogry_list.remove('far_left')
    alignment_value_to_rank = {TOPNUM - i: i+1 for i in range(TOPNUM)}

    top_n = 5
    cateogry_list = {x: [''] * top_n for x in cateogry_list}

    node_info = json.load(open(RETWEET_GRAPH_JSON_PATH(year)))

    node_dict = {}
    ci_weights = []
    user_id_map = {}
    remove_later = []
    user_to_ranks = {}
    for item in node_info['nodes']:
        user_id = item['userid']
        username = item['username']
        ci_weight = item['kout']
        rank = item['CIoutrank']
        grouprank = item['grouprank']

        if rank > 25: #25
            continue

        user_id_map[user_id] = username

        colors = []
        fracs = []
        labels = []
        for alignment in item['proportions']:
            category = alignment['group']
            value = alignment['value']

            # convert value to rank
            alignment_rank = alignment_value_to_rank[value]

            if ignore_extreme_left and (category == 'far_left' or category == 'extreme_bias_left'):
                continue

            labels.append(category)
            fracs.append(alignment['value'])

            # these proportions define a users place in the top rankings of each media category, save
            # this information if the rank is in the top 5. 
            if alignment_rank <= top_n:
                cateogry_list[category][alignment_rank - 1] = user_id

        fracs = np.array(fracs)
        fracs = fracs / np.sum(fracs)

        colors = [assign_color(x) for x in labels]

        if len(labels) == 0:
            continue


        ci_weights.append(ci_weight)
        node_dict[user_id] = {'ci_weight': ci_weight, 'rank': rank, 'colors': colors, 'fracs': fracs, 'labels': labels}


    count = 1
    cateogry_list_order = {}
    for grouprank in cateogry_list:
        for rank in cateogry_list[grouprank]:
            if rank not in cateogry_list_order:
                cateogry_list_order[rank] = count
                count += 1

    ci_weights = np.array(ci_weights).reshape(-1, 1)
    scalar = MinMaxScaler((0.4, 2.5))
    ci_weights = scalar.fit_transform(ci_weights).flatten()

    for i, node_id in enumerate(list(node_dict.keys())):
        node_dict[node_id]['ci_weight'] = ci_weights[i]

    edgelist = []
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i,j] > 0.0:
                user_i = nodes[i]
                user_j = nodes[j]
                try:
                    info_i = node_dict[user_i]
                    info_j = node_dict[user_j]

                    edgelist.append((i, j, {'weight': M[i,j]}))
                except:
                    continue

    G = nx.Graph(edgelist)
    
    fake_tag = mpatches.Patch(color='#282828', label='Fake news')
    extreme_bias_right_tag = mpatches.Patch(color='#4F0906', label='Extreme bias\n right')
    right_tag = mpatches.Patch(color='#8F100B', label='Right')
    lean_right_tag = mpatches.Patch(color='#DB4742', label='Leaning right')
    center_tag = mpatches.Patch(color='#CFDB00', label='Center')
    lean_left_tag = mpatches.Patch(color='#4495DB', label='Leaning left')
    left_tag = mpatches.Patch(color='#0E538F', label='Left')
    extreme_bias_left_tag = mpatches.Patch(color='#082E4F', label='Extreme bias\n left')


    partition = community_louvain.best_partition(G, random_state=seed)
    #print(partition)
    tmp_partition = {}
    for node, comm in partition.items():
        if comm != 1:
            tmp_partition[node] = comm
        else:
            tmp_partition[node] = 2
    partition = tmp_partition.copy()

    get_partition_stats(G, nodes, partition, node_dict)

    # community_separation(G, nodes, partition)

    nx.set_node_attributes(G, partition, 'community')
    labels = nx.get_node_attributes(G, 'community') 
    
    pos = community_layout(G, partition)
    pos = community_based_pos(pos, partition)

    edge_colors = ['#999'] * len(G.edges())
    node_colors = ['white'] * len(G.nodes())

    # G = remove_graph_edges(G, partition)
    G = cap_node_degree(G, partition)

    print(list(nx.isolates(G)))

    pickle.dump(G, open(join(SAVE_DIR, "2020_figure_network.pkl"), "wb"))
    

    edge_colors = get_edge_colors_community(G, partition)
    nx.draw_networkx_edges(G, pos=pos, edge_color = edge_colors, width=0.5, alpha=0.6)
    from collections import defaultdict
    colors_to_node = defaultdict(list)
    for node in G.nodes:
        user_id = nodes[node]
        
        fracs = node_dict[user_id]['fracs']
        colors = node_dict[user_id]['colors']
        labels = node_dict[user_id]['labels']

        for color in colors:
            colors_to_node[color].append(node)

        # radius = node_dict[user_id]['ci_weight'] * 0.000038
        radius = node_dict[user_id]['ci_weight']

        # print(radius)

        rank_labels = [''] * len(colors)
        if user_id in cateogry_list_order:
            rank_labels[0] = str(cateogry_list_order[user_id])
        

        a = plt.pie(
            fracs, # s.t. all wedges have equal size
            labels = rank_labels,
            textprops={'color': 'green', 'fontsize': 6, 'ha': 'center', 'va': 'top', 'x': pos[node][0], 'y': pos[node][1] + 2, 'weight': 'bold'},#, 'family':'Arial'},
            center=pos[node], 
            colors = colors,
            radius=radius)
    
    # if ignore_extreme_left:
    #     plt.legend(loc='upper right', prop={'family':'Arial'}, handles=[fake_tag, extreme_bias_right_tag, right_tag, lean_right_tag, center_tag, lean_left_tag, left_tag])
    # else:
    #     plt.legend(loc='upper right', prop={'family':'Arial'}, handles=[fake_tag, extreme_bias_right_tag, right_tag, lean_right_tag, center_tag, lean_left_tag, left_tag, extreme_bias_left_tag])
    
    #print(colors_to_node)

    #lim_range = 40
    lim_range = 40
    #plt.ylim(-lim_range,lim_range)
    #plt.xlim(-lim_range,lim_range)
    plt.ylim(-lim_range,lim_range)
    plt.xlim(-28,30)


    plt.savefig(join(SAVE_DIR, 'figure5_community_network_' + str(year) + '.pdf'), dpi=500, format='pdf', bbox_inches='tight')
    plt.savefig(join(SAVE_DIR, 'figure5_community_network_' + str(year) + '.svg'), dpi=500, format='svg', bbox_inches='tight')