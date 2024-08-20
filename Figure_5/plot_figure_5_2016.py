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
from sklearn.preprocessing import MinMaxScaler, normalize

font = font_manager.FontProperties(family='Arial')

seed = 12
np.random.seed(seed)
random.seed(seed)


SIM_NETWORK_PATH = '../data/similarity/sim_network_joined_large_2016_v1.pkl' # 'data/2016/sim_network_large_with_figure3_influencers.pkl'
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

    tmp_pos = {}
    for node, xy in pos.items():
        tmp_pos[node] = xy * 1.5
    pos = tmp_pos.copy()

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
    print('Overall threshold:', threshold)
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

def remove_graph_edges_in_community(G, partition):    
    threshold = find_cut_threshold(G, partition) * 0.4
    print('Intra-community threshold:', threshold)

    num_partitions = list(set(list(partition.values())))

    delete_edge = []
    for partition_val in num_partitions:
        inner_nodes = []
        ignore_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)

        for i in inner_nodes:
            for j in inner_nodes:
                if i != j and G.has_edge(i, j):
                    if i not in ignore_nodes and j not in ignore_nodes:
                        weight = G[i][j]['weight']
                        if weight < threshold and (j, i) not in delete_edge:
                            delete_edge.append((i, j))

    for edge in delete_edge:
        i, j = edge
        G.remove_edge(i, j)

    return G

def get_edge_weights(G, partition, method='inverse'):    
    num_partitions = list(set(list(partition.values())))

    weights = {(u,v): G[u][v]['weight'] for u,v in G.edges()}

    weights = np.array(list(weights.values())).reshape(-1, 1)
    weights = list(MinMaxScaler((1, 20)).fit_transform(weights).flatten())
    weights = [int(x) for x in weights]
    return weights

def get_edge_colors_binned(G):
    edge_colors = ['#F7AC57'] * len(G.edges())
    weights = [G[u][v]['weight'] for u,v in G.edges()]

    lower_bound = 0.4

    lower_color = '#F7AC57'
    upper_color = '#803EF7'

    for i in range(len(weights)):
        weight = weights[i]
        if weight <= lower_bound:
            edge_colors[i] = lower_color
        else:
            edge_colors[i] = upper_color
            
    return edge_colors

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

def community_based_pos(pos, partition):
    target_partition = 2 # 1 = right-leaning echo chamber, 2 = left-leaning echo chamber
    target_repos = 12.3
    repos = 7.3
    y_shift = 8.5
    
    tmp_pos = {}
    for node, xy in pos.items():
        if partition[node] == target_partition:
            tmp_pos[node] = xy * target_repos
            tmp_pos[node][1] = tmp_pos[node][1] + y_shift
        else:
            tmp_pos[node] = xy * repos
            tmp_pos[node][1] = tmp_pos[node][1] + y_shift
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
    def get_label_arr(user_id, node_dict, label_temp):
        fracs = node_dict[user_id]['fracs']
        labels = node_dict[user_id]['labels']
        label_arr = np.zeros(len(set(label_temp.values())))
        for i in range(len(labels)):
            label_arr[label_temp[labels[i]]] += fracs[i]
        return label_arr

    label_arr_template = {'fake': 0, 'extreme_bias_right': 1, 'far_right': 1, 'right': 1, 'lean_right': 1, 'center': 2, 'lean_left': 3, 'left': 3, 'extreme_bias_left': 3, 'far_left': 3}
    #label_arr_template = {'fake': 0, 'right_extreme': 1, 'far_right': 1, 'right': 1, 'right_leaning': 1, 'center': 2, 'left_leaning': 3, 'left': 3, 'left_extreme': 3, 'far_left': 3}

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
    year = 2016
    ignore_extreme_left = True

    # data = pickle.load(open('../' + str(year) + '/sim_network/sim_network_large.pkl', 'rb'))
    #data = pickle.load(open('../data/similarity/sim_network_large_with_figure3_influencers.pkl', 'rb'))
    #data = pickle.load(open('../data/similarity/sim_network_large_with_figure3_influencers_bren_800.pkl', 'rb'))
    data = pickle.load(open(SIM_NETWORK_PATH, 'rb'))
    #data = pickle.load(open('../data/similarity/sim_network_large_joined_influencers_bren.pkl', 'rb'))
    
    #data = pickle.load(open('temp_data/sim_network_large_with_figure3_influencers_noextra.pkl', 'rb'))

    M = data['sim_matrix'] 
    nodes = data['nodes']
    tags = data['tags']

    cateogry_list = ['far_left', 'left', 'lean_left', 'center', 'lean_right', 'right', 'far_right', 'fake']
    if ignore_extreme_left:
        cateogry_list.remove('far_left')

    top_n = 5
    cateogry_list = {x: [''] * top_n for x in cateogry_list}

    #if year == 2016:
    #    
    #    node_info = json.load(open('data/retweet_graph_top_combined_topnum30_2016_UserFix.json'))
    #elif year == 2020:
    #    node_info = json.load(open('data/retweet_graph_top_combined_topnum30_2020.json'))

    node_info = json.load(open(RETWEET_GRAPH_JSON_PATH(year)))
    #node_info = json.load(open('../data/retweet_graph_top_combined_topnum30_2016_UserFix.json'))

    node_dict = {}
    ci_weights = []
    for item in node_info['nodes']:
        user_id = item['userid']
        username = item['username']
        ci_weight = item['kout']
        rank = item['CIoutrank']
        grouprank = item['grouprank']

        if rank > 25: # Limit top 30 to top 25
            continue

        colors = []
        fracs = []
        labels = []
        for alignment in item['proportions']:
            category = alignment['group']

            if ignore_extreme_left and (category == 'far_left' or category == 'extreme_bias_left'):
                continue

            labels.append(category)
            fracs.append(alignment['value'])
        
        fracs = np.array(fracs)
        fracs = fracs / np.sum(fracs)

        colors = [assign_color(x) for x in labels]

        if len(labels) == 0:
            continue

        if grouprank in cateogry_list:
            if rank <= top_n:
                cateogry_list[grouprank][rank - 1] = user_id

        ci_weights.append(ci_weight)
        node_dict[user_id] = {'ci_weight': ci_weight, 'rank': rank, 'colors': colors, 'fracs': fracs, 'labels': labels}

    cateogry_list['lean_right'][3] = 25073877
    cateogry_list['right'][1] = 25073877
    cateogry_list['right'][3] = 14669951
    cateogry_list['far_right'][4] = 14669951
    cateogry_list['fake'][4] = 25073877

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

    print(len(G.nodes), len(node_dict))

    fake_tag = mpatches.Patch(color='#282828', label='Fake news')
    extreme_bias_right_tag = mpatches.Patch(color='#4F0906', label='Extreme bias\n right')
    right_tag = mpatches.Patch(color='#8F100B', label='Right')
    lean_right_tag = mpatches.Patch(color='#DB4742', label='Leaning right')
    center_tag = mpatches.Patch(color='#CFDB00', label='Center')
    lean_left_tag = mpatches.Patch(color='#4495DB', label='Leaning left')
    left_tag = mpatches.Patch(color='#0E538F', label='Left')
    extreme_bias_left_tag = mpatches.Patch(color='#082E4F', label='Extreme bias\n left')

    # partition = random_partitition(G, k=2)

    # partition = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=500, seed=seed)
    # tmp_partition = {}
    # for i, comm in enumerate(partition):
    #     for node in comm:
    #         tmp_partition[node] = i
    # partition = tmp_partition.copy()

    partition = community_louvain.best_partition(G, random_state=seed)
    tmp_partition = {}
    node_names = []
    for node, comm in partition.items():
        node_names.append(nodes[node])
        if comm != 1:
            tmp_partition[node] = comm
        else:
            tmp_partition[node] = 2
    partition = tmp_partition.copy()

    #print(partition)
    #partition = {0: 0, 1: 0, 2: 2, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 54: 0, 63: 0, 68: 0, 70: 0, 89: 0, 92: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 115: 0, 121: 0, 124: 0, 125: 0, 138: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 2, 167: 0, 168: 0, 169: 0, 210: 2, 211: 0, 212: 2, 213: 2, 214: 2, 215: 2, 216: 0, 217: 2, 218: 2, 219: 0, 220: 2, 221: 2, 256: 2, 263: 2, 264: 2, 265: 2, 266: 2, 267: 2, 268: 2, 269: 2, 270: 2, 271: 2, 272: 2, 273: 2, 274: 2, 275: 2, 276: 2, 277: 2, 278: 2, 279: 2, 280: 2, 281: 2, 282: 2, 312: 2, 313: 2, 323: 2, 339: 2, 340: 2, 341: 2, 342: 2, 343: 2, 344: 2, 345: 2, 346: 2, 347: 2, 348: 2, 349: 2, 350: 2, 351: 2, 352: 2, 353: 2, 354: 2, 355: 2, 356: 2, 357: 2, 358: 2, 359: 2, 360: 2, 401: 2, 421: 2, 422: 2, 423: 2, 424: 2, 425: 2, 426: 2, 427: 2, 428: 2, 429: 2, 430: 2, 431: 2, 432: 2, 433: 2, 434: 2, 435: 2, 436: 2, 437: 2, 438: 2, 439: 2, 440: 2, 441: 2, 446: 2}
    print(sorted(node_names))

    get_partition_stats(G, nodes, partition, node_dict)

    # community_separation(G, nodes, partition)

    nx.set_node_attributes(G, partition, 'community')
    labels = nx.get_node_attributes(G, 'community') 
    
    pos = community_layout(G, partition)

    pos = community_based_pos(pos, partition)

    # 352, 445

    # x = -0.9511019
    # y = -12.63289746
    # pos[445] = np.array([x, y])

    x = 10.9511019
    y = -12.63289746
    pos[352] = np.array([x, y])

    edge_colors = ['#999'] * len(G.edges())
    node_colors = ['white'] * len(G.nodes())

    # G = remove_graph_edges(G, partition)
    G = cap_node_degree(G, partition)

    print(list(nx.isolates(G)))

    # weights = {(u,v): G[u][v]['weight'] for u,v in G.edges()}
    # pickle.dump(weights, open("output/2016_edge_weights.pkl", "wb"))

    
    pickle.dump(G, open(join(SAVE_DIR, "2016_figure_network.pkl"), "wb"))

    # fig = plt.figure()
    # ax = plt.axes()

    # nx.draw(G, pos, node_color=node_colors, edge_color = edge_colors, node_size = node_sizes)

    # edge_weights = get_edge_weights(G, partition)
    # nx.draw_networkx_edges(G, pos=pos, edge_cmap = plt.cm.spring, edge_color=edge_weights, width=0.5)

    edge_colors = get_edge_colors_community(G, partition)
    nx.draw_networkx_edges(G, pos=pos,  edge_color=edge_colors, width=0.5, alpha=0.6)
    l = []
    for node in G.nodes:
        user_id = nodes[node]
        fracs = node_dict[user_id]['fracs']
        colors = node_dict[user_id]['colors']
        labels = node_dict[user_id]['labels']

        # radius = node_dict[user_id]['ci_weight'] * 0.000038
        radius = node_dict[user_id]['ci_weight']

        # print(radius)

        rank_labels = [''] * len(colors)
        if user_id in cateogry_list_order:
            rank_labels[0] = str(cateogry_list_order[user_id])
            l.append(cateogry_list_order[user_id])
        
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
    
    lim_range = 40
    plt.ylim(-lim_range,lim_range)
    #plt.xlim(-lim_range,lim_range)
    plt.xlim(-30,35)
    plt.savefig(join(SAVE_DIR, 'figure5_community_network_' + str(year) + '.pdf'), dpi=500, format='pdf', bbox_inches='tight')
    plt.savefig(join(SAVE_DIR, 'figure5_community_network_' + str(year) + '.svg'), dpi=500, format='svg', bbox_inches='tight')