#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:05:05 2017

@author: alex
"""

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import time
import pickle
import graph_tool.all as gt
import networkx as nx
import pandas as pd

import numpy as np


year = '2020' #'2020' '2016
save_dir =  '../data/urls/{}/'.format(year)
network_dir = '../data/ci_output/graphs/{}/'.format(year)

# create the output directory
os.makedirs(os.path.join(save_dir, 'retweet_graphs_to_draw'), exist_ok=True)

# dark color for max k_out
colors_max = {'fake'  :  '#633517',
          'extreme_bias_right'  :  '#a6001a',
          'right' : 	'#D96C09',
          'lean_right' : 	'#554F1A',
          'center':  '#004d33',
          'lean_left' : '#00477e',
          'left' : '#3F0452',
          'extreme_bias_left' : '#55002D'}
# light color for min k_out
          
colors_min = {'fake'  :  '#ED8E53',
          'extreme_bias_right'  :  '#FF514A',
          'right' : 	'#FFC342',
          'lean_right' : 	'#C0B24D',
          'center':  '#2BC889',
          'lean_left' : '#40A2FF',
          'left' : '#AB2BDE',
          'extreme_bias_left' : '#F4388A'}

#with open(os.path.join(save_dir, 'retweet_graphs_to_draw',
#                                 'force_layout_for_drawing.html'), 'r') as fopen:
#    html_file = fopen.read()
    

#media_types = ['fake', 'right_extreme', 'right', 'right_leaning',
#                   'center', 'left_leaning', 'left', 'left_extreme']
media_types = {
    '2020': ['fake', 'right_extreme', 'right', 'right_leaning',
             'center', 'left_leaning', 'left', 'left_extreme'],
    #'2016': ['fake', 'extreme_bias_right', 'right', 'lean_right',
    #         'center', 'lean_left', 'left', 'extreme_bias_left']
    '2016': ['fake', 'right_extreme', 'right', 'right_leaning',
             'center', 'left_leaning', 'left', 'left_extreme']
               }

media_type_to_group = {
    'fake': 'fake',
    'center': 'center',
    'left_leaning': 'lean_left',
    'right_leaning': 'lean_right',
    'extreme_bias_right': 'far_right',
    'extreme_bias_left': 'far_left',
    'right_extreme': 'far_right',
    'left_extreme': 'far_left',
    'lean_left': 'lean_left',
    'lean_right': 'lean_right',
    'left': 'left',
    'right': 'right'
}

#raise Exception


#%% get top influencers of each networks
period = 'june-nov'
top_num = 30
ranker = 'CI_out'

#ranking = pd.read_pickle(os.path.join(save_dir, 'influencer_rankings_simple_{}.pickle'.format(year)))
ranking = pd.read_pickle(os.path.join(save_dir, 'influencer_rankings_{}.pickle'.format(year)))
infl_uids = set()
    
for media_type in media_types[year]:
    infl_uids.update(ranking[media_type][ranker]['user_id'][:top_num])
    
uids_to_screename = dict()
for media_type in media_types[year]:
    for i, row in ranking[media_type][ranker][['user_id','screenname']][:top_num].iterrows():
        if row.screenname == '@???????':
            uids_to_screename[row.user_id] = 'deleted'
        else:
            uids_to_screename[row.user_id] = row.screenname
    
    
    

#%% combined all graphs

Graphs_simple = dict()
for media_type in media_types[year]:
    Graphs_simple[media_type] = dict()
    #Graphs_simple[media_type] = gt.load_graph(os.path.join(network_dir, media_type + '_' + year + '_simple_ci.gt'))
    Graphs_simple[media_type] = gt.load_graph(os.path.join(network_dir, media_type + '_' + year + '_ci.gt'))
    
Glist_simple = [Graphs_simple[key] for key in media_types[year]]

#load combined graph
#Gcomb= gt.load_graph(os.path.join(save_dir, 
#                 'retweet_graph_' + \
#                 'combined' + '_simple_' + period + '.gt'))

    
#all uids:
uids = set()
for G in Glist_simple:
    if G.vp.user_id.a is None:
        uids.update(np.array([G.vp.user_id[v] for v in G.vertices()], dtype=np.int64))
    else:
        uids.update(G.vp.user_id.a)

# user id to verterx id of new graph    
#uid_to_vid = {uid : i for i, uid in enumerate(uids)}


#Gcomb.add_vertex(len(uids))

#add edges
edge_list_list = []
for G in Glist_simple:
    print(G.gp.name)
    e_array = G.get_edges([G.ep.tweet_id])
    # translate to new vids
    new_source_iter = (G.vp.user_id[s] for s in e_array[:,0].tolist())
    new_target_iter = (G.vp.user_id[t] for t in e_array[:,1].tolist())
    #new_tweet_id_iter = (G.ep.tweet_id.a[i] for i in e_array[:,2].tolist())

    
    new_sources = np.fromiter(new_source_iter, np.int64, e_array.shape[0])
    new_targets = np.fromiter(new_target_iter, np.int64, e_array.shape[0])
    new_tweetids = e_array[:,2]
    #new_tweetids = np.fromiter(new_tweet_id_iter, np.int64, e_array.shape[0])
    
    edge_list_list.append(np.vstack((new_sources,new_targets,new_tweetids)).T)

Gcomb = gt.Graph()        

tweet_id_eprop = Gcomb.new_edge_property('int64_t')

user_id_vprop = Gcomb.add_edge_list(np.vstack(edge_list_list), hashed=True,
                                    eprops=[tweet_id_eprop])
    
Gcomb.ep['tweetid'] = tweet_id_eprop

gt.remove_parallel_edges(Gcomb)

Gcomb.vp['userid'] = user_id_vprop
Gcomb.vp['kout'] = Gcomb.degree_property_map('out')
Gcomb.vp['kin'] = Gcomb.degree_property_map('in')
#


# add membership info
Gcomb.ep['media'] = Gcomb.new_edge_property('string')
tweetmedia = np.zeros_like(tweet_id_eprop.a, dtype='|S2')
for G in Glist_simple:
    Gcomb.vp['isin' + G.gp.name] = Gcomb.new_vertex_property('bool')
    Gcomb.vp['isin' + G.gp.name].a = np.isin(Gcomb.vp.userid.a,G.vp.user_id.a)
    
    if G.gp.name == 'fake':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'fk'
    if G.gp.name == 'right_extreme':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'fr'
    if G.gp.name == 'right':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'ri'
    if G.gp.name == 'right_leaning':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'lr'
    if G.gp.name == 'center':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'ce'
    if G.gp.name == 'left_leaning':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'll'
    if G.gp.name == 'left':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'le'
    if G.gp.name == 'left_extreme':
        tweetmedia[np.isin(Gcomb.ep.tweetid.a,G.ep.tweet_id.a)] = 'fl'
        
Gcomb.ep['media'].set_2d_array(tweetmedia)

# keep only GC
gc = gt.label_largest_component(Gcomb, directed=False)
Gcomb = gt.GraphView(Gcomb,vfilt=gc)
Gcomb = gt.Graph(Gcomb, prune=True)

        
Gcomb.save(os.path.join(network_dir, 'combined_' + year + '_ci.gt'))
    
Gcomb.save(os.path.join(save_dir, 'retweet_graphs_to_draw', 'combined_' + year + '.graphml'))

# scgc
#scgc = gt.label_largest_component(Gcomb, directed=True)
#Gcomb = gt.GraphView(Gcomb,vfilt=scgc)
#Gcomb = gt.Graph(Gcomb, prune=True)  
#
#        
#Gcomb.save(os.path.join(save_dir, 
#                 'retweet_graph_' + \
#                 'combined_scgc' + '_simple_' + period + '.gt'))
#    
#Gcomb.save(os.path.join(save_dir, 
#                 'retweet_graph_' + \
#                 'combined_scgc' + '_simple_' + period + '.graphml'))


Gcombtop = gt.Graph(gt.GraphView(Gcomb,
                                 vfilt=lambda v: Gcomb.vp['userid'][v] in infl_uids),
                                prune=True)

Gcombtop.vp['username'] = Gcombtop.new_vertex_property('string')
   
for v in Gcombtop.vertices():
    Gcombtop.vp['username'][v] = uids_to_screename[Gcombtop.vp.userid[v]]
    
Gcombtop.vp['label'] = Gcombtop.vp['userid']


for mediatype in media_types[year]:
    #import pdb
    #pdb.set_trace()
    #Gcombtop.vp['isin' + mediatype] = Gcombtop.new_vertex_property('bool')
    #Gcombtop.vp['isin' + mediatype] = Gcombtop.vp['isin_' + mediatype]
    del Gcombtop.vp['isin' + mediatype + '_{}'.format(year)]

#Gcombtop.vp['isinfarright'] = Gcombtop.new_vertex_property('bool')
#Gcombtop.vp['isinfarright'] = Gcombtop.vp['isinfar_right']
#del Gcombtop.vp['isinfar_right']
#
#Gcombtop.vp['isinleanright'] = Gcombtop.new_vertex_property('bool')
#Gcombtop.vp['isinleanright'] = Gcombtop.vp['isinlean_right']
#del Gcombtop.vp['isinlean_right']
#
#Gcombtop.vp['isinleanleft'] = Gcombtop.new_vertex_property('bool')
#Gcombtop.vp['isinleanleft'] = Gcombtop.vp['isinlean_left']
#del Gcombtop.vp['isinlean_left']
#
#Gcombtop.vp['isinfarleft'] = Gcombtop.new_vertex_property('bool')
#Gcombtop.vp['isinfarleft'] = Gcombtop.vp['isinfar_left']
#del Gcombtop.vp['isinfar_left']

Gcombtop.save(os.path.join(save_dir,
                 'retweet_graphs_to_draw',
                 'combined_' + year + '_topnum' + str(top_num) + '.gml'))
#
#Gcombtop.save(os.path.join(save_dir, 
#                 'retweet_graphs_to_draw',
#                 'combined' + '_simple_' + period + '_topnum' + str(top_num) + '.graphml'))

#%% convert graph to gml and json


#%% read gml with neworkx
def createJSON(G,jsonFile,path=os.path.join(save_dir, 'retweet_graphs_to_draw')):

    from networkx.readwrite import json_graph
    import json
    
    data = json_graph.node_link_data(G)
    s = json.dumps(data)
    
    with open(os.path.join(path,jsonFile),'w') as f:
        f.write(s)
    
    

G = nx.read_gml(os.path.join(save_dir, 'retweet_graphs_to_draw',
                'combined_' + year + '_topnum' + str(top_num) + '.gml'))

#%% add membership
for i, media_type in enumerate(media_types[year]):
    
    for rank, uid in enumerate(ranking[media_type][ranker]['user_id'][:top_num].tolist()):
        
        if 'proportions' not in G.nodes[uid].keys():
            G.nodes[uid]['proportions'] = []
        
        G.nodes[uid]['proportions'].append({'group': media_type_to_group[media_type], 'value': top_num - rank,
                                           'kout': G.nodes[uid]['kout'],
                                           'username': G.nodes[uid]['username']})
        
        if 'CIoutrank' not in G.nodes[uid].keys():
            G.nodes[uid]['CIoutrank'] = rank + 1
            G.nodes[uid]['grouprank'] = media_type_to_group[media_type]
        elif G.nodes[uid]['CIoutrank'] > rank + 1:
            #replace with best rank
            G.nodes[uid]['CIoutrank'] = rank + 1
            G.nodes[uid]['grouprank'] = media_type_to_group[media_type]
        
            



jsonFile = 'retweet_graph_top' + '_combined' +'_topnum' + str(top_num) +'.json'
createJSON(G,jsonFile)

#%% add CI rank to graph-tool graph

Gcombtop.vp['CIoutrank'] = Gcombtop.new_vertex_property('int',val=top_num+1)
Gcombtop.vp['grouprank'] = Gcombtop.new_vertex_property('string')

for i, media_type in enumerate(media_types[year]):
    
    for rank, uid in enumerate(ranking[media_type][ranker]['user_id'][:top_num].tolist()):
        
        nodeid = np.where(Gcombtop.vp.userid.a == uid)[0][0]
        
        if Gcombtop.vp['CIoutrank'][nodeid] > rank + 1:
            #replace with best rank
            Gcombtop.vp['CIoutrank'][nodeid] = rank + 1
            Gcombtop.vp['grouprank'][nodeid] = media_type_to_group[media_type]
            
Gcombtop.save(os.path.join(save_dir, 
                 'retweet_graphs_to_draw',
                 'combined_' + year + '_topnum' + str(top_num) + '.graphml'))

#%% create d3js file

#with open(os.path.join(save_dir, 'retweet_graphs_to_draw',
#                           'd3js_network_' + media_type + '.html'), 'w') as fopen:
#    fopen.write(html_file % (colors_max[media_type],
#                                     colors_min[media_type], jsonFile))
