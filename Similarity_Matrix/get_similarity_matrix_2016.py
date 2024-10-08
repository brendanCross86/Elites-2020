# Copyright (C) 2021-2026, James Flamino


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
#http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
from os import listdir, makedirs
from os.path import isfile, join
import collections
import sys

import numpy as np
import sqlite3
from numba import njit
from numba import types
from numba.typed import Dict
import json

influencer_dir = '../data/influencers/top_100_anon/'
raw_retweets_2016 = '/home/pub/hernan/Election_2016/complete_trump_vs_hillary_db.sqlite' # '/path/to/2016_election_sqlite3/data/complete_trump_vs_hillary_db.sqlite'
save_dir = '../data/similarity'

# this flag is true if the users will be anonymized in this files output.
# if this flag is set, we expect the input data to have been anonymized as well.
# if regenerating from raw data, set the flag to false. 
anonymized = False
if anonymized:
    user_id_to_anon_id = json.load(open('../data/maps/user_id_to_anon_id_extended.json','r'))

biases = ['fake', 'right_extreme', 'right', 'right_leaning', 'center', 'left_leaning', 'left', 'left_extreme']

def get_top_n_influencers(top_n = 33):
    bnames = [f for f in listdir(influencer_dir) if isfile(join(influencer_dir, f))]
    top_influencers = []
    for bname in bnames:
        if '2016' in bname:
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(influencer_dir + bname, 'rb'))
            top_influencers = top_influencers + list(target_influencers.keys())[:100]

    top_influencers = dict(collections.Counter(top_influencers))
    top_influencers = dict(sorted(top_influencers.items(), key=lambda item: item[1], reverse=True))
    top_influencers = list(top_influencers.keys())[:top_n]

    top_influencers = sorted(list(set(top_influencers)))

    keep_order = False
    influencer_tags = []

    return top_influencers, influencer_tags, keep_order

def get_joined_top_n_by_news(top_n = 5):
    keep_order = True

    #bnames = ['top_100_fake_2020.pkl', 'top_100_extreme_bias_right_2020.pkl', 'top_100_right_2020.pkl', 'top_100_lean_right_2020.pkl', 'top_100_center_2020.pkl', 'top_100_lean_left_2020.pkl', 'top_100_left_2020.pkl', 'top_100_extreme_bias_left_2020.pkl']    
    bnames = ['top_100_{}_2020.pkl'.format(bias) for bias in biases]
    top_influencers_2020 = []
    influencer_tags_2020 = []
    for bname in bnames:
        if '2020' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2020.pkl', '')
            target_influencers = pickle.load(open(influencer_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2020:
                    influencer_tags_2020.append(influencer_name)
                    top_influencers_2020.append(influencer) 

    assert len(top_influencers_2020) == len(influencer_tags_2020)

    tag_map_2020 = {}
    for i, influencer in enumerate(top_influencers_2020):
        tag_map_2020[influencer] = influencer_tags_2020[i]

    #bnames = ['top_100_fake_2016.pkl', 'top_100_extreme_bias_right_2016.pkl', 'top_100_right_2016.pkl', 'top_100_lean_right_2016.pkl', 'top_100_center_2016.pkl', 'top_100_lean_left_2016.pkl', 'top_100_left_2016.pkl', 'top_100_extreme_bias_left_2016.pkl']
    bnames = ['top_100_{}_2016.pkl'.format(bias) for bias in biases]
    top_influencers_2016 = []
    influencer_tags_2016 = []
    for bname in bnames:
        if '2016' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(influencer_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2016:
                    influencer_tags_2016.append(influencer_name)
                    top_influencers_2016.append(influencer)

    assert len(top_influencers_2016) == len(influencer_tags_2016)

    tag_map_2016 = {}
    for i, influencer in enumerate(top_influencers_2016):
        tag_map_2016[influencer] = influencer_tags_2016[i]

    joined_influencers = list(set(top_influencers_2020) & set(top_influencers_2016))
    joined_tags = []

    for influencer in joined_influencers:
        try: # For 2020 version, switch try and except calls
            tag = tag_map_2016[influencer]
        except:
            tag = tag_map_2020[influencer]

        joined_tags.append(tag)

    assert len(joined_influencers) == len(joined_tags)

    return joined_influencers, joined_tags, keep_order

def add_extra_influencers():
    additional_influencers = pickle.load(open('/path/to/influencer_pkls/influencers_2016.pkl', 'rb'))
    user_ids = []
    user_tags = []
    for profile in additional_influencers:
        if int(profile['user_id']) not in user_ids:
            user_tags.append(profile['category'].strip().replace(' ', '_'))
            user_ids.append(int(profile['user_id']))
    
    return user_ids, user_tags

def get_top_n_by_news(top_n = 5, add_extra=False):
    #bnames = ['top_100_fake_2016.pkl', 'top_100_extreme_bias_right_2016.pkl', 'top_100_right_2016.pkl', 'top_100_lean_right_2016.pkl', 'top_100_center_2016.pkl', 'top_100_lean_left_2016.pkl', 'top_100_left_2016.pkl', 'top_100_extreme_bias_left_2016.pkl']
    bnames = ['top_100_{}_2016.pkl'.format(bias) for bias in biases]

    top_influencers = []
    influencer_tags = []
    for bname in bnames:
        if '2016' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(influencer_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers:
                    influencer_tags.append(influencer_name)
                    top_influencers.append(influencer)

    keep_order = True

    if add_extra:
        extra_users, extra_tags = add_extra_influencers()
        for i, user_id in enumerate(extra_users):
            if user_id not in top_influencers:
                top_influencers.append(user_id)
                influencer_tags.append(extra_tags[i])

    assert len(top_influencers) == len(influencer_tags)
    return top_influencers, influencer_tags, keep_order

def get_retweet_edges(top_influencers_map):
    edges = {}
    ftarget_dir = "/path/to/retweet_networks_not_anonymous/"
    fnames = [f for f in listdir(ftarget_dir) if isfile(join(ftarget_dir, f))]
    for fname in fnames:
        if '.csv' in fname:
            print(fname)
            with open(ftarget_dir + fname, 'r') as reader:
                for line in reader:                    
                    line = line.rstrip('\n')
                    line = line.split(',')
                    infl_id = int(line[0])
                    auth_id = int(line[1])
                    tid = int(line[2])

                    # NOTE: Alternate conditionals
                    try: # infl_id must be one of the top influencer
                        check = top_influencers_map[infl_id]
                    except:
                        continue

                    try: # auth_id must NOT be a top influencer
                        check = top_influencers_map[auth_id]
                        continue
                    except:
                        pass

                    try:
                        edges[infl_id].append(auth_id)
                    except:
                        edges[infl_id] = [auth_id]
    return edges

def get_full_retweet_edges(top_influencers_map):
    #conn = sqlite3.connect('/path/to/2016_election_sqlite3/data/complete_trump_vs_hillary_db.sqlite')
    conn = sqlite3.connect(raw_retweets_2016)
    c = conn.cursor()

    edges = {}
    full_auth_list = []
    for row in c.execute('SELECT * FROM tweet_to_retweeted_uid'):
        
        infl_id = row[1]
        auth_id = row[2]
        
        if anonymized:
            # after getting the ids from the table, anonymize them so that they match the top100 influencer
            # data (which should also be anonymized)
            try:
                infl_id = user_id_to_anon_id[str(infl_id)]
                auth_id = user_id_to_anon_id[str(auth_id)]
            except:
                continue

        # NOTE: Alternate conditionals
        try: # infl_id must be one of the top influencer
            check = top_influencers_map[infl_id]
        except:
            continue

        try: # auth_id must NOT be a top influencer
            check = top_influencers_map[auth_id]
            continue
        except:
            pass

        try:
            edges[infl_id].append(auth_id)
            full_auth_list.append(auth_id)
        except:
            edges[infl_id] = [auth_id]
            full_auth_list.append(auth_id)

    full_auth_list = np.array(sorted(list(set(full_auth_list))))
    
    return edges, full_auth_list


def get_weighted_vec(occurrences, full_auth_list):
    vec = np.zeros(len(full_auth_list))

    inds = np.array(list(occurrences.keys()))
    vals = np.array(list(occurrences.values()))

    vec[np.searchsorted(full_auth_list, inds)] = vals

    return vec

@njit
def cos_sim_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu*vv)
    return cos_theta

@njit
def compute_similarity(edges):
    N = len(edges)
    M = np.full((N, N), -1.0)
    sims = []
    for i, edge_i in enumerate(edges):
        print(i + 1, '/', N)
        vec_i = edges[edge_i]
        for j, edge_j in enumerate(edges):
            if i != j and M[j,i] == -1.0:
                vec_j = edges[edge_j]

                res = cos_sim_numba(vec_i, vec_j)
                if not np.isnan(res):
                    sims.append(res)
                    M[i,j] = res
                else:
                    M[i,j] = 0.0

            elif i == j:
                M[i,i] = 1.0

    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i,j] == -1.0:
                M[i,j] = M[j,i]

    return M, sims

if __name__ == '__main__':
    print('Getting top influencers')
    #top_influencers, influencer_tags, keep_order = get_joined_top_n_by_news(top_n = 800) # 100, 200, 1000
    top_influencers, influencer_tags, keep_order = get_top_n_by_news(top_n = 800) # 100, 200, 1000

    top_influencers_map = {}
    for influencer in top_influencers:
        top_influencers_map[influencer] = 1

    print('Number of influencers:', len(top_influencers_map))

    print('Getting edges')
    edges, full_auth_list = get_full_retweet_edges(top_influencers_map)

    print('Formatting edges')

    typed_edges = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:],
    )

    if not keep_order:
        for edge in edges:
            typed_edges[edge]= get_weighted_vec(dict(collections.Counter(edges[edge])), full_auth_list)
    else:
        tmp_top_influencers = []
        tmp_influencer_tags = []
        for i, influencer in enumerate(top_influencers):
            if influencer in edges:
                tmp_top_influencers.append(influencer)
                tmp_influencer_tags.append(influencer_tags[i])

        for influencer in tmp_top_influencers:
            typed_edges[influencer] = get_weighted_vec(dict(collections.Counter(edges[influencer])), full_auth_list)
        
        influencer_tags = tmp_influencer_tags[:]

    nodes = list(typed_edges.keys())
    nodes = [int(x) for x in nodes]
    del edges

    print('Computing edge similarities')
    
    M, sims = compute_similarity(typed_edges)

    print('Max similarity:', np.max(sims))
    print('Min similarity:', np.min(sims))
    print('Mean similarity:', np.mean(sims))
    print('Median similarity:', np.median(sims))

    print(nodes)

    res = {'sim_matrix': M, 'nodes': nodes}

    if len(influencer_tags) > 0:
        res['tags'] = influencer_tags

    makedirs(join(save_dir), exist_ok=True)
    pickle.dump(res, open(join(save_dir, 'sim_network_large_2016.pkl'), 'wb'))