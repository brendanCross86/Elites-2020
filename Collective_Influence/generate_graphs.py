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

import graph_tool as gt
#import graph_tool.stats as gts
import graph_tool.generation as gtg

# Edit this to point at your <bias>_retweet_edges.csv files.
BASE_PATH = '../data'

def load_graph_csv(file, stance, year, for_ci = True):
    print('Reading csv')
    edge_list = []
    first_line = True
    with open(file, 'r') as reader:
        for line in reader:
            if first_line:
                first_line = False
                continue

            line = line.rstrip('\n')
            line = line.split(',')

            if year == 2020:
                # 2020 edgelists are ordered: tweet_id, author_id, influencer_id
                edge_list.append((line[2], line[1], line[0])) # infid, authid, tid
            elif year == 2016:
                # 2016 edgelists are ordered: influencer_id, author_id, tweet_id
                edge_list.append((line[0], line[1], line[2])) # infid, authid, tid

    print('Loading graph')
    G = gt.Graph(directed=True)
    G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
    G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    G.edge_properties['source_id'] = G.new_edge_property('int64_t')
    G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id, G.ep.source_id])

    G.gp['name'] = G.new_graph_property('string', stance + '_' + str(year))

    if for_ci: # remove parallel edges for CI analysis
        gtg.remove_parallel_edges(G)
        gtg.remove_self_loops(G)

    G.save(BASE_PATH + str(year) + '/' + stance + '_' + str(year) + '.gt')

if __name__ == '__main__':
    # EDIT THIS
    #base_path = BASE_PATH # '../data/'

    years = [2016, 2020]
    #years = [2016]#[2020]

    #stances = ['center', 'extreme_bias_left', 'extreme_bias_right', 'fake', 'lean_left', 'lean_right', 'left', 'right']
    stances = ['center', 'left_extreme', 'right_extreme', 'fake', 'left_leaning', 'right_leaning', 'left', 'right']

    for year in years:
        for stance in stances:
            fname = BASE_PATH + str(year) + '/' + stance + '_' + 'retweet_edges.csv'
            print(fname)
            load_graph_csv(fname, stance, year)