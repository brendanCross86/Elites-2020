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
import graph_tool as gt
import graph_tool.centrality as gc

from os import listdir, makedirs
from os.path import isfile, join

top_n = 100
year = 2016 # 2016 2020
biases = ['center', 'left', 'right', 'fake', 'left_leaning', 'right_leaning', 'left_extreme', 'right_extreme']

#target_dir = "/path/to/generate_ci/graphs/" + str(year) + '/'
# This is the path to the <bias>_<year>_ci.gt files, produced by compute_CI_retweet_networks.py script in Collective_Influence
target_dir = "../data/ci_output/graphs/" + str(year) + '/'
out_dir = '../data/influencers/top_100'

fnames = [f for f in listdir(target_dir) if isfile(join(target_dir, f))]

# Create output directory
makedirs(join(out_dir), exist_ok=True)


for fname in fnames:
    if 'combined' in fname: 
        continue
    if ".gt" in fname:
        oname = fname.replace('_ci.gt', '')

        top_100 = {}

        print('Loading', fname)

        graph = gt.load_graph(target_dir + fname)

        # print('Assembling CI list')
        res = []
        for vertex in graph.vertices():
            res.append((graph.vp.CI_in[vertex], graph.vp.CI_out[vertex], graph.vp.user_id[vertex]))

        # print('Sorting by CI-out')
        res = sorted(res, key=lambda x: x[1], reverse=True)

        for rank, item in enumerate(res[:top_n]):  
            top_100[int(item[2])] = 1            

        
        pickle.dump(top_100, open(join(out_dir, 'top_100_' + oname + '.pkl'), 'wb'))
