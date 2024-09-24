import pandas as pd
import json
import os


AFFILIATION_MAP = './infl_affiliation_map.json'
# provide the path to the top_influencer_<bias>.csv files, produced by the elites_network_analysis.py file
TOP_INFLUENCERS = './'

anon_id_to_handle = json.load(open(AFFILIATION_MAP, 'r'))
bias_name_map = {'left': 'left', 'left leaning': 'lean_left', 'right leaning': 'lean_right', 'fake': 'fake',
                 'left extreme': 'extreme_bias_left', 'right extreme': 'extreme_bias_right', 'right': 'right', 'center': 'center'}
tops = {}
for year in [2016, 2020]:
    rank_key = 'rank_2016' if year == 2016 else 'rank_2020'
    for bias in ['left', 'left leaning', 'center', 'right leaning', 'right', 'right extreme', 'fake']:
        df = pd.read_csv(os.path.join(TOP_INFLUENCERS, 'top_influencers_{}.csv'.format(bias)))
        top = df[df[rank_key] <= 100].copy()
        top['type'] = bias_name_map[bias]
        top['name'] = top['user_id'].map(lambda x: anon_id_to_handle[str(x)] if str(x) in anon_id_to_handle else '')

        tops[bias] = top

    merged = pd.concat([x for x in tops.values()])

    # due to being removed from twitter, the rank 2 user from 2016 was truncated from the top influencer table, adding it back in
    merged = pd.concat([merged, pd.DataFrame([{'user_id': 3163930, 'name': anon_id_to_handle["3163930"],  'CI_2016': 0, 'CI_2020': 0, 'rank_2016': 2.0, 
                        'bias': 'fake', 'type': 'fake', 'user_handle': anon_id_to_handle["3163930"], 'verified': 1}])])

    merged = merged.sort_values(['type', rank_key])[['user_id', 'name', 'type', rank_key]].rename(columns={'user_id': 'id', rank_key: 'rank'})
  
    merged.to_csv('anon_all_influencers_data_{}.csv'.format(year), index=False)
    merged[merged['rank'] <= 5].to_csv('anon_all_top_5_cat_{}.csv'.format(year), index=False)