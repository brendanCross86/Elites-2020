from __future__ import unicode_literals

from os import listdir, makedirs
from os.path import isfile, join
import collections
import sys, traceback

import numpy as np

import pickle
import orjson as json

import pandas as pd

import time

INFLUENCER_DIR = '../data/influencers/top_100/'
USER_PROFILE_DIR = './user_profiles/'
SAVE_DIR = '../data/maps/'
biases = ['fake', 'right_extreme', 'right', 'right_leaning', 'center', 'left_leaning', 'left', 'left_extreme']

def get_target_influencers(top_n = 25):
    #btarget_dir = "/home/flamij/ci/ci/analysis/top_100_influencer_pkls/"
    btarget_dir = INFLUENCER_DIR
    #bnames = ['top_100_fake_2020.pkl', 'top_100_extreme_bias_right_2020.pkl', 'top_100_right_2020.pkl', 'top_100_lean_right_2020.pkl', 'top_100_center_2020.pkl', 'top_100_lean_left_2020.pkl', 'top_100_left_2020.pkl', 'top_100_extreme_bias_left_2020.pkl']    
    bnames = ['top_100_{}_2020.pkl'.format(bias) for bias in biases]
    top_influencers_2020 = []
    influencer_tags_2020 = {}
    for bname in bnames:
        if '2020' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2020.pkl', '')
            target_influencers = pickle.load(open(btarget_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2020:
                    try:
                        influencer_tags_2020[influencer].append(influencer_name)
                    except:
                        influencer_tags_2020[influencer] = [influencer_name]
                    
                    top_influencers_2020.append(influencer) 

    #bnames = ['top_100_fake_2016.pkl', 'top_100_extreme_bias_right_2016.pkl', 'top_100_right_2016.pkl', 'top_100_lean_right_2016.pkl', 'top_100_center_2016.pkl', 'top_100_lean_left_2016.pkl', 'top_100_left_2016.pkl', 'top_100_extreme_bias_left_2016.pkl']
    bnames = ['top_100_{}_2016.pkl'.format(bias) for bias in biases]
    top_influencers_2016 = []
    influencer_tags_2016 = {}
    for bname in bnames:
        if '2016' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(btarget_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2016:
                    try:
                        influencer_tags_2016[influencer].append(influencer_name)
                    except:
                        influencer_tags_2016[influencer] = [influencer_name]
                    
                    top_influencers_2016.append(influencer)

    return top_influencers_2020, influencer_tags_2020, top_influencers_2016, influencer_tags_2016

def get_user_profiles():
    fnames = [f for f in listdir(USER_PROFILE_DIR) if isfile(join(USER_PROFILE_DIR, f))]
    fnames = sorted(fnames)

    found_profiles = {}
    for fname in fnames:
        if '.pkl' in fname and 'missing' not in fname:
            profiles = pickle.load(open(USER_PROFILE_DIR + fname, 'rb'))
            found_profiles.update(profiles)

    return found_profiles

if __name__ == '__main__':
    top_influencers_2020, influencer_tags_2020, top_influencers_2016, influencer_tags_2016 = get_target_influencers()
    total_influencers = list(set(top_influencers_2020 + top_influencers_2016))
    
    print('2020', len(set(top_influencers_2020)))
    print('2016', len(set(top_influencers_2016)))

    user_profiles = get_user_profiles()
    w = open(join(SAVE_DIR, 'survey.tsv'), 'w', encoding='utf-8-sig')
    w.write('id\tfull_name\tlink\tdescription\tnews categories\tnew influencer\tlinked to media outlet\tlinked to political party\tindependent\tother\n')

    # id,full_name,link,description,news categories,new influencer,linked to media outlet,linked to political party,independent,other
    count = 0
    link_map = {}
    for influencer in total_influencers:
        if influencer in user_profiles:
            profile = user_profiles[influencer]

            id = str(int(profile['id']))
            screen_name = profile['screen_name']
            full_name = profile['name']
            link = 'https://twitter.com/' + screen_name
            link_map[link] = int(id)
  
            if 'math' in link:
                print('yes', link, id)
            
            if 'description' in profile:
                description = str(profile['description']).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
            else:
                description = ''

            if influencer in top_influencers_2020 and influencer not in top_influencers_2016:
                new_influencer = 1
            else:
                new_influencer = 0

            if new_influencer:
                try:
                    news_categories = influencer_tags_2020[influencer]
                except:
                    news_categories = influencer_tags_2016[influencer]
            else:
                try:
                    news_categories = influencer_tags_2016[influencer]
                except:
                    news_categories = influencer_tags_2020[influencer]

            news_categories = list(set(news_categories))
            news_categories = ','.join(news_categories)

            linked_to_media_output = '0'
            linked_to_political_party = '0'
            independent = '0'
            other = '0'

            w.write(id + '\t' + full_name + '\t' + link + '\t' + description + '\t' + news_categories + '\t' + str(new_influencer) + '\t' + linked_to_media_output + '\t' + linked_to_political_party + '\t' + independent + '\t' + other + '\n')
            count += 1
        else:
            continue


    makedirs(join(SAVE_DIR), exist_ok=True)
    pickle.dump(link_map, open(join(SAVE_DIR, 'link_map.pkl'), 'wb'))

    print(count, 'out of', len(total_influencers), 'influencers put into survey. The rest are permanently missing')

    w.close()