import pickle
import pandas as pd
import numpy as np

from os import listdir, makedirs
from os.path import isfile, join

import json

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
 
import matplotlib.cm as cm

from itertools import cycle, islice

import sys

font = font_manager.FontProperties(family='Arial')


INFLUENCER_DIR = '../data/influencers/top_100/'
USER_PROFILE_DIR = './user_profiles/'
ANSWER_DIR = '../data/survey_answers/'
MAPS_DIR = '../data/maps/'
SAVE_DIR = '../data/maps/'
makedirs(join(SAVE_DIR), exist_ok=True)

biases = ['fake', 'right_extreme', 'right', 'right_leaning', 'center', 'left_leaning', 'left', 'left_extreme']

def get_top_n_influencers(category, year, top_n = 25):
    bname = 'top_100_' + category + '_' + str(year) + '.pkl'
    print('Loading influencers from', bname)
    target_influencers = pickle.load(open(INFLUENCER_DIR + bname, 'rb'))
    target_influencers = list(target_influencers.keys())[:top_n]
    return target_influencers

def get_data(classifications, category, year, top_n = 25):
    top_n_influencers = get_top_n_influencers(category, year, top_n = top_n)

    classification_count = {'media': 0, 'political': 0, 'independent': 0, 'other': 0}
    for influencer in top_n_influencers:
        classification_count[classifications[influencer]] += 1

    return classification_count

def set_target_influencers(top_n = 25):
    bnames = ['top_100_{}_2020.pkl'.format(bias) for bias in biases]
    top_influencers_2020 = []
    influencer_tags_2020 = []
    for bname in bnames:
        if '2020' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2020.pkl', '')
            target_influencers = pickle.load(open(INFLUENCER_DIR + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2020:
                    influencer_tags_2020.append(influencer_name)
                    top_influencers_2020.append(influencer) 

    assert len(top_influencers_2020) == len(influencer_tags_2020)

    tag_map_2020 = {}
    for i, influencer in enumerate(top_influencers_2020):
        tag_map_2020[influencer] = influencer_tags_2020[i]

    bnames = ['top_100_{}_2016.pkl'.format(bias) for bias in biases]
    top_influencers_2016 = []
    influencer_tags_2016 = []
    for bname in bnames:
        if '2016' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(INFLUENCER_DIR + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2016:
                    influencer_tags_2016.append(influencer_name)
                    top_influencers_2016.append(influencer)

    assert len(top_influencers_2016) == len(influencer_tags_2016)

    tag_map_2016 = {}
    for i, influencer in enumerate(top_influencers_2016):
        tag_map_2016[influencer] = influencer_tags_2016[i]

    joined_influencers = list(set(top_influencers_2016 + top_influencers_2020))

    return joined_influencers, top_influencers_2016, top_influencers_2020

if __name__ == '__main__':
    joined_influencers, top_influencers_2016, top_influencers_2020 = set_target_influencers()
    #missing_users = pickle.load(open('user_profiles/missing_users.pkl', 'rb'))
    #link_map = pickle.load(open(join(MAPS_DIR, 'link_map.pkl'), 'rb'))

    fnames = [f for f in listdir(ANSWER_DIR) if isfile(join(ANSWER_DIR, f))]

    print('=' * 60)
    print('Loading data')
    votes = {}
    info = {}
    for fname in fnames:
        if '.xlsx' in fname:
            print('Loading spreadsheet', fname)
            df = pd.read_excel(ANSWER_DIR + fname, engine='openpyxl')
            
            ids = df['id'].values
            meida_link = df['linked to media outlet'].values
            political_link = df['linked to political party'].values
            independent = df['independent'].values
            other = df['other'].values

            #full_names = df['full_name'].values
            #alignments = df['news categories'].values
            #links = df['link'].values

            #for i, id in enumerate(ids):
            #    new_id = link_map[links[i]]
            #    ids[i] = new_id

            for i, id in enumerate(ids):
                is_media = bool(meida_link[i])
                is_political = bool(political_link[i])
                is_independent = bool(independent[i])
                is_other = bool(other[i])

                #if id not in info:
                #    info[id] = {'full name': full_names[i], 'news category': alignments[i], 'link': links[i]}

                try:
                    isnan = np.isnan(float(links[i]))
                except:
                    isnan = False


                if not isnan:
                    if id == 2025317:
                        is_media = 1000
                    elif id == 3436929:
                        is_political = 1000
                    elif id == 1517960:
                        is_media = 1000
                    elif id == 2738861:
                        is_media = 1000
                    elif id == 776729:
                        is_media = 1000
                    elif id == 910429:
                        is_other = 1000
                    elif id == 2075698:
                        is_other = 1000
                    elif id == 2473454:
                        is_independent = 1000
                    elif id == 223297:
                        is_other = 1000
                    elif id == 4689768:
                        is_independent = 1000

                if id not in votes:
                    votes[id] = {'media': 0, 'political': 0, 'independent': 0, 'other': 0}
                    votes[id]['media'] += int(is_media)
                    votes[id]['political'] += int(is_political)
                    votes[id]['independent'] += int(is_independent)
                    votes[id]['other'] += int(is_other)
                else:
                    votes[id]['media'] += int(is_media)
                    votes[id]['political'] += int(is_political)
                    votes[id]['independent'] += int(is_independent)
                    votes[id]['other'] += int(is_other)

    print(len(votes), 'influencers scanned')


    print('=' * 60)
    print('Analyzing data')
    infl_classification = {}
    output_data = {}
    count = 0
    for influencer in set(list(joined_influencers) + list(votes.keys())):
        if influencer in votes:
            infl_votes = votes[influencer]

            # If tie, opt for media > political > independent > other 
            vote_types = np.array(list(infl_votes.keys()))
            vote_count = np.array(list(infl_votes.values()))
            
            if len(np.where(vote_count == np.max(vote_count))[0]) > 1:
                print(influencer, 'has a tie:', votes[influencer])

            #output_data[influencer] = info[influencer]
            #output_data[influencer]['affiliation'] = vote_types[np.argmax(vote_count)]

            infl_classification[influencer] = vote_types[np.argmax(vote_count)]
            count += 1
        else:
            infl_classification[influencer] = 'other'

    rename = {'media': 'media', 'political': 'polit', 'independent': 'indep', 'other': 'other'}
    for influencer in infl_classification:
        if influencer in top_influencers_2016 and influencer in top_influencers_2020:
            infl_classification[influencer] = rename[infl_classification[influencer]] + '_both'
        elif influencer in top_influencers_2016 and influencer not in top_influencers_2020:
            infl_classification[influencer] = rename[infl_classification[influencer]] + '_2016'
        elif influencer not in top_influencers_2016 and influencer in top_influencers_2020:
            infl_classification[influencer] = rename[infl_classification[influencer]] + '_2020'
        else:
            infl_classification[influencer] = rename[infl_classification[influencer]] + '_both'

    pickle.dump(infl_classification, open(join(SAVE_DIR, "infl_affiliation_map_no_handles.pkl"), "wb"))

    infl_classification_json = {}
    for influencer in infl_classification:
        infl_classification_json[int(influencer)] = infl_classification[influencer]
    

    with open(join(SAVE_DIR, "infl_affiliation_map_no_handles.json"), "w") as outfile:
        json.dump(infl_classification_json, outfile)


    # also create a version that has the user handles of allowed users
    
    allowed_users = json.load(open(join(MAPS_DIR, 'allowed_users_anon_id_to_handle.json')))
    for influencer in infl_classification_json:
        if str(influencer) in allowed_users:
            infl_classification_json[influencer] = allowed_users[str(influencer)]

    with open(join(SAVE_DIR, "infl_affiliation_map.json"), "w") as outfile:
        json.dump(infl_classification_json, outfile)