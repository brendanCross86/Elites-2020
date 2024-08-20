import pickle
import pandas as pd
import numpy as np

from os import listdir, makedirs
from os.path import isfile, join

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
 
import matplotlib.cm as cm

from itertools import cycle, islice

import sys

font = font_manager.FontProperties(family='Arial')


user_data = './data/'# '../data/user_profiles'
influencer_data = '../data/influencers/top_100/'
infl_classification_surveys = './answers/' #'../data/surveys/'
save_dir = './myplot/'
makedirs(join(save_dir), exist_ok=True)


biases = ['fake', 'right_extreme', 'right', 'right_leaning', 'center', 'left_leaning', 'left', 'left_extreme']


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="//", **kwargs):
    # Source: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas

    # my_colors = {'center': ['#CFDB00', '#595C1C'], 'extreme_bias_left': ['#082E4F', '#3E709C'], 'extreme_bias_right': ['#4F0906', '#9C3E3B'], 'fake': ['#282828', '#755252'], 'lean_left': ['#4495DB', '#89B8E1'], 'lean_right': ['#DB4742', '#E18A87'], 'left': ['#0E538F', '#589EDB'], 'right': ['#8F100B', '#DB5853']}
    my_colors = list(islice(cycle(['#7F00DB', '#01DB4E', '#FC9088', '#FCDF62', '#000000']), None, len(dfall[0])))

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=my_colors)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) - 0.2 + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0, fontname='Arial')
    # axe.set_ylabel('', fontname='Arial')
    # axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5], prop=font)
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1], prop=font) 
    axe.add_artist(l1)
    return axe

def get_top_n_influencers(category, year, top_n = 25):
    #btarget_dir = "/home/flamij/ci/ci/analysis/top_100_influencer_pkls/"
    btarget_dir = influencer_data
    bname = 'top_100_' + category + '_' + str(year) + '.pkl'
    print('Loading influencers from', bname)
    target_influencers = pickle.load(open(btarget_dir + bname, 'rb'))
    target_influencers = list(target_influencers.keys())[:top_n]
    return target_influencers

def get_data(classifications, category, year, top_n = 25):
    top_n_influencers = get_top_n_influencers(category, year, top_n = top_n)

    classification_count = {'media': 0, 'political': 0, 'independent': 0, 'other': 0}
    for influencer in top_n_influencers:
        classification_count[classifications[influencer]] += 1

    return classification_count

def set_target_influencers(top_n = 25):
    #btarget_dir = "/home/flamij/ci/ci/analysis/top_100_influencer_pkls/"
    btarget_dir = influencer_data
    #bnames = ['top_100_fake_2020.pkl', 'top_100_extreme_bias_right_2020.pkl', 'top_100_right_2020.pkl', 'top_100_lean_right_2020.pkl', 'top_100_center_2020.pkl', 'top_100_lean_left_2020.pkl', 'top_100_left_2020.pkl']    
    bnames = ['top_100_{}_2020.pkl'.format(bias) for bias in biases]
    top_influencers_2020 = []
    influencer_tags_2020 = []
    for bname in bnames:
        if '2020' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2020.pkl', '')
            target_influencers = pickle.load(open(btarget_dir + bname, 'rb'))
            target_influencers = list(target_influencers.keys())[:top_n]
            for influencer in target_influencers:
                if influencer not in top_influencers_2020:
                    influencer_tags_2020.append(influencer_name)
                    top_influencers_2020.append(influencer) 

    assert len(top_influencers_2020) == len(influencer_tags_2020)

    tag_map_2020 = {}
    for i, influencer in enumerate(top_influencers_2020):
        tag_map_2020[influencer] = influencer_tags_2020[i]

    #bnames = ['top_100_fake_2016.pkl', 'top_100_extreme_bias_right_2016.pkl', 'top_100_right_2016.pkl', 'top_100_lean_right_2016.pkl', 'top_100_center_2016.pkl', 'top_100_lean_left_2016.pkl', 'top_100_left_2016.pkl']
    bnames = ['top_100_{}_2016.pkl'.format(bias) for bias in biases]
    top_influencers_2016 = []
    influencer_tags_2016 = []
    for bname in bnames:
        if '2016' in bname:
            print('Loading', bname)
            influencer_name = bname.replace('top_100_', '').replace('_2016.pkl', '')
            target_influencers = pickle.load(open(btarget_dir + bname, 'rb'))
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

    return joined_influencers

if __name__ == '__main__':
    joined_influencers = set_target_influencers()
    #missing_users = pickle.load(open(join(user_data, 'missing_users.pkl'), 'rb'))

    link_map = pickle.load(open(join(user_data, 'link_map.pkl'), 'rb'))

    #target_dir = 'answers/'
    target_dir = infl_classification_surveys
    fnames = [f for f in listdir(target_dir) if isfile(join(target_dir, f))]

    print('=' * 60)
    print('Loading data')
    votes = {}
    info = {}
    for fname in fnames:
        if '.xlsx' in fname:
            print('Loading spreadsheet', fname)
            df = pd.read_excel(target_dir + fname, engine='openpyxl')
            
            ids = df['id'].values
            meida_link = df['linked to media outlet'].values
            political_link = df['linked to political party'].values
            independent = df['independent'].values
            other = df['other'].values

            full_names = df['full_name'].values
            alignments = df['news categories'].values
            links = df['link'].values

            for i, id in enumerate(ids):
                new_id = link_map[links[i]]

                ids[i] = new_id

            for i, id in enumerate(ids):
                is_media = bool(meida_link[i])
                is_political = bool(political_link[i])
                is_independent = bool(independent[i])
                is_other = bool(other[i])

                if id not in info:
                    info[id] = {'full name': full_names[i], 'news category': alignments[i], 'link': links[i]}

                try:
                    isnan = np.isnan(float(links[i]))
                except:
                    isnan = False

                if not isnan:
                    if 'jsolomonReports' in links[i]:
                        is_media = 1000
                    elif 'KellyannePolls' in links[i]:
                        is_political = 1000
                    elif 'FrankelJeremy' in links[i]:
                        is_media = 1000
                    elif 'dbongino' in links[i]:
                        is_media = 1000
                    elif 'TomFitton' in links[i]:
                        is_media = 1000
                    elif 'TAftermath2020' in links[i]:
                        is_other = 1000
                    elif 'JudgeJeaninefan' in links[i]:
                        is_other = 1000
                    elif 'RealMuckmaker' in links[i]:
                        is_independent = 1000
                    elif 'FedtheEffUp1' in links[i]:
                        is_other = 1000
                    elif 'j_starace' in links[i]:
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

            output_data[influencer] = info[influencer]
            output_data[influencer]['affiliation'] = vote_types[np.argmax(vote_count)]

            infl_classification[influencer] = vote_types[np.argmax(vote_count)]
            count += 1
        else:
            infl_classification[influencer] = 'other'

    print(len(infl_classification))
    # sys.exit()

    print(count, 'classifications assigned out of', len(joined_influencers))

    print(output_data)

    # pickle.dump(output_data, open('output/survey_links_v6.pkl', 'wb'))

    print('=' * 60)
    print('Sorting data')
    top_n = 25
    #categories = ['left', 'lean_left', 'center', 'lean_right', 'right', 'extreme_bias_right', 'fake']
    categories = [x for x in reversed(biases) if x != 'left_extreme']
    data_2020 = []
    data_2016 = []
    for category in categories:
        group_size = get_data(infl_classification, category, 2016, top_n = top_n)
        media_pct_2016 = float(group_size['media'] / sum(list(group_size.values()))) * 100
        political_pct_2016 = float(group_size['political'] / sum(list(group_size.values()))) * 100
        independent_pct_2016 = float(group_size['independent'] / sum(list(group_size.values()))) * 100
        other_pct_2016 = float(group_size['other'] / sum(list(group_size.values()))) * 100
        data_2016.append([media_pct_2016, political_pct_2016, independent_pct_2016, other_pct_2016])
        
        group_size = get_data(infl_classification, category, 2020, top_n = top_n)
        media_pct_2020 = float(group_size['media'] / sum(list(group_size.values()))) * 100
        political_pct_2020 = float(group_size['political'] / sum(list(group_size.values()))) * 100
        independent_pct_2020 = float(group_size['independent'] / sum(list(group_size.values()))) * 100
        other_pct_2020 = float(group_size['other'] / sum(list(group_size.values()))) * 100
        data_2020.append([media_pct_2020, political_pct_2020, independent_pct_2020, other_pct_2020])

    data_2016 = np.array(data_2016)
    data_2020 = np.array(data_2020)

    labels = ['Left', 'Left leaning', 'Center', 'Right leaning', 'Right', 'Extreme bias right', 'Fake news']

    df_2016 = pd.DataFrame(data_2016,
                    index=labels,
                    columns=['Media', 'Poli', 'Indp', 'Other'])
    df_2020 = pd.DataFrame(data_2020,
                    index=labels,
                    columns=['Media', 'Poli', 'Indp', 'Other'])

    print('=' * 60)
    print('Plotting data')
    # plt.figure(figsize=(10,6))
    ax = plot_clustered_stacked([df_2016, df_2020],['2016', '2020'])
    # plt.yticks(list(plt.yticks()[0]) + [50])
    for label in ax.get_xticklabels():
        label.set_fontproperties('Arial')    

    print(df_2016)
    print(df_2020)

    plt.ylabel('Percentage of influencer account types', **{'fontname':'Arial'})
    # plt.title('Hello', **{'fontname':'Arial'})
    # plt.axhline(y=50, color='black', linestyle='-')
    plt.xlabel
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(join(save_dir,'categorical_links_v6_no_ext_left_3.pdf'), dpi=500, format='pdf')