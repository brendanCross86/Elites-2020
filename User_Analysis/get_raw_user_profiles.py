
import orjson as json
import sys, traceback
from datetime import datetime
import datetime as dt
import pickle
from os import makedirs
from os.path import join

duplicates = {}
user_id_data = {}
target_user_ids = {}


INFLUENCER_DIR = '../data/influencers/top_100/'
SAVE_DIR = './user_profiles/'
biases = ['fake', 'right_extreme', 'right', 'right_leaning', 'center', 'left_leaning', 'left', 'left_extreme']


def catch_key(key, d):
    try:
        return d[key]
    except:
        return -1

def catch_key_extension(extension, key, d):
    try:
        return d[extension][key]
    except:
        return -1

def catch_key_bool_extension(extension, key, d):
    try:
        if bool(d[extension][key]) == True:
            return 1
        else:
            return 0
    except:
        return 0

def submit_tweet(data):
    id = data["id"]
    
    try:
        duplicates[id] += 1
        return 
    except:
        duplicates[id] = 1
    
    user_id = catch_key_extension("user", "id", data)
    
    if user_id != -1:
        try:
            target_user_ids[user_id] += 1
            user_id_data[user_id] = data["user"]
        except:
            pass

        retweet_id = catch_key_extension("retweeted_status", "id", data)
        quoted_id = catch_key_extension("quoted_status", "id", data)
        
        if retweet_id != -1:
            retweeted_data = data["retweeted_status"]
            submit_tweet(retweeted_data)

        if quoted_id != -1:
            quoted_data = data["quoted_status"]
            submit_tweet(quoted_data)

def extract_fname(fname):
    if "/" in fname:
        fname_list = fname.split("/")
        target_fname = fname_list[len(fname_list) - 1]
        target_fname_list = target_fname.split(".")
        return target_fname_list[0]
    else:
        fname_list = fname.split(".")
        return fname_list[0]

def set_target_influencers(top_n = 25):
    #btarget_dir = "/home/flamij/ci/ci/analysis/top_100_influencer_pkls/"
    btarget_dir = INFLUENCER_DIR

    #bnames = ['top_100_fake_2020.pkl', 'top_100_extreme_bias_right_2020.pkl', 'top_100_right_2020.pkl', 'top_100_lean_right_2020.pkl', 'top_100_center_2020.pkl', 'top_100_lean_left_2020.pkl', 'top_100_left_2020.pkl', 'top_100_extreme_bias_left_2020.pkl']    
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

    #bnames = ['top_100_fake_2016.pkl', 'top_100_extreme_bias_right_2016.pkl', 'top_100_right_2016.pkl', 'top_100_lean_right_2016.pkl', 'top_100_center_2016.pkl', 'top_100_lean_left_2016.pkl', 'top_100_left_2016.pkl', 'top_100_extreme_bias_left_2016.pkl']
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

    for influencer in joined_influencers:
        target_user_ids[influencer] = 1


if __name__ == "__main__":
    print("Program Begin.")

    
    if (len(sys.argv) > 1):

        # as an argument we take a file path pointing to the raw 202001.lj - 202011.lj
        # files containing raw tweet data
        fname = sys.argv[1]

        set_target_influencers()
        print(len(target_user_ids), 'target users needed')

        count = 0
        with open(fname, "r") as ins:
            for line in ins:
                count += 1
                # print(count, end="\r", flush=True)
                try:
                    data = json.loads(line)
                except Exception:
                    print("-"*100)
                    print("orjson.loads failed for:", line)
                    print("Exception in code:")
                    print("-"*100)
                    traceback.print_exc(file=sys.stdout)
                    print("-"*100)
                    continue
                submit_tweet(data)

        print(len(user_id_data), 'of users found out of', len(target_user_ids), 'required')

        makedirs(join(SAVE_DIR), exist_ok=True)
        pickle.dump(user_id_data, open(join(SAVE_DIR, extract_fname(fname) + '_user_profiles.pkl'), 'wb'))

    else:
        print("No argument given.")

    print("Program End.")