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
from os import listdir
from os.path import isfile, join
import sys
import json

user_map = {}
# edit this to choose which years to run
year = [2016, 2020]
target_dir = '/path/to/raw/users/'
out_dir = '../data/maps/'
anonymize = True
anonymous_map = '../data/maps/'

if anonymize:
    users_to_anon = json.load(open(join(anonymous_map, 'user_id_to_anon_id_extended.json'), 'r'))

if 2020 in year:
    print('Assembling 2020 user map')
    fnames = [f for f in listdir(target_dir) if isfile(join(target_dir, f))]
    for fname in fnames:
        if '.csv' in fname:
            print('Reading', fname)

            with open(target_dir + fname, 'r') as reader:
                for line in reader:
                    line = line.rstrip('\n')
                    line = line.split(',')

                    # print(line)

                    try:
                        id = int(line[0])
                        verified = int(line[-1])
                        created_at = str(line[1])
                        name = str(line[-2])
                            
                        if anonymize:
                            id = users_to_anon[str(id)]
                            name = "" # don't display handle for anonymized users. If certain users are allowed, handle it in future step

                        user_map[id] = {'created_at': created_at, 'name': name, 'verified': verified}
                    except:
                        continue

    print('Map size:', len(user_map))
    pickle.dump(user_map, open(out_dir + 'user_map_2020.pkl', 'wb'))

if 2016 in year:
    print('Assembling 2016 user map')
    import sqlite3

    user_map = {}

    SQL_PATH = '/path/to/2016_election_sqlite3/data/complete_trump_vs_hillary_db.sqlite'
    conn = sqlite3.connect(SQL_PATH)
    c = conn.cursor()

    counts = {'in': 0, 'not in': 0}
    todo = []
    for row in c.execute('SELECT * FROM influencer_rank_date'):
        # print(row)

        id = int(row[5])
        created_at = str(row[4])
        name = str(row[6]).replace('@', '')

        
        if anonymize:
            if str(id) in users_to_anon:
                counts['in'] += 1
                id = users_to_anon[str(id)]
                name = "" # don't display handle for anonymized users. If certain users are allowed, handle it in future step
            else:
                counts['not in'] += 1
                #if '???' not in name:
                todo.append((id, name))
                #print('not in:', id, name)
                # add the 

        if '???' not in name:
            user_map[id] = {'created_at': created_at, 'name': name}

    print('in:', counts['in'], 'not in:', counts['not in'])
    print('Map size:', len(user_map))

    pickle.dump(user_map, open(out_dir + 'user_map_2016.pkl', 'wb'))
