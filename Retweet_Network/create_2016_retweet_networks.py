# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0


import os

import sqlite3
import numpy as np
from datetime import datetime
import pandas as pd


import time


def buildGraphSqlite(conn, graph_type, start_date, stop_date,
                       save_filename=None,
                       additional_sql_select_statement=None,):
    """ Returns graph for interaction types in `graph_type` from sqldatabase,
        using the graph library graph_lib.

        Notes
        -----
        tweets are selected such that `start_date` <= tweet timestamp < `stop_date`.


        `additional_sql_select_statement` can be use to add a condition of the
        tweet ids. Must start by "SELECT tweet_id FROM ... "


        edge_list returns a numpy array of edges with (influencer_id, tweet_author_id, tweet_id)

    """

    c = conn.cursor()


    # transform the list of graph types to a list of table names
    graph_type_table_map = {'retweet': 'tweet_to_retweeted_uid',
                            'reply' : 'tweet_to_replied_uid',
                            'mention' : 'tweet_to_mentioned_uid',
                            'quote' : 'tweet_to_quoted_uid'}

    # table_name to influencer col_name
    table_to_col_map = {'tweet_to_retweeted_uid' : 'retweeted_uid',
                            'tweet_to_replied_uid': 'replied_uid',
                            'tweet_to_mentioned_uid' : 'mentioned_uid',
                            'tweet_to_quoted_uid' : 'quoted_uid'}

    table_names = []
    if isinstance(graph_type, str):
        if graph_type == 'all':
            table_names = list(graph_type_table_map.values())
        else:
            graph_type = [graph_type]

    if isinstance(graph_type, list):
        for g_type in graph_type:
            if g_type in graph_type_table_map.keys():
                table_names.append(graph_type_table_map[g_type])
            else:
                raise ValueError('Not implemented graph_type')



    table_queries = []
    values = []
    for table in table_names:

        sql_select = """SELECT tweet_id, {col_name}, author_uid
                     FROM {table}
                     WHERE tweet_id IN
                         (
                         SELECT tweet_id
                         FROM tweet
                         WHERE datetime_EST >= ? AND datetime_EST < ?
                         )""".format(table=table, col_name=table_to_col_map[table])


        values.extend([start_date, stop_date])



        if additional_sql_select_statement is not None:
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         """ + additional_sql_select_statement + """
                         )
                         """])




        table_queries.append(sql_select)

    # take union of all the interaction type tables
    sql_query = '\nUNION \n'.join(table_queries)


#    print(sql_query)
    c.execute(sql_query, values)

 
    G = np.array([(infl_uid, auth_uid, tweet_id ) for tweet_id,
                              infl_uid, auth_uid in c.fetchall()],
                              dtype=np.int64)


    return G



save_dir = '../data'

#%% load user and tweet list

tweet_db_file1 = '../databases_ssd/complete_trump_vs_hillary_db.sqlite'
tweet_db_file2 = '../databases_ssd/complete_trump_vs_hillary_sep-nov_db.sqlite'
urls_db_file = '../databases_ssd/urls_db.sqlite'

sql_retweets = """SELECT tweet_id, retweeted_uid, author_uid FROM tweet_to_retweeted_uid 
   WHERE tweet_id IN (
            SELECT tweet_id FROM tweet
            WHERE datetime_EST > '2016-06-01' AND datetime_EST < '2016-11-09'
	)"""


sql_query_urls = """SELECT tweet_id FROM urls.urls
                         WHERE final_hostname IN (
                                 SELECT hostname FROM hosts_{med_type}_rev_stat
                                 WHERE perccum > 0.01)
                         """
                         

                         
media_types = ['fake', 'far_right', 'right', 'lean_right', 'center', 'lean_left', 'left',
            'far_left']



t00 = time.time()            
t0 = time.time()            
#
start_date = datetime(2016,6,1)
stop_date = datetime(2016,11,9)
edges_db_file = dict()

# get edges list

#set tmp dir
os.environ['SQLITE_TMPDIR'] = '/home/tmp'

for tweet_db_file in [tweet_db_file1, tweet_db_file2]:
    print(tweet_db_file)
    
    edges_db_file[tweet_db_file] = dict()
    with sqlite3.connect(tweet_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
    
        c = conn.cursor()
        c.execute("ATTACH '{urls_db_file}' AS urls".format(urls_db_file=urls_db_file))
        

        
    for media_type in media_types:
        print(media_type)
        edges_db_file[tweet_db_file][media_type] = buildGraphSqlite(conn, 
                                  graph_type='retweet', 
                                  start_date=start_date,
                                  stop_date=stop_date,
                                  additional_sql_select_statement=sql_query_urls.format(med_type=media_type),
                                  graph_lib='edge_list')
    
        print(time.time() - t0)
             

print(time.time() - t0)




#%%
# build and save each graph    
t0 = time.time()        
for media_type in edges_db_file[tweet_db_file1].keys():
                        
    print(media_type)

    edges_array = np.concatenate((edges_db_file[tweet_db_file1][media_type],
                                              edges_db_file[tweet_db_file2][media_type]))
    
    
    pd.DataFrame(edges_array).to_csv(os.path.join(save_dir, media_type + 'retweet_edges.csv'),
                                     header=False, index=False)
                

        
        

print('total time')
print(time.time() - t00)