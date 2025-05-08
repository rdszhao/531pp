# %%
import os
import json
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
# %%
datadir = '../data/weekslite2'
modeldirs = '../data/topicmodels2'
topicjson = '../data/topics.json'
categories = '../data/topicbuckets.json'

files = [f"{datadir}/{fname}" for fname in os.listdir(datadir)]
modelpaths = sorted([f for f in os.listdir(modeldirs) if '.' not in f])

with open(topicjson, 'r') as f:
    topicmap = json.load(f)

with open(categories, 'r') as f:
    buckets = json.load(f)

id_buckets = [topic for topic in topicmap.keys()]
topicmap = {smalltopic: bigtopic for bigtopic, smalltopics in topicmap.items() for smalltopic in smalltopics}

cluster_labels = [''] * len(id_buckets[:-1])
for label, vals in buckets.items():
    for idx in vals['clusters']:
        cluster_labels[idx] = label

bigmap = dict(zip(id_buckets[:-1], cluster_labels))

outlet_weights = {
    'thenation' : -1,
    'democracynow' : -1,
    'nyt' : -1,
    'cnn' : -1,
    'nypost' : 1,
    'fox' : 1,
    'dailywire' : 1,
    'breitbart' : 1
}
# %%
dfs  = []
for file in tqdm(modelpaths):
    weeknum = int(file.split('_')[-1])
    modeldir = f"{modeldirs}/{file}"
    filepath = f"{datadir}/{file}.csv.gz"

    df = pd.read_csv(filepath)
    df['week'] = weeknum
    docs = df['text'].to_list()

    topic_model = BERTopic.load(modeldir)
    topic_model.reduce_topics(docs, nr_topics=45)
    reps = topic_model.get_document_info(docs)['Representation'].apply(lambda x: '_'.join(x)).to_list()

    df['weektopic'] = reps
    topic_mode_alignment = df.groupby('weektopic')['alignment'].agg(lambda x: x.mode().iloc[0])
    topic_labels = topic_mode_alignment.map(outlet_weights)
    df['position'] = df['weektopic'].map(topic_labels)
    df['bigtopic'] = df['weektopic'].map(topicmap)
    df['bucket'] = df['bigtopic'].map(bigmap)
    cols = ['retweetCount', 'likeCount', 'viewCount']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    dfs.append(df)

bigdf = pd.concat(dfs)
bigdf.to_csv('../data/processed.csv.gz', index=False) 
bigdf.drop('text', axis=1).to_csv('../data/processed_notext.csv.gz', index=False) 