# %%
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from collections import Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import HDBSCAN
from umap import UMAP
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# %%
def clean_representation(words) -> str:
    cleaned = []
    seen = set()
    for w in words:
        w_low = w.lower()
        if w_low in ENGLISH_STOP_WORDS or 'http' in w_low:
            continue
        lemma = lemmatizer.lemmatize(w_low)
        if lemma not in seen:
            seen.add(lemma)
            cleaned.append(lemma)
    return '_'.join(cleaned)

datadir = '../data/weekslite2'
modeldirs = '../data/topicmodels2'
modelpaths = sorted([f for f in os.listdir(modeldirs) if '.' not in f])

alltopics = []
for file in tqdm(modelpaths):
    modeldir = f"{modeldirs}/{file}"
    filepath = f"{datadir}/{file}.csv.gz"
    df = pd.read_csv(filepath)
    docs = df['text'].to_list()
    topic_model = BERTopic.load(modeldir)
    topic_model.reduce_topics(docs, nr_topics=45)
    reps = topic_model.get_topic_info()['Representation'].to_list()
    alltopics += ['_'.join(rep) for rep in reps]

alltopics = list(set(alltopics))
allwords = [word for topic in alltopics for word in topic.split('_')]
allwords = list(set(allwords))
cleaned = [word for word in allwords if not (word == '' or word.lower() in ENGLISH_STOP_WORDS or 'http' in word.lower())]
model = SentenceTransformer('all-MiniLM-L6-v2')
word_embeddings = model.encode(cleaned, show_progress_bar=True)
wordmap = dict(zip(cleaned, word_embeddings))
embeddings = [np.mean([wordmap[word] for word in topic.split('_') if word in cleaned], axis=0) for topic in alltopics]
# %%
reducer = UMAP(n_neighbors=100, n_components=5, min_dist=0.0, metric='cosine')
embeddings = reducer.fit_transform(embeddings)
clusterer = HDBSCAN(min_cluster_size=20, metric='cosine')
labels = clusterer.fit_predict(embeddings)

vectorizer = TfidfVectorizer(
    tokenizer=lambda s: s.split('_'),
    lowercase=False,
    token_pattern=None,
    stop_words='english'

)
X = vectorizer.fit_transform(alltopics)
terms = vectorizer.get_feature_names_out()

cluster_ids = sorted(set(labels))
top_n = 10
cluster_keywords = {}
for cid in cluster_ids:
    if cid == -1:
        continue
    member_mask = (labels == cid)
    cluster_tfidf_sum = X[member_mask].sum(axis=0).A1
    top_indices = np.argsort(cluster_tfidf_sum)[::-1][:top_n]
    cluster_keywords[cid] = [terms[i] for i in top_indices]

counter = Counter(labels)
print(len(counter))
print(counter[-1])

labels[labels == -1] = labels.max() + 1
for cid, kws in cluster_keywords.items():
    print(f"cluster {cid:>2}: {kws}")

topicbuckets = [['_'.join(kws), []] for kws in cluster_keywords.values()]
topicbuckets.append(['other', []])
for topic, cluster_id in zip(alltopics, labels):
    topicbuckets[cluster_id][1].append(topic)
# %%
topicdict = {tstring: topics for tstring, topics in topicbuckets}
with open('../data/topics2.json', 'w') as f:
    json.dump(topicdict, f, indent=2)
# %% fitting
results = []
while len(results) < 50:
    reducer = UMAP(n_neighbors=100, n_components=5, min_dist=0.0, metric='cosine')
    embeddings = reducer.fit_transform(embeddings)
    clusterer = HDBSCAN(min_cluster_size=18, metric='cosine')
    labels = clusterer.fit_predict(embeddings)

    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: s.split('_'),
        lowercase=False,
        token_pattern=None,
        stop_words='english'

    )
    X = vectorizer.fit_transform(alltopics)
    terms = vectorizer.get_feature_names_out()

    cluster_ids = sorted(set(labels))
    top_n = 10
    cluster_keywords = {}
    for cid in cluster_ids:
        if cid == -1:
            continue
        member_mask = (labels == cid)
        cluster_tfidf_sum = X[member_mask].sum(axis=0).A1
        top_indices = np.argsort(cluster_tfidf_sum)[::-1][:top_n]
        cluster_keywords[cid] = [terms[i] for i in top_indices]

    counter = Counter(labels)
    num_clusters = len(counter)
    num_outliers = counter[-1]

    labels[labels == -1] = labels.max() + 1
    keywords = set([kw for kws in cluster_keywords.values() for kw in kws])
    if 'genocide' not in keywords or num_outliers >= 100:
        continue

    topicbuckets = [['_'.join(kws), []] for kws in cluster_keywords.values()]
    topicbuckets.append(['other', []])
    for topic, cluster_id in zip(alltopics, labels):
        topicbuckets[cluster_id][1].append(topic)

    topicdict = {tstring: topics for tstring, topics in topicbuckets}
    results.append({
        'num_clusters': num_clusters,
        'num_outliers': num_outliers,
        'topicdict': topicdict
    })

results = sorted(results, key=lambda x: (x['num_outliers'], x['num_clusters']))
# %%
best_result = results[0]
with open('../data/topics2.json', 'w') as f:
    json.dump(best_result['topicdict'], f, indent=2)

for cid, topic in enumerate(best_result['topicdict'].keys()):
    print(f"cluster {cid:>2}: {topic}")