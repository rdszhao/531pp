# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic

datadir = '../data/processed/uscelection/weeks'
files = [f"{datadir}/{fname}" for fname in os.listdir(datadir)]
# %%
file = files[0]
df = pd.read_csv(file)
docs = df['text'].to_list()
# %%
topic_model = BERTopic(embedding_model='sentence-transformers/all-MiniLM-L6-v2', nr_topics=50)
topics, _ = topic_model.fit_transform(docs)
topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=10, separator='_')
topic_model.set_topic_labels(topic_labels)
topic_model.get_topic_info()
# %%
