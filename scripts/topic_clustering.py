# %%
import os
import re
import torch
import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.faiss import FAISS
from sklearn.feature_extraction.text import CountVectorizer
from cuml import UMAP
from cuml import HDBSCAN
from tqdm.notebook import tqdm
# %%
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'torch_dtype': 'float16'})
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
device = torch.device('cuda')
txt_dir = 'data/weekslite'
modeldir = 'data/topicmodels'
txtfiles = sorted([f"{txt_dir}/{filename}" for filename in os.listdir(txt_dir)])
headline_df = pd.read_csv('data/headlines.csv')

def remove_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|ftp://\S+|www\.\S+')
    return url_pattern.sub('', text)

def get_alignments(embeddings, vs, threshold=1):
    print('getting alignments...')
    mask = []
    alignments = []

    for emb in tqdm(embeddings):
        matched = vs.similarity_search_with_score_by_vector(emb, k=1, score_threshold=threshold)
        if matched:
            matched = matched[0][0]
            mask.append(True)
            alignments.append(matched.page_content)
        else:
            mask.append(False)

    mask = np.array(mask)
    print(f"retained {mask.sum() / len(mask) * 100:.2f}%")
    return embeddings[mask], mask, alignments
# %%
for file in tqdm(txtfiles):
    try:
        weeknum = int(file.split('/')[-1].split('.')[0])
        print(f"processing week {weeknum}...")
        df = pd.read_csv(file)
        df['text'] = df['text'].apply(remove_urls)

        week_df = headline_df[headline_df['week'] == weeknum].drop(columns=['week'])
        news_embeddings = encoder.encode(week_df['headline'].tolist())
        vs = FAISS.from_embeddings(list(zip(week_df['source'].tolist(), news_embeddings)), encoder)
        text_embeddings = np.array(encoder.encode(df['text'].to_list(), show_progress_bar=True))
        embeddings, mask, alignments = get_alignments(text_embeddings, vs)

        df = df[mask]
        df['alignment'] = alignments
        docs = df['text'].tolist()

        topic_model = BERTopic(
            embedding_model=encoder,
            umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0),
            hdbscan_model=HDBSCAN(min_samples=10),
            vectorizer_model = CountVectorizer(stop_words='english'),
            representation_model=KeyBERTInspired(),
            nr_topics='auto',
            verbose=True
        )
        topics, probs = topic_model.fit_transform(docs, embeddings)
        df['topic'] = topics
        topic_model.save(f"{modeldir}/{weeknum}", serialization='safetensors', save_ctfidf=True)
        df.to_csv(file, index=False)
        print(f"done processing {weeknum} !")
    except Exception as e:
        print(f"error processing {file}: {e}")
        continue
# %%
