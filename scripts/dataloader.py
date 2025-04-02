# %%
import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

partsdir = '../data/usc-x-24-us-election'
processed_dir = '../data/processed/uscelection'
parts = [folder for folder in os.listdir(partsdir) if 'part' in folder]
prefiles = [f"{partsdir}/{part}/{fname}" for part in parts for fname in os.listdir(f"{partsdir}/{part}")]
outdirs = [f"{processed_dir}/{part}/{fname.split('.')[0]}_processed.tsv.gz" for part in parts for fname in os.listdir(f"{partsdir}/{part}")]
filemap = dict(zip(outdirs, prefiles))

if os.path.exists(processed_dir):
	alr_processed = [folder for folder in os.listdir(processed_dir) if 'part' in folder]
	processed_files = [f"{processed_dir}/{part}/{fname}" for part in alr_processed for fname in os.listdir(f"{processed_dir}/{part}")]
	for file in processed_files:
		filemap.pop(file, None)

def extract(text, val, listed=False):
	try:
		pattern = re.compile(rf"'{val}': '(\d+)'")
		matches = pattern.findall(text)
		if len(matches) > 1 or listed:
			return list(set(matches))
		else:
			return matches[0]
	except:
		return None

attrs = ['id', 'user', 'text', 'viewCount', 'likeCount', 'retweetCount', 'lang', 'date']

for outdir, prefile in tqdm(filemap.items(), total=len(filemap)):
	df = pd.read_csv(prefile, compression='gzip', usecols=attrs)
	df = df[df['lang'] == 'en']
	df = df.drop('lang', axis=1)
	df['text'] = df['text'].astype(str).str.strip().replace(r'[^\w\s\-\.,;:!?()/\'"&]', '')
	df['user'] = df['user'].apply(lambda x: extract(x, 'id_str'))
	df['likeCount'] = df['likeCount'].astype(int, errors='ignore')
	df['viewCount'] = df['viewCount'].apply(lambda x: extract(x, 'count')).astype(int, errors='ignore')
	Path('/'.join(outdir.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
	df.to_csv(outdir)