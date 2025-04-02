# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# %% 
# assign theoretical clusters to each entry - comment this out for later
data = pd.read_csv('../week_18.csv')
data['topic'] = np.random.randint(1,13,size=len(data)) # number of topics = 12

# take mean and std of each topic
data['key_metric'] = np.where(data['likeCount'] != 0, data['viewCount'] / data['likeCount'], 0)
mean_std = data.groupby('topic')['key_metric'].agg(['mean', 'std']).reset_index()
# mean_std

# calc z-score per tweet for view/like ratio (assume it is population)
data_merge = data.merge(mean_std, on='topic', how='left')
data_merge['z_score'] = (data_merge['key_metric'] - data_merge['mean']) / data_merge['std']
data_merge['p_value'] = 2 * (1-norm.cdf(abs(data_merge['z_score'])))
# data_merge.head()

# data check
# print(data_merge['p_value'].max(), data_merge['p_value'].min())

# take average p-value for each topic
avg_pval = data_merge.groupby('topic')['p_value'].agg(['mean']).reset_index()
# avg_pval

