#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import textwrap
import seaborn as sns
import scikit_posthocs as sp
import statsmodels.api as sm
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor

chunksize = 100_000
chunks = []

for chunk in pd.read_csv(
    "processed_notext.csv.gz", compression="gzip", chunksize=chunksize
):
    chunks.append(chunk)
    print("chunk done")

df = pd.concat(chunks, ignore_index=True)


# count most freq. outlet per agg group
top_align = (
    df.groupby("bucket")["alignment"]
    .value_counts()
    .groupby(level=0)
    .head(1)
    .reset_index(name="align_count")
)
print(top_align)

# count most freq. outlet per agg group, position
top_align_pos = (
    df.groupby(["bucket", "position"])["alignment"]
    .value_counts()
    .groupby(level=[0, 1])
    .head(1)
    .reset_index(name="align_count")
)
print(top_align_pos)

# plot news alignment per bucket
all_res = df.groupby("bucket")["alignment"].value_counts().unstack(fill_value=0)


all_res.plot(kind="bar", figsize=(12, 8))
plt.title("News Source Alignment Counts per Bucket")
plt.xlabel("Bucket")
plt.xticks(rotation=40)
plt.ylabel("Count")
plt.legend(title="News Source")
plt.tight_layout()
plt.show()


print(all_res.columns)

all_res_new = np.log10(all_res)
wrapped_labels = [textwrap.fill(label, width=16) for label in all_res_new.index]

ax = all_res_new.plot(kind="bar", figsize=(12, 8))
ax.set_title("News Source Alignment Counts per Bucket Topic", fontsize=20)
ax.set_xlabel("Bucket Topic", fontsize=18)
ax.set_ylabel("Logarithmic Count", fontsize=18)

ax.set_xticklabels(wrapped_labels, rotation=40, ha="right", fontsize=14)
ax.tick_params(axis="y", labelsize=16)

ax.legend(title="News Source", fontsize=16, title_fontsize=16)

plt.tight_layout()
plt.show()


# %%
df["rawMetric"] = df["likeCount"] / df["viewCount"]

# ANOVA
buckets = [
    "mixed/pop culture crossover & miscellaneous",
    "election & political personalities",
    "trump world / legal drama",
    "internet culture, media, and meme politics",
    "foreign policy & global conflict",
    "major domestic issues",
]

bucket_dict = {
    "mixed/pop culture crossover & miscellaneous": np.log(4384730),
    "election & political personalities": np.log(6392826),
    "trump world / legal drama": np.log(4983763),
    "internet culture, media, and meme politics": np.log(36009),
    "foreign policy & global conflict": np.log(257799),
    "major domestic issues": np.log(133355),
} # number of tweets per global topic
df["keyMetric"] = df["bucket"].map(bucket_dict) * df["rawMetric"]

split_buckets = [
    df[df["bucket"] == bucket]["keyMetric"].dropna().values for bucket in buckets
]
df.head()

# check sample size per bucket
for bucket in split_buckets:
    print(len(bucket))

# statistical analysis
f_stat, p_val = f_oneway(*split_buckets)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

f_stat, p_val = kruskal(*split_buckets)
print(f"Kruskal-Wallis H-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")


#
print(df.groupby("bucket")["keyMetric"].agg(["count", "std"]))


# dunn's test
dunn = sp.posthoc_dunn(
    df, val_col="keyMetric", group_col="bucket", p_adjust="bonferroni"
)
print(dunn)

# temporal analysis 
df_plot = df.sort_values('week')
df_plot = df_plot.groupby(['week', 'bucket'], as_index=False)['keyMetric'].mean()



plt.figure(figsize=(12, 8))
sns.lineplot(df_plot, x='week', y='keyMetric', hue='bucket', marker='o')

plt.title('Key Metric per Week by Bucket', fontsize=18)
plt.xlabel('Week', fontsize=16)
plt.xticks(fontsize=12)
plt.ylabel('Metric', fontsize=16)
plt.yticks(fontsize=12)
plt.legend(title='Bucket')
plt.grid(True)
plt.tight_layout()
plt.show()



