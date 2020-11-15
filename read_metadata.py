# %% [markdown]
## Required Libraries ----
# For this particular exercise, we are using [Surprise](https://github.com/NicolasHug/Surprise).Surprise is a Python SciKit that comes with various recommender algorithms and similarity metrics to make it easy to build and analyze recommenders.<br>

# %%
import pandas as pd

# %% [markdown]
## Reading the data ----
# Source: http://jmcauley.ucsd.edu/data/amazon/

# %%
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('raw/reviews_Patio_Lawn_and_Garden_5.json.gz')
df_meta = getDF('raw/meta_Patio_Lawn_and_Garden.json.gz')

# %% [markdown]
## EDA ----
# The following is an HTML report of the data.<br>
# Also, printing a few lines of the dataset.

# %%
from pandas_profiling import ProfileReport

# profile = ProfileReport(df, title="Patio_Lawn_and_Garden", minimal=False)
# profile.to_file("storage/df_report.html")


# %%


df.iloc[1]

df['category'].notnull().sum()
df.shape
# %%
