# %% [markdown]
## Required Libraries
# For this particular exercise, we are using [Surprise](https://github.com/NicolasHug/Surprise).Surprise is a Python SciKit that comes with various recommender algorithms and similarity metrics to make it easy to build and analyze recommenders.<br>

# %%
import pandas as pd

# %% [markdown]
## Reading the data<br>
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

# %% [markdown]
# EDA ----
# The following is an HTML report of the data.<br>
# Also, printing a few lines of the dataset.

# %%
from pandas_profiling import ProfileReport

# profile = ProfileReport(df, title="Patio_Lawn_and_Garden", minimal=False)
# profile.to_file("storage/df_report.html")

# %%
print('Dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::20000, :])

# %% [markdown]
### Plots

#### Ratings distribution

# The first thing is to analyze the distribution of ratings to understand among other things:
# - The rating scale: is it 0-5, 1-5, 1-10....
# - The distribution of the ratings: are the users biased to high or low rates?
# - Are the users using the entire rating scale? e.g.: sometimes people performs binary ratings even if you provide them with a more granular scale (5 good, 1 bad and nothing in between)

# %%
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# Count the number of times each rating appears in the dataset
data = df['overall'].value_counts().sort_index(ascending=False)

# Create the histogram
trace = go.Bar(x = data.index,
               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
               textposition = 'auto',
               textfont = dict(color = '#000000'),
               y = data.values,
               )
# Create layout
layout = dict(title = 'Distribution Of {} Ratings'.format(df.shape[0]),
              xaxis = dict(title = 'Rating'),
              yaxis = dict(title = 'Count'))
# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# %% [markdown]
# We can see the distribution of the ratings is skewed to the high values. 

# ## Numbers of ratings per item
# The following analysis checks how many ratings (interactions) each item has.<br>
# The degree in which this long tail phenomena occurs will be related to aspects such as serendipity or overespecialization of the recommendations and the recommendation methodology that you should select.

# %%

# Number of ratings per movie
data = df.groupby('asin')['overall'].count()

# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per Product Id',
                   xaxis = dict(title = 'Number of Ratings Per Product Id'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# %% [markdown]
# ## Number of ratings per user
# Now it's time to see how many ratings we have per user.

# %%
# Number of ratings per user
data = df.groupby('reviewerID')['overall'].count()

# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0, size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per User',
                   xaxis = dict(title = 'Ratings Per User'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# %% [markdown]
# As expected similar distribution as per item.

# %% [markdown]
## Managing data with Surprise
# It's now time to convert our datasets to the format required by the Surprise library.<br>
# To load a data set from the above pandas data frame, we will use the *load_from_df()* method, we will also need a Reader object, and the rating_scale parameter must be specified. The data frame must have three columns, corresponding to the user ids, the item ids, and the ratings in this order. Each row thus corresponds to a given rating.<br>
# <br>
# Split the dataset
### Data selection and preprocessing
# Created a training dataset and a testing dataset from therein for the experiment. <br>
# A recommended standard pre-processing strategy is that: for each user, randomly select 80% of ratings as the training ratings and use the remaining 20% ratings as testing ratings.

# %%
import numpy as np
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['reviewerID'])

# %%
# Collaborative Filtering
from surprise import Dataset
from surprise import Reader

reader = Reader(rating_scale=(1, 5))
data_train_cf = Dataset.load_from_df(data_train[['reviewerID', 'asin', 'overall']], reader)
data_test_cf = Dataset.load_from_df(data_test[['reviewerID', 'asin', 'overall']], reader)

# %% [markdown]
# # First Try: Neighbourhood-based Collaborative Filtering
# We are going to start by using one of the simplest recommendation methodologies (Neighbourhood-based CF). This technique is pretty simple and fast, but it provides accurate results for many scenarios.<br>
# To specify the parameters of the execution, you simply have to configure the function by passing a dictionary as an argument to the recommender function. The dictionary should have the required keys, such as the following:
# - **name** contains the similarity metric to use. Options are cosine, msd, pearson, or pearson_baseline. The default is msd.
# - **user_based** is a boolean that tells whether the approach will be user-based or item-based. The default is True, which means the user-based approach will be used.
# - **min_support** is the minimum number of common items needed between users to consider them for similarity. For the item-based approach, this corresponds to the minimum number of common users for two items.
# In particular, I will use the cosine distance as similarity metric for finding the neighbours, using the item-based approach

# %%
from surprise import KNNBaseline
from surprise.model_selection import cross_validate

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
knn = KNNBaseline(sim_options=sim_options)

results = cross_validate(knn, data_train_cf, measures=['RMSE'], cv=3, verbose=True, n_jobs=-1)

# %% [markdown]
# #Second Try: Benchmarking
# In the following exercise we are going to experiment with different algorithms to check which one of them offers the best results.

# %%
from surprise import SVD, BaselineOnly, NMF, SlopeOne, CoClustering, SVDpp, NormalPredictor

benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), BaselineOnly(), CoClustering()]:
    
    print("Testing {}".format(algorithm))
    # Perform cross validation
    results = cross_validate(algorithm, data_train_cf, measures=['RMSE'], cv=3, verbose=False, n_jobs=-1)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    

# %% [markdown]
# # Hyperparameter optimization

# %%
# https://towardsdatascience.com/svd-where-model-tuning-goes-wrong-61c269402919
# https://surprise.readthedocs.io/en/stable/matrix_factorization.html?highlight=SVDpp#surprise.prediction_algorithms.matrix_factorization.SVDpp
from surprise.model_selection.search import GridSearchCV

param_grid = {'n_factors':[50,100,150], 'n_epochs':[20,30], 'lr_all':[0.005,0.01,0.007], 'reg_all':[0.02,0.1]}
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(data_train_cf)

# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# %% [markdown]
# Using the `test` dataset.

# %%
svdpp_tuned = SVDpp(**gs.best_params['rmse'], random_state=42)
cross_validate(svdpp_tuned, data_test_cf, measures=['RMSE'], cv=3, verbose=True, n_jobs=-1)

# %% [markdown]
# It can be noticed a small `overfitting` from our model.


# Personal model
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html?highlight=fit#the-fit-method