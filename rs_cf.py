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

# df = getDF('raw/reviews_Cell_Phones_and_Accessories_5.json.gz')
df = getDF('raw/reviews_Patio_Lawn_and_Garden_5.json.gz')

# %% [markdown]
## EDA ----
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
### Plots ----

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

### Numbers of ratings per item ----
# The following analysis checks how many ratings (interactions) each item has.<br>
# The degree in which this long tail phenomena occurs will be related to aspects such as serendipity or overespecialization of the recommendations and the recommendation methodology that you should select.

# %%

# Number of ratings per item
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
### Number of ratings per user ----
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
## Managing data with Surprise ----
# It's now time to convert our datasets to the format required by the Surprise library.<br>
# To load a data set from the above pandas data frame, we will use the *load_from_df()* method, we will also need a Reader object, and the rating_scale parameter must be specified. The data frame must have three columns, corresponding to the user ids, the item ids, and the ratings in this order. Each row thus corresponds to a given rating.<br>

# %% [markdown]
## Split the dataset ----
### Data selection and preprocessing
# Created a training dataset and a testing dataset from therein for the experiment. <br>
# A recommended standard pre-processing strategy is that: for each user, randomly select 80% of ratings as the training ratings and use the remaining 20% ratings as testing ratings.

# %%
import numpy as np
from sklearn.model_selection import train_test_split

# df2 = df
# df2['reviewerID|asin'] = df[['reviewerID', 'asin']].agg('|'.join, axis=1)
# df2 = df2.groupby('reviewerID|asin').filter(lambda x: len(x) >= 2)
# data_train, data_test = train_test_split(df2, test_size=0.20, random_state=42, stratify=df2['reviewerID'])
data_train, data_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['reviewerID'])
# Clear index
data_train.reset_index(drop=True, inplace=True)
data_test.reset_index(drop=True, inplace=True)

# %%
## Collaborative Filtering ----
from surprise import Dataset
from surprise import Reader

reader = Reader(rating_scale=(1, 5))
data_train_cf = Dataset.load_from_df(data_train[['reviewerID', 'asin', 'overall']], reader)
data_test_cf = Dataset.load_from_df(data_test[['reviewerID', 'asin', 'overall']], reader)

# %% [markdown]
### Neighbourhood-based Collaborative Filtering ----
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

### Personal model ----
# https://surprise.readthedocs.io/en/stable/building_custom_algo.html?highlight=fit#the-fit-method
# https://github.com/NicolasHug/Surprise/blob/fa7455880192383f01475162b4cbd310d91d29ca/examples/building_custom_algorithms/with_baselines_or_sim.py

# %%
from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import PredictionImpossible

class GroupAlgorithm(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Compute baselines and similarities
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Compute similarities between u and v, where v describes all other users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
        # Sort these neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        print('The 3 nearest neighbors of user', str(u), 'are:')
        for v, sim_uv in neighbors[:3]:
            print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

        # Return the baseline estimate
        bsl = self.trainset.global_mean + self.bu[u] + self.bi[i]
        return bsl

# %% [markdown]
### Benchmarking ----
# In the following exercise we are going to experiment with different algorithms to check which one of them offers the best results.

# %%
from surprise import SVD, BaselineOnly, NMF, SlopeOne, CoClustering, SVDpp, NormalPredictor

benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), BaselineOnly(), CoClustering(), GroupAlgorithm()]:
    
    print("Testing {}".format(algorithm))
    # Perform cross validation
    results = cross_validate(algorithm, data_train_cf, measures=['RMSE'], cv=3, verbose=False, n_jobs=-1)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    

# %% [markdown]
### Hyperparameter optimization ----

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

# %%
#### Train ----
cross_validate(svdpp_tuned, data_train_cf, measures=['RMSE'], cv=3, verbose=True, n_jobs=-1)

# %%
#### Test ----
cross_validate(svdpp_tuned, data_test_cf, measures=['RMSE'], cv=3, verbose=True, n_jobs=-1)

# %% [markdown]
# It can be noticed a small `overfitting` from our model.

# %% [markdown]
## Content-Based recommendation system using TFIDF with recommendations per user and metrics on results ----
### Exploratory Analysis ----

# To verify whether the preprocessing happened correctly, we’ll make a word cloud using the wordcloud package to get a visual representation of most common words. It is key to understanding the data and ensuring we are on the right track, and if any more preprocessing is necessary before training the model.

# %%
# https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(df['reviewText'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# %%
# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df['reviewText'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

# %% [markdown]
### Cosine similarity ----

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def cosine_distance(data):
    #Construct the required TF-IDF matrix by fitting and transforming the data
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b')
    vectorizer = vectorizer.fit(data)  
    tfidf_matrix = vectorizer.transform(data)
    # Calculate the distances
    return linear_kernel(tfidf_matrix, tfidf_matrix)

# %% [markdown]
### Get recommendations ----

# %%

# Provides an `asin`, and this function will return a list of recommendations.
def get_recommendations_based_on_reviewText(asin_base, data_distance, data_distance_cosine, num_recommendations):
    # Get the index of the item that matches the title
    idx_item = data_distance.loc[data_distance['asin'].isin([asin_base])]
    idx_item = idx_item.index
    # Get the pairwise similarity scores of all items with that item
    sim_scores_item = list(enumerate(data_distance_cosine[idx_item][0]))
    # Sort the items based on the similarity scores
    sim_scores_item = sorted(sim_scores_item, key=lambda x: x[1], reverse=True)
    # Get the scores of the XX most similar items
    sim_scores_item = sim_scores_item[1:num_recommendations]
    # Get the item indices
    item_indices = [i[0] for i in sim_scores_item]
    # Return the top 2 most similar items
    return data_distance['asin'].iloc[item_indices]

def get_recommendation_content_model(reviewerID_base, data_review, data_distance, data_distance_cosine, num_recommendations=3):
    recommended_item_list = []
    recommended_item_list_already = []

    # List of items reviewed by `reviewerID_base` and only items that the user likes
    data_review_filtered = data_review[(data_review["reviewerID"] == reviewerID_base) & (data_review['overall'] >= 4)]

    # Looking for recommendations per items reviewed by the `reviewerID_base`
    for index, item in enumerate(data_review_filtered['asin']):
        for key, item_recommended in get_recommendations_based_on_reviewText(item, data_distance, data_distance_cosine, num_recommendations).iteritems():
            # print(item_recommended)
            recommended_item_list.append(item_recommended)

    # Removing already reviewed item from recommended list    
    for item_title in recommended_item_list:
        if item_title in data_review_filtered['asin']:
            recommended_item_list.remove(item_title)
            recommended_item_list_already.append(item_title)
    
    return recommended_item_list, recommended_item_list_already

# %%
### Apply model ----
#### Train ----
data_distance_cosine = cosine_distance(data_train['reviewText'])
get_recommendation_content_model('A3497NDGXXH92J', data_train, data_train, data_distance_cosine, 3)

# %%
#### Test ----
data_distance_cosine = cosine_distance(data_train['reviewText'])
get_recommendation_content_model('A2HTPS0JV3Q8ZD', data_test, data_train, data_distance_cosine, 3)  # Tiene problemas


# %%
get_recommendations_based_on_reviewText('B00E1CS5JQ', data_train, data_distance_cosine, 3)

# %%





# %%


data_review_filtered = data_test[(data_test["reviewerID"] == reviewerID_base) & (data_test['overall'] >= 4)]

data_review_filtered['asin']

data_review_filtered = data_test[(data_test["reviewerID"] == reviewerID_base) & (data_test['overall'] >= 4)]...
3129     B000FLV9H2
6727     B002ITKYRK
1892     B000A0VOD2
12242    B00E1CS5JQ  >> Problema
6018     B001H1JKN4
3373     B000H6JJEA
12122    B00DRBBRQU  >> Problema


# %%

asin_base = 'B00E1CS5JQ'
data_distance = data_train.reset_index(drop=True)
num_recommendations =3

# Get the index of the item that matches the title
idx_item = data_distance.loc[data_distance['asin'].isin([asin_base])]
idx_item = idx_item.index
# Get the pairwise similarity scores of all items with that item
sim_scores_item = list(enumerate(data_distance_cosine[idx_item][0]))
# Sort the items based on the similarity scores
sim_scores_item = sorted(sim_scores_item, key=lambda x: x[1], reverse=True)
# Get the scores of the XX most similar items
sim_scores_item = sim_scores_item[1:num_recommendations]
# Get the item indices
item_indices = [i[0] for i in sim_scores_item]
# Return the top 2 most similar items
data_distance['asin'].iloc[item_indices]

# data_distance.shape
# data_distance_cosine.shape





# %%
#### Train + Test ----
data = data_train.append(data_test)
get_recommendation_content_model('A6HOWM08PLFZ5', data, data, 3)

# %% [markdown]
## Hybrid model ----
### Function definition ----

# %%
def hybrid_content_svdpp_per_reviewer(reviewerID_base, data_review, data_distance, svdpp_tuned):
    recommended_items_by_content_model = get_recommendation_content_model(reviewerID_base, data_review, data_distance)[0]
    rating_=[]
    for item in recommended_items_by_content_model:
        predict = svdpp_tuned.predict(reviewerID_base, item)
        rating_.append([reviewerID_base, item, predict.est])
    rating_ = pd.DataFrame(rating_, columns=['reviewerID', 'item', 'predict'])
    rating_ = rating_.sort_values(by='predict', ascending=False)
    return rating_

# %% [markdown]
### Apply to all the reviewer ----

# %%
first_reviewers = 4
top_n = 5

# SVD++ model fitted to the trainset
# https://surprise.readthedocs.io/en/stable/getting_started.html#train-on-a-whole-trainset-and-the-predict-method
trainset = data_train_cf.build_full_trainset()
svdpp_tuned.fit(trainset)

recommendation_ = pd.DataFrame()
for reviewerID_base in data['reviewerID'].unique()[:first_reviewers]:
    case = hybrid_content_svdpp_per_reviewer(reviewerID_base, data_test, data_train, svdpp_tuned).head(top_n)
    recommendation_ = recommendation_.append(case)
# %%
recommendation_

# %%












# %% [markdown]
### Performance ----

# %%
match = 0

for user_id in data['reviewerID'].unique():
    recom = recommendations_for_all[recommendations_for_all['reviewerID']==user_id]['asin']
    test = ratings_test[ratings_test['reviewerID']==user_id]['asin']

    match = recom.isin(test).sum()

match

# %%
f'My accuracy is {match/recommendations_for_all.shape[0]}.'

# %%
