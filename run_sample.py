# -*- coding:utf-8 -*-

from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

songs = datasets.load_sample_songs()
data = songs

print data.user_ids
# {1: 'Jack Matthews',
#  2: 'Mick LaSalle',
#  3: 'Claudia Puig',
#  4: 'Lisa Rose',
#  5: 'Toby',
#  6: 'Gene Seymour',
#  7: 'Michael Phillips'}

print data.item_ids
# {1: 'Lady in the Water',
#  2: 'Snakes on a Planet',
#  3: 'You, Me and Dupree',
#  4: 'Superman Returns',
#  5: 'The Night Listener',
#  6: 'Just My Luck'}


# Build the data model
model = MatrixPreferenceDataModel(data.data)

# Build the similarity
similarity = UserSimilarity(model, pearson_correlation)


# Build the User based recommender
recommender = UserBasedRecommender(model, similarity, with_preference=True)

# Recommend items for the user 5 (Toby)
print recommender.recommend(5, how_many=3)
# [(5, 3.3477895267131013), (1, 2.8572508984333034), (6, 2.4473604699719846)]