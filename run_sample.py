# -*- coding:utf-8 -*-

from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

songs = datasets.load_sample_songs()
data = songs

print data.user_ids
# {1: 'Angelica',
#  2: 'Veronica',
#  3: 'Sam',
#  4: 'Jordyn',
#  5: 'Dan',
#  6: 'Bill',
#  7: 'Chan',
#  8: 'Hailey'}

print data.item_ids
# {1: 'The Strokes',
#  2: 'Blues Traveler',
#  3: 'Phoenix',
#  4: 'Broken Bells',
#  5: 'Norah Jones',
#  6: 'Slightly Stoopid',
#  7: 'Vampire Weekend',
#  8: 'Deadmau5'}

# Build the data model
model = MatrixPreferenceDataModel(data.data)

# Build the similarity
similarity = UserSimilarity(model, pearson_correlation)

# Build the User based recommender
recommender = UserBasedRecommender(model, similarity, with_preference=True)

# Recommend items for the user 5 (Toby)
print recommender.recommend(5, how_many=3)
# [(5, 3.3477895267131013), (1, 2.8572508984333034), (6, 2.4473604699719846)]
# [(5, 4.0)]
