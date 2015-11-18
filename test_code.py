from scikits.crab import datasets
from scikits.crab.models import NumericalDataModel
from scikits.crab import similarities
from scikits.crab.metrics import PearsonCorrelation
from scikits.crab.recommenders.knn import RecommendItemsBasedOnUser
import numpy as np

songs = datasets.fetch_sample_songs()
print songs.user_mapping
# {'Angelica': 0,
#  'Veronica': 1,
#  'Sam': 2,
#  'Chan': 6,
#  'Dan': 4,
#  'Bill': 5,
#  'Jordyn': 3,
#  'Hailey': 7}
print songs.item_mapping
# {'The Strokes': 0,
#  'Blues Traveler': 1,
#  'Phoenix': 2,
#  'Broken Bells': 3,
#  'Deadmau5': 7,
#  'Norah Jones': 4,
#  'Slightly Stoopid': 5,
#  'Vampire Weekend': 6}
data_model = NumericalDataModel(songs.data)
user_similarity = similarities.User(data_model, PearsonCorrelation)
recommender = RecommendItemsBasedOnUser(data_model, user_similarity)
recommend_data = recommender.recommend(4)
print recommend_data
# {3: 3.8630820047385939, 6: 2.3626939383442211, 7: 3.0200696734781376}