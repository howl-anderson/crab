from scikits.crab import datasets
from scikits.crab.models import NumericalDataModel
from scikits.crab import similarities
from scikits.crab.metrics import PearsonCorrelation
from scikits.crab.recommenders.knn import RecommendItemsBasedOnUser
import numpy as np

songs = datasets.fetch_sample_movies()
print songs.user_mapping
# {'Jack Matthews': 0,
#  'Mick LaSalle': 1,
#  'Claudia Puig': 2,
#  'Lisa Rose': 3,
#  'Toby': 4,
#  'Gene Seymour': 5,
#  'Michael Phillips': 6}
print songs.item_mapping
# {'Lady in the Water': 0,
#  'Just My Luck': 5,
#  'Superman Returns': 3,
#  'The Night Listener': 4,
#  'Snakes on a Planet': 1,
#  'You, Me and Dupree': 2}
data_model = NumericalDataModel(songs.data)
user_similarity = similarities.User(data_model, PearsonCorrelation)
recommender = RecommendItemsBasedOnUser(data_model, user_similarity)
recommend_data = recommender.recommend(4)
print recommend_data
# {1: 14.279912709529622, 2: 10.214895461462383, 3: 15.036860450715638, 4: 12.89975185847269}