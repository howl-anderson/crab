"""
Strategies for users selection to be a
possible candidate to be member of a user neighborhood.

Please check the base.BaseUserNeighborhoodStrategy before
implement your own strategy.

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from base import BaseUserNeighborhoodStrategy
import numpy as np
from ...similarities.basic_similarities import UserSimilarity
from ...metrics.pairwise import euclidean_distances


class AllNeighborsStrategy(BaseUserNeighborhoodStrategy):
    '''
    Returns
    --------
    Returns all users in the model.
    This strategy is not recommended for large datasets and
    it is the dummiest one.
    '''
    def user_neighborhood(self, user_id, data_model, similarity='user_similarity',
        distance=None, nhood_size=None, **params):
        '''
        Computes a neighborhood consisting of the  n users to a given user
        based on the strategy implemented in this method.

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: DataModel instance
            The data model that will be the source for the possible
            candidates

        similarity: string
            The similarity to compute the neighborhood  (default = 'user_similarity')
            |user_similarity'|

        distance: function
            Pairwise metric to compute the similarity between the users.

        nhood_size: int
            The neighborhood size (default = None all users)

        '''
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids


class NearestNeighborsStrategy(BaseUserNeighborhoodStrategy):
    """
    Returns
    --------
    Returns the neighborhood consisting of the nearest n
    users to a given user. "Nearest" in this context is
    defined by the Similarity.

    Parameters
    -----------
    user_id:  int or string
        ID of user for which to find most similar other users

    data_model: DataModel instance
        The data model that will be the source for the possible
        candidates

    similarity: string
        The similarity to compute the neighborhood  (default = 'user_similarity')
        |user_similarity'|

    distance: function
        Pairwise metric to compute the similarity between the users.

    nhood_size: int
        The neighborhood size (default = None all users)

    """

    def __init__(self):
        self.similarity = None

    def _sampling(self, data_model, sampling_rate):
        # TODO: Still to be implemented in a best way
        return data_model

    def _set_similarity(self, data_model, similarity, distance, neighborhood_size):
        if not isinstance(self.similarity, UserSimilarity) or not distance == self.similarity.distance:
            neighborhood_size = neighborhood_size if not neighborhood_size else neighborhood_size + 1
            self.similarity = UserSimilarity(data_model, distance, neighborhood_size)

    def user_neighborhood(self, user_id, data_model, n_similarity='user_similarity',
                          distance=None, neighborhood_size=None, **kwargs):
        """
        Computes a neighborhood consisting of the  n users to a given
        user based on the strategy implemented in this method.
        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: DataModel instance
            The data model that will be the source for the possible
            candidates

        n_similarity: string
            The similarity to compute the neighborhood (Default = 'user_similarity')

        neighborhood_size: int
            The neighborhood size (default = None all users)

        Optional Parameters
        --------------------
        minimal_similarity: float
            minimal similarity required for neighbors (default = 0.0)

        sampling_rate: int
            percentage of users to consider when building neighborhood
                (default = 1)

        """

        minimal_similarity = kwargs.get('minimal_similarity', 0.0)
        sampling_rate = kwargs.get('sampling_rate', 1.0)

        data_model = self._sampling(data_model, sampling_rate)
        # set the neighborhood_size at Similarity , and use Similarity to get the top_users
        if distance is None:
            distance = euclidean_distances
        if n_similarity == 'user_similarity':
            self._set_similarity(data_model, n_similarity, distance, neighborhood_size)
        else:
            raise ValueError('similarity argument must be user_similarity')

        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id] if not np.isnan(score) and score >= minimal_similarity and to_user_id != user_id]

        return neighborhood
