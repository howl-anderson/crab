
"""
Base IO code for all datasets
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

import os

import numpy as np


class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes.
    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(**kwargs)
        self.__dict__ = self


def load_movielens_r100k(load_timestamp=False):
    """
    Load and return the MovieLens dataset with 100,000 ratings (only the user ids, item ids, timestamps and ratings).

    Parameters
    ----------
    load_timestamp: bool, optional (default=False)
        Whether it loads the timestamp.

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    """
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    # Read data
    if load_timestamp:
        data_ndarray = np.loadtxt(os.path.join(base_dir, 'movielens100k.data'), delimiter='\t', dtype=int)
        data_movies = {}
        for user_id, item_id, rating, timestamp in data_ndarray:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = (timestamp, int(rating))
    else:
        data_ndarray = np.loadtxt(os.path.join(base_dir, 'movielens100k.data'), delimiter='\t', usecols=(0, 1, 2), dtype=int)

        data_movies = {}
        for user_id, item_id, rating in data_ndarray:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = int(rating)

    # Read the titles
    data_titles_ndarray = np.loadtxt(os.path.join(base_dir, 'movielens100k.item'), delimiter='|', usecols=(0, 1), dtype=str)

    data_t = []
    for item_id, label in data_titles_ndarray:
        data_t.append((int(item_id), label))
    data_titles = dict(data_t)

    fd = open(os.path.join(os.path.dirname(__file__), 'descr', 'movielens100k.rst'))
    desc = fd.read()
    fd.close()

    return Bunch(data=data_movies, item_ids=data_titles, user_ids=None, DESCR=desc)


def load_sample_songs():
    """
    Load and return the songs dataset with 49 ratings (only the user ids, item ids and ratings).

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    """
    base_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Read data
    data_m = np.loadtxt(os.path.join(base_dir, 'sample_songs.csv'), delimiter=',', dtype=str)
    item_id_list = []
    user_id_list = []
    data_songs = {}
    for user_id, item_id, rating in data_m:
        if user_id not in user_id_list:
            user_id_list.append(user_id)
        if item_id not in item_id_list:
            item_id_list.append(item_id)

        current_user_id = user_id_list.index(user_id) + 1
        current_item_id = item_id_list.index(item_id) + 1
        data_songs.setdefault(current_user_id, {})
        data_songs[current_user_id][current_item_id] = float(rating)

    data_t = []
    for no, item_id in enumerate(item_id_list):
        data_t.append((no + 1, item_id))
    data_titles = dict(data_t)

    data_u = []
    for no, user_id in enumerate(user_id_list):
        data_u.append((no + 1, user_id))
    data_users = dict(data_u)

    fd = open(os.path.join(os.path.dirname(__file__), 'descr', 'sample_songs.rst'))
    desc = fd.read()
    fd.close()

    return Bunch(data=data_songs, item_ids=data_titles, user_ids=data_users, DESCR=desc)


def load_sample_movies():
    """
    Load and return the movies dataset with n ratings (only the user ids, item ids and ratings).

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    """
    base_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Read data
    raw_data = np.loadtxt(os.path.join(base_dir, 'sample_movies.csv'), delimiter=';', dtype=str)
    item_name_list = []
    user_name_list = []
    data_songs = {}
    for user_name, item_name, rating in raw_data:
        if user_name not in user_name_list:
            user_name_list.append(user_name)
        if item_name not in item_name_list:
            item_name_list.append(item_name)
        user_index = user_name_list.index(user_name) + 1
        item_index = item_name_list.index(item_name) + 1
        data_songs.setdefault(user_index, {})
        data_songs[user_index][item_index] = float(rating)

    item_data = []
    for item_index, item_name in enumerate(item_name_list):
        item_data.append((item_index + 1, item_name))
    item_mapping = dict(item_data)

    user_data = []
    for item_index, user_name in enumerate(user_name_list):
        user_data.append((item_index + 1, user_name))
    user_mapping = dict(user_data)

    fd = open(os.path.join(os.path.dirname(__file__), 'descr', 'sample_movies.rst'))
    desc = fd.read()
    fd.close()

    return Bunch(data=data_songs, item_ids=item_mapping, user_ids=user_mapping, DESCR=desc)


def fetch_sample_songs():
    """
    This class designed to replace the class of load_sample_songs
    """
    base_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Read data
    raw_data = np.loadtxt(os.path.join(base_dir, 'sample_songs.csv'), delimiter=',', dtype=str)

    item_list = []
    user_list = []
    data = {}
    for user_id, item_id, rating in raw_data:
        if user_id not in user_list:
            user_list.append(user_id)
        if item_id not in item_list:
            item_list.append(item_id)

        current_user_id = user_list.index(user_id)
        current_item_id = item_list.index(item_id)
        data.setdefault(current_user_id, {})
        data[current_user_id][current_item_id] = float(rating)

    item_mapping = {}
    for id, item_tag in enumerate(item_list):
        item_mapping[item_tag] = id

    user_mapping = {}
    for id, user_tag in enumerate(user_list):
        user_mapping[user_tag] = id

    fd = open(os.path.join(os.path.dirname(__file__), 'descr', 'sample_songs.rst'))
    description = fd.read()
    fd.close()

    return Bunch(data=data, item_mapping=item_mapping, user_mapping=user_mapping, description=description)