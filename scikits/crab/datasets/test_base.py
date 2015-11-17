import unittest

from scikits.crab.datasets import load_movielens_r100k
from scikits.crab.datasets import load_sample_songs
from scikits.crab.datasets import load_sample_movies


class Test(unittest.TestCase):

    def test_load_movielens_r100k(self):
        movies = load_movielens_r100k()
        self.assertEquals(len(movies['data']), 943)
        self.assertEquals(len(movies['item_ids']), 1682)

    def test_load_sample_songs(self):
        songs = load_sample_songs()
        self.assertEquals(len(songs['data']), 8)
        self.assertEquals(len(songs['item_ids']), 8)

    def test_load_sample_movies(self):
        movies = load_sample_movies()
        self.assertEquals(len(movies['data']), 7)
        self.assertEquals(len(movies['item_ids']), 6)