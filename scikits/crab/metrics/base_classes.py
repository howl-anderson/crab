from scipy.spatial import distance
import numpy as np


class PearsonCorrelation(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def compute(self):
        X, Y = zip(*[(self.X[k], self.Y[k]) for k, _ in enumerate(self.X) if not np.isnan(self.X[k]) and not np.isnan(self.Y[k])])
        return self.pearson_correlation(X, Y)

    @staticmethod
    def pearson_correlation(X, Y):
        """
        """
        XY = distance.cdist([X], [Y], 'correlation', 2)

        return 1 - XY[0][0]