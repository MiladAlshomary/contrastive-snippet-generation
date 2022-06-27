import logging
from typing import List

import numpy as np
from sklearn.base import TransformerMixin

from shared import myutils as utils
from shared.Argument import Argument


class CentralityScorer(TransformerMixin):
    """
    Determines the pairwise similarity matrix and stores it in centrality scores.
    """

    def __init__(self, contrast_discount_weight: float = 0.0):
        self.contrast_discount_weight = contrast_discount_weight
        self.log = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[Argument]):
        for argument in X:
            similarity = self.compute_sim_matrix(argument)
            argument.centrality_scores = self.compute_coefficient_matrix(similarity)
        return X

    @staticmethod
    def compute_sim_matrix(argument: Argument):
        """
        Computes the pairwise similarity between all sentences of the given argument.

        :param argument: argument of which the sentences-wise similarity matrix should be computed
        :return: matrix representing pairwise similarities
        """
        _sim = np.zeros(shape=(len(argument.sentences), len(argument.sentences)))
        for i, u in enumerate(argument.sentence_embeddings):
            for j, v in enumerate(argument.sentence_embeddings):
                _sim[i, j] = utils.cosine_similarity(u, v)
        return np.array(_sim)

    @staticmethod
    def compute_coefficient_matrix(sim_mat):
        """
        Computes a matrix containing all similarity coefficients, i.e.,

            :math:`\\sum_{s_j \\ne s_i} \\frac{sim(s_i, s_j)}{\\sum_{s_j \\ne s_k} sim(s_j, s_k)}`

        :param sim_mat: sentence-wise similarity matrix
        :return: coefficient matrix
        """
        # coefficient matrix
        B = np.full((len(sim_mat), len(sim_mat)), .0)

        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    # similarity of s_i and s_j
                    sim_s_i_s_j = sim_mat[i, j]

                    # similarity of s_j to all other sentences s_k (s_j != s_k) is equal to the
                    # sum of the row minus  the similarity to itself
                    sum_sim_s_j_s_k = np.sum(sim_mat[j, :]) - sim_mat[j, j]

                    B[i, j] = sim_s_i_s_j / sum_sim_s_j_s_k
            B[i, i] = 1
        return B
