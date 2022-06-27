import logging
from typing import List

import numpy as np
from discreteMarkovChain import markovChain
from sklearn.base import BaseEstimator

from shared.Argument import Argument


class ContraLexRank(BaseEstimator):
    """
    Compute the final scoring and selects the summary sentences.

        :math:`P(s_i) = (1-\\alpha) \\cdot \\sum_{s_j \ne s_i} \\frac{sim(s_i, s_j)}{\\sum_{s_j \\ne s_k} sim(s_j,
        s_k)} P(s_j) + \\alpha \\cdot \\frac{arg(s_i)}{\\sum_{s_k}arg(s_k)}`
    """

    def __init__(self, d_1: float, d_2: float, d_3: float, limit: int = 2):
        """
        Initializes the ContraLexRank.
        :param d_1: factor to weight representativeness
        :param d_2: factor to weight argumentativeness
        :param d_3: factor to weight contrastiveness (has negative influence)
        :param limit: number of sentences to extract as an excerpt
        """
        self.log = logging.getLogger(__name__)
        self.d_1 = d_1
        self.d_2 = d_2
        self.d_3 = d_3
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def predict(self, X: List[Argument]):
        self.final_scoring(X)

        for argument in X:
            if argument.scores.min() >= 0:
                # take the sentence indices ordered by score, but only as much as defined by limit
                idx = np.argsort(argument.scores)[-self.limit:]
                # excerpts should preserve sentence ordering form original
                idx = sorted(idx)
                self.log.debug(f'Selecting sentences {idx}.')
                argument.excerpt_indices = idx
                excerpt = list()
                for i in idx:
                    excerpt.append(argument.sentences[i])

                argument.excerpt = excerpt
            else:
                argument.excerpt = list()
                self.log.warning(f'No snippet for {argument.arg_id}.')
        return X

    def final_scoring(self, X):
        """
        Performs the final scoring, that is weighting of centrality_scores, argumentativeness_scores,
        and context_similarity.

        "A right stochastic matrix means each row sums to 1, whereas a left stochastic matrix means each column sums
        to 1. In a doubly stochastic matrix, both the rows and the columns sum to 1." (
        https://deepai.org/machine-learning-glossary-and-terms/markov-matrix)

        :param X: list of arguments
        """
        for argument in X:
            if argument.centrality_scores.shape != argument.argumentativeness_scores.shape != \
                    argument.context_similarity.shape:
                self.log.warning('Something went wrong.')
            self.log.debug(f'Working on {argument.arg_id}')

            matrix = self.d_1 * argument.centrality_scores \
                     + self.d_2 * argument.argumentativeness_scores \
                     - self.d_3 * argument.context_similarity
            matrix = self.normalize_by_row_sum(matrix)

            if not ((matrix.min() >= 0) or self.is_right_stochastic_matrix(matrix)):
                self.log.warning(
                    f'Matrix miss requirements ({argument.arg_id}) '
                    f'({self.d_1, self.d_2, self.d_3}) {matrix}')
            try:
                mc = markovChain(matrix.T)
                mc.computePi('power')
                mc_sol = mc.pi
                argument.scores = np.array(mc_sol)
            except AssertionError as e:
                self.log.warning(
                    f'MarkovChain throws exception: {e}, ({argument.arg_id}) '
                    f'({self.d_1, self.d_2, self.d_3}) '
                    f'matrix={matrix}')

    def power_iteration(self, M, epsilon: float = 10e-8):
        """
        Applies power iteration algorithm on the given matrix.

        Note: Not all problems can be solved using power methods.

        :param M: matrix
        :param epsilon: threshold of residual to stop iteration
        :return: solution vector
        """
        # Initialize the solution with equal values.
        r_t = np.full((len(M)), 1 / len(M))
        run = True
        iterations = 0

        while run:
            r_t_prev = r_t

            # calculate the matrix-by-vector product Mr
            r_t1 = np.dot(M, r_t)

            # calculate the norm
            r_t1_norm = np.linalg.norm(r_t1)

            # re normalize the vector
            r_t = r_t1 / r_t1_norm

            # Runtime control
            if np.isnan(r_t).any():
                self.log.error('Found nan value.')
                run = False

            residual = np.linalg.norm(r_t - r_t_prev)

            if residual < epsilon:
                run = False

            if iterations >= 10000:
                self.log.error(f'Max number of iterations reached. Stopping power iteration with '
                               f'residual {residual}.')
                run = False
            iterations += 1

        return r_t

    @staticmethod
    def is_right_stochastic_matrix(M):
        result = True
        for row in M:
            if not np.allclose(np.sum(row), 1, .1):
                result = False
                break
        return result

    @staticmethod
    def normalize_by_row_sum(M):
        for row in M:
            row_sum = np.sum(row)
            row *= 1 / row_sum

            row[row < 0] = .00001  # set negative probability to very small value

        return M
