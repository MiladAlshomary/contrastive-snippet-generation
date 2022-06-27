import logging

import numpy as np
from sklearn.base import TransformerMixin

from shared.myutils import tokenize


class ArgumentativenessScorer(TransformerMixin):
    def __init__(self, calculation=None, discourse_markers='discourse-markers.txt',
                 claim_lexicon='ClaimLexicon.txt'):
        self.log = logging.getLogger(__name__)
        self.calculation = calculation

        with open(discourse_markers, 'r', encoding='utf-8') as f:
            self.transition_markers = f.read().split(', ')
        self.log.info('%s discourse transition markers loaded.', len(self.transition_markers))

        with open(claim_lexicon, 'r', encoding='utf-8') as f:
            self.claim_markers = f.read().split(', ')
        self.log.info('%s claim markers are loaded.', len(self.claim_markers))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for argument in X:
            self.compute_arg_matrix(argument)
        return X

    def arg(self, s):
        """
        Scores the argumentativeness of a sentence.

        Different scoring measures are available:
        'log-ratio'
        'inverse'
        """
        count_markers = 1
        for marker in self.transition_markers:
            if marker.lower() in s.lower():
                count_markers += 1
        if any(claim_marker in s.lower() for claim_marker in self.claim_markers):
            count_markers += 1

        if self.calculation == 'log-ratio':
            arg_sc = np.log(1 + count_markers / len(tokenize(s)))
        elif self.calculation == 'inverse':
            arg_sc = 1 - (1 / (1 + count_markers))
        else:  # default
            arg_sc = count_markers

        return arg_sc

    def compute_arg_matrix(self, argument):
        """
        Computes the argumentativeness values for sentences of given arguments.

        :param argument: argument of which sentences argumentativeness should be determined
        :return: argument with argumentativeness score for each sentence as matrix (len(
        sentences) x len(sentences)),
        all rows are equal.
        """
        argument.argumentativeness_scores = list()
        raw_arg_scores = list()
        for k, s_k in enumerate(argument.sentences):
            raw_arg_scores.append(self.arg(s_k))

        _row_sum = sum(raw_arg_scores)
        row = list()
        if _row_sum > 0:
            for a in raw_arg_scores:
                row.append(a / _row_sum)
        else:
            self.log.warning('No markers were found for argument %s.', argument.arg_id)
            row = raw_arg_scores

        score_matrix = list()
        for _ in argument.sentences:
            score_matrix.append(row)

        argument.argumentativeness_scores = np.array(score_matrix)
        return argument
