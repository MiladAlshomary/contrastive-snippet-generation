import json
import logging
from typing import List

import numpy as np
from scipy.stats import kendalltau

from Argument import Argument
from DataHandler import DataHandler
from myutils import cosine_similarity


class EdgeCorrelation:
    """
    Computes Kendall's rank correlation coefficient between similarity matrix and occurrence matrix.
    Computation are based on the sentence embeddings, their similarity, and their belonging to an
    argument.

    Usage:
    >>> arguments: List[Argument]
    >>> ec = EdgeCorrelation()
    >>> scores = ec.edge_correlation(arguments)
    """

    def __init__(self, save_path=None):
        """
        :param save_path: if set, then results are saved at the specified location
        """
        self.log = logging.getLogger(EdgeCorrelation.__name__)
        self.save_path = save_path

    def edge_correlation(self, arguments: List[Argument]):
        contexts = DataHandler.get_query_context_keys(arguments)
        results = dict()
        for context in contexts:
            contextual_arguments = DataHandler.get_query_context(arguments, context)
            self.log.debug(
                f'Compute edge correlation for {len(contextual_arguments)} in context {context}.')
            results[context] = self.edge_correlation_for_context(contextual_arguments)

        if self.save_path is not None:
            self.save_results(results)
        return results

    def edge_correlation_for_context(self, arguments: List[Argument]):
        similarity_matrix, sentence_arg_mapping = self._compute_similarity_matrix(arguments)
        occurrence_matrix = self._compute_occurrence_matrix(sentence_arg_mapping)
        return self._rank_correlation(similarity_matrix, occurrence_matrix)

    def _compute_similarity_matrix(self, data: List[Argument]):
        # value at position i is the index of the argument to which the i-th sentence belongs
        sentence_arg_mapping = list()
        all_sentences = list()
        for i, arg in enumerate(data):
            for s in arg.sentence_embeddings:
                all_sentences.append(s)
                sentence_arg_mapping.append(i)
        self.log.debug(f'Found {len(all_sentences)} sentences.')

        similarity_matrix = np.zeros((len(all_sentences), len(all_sentences)))
        for i, s in enumerate(all_sentences):
            for j, s_prime in enumerate(all_sentences):
                if i == j:
                    similarity_matrix[i, j] = 1
                else:
                    similarity_matrix[i, j] = cosine_similarity(s, s_prime)

        return similarity_matrix, sentence_arg_mapping

    def _compute_occurrence_matrix(self, sentence_arg_mapping):
        occurrence_matrix = np.zeros((len(sentence_arg_mapping), len(sentence_arg_mapping)))
        for s_idx, arg_idx in enumerate(sentence_arg_mapping):
            for s_prime_idx, arg_prime_idx in enumerate(sentence_arg_mapping):
                if arg_idx == arg_prime_idx:
                    occurrence_matrix[s_idx, s_prime_idx] = 1

        return occurrence_matrix

    def _rank_correlation(self, similarity_matrix, occurrence_matrix):
        correlation = kendalltau(similarity_matrix, occurrence_matrix)
        self.log.debug(f'Correlation: {correlation}')
        return correlation

    def save_results(self, result):
        with open(self.save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file)
