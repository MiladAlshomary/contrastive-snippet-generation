import logging
from typing import List, Tuple

import numpy as np

from Argument import Argument
from SBERTHandler import SBERTHandler
from myutils import cosine_similarity


class SentenceArgReAllocator:
    """
    Argument pseudo-clustering as preparation for scoring.

    Usage:

    >>> arguments: List[Argument]
    >>> sara = SentenceArgReAllocator()

    >>> sara.prepare_snippet_embeddings(arguments)
    >>> sara.re_allocate(arguments)
    >>> reallocated_arguments = sara.convert_to_argument()
    """

    def __init__(self):
        self.log = logging.getLogger(SentenceArgReAllocator.__name__)
        self.new_sentence_allocation = dict()
        self.summary_embeddings = dict()
        self.embedding_model = SBERTHandler()

    def prepare_snippet_embeddings(self, arguments: List[Argument]):
        for argument in arguments:
            excerpt_embeddings = np.array(
                [argument.sentence_embeddings[v] for v in argument.excerpt_indices])
            self.summary_embeddings[(argument.query, argument.arg_id)] = np.mean(excerpt_embeddings,
                                                                                 axis=0)
            self.new_sentence_allocation[(argument.query, argument.arg_id)] = list()

    def re_allocate(self, arguments: List[Argument]):
        """
        Iterates over all sentences, and re-allocates a sentence to the closest summary.
        :param arguments: arguments to reallocate
        """
        for argument in arguments:
            for s in argument.sentence_embeddings:
                query_and_new_arg_id = self._find_closest_summary_snippet(s, argument.query)
                self.new_sentence_allocation[query_and_new_arg_id].append(s)

    def convert_to_argument(self) -> List[Argument]:
        """
        Converts the internal dictionary to a list of Argument items.
        :return: list of arguments
        """
        new_arguments = list()
        for query_and_new_arg_id, sentences in self.new_sentence_allocation.items():
            a = Argument(arg_id=query_and_new_arg_id[1], query=query_and_new_arg_id[0])
            a.sentence_embeddings.extend(sentences)
            new_arguments.append(a)

        return new_arguments

    def _find_closest_summary_snippet(self, sentence, given_query) -> Tuple[str, str]:
        """
        Finds the arg_id to which the given sentence is most similar. arg_id is bound to the
        snippet.
        :param sentence: vector to which the closest summary should be found
        :param given_query: query the sentence belongs to
        :return: id and query of the argument (snippet) to which the sentence is most similar
        """
        _similarities = dict()
        for query_and_new_arg_id, embedding in self.summary_embeddings.items():
            if query_and_new_arg_id[0] == given_query:
                value = cosine_similarity(embedding, sentence)
                if type(value) == np.float64 or type(value) == np.float32 or type(value) == float:
                    _similarities[query_and_new_arg_id] = value
                else:
                    self.log.warning(f'Failure computing similarities resulting in {value}.')
                    if type(value) == np.ndarray:
                        self.log.warning('First similarity taken')
                        _similarities[query_and_new_arg_id] = value[0]
                    else:
                        self.log.warning('Setting similarity to zero')
                        _similarities[query_and_new_arg_id] = 0

        return max(_similarities, key=lambda k: _similarities[k])
