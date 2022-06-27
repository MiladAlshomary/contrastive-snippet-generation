import logging
from typing import List

import numpy as np
from sklearn.base import TransformerMixin

from shared import myutils as u
from shared.Argument import Argument
from shared.DataHandler import DataHandler


class ContrastivenessScorer(TransformerMixin):
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        last = ('', '')
        for argument in X:
            if argument.query == last[0]:
                context = last[1]
            else:
                context = DataHandler.get_query_context(X, argument.query)
                last = (argument.query, context)
            self.context_similarity_vector(argument, context)
        return X

    @staticmethod
    def context_similarity_vector(argument: Argument, context: List[Argument]):
        """
        Computes the similarity between each sentence of the given argument and each argument in the context, sums
        it up over the context.

        :param argument: argument of which sentences should be scored
        :param context: other arguments that form the context
        """
        similarities = list()
        for i, sent_vector in enumerate(argument.sentence_embeddings):
            similarity = 0
            for j, arg in enumerate(context):
                # similarity between sentence and each context is equally weighted
                similarity += (1 / len(context)) * u.cosine_similarity(sent_vector, arg.argument_embedding)
            similarities.append(similarity)

        argument.context_similarity = np.array([similarities] * len(argument.sentences))
