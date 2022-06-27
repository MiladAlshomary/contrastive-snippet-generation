import logging

import numpy as np
from sklearn.base import TransformerMixin

from Argument import Argument
from SBERTHandler import SBERTHandler
from myutils import cosine_similarity


class TradeOffScorer(TransformerMixin):
    """
    Computes the similarity between a snippet and the average argument.
    """

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.handler = SBERTHandler()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for argument in X:
            if argument.excerpt is not None and len(argument.excerpt) > 0:
                excerpt_embeddings = [self.handler.sentence_vector(sentence) for sentence in
                                      argument.excerpt]
            elif argument.excerpt_indices is not None and len(argument.excerpt_indices) > 0:
                excerpt_embeddings = np.take(argument.sentence_embeddings, argument.excerpt_indices,
                                             axis=0)
            else:
                self.log.debug(f'Skipped {argument.arg_id}.')
                continue

            excerpt_vector = np.mean(np.array(excerpt_embeddings), axis=0)
            if argument.argument_embedding is None or len(argument.argument_embedding) == 0:
                self.compute_arg_embedding(argument)
            # Summary-original comparison
            soc_ex = cosine_similarity(excerpt_vector, argument.argument_embedding)

            argument.soc_ex = soc_ex

            if argument.snippet is not None and len(argument.snippet) > 0 and False:
                # Summary-summary comparison
                snippet_embeddings = [self.handler.sentence_vector(sentence) for sentence in
                                      argument.snippet]
                snippet_vector = np.mean(np.array(snippet_embeddings), axis=0)

                ssc = cosine_similarity(excerpt_vector, snippet_vector)
                argument.ssc = ssc

                if argument.argument_embedding is None or len(argument.sentence_embeddings) == 0:
                    self.compute_arg_embedding(argument)
                # Summary-original comparison
                soc_sn = cosine_similarity(snippet_vector, argument.argument_embedding)
                argument.soc_sn = soc_sn

                self.log.debug(f'{argument.arg_id}:\tssc={ssc} \tsoc_ex={soc_ex} \tsoc_sn={soc_sn}')

    def compute_arg_embedding(self, argument: Argument):
        argument.argument_embedding = self.handler.sentence_vector(" ".join(argument.sentences))
