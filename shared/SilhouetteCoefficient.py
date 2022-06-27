import json
import logging
from typing import List

from sklearn.metrics import silhouette_score

from Argument import Argument


class SilhouetteCoefficient:
    """
    Performs the silhouette analysis for a given set of arguments.

    Usage:
    >>> arguments: List[Argument]
    >>> sc = SilhouetteCoefficient()
    >>> scores = sc.silhouette_coefficient(arguments)
    """

    def __init__(self, save_path=None):
        """
        :param save_path: if specified, the results will be dumped to this location
        """
        self.log = logging.getLogger(SilhouetteCoefficient.__name__)
        self.save_path = save_path

    def silhouette_coefficient(self, arguments: List[Argument]):
        """
        Computes silhouette scores. The implementation accounts for the different contexts of the
        given arguments. Thus, you can pass in all you arguments independently of their context.

        :param arguments: arguments of which silhouette should be computed
        :return:
        """
        contexts = set([a.query for a in arguments])
        result = dict()
        for context in contexts:
            contextual_arguments = list(filter(lambda a: a.query == context, arguments))
            self.log.debug(f'{len(contextual_arguments)} arguments in context {context}.')

            X = list()
            labels = list()
            for i, argument in enumerate(contextual_arguments):
                for sentence in argument.sentence_embeddings:
                    X.append(sentence)
                    labels.append(i)

            assert len(X) == len(labels)
            result[context] = silhouette_score(X, labels, metric='cosine')

        if self.save_path is not None:
            self.save_results(result)

        return result

    def save_results(self, result):
        # Convert from numpy's float to python float
        result = {k: float(v) for k, v in result.items()}
        with open(self.save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file)
