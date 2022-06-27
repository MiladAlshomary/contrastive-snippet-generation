from typing import List

import numpy as np
import torch

from shared.Argument import Argument


class FeaturedArgument(Argument):
    """
    Extends `Argument` to represent surface features and provide convenience methods for conversion.
    """

    def __init__(self, topic: str = None, query: str = None, arg_id: str = None,
                 sentences: list = None,
                 snippet: list = None, position: List[int] = None, word_count: List[int] = None,
                 noun_count: List[int] = None, tfisf=None,
                 btfisf=None, lr=None, sentence_embeddings: list = None):
        super(FeaturedArgument, self).__init__(topic, query, arg_id, sentences, snippet)
        self.position = position
        self.word_count = word_count
        self.noun_count = noun_count
        self.tfisf = tfisf
        self.btfisf = btfisf
        self.lr = lr
        self.number_of_surface_features = 6
        self.sentence_embeddings = sentence_embeddings
        self.length = len(self.sentences)
        self.surface_features = None  # = self._make_surface_feature_vectors()

    def _make_surface_feature_vectors(self):
        n = len(self.sentences)
        sentence_features = list()
        for i in range(n):
            sf = list()
            sf.append(self.position[i])
            sf.append(self.word_count[i])
            sf.append(self.noun_count[i])
            sf.append(self.tfisf[i])
            sf.append(self.lr[i])
            sentence_features.append(sf)
        assert n == len(sentence_features)
        self.surface_features = torch.tensor(sentence_features)
        return self.surface_features

    @property
    def sentence_embeddings_tensor(self):
        return torch.tensor(self.sentence_embeddings)

    @staticmethod
    def flatten_argument_feature_structure(arguments):
        surface_features = list()
        for argument in arguments:
            surface_features.extend(argument.surface_features)
        return np.array(surface_features)

    @classmethod
    def from_argsme_json(cls, json):
        arg = FeaturedArgument(topic=json['context']['discussionTitle'],
                               query=json['context']['discussionTitle'],
                               arg_id=json['id'],
                               sentences=json['premises'][0]['sentences'],
                               position=json['premises'][0]['position'],
                               word_count=json['premises'][0]['word_counts'],
                               noun_count=json['premises'][0]['noun_counts'],
                               tfisf=json['premises'][0]['tfisf'],
                               lr=json['premises'][0]['lr'],
                               snippet=json['reference'],
                               sentence_embeddings=json['sentence_embeddings'])
        return arg

    @staticmethod
    def only_snippet_argument_dummy(argument, indices=None):
        """
        Constructs a dummy argument from the given. The dummy only contains information regarding
        the snippet, i.e.,
        only a subset of the sentences.
        """
        if indices is None:
            indices = argument.snippet

        fa = FeaturedArgument(
            topic=argument.topic,
            query=argument.query,
            arg_id=argument.arg_id,
            sentences=np.take(argument.sentences, indices, axis=0),
            position=np.take(argument.position, indices, axis=0),
            word_count=np.take(argument.word_count, indices, axis=0),
            noun_count=np.take(argument.noun_count, indices, axis=0),
            tfisf=np.take(argument.tfisf, indices, axis=0),
            lr=np.take(argument.lr, indices, axis=0),
            snippet=indices,
            sentence_embeddings=np.take(argument.sentence_embeddings, indices, axis=0),
        )
        fa.surface_features = torch.index_select(argument.surface_features, dim=0,
                                                 index=torch.tensor(indices))
        fa.argumentativeness_scores = argument.argumentativeness_scores
        fa.number_of_surface_features = argument.number_of_surface_features
        return fa

    @staticmethod
    def context_argument_dummy(arguments):
        """
        Constructs a dummy argument from the given arguments. The dummy contains all sentences (
        and its information) that the given arguments contain. Order is preserved.

        Used for all remaining arguments of a context.
        """
        fa = FeaturedArgument(
            topic='',
            query='',
            arg_id='',
            sentences=[],
            position=[],
            word_count=[],
            noun_count=[],
            tfisf=[],
            lr=[],
            sentence_embeddings=[],
        )
        for arg in arguments:
            fa.sentences.extend(arg.sentences)
            fa.position.extend(arg.position)
            fa.word_count.extend(arg.word_count)
            fa.noun_count.extend(arg.noun_count)
            fa.tfisf.extend(arg.tfisf)
            fa.lr.extend(arg.lr)
            fa.sentence_embeddings.extend(arg.sentence_embeddings)
        fa.number_of_surface_features = arguments[0].number_of_surface_features
        fa.query = arguments[0].query
        fa.length = len(fa.sentences)
        fa.surface_features = torch.empty([fa.length, fa.number_of_surface_features],
                                          dtype=torch.float64)
        idx = 0
        for arg in arguments:
            for s in arg.surface_features:
                fa.surface_features[idx] = s
                idx += 1

        return fa
