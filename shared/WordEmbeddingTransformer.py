from sklearn.base import TransformerMixin
from tqdm import tqdm

from SBERTHandler import SBERTHandler


class WordEmbeddingTransformer(TransformerMixin):

    def __init__(self):
        # self.handler = Word2VecHandler('../models/GoogleNews-vectors-negative300.bin')
        self.handler = SBERTHandler()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for argument in tqdm(X):
            if argument.sentence_embeddings is None or len(argument.sentence_embeddings) == 0:
                argument.sentence_embeddings = self.handler.sentence_vector(argument.sentences)
                # for sentence in argument.sentences:
                #     argument.sentence_embeddings.append(self.handler.sentence_vector(
                #         sentence
                #     ))
            if len(argument.argument_embedding) == 0:
                argument.argument_embedding = self.handler.document_vector(argument)
        return X
