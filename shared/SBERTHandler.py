import numpy as np
from sentence_transformers import SentenceTransformer


class SBERTHandler:
    """
    Handles the conversion from sentences, and documents to numeric embeddings.
    """

    def __init__(self):
        self.model = SentenceTransformer('stsb-roberta-base')

    def sentence_vector(self, sent):
        """
        Gets the vector for a sentence using SBERT model.
        :param sent: string or list of strings a.k.a. sentence(s)
        :return: the sentence embedding
        """
        return self.model.encode(sent, show_progress_bar=False)

    def document_vector(self, argument):
        """
        Averages the sentence embeddings of a given argument.
        :param argument: argument of which a document embedding should be build
        :return: the document embedding
        """
        return np.mean(self.sentence_vector(argument.sentences), axis=0)
