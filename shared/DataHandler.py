import json
import logging
import pickle
from typing import List, Callable

from Argument import Argument


class DataHandler:
    """
    Encapsulates file access of the data
    """

    def __init__(self, arguments: List[Argument] = None):
        """
        Initializes the handler with the path of the text da
        """
        self.log = logging.getLogger(__name__)
        self.log.debug('%s initialized.', __name__)
        if arguments is None:
            self.__data = list()
        else:
            self.__data = arguments

    def load_json(self, path, results=False):
        """
        Loads the data from json file.
        :param path: path to json file
        :param results: indicates whether results should be loaded
        """
        with open(path, 'r') as file:
            _data_json = json.load(file)
        for arg in _data_json:
            self.__data.append(Argument.from_json(arg, results))
        self.log.debug(f'{len(self.__data)} arguments loaded.')

    def load_argsme_json(self, path):
        """
        Loads the data from json file. Data must be formatted like args.me corpus
        :param path: path to json file
        """
        with open(path, 'r') as file:
            _data_json = json.load(file)
        for arg in _data_json:
            self.__data.append(Argument.from_argsme_json(arg))
        self.log.debug(f'{len(self.__data)} arguments loaded.')

    def save_results(self, path):
        d = [arg.to_json() for arg in self.__data]
        with open(path, 'w') as file:
            file.write(json.dumps(d))

    def get_arguments(self) -> List[Argument]:
        """
        Gets the data stored by the handler.
        :return: list of `Argument`
        """
        return self.__data

    def get_filtered_arguments(self, filter_fn: List[Callable]) -> List[Argument]:
        """
        Gets the data stored by the handler that fulfills the filter condition.

        :param filter_fn: functions to filter arguments
        :return: list of `Argument`
        """
        if len(filter_fn) == 0:
            return self.__data
        elif len(filter_fn) == 1:
            return list(filter(lambda a: filter_fn[0](a, self.__data), self.__data))
        else:
            filtered_data = self.__data.copy()
            for ffn in filter_fn:
                filtered_data = list(filter(lambda a: ffn(a, filtered_data), filtered_data))
            return filtered_data

    def dump_data(self, path, limit: int = None):
        """
        Dumps the data in a pickle.
        :param path: path to pickle
        :param limit: restricts the number of persists elements
        """
        with open(path, 'wb') as f:
            if limit is not None:
                pickle.dump(self.__data[:limit], f)
            else:
                pickle.dump(self.__data, f)

    def load_bin(self, path):
        """
        Loads the data from pickle.
        :param path: path to pickle
        """
        with open(path, 'rb') as f:
            self.__data = pickle.load(f)
        self.log.debug('%s arguments loaded.', len(self.__data))

    @staticmethod
    def get_query_context_keys(arguments: List[Argument]):
        return set([a.query for a in arguments])

    @staticmethod
    def get_query_context(arguments: List[Argument], query: str) -> List[Argument]:
        return list(filter(lambda a: a.query == query, arguments))

    @staticmethod
    def flatten_argument_sentence_structure(arguments: List[Argument], embeddings=True):
        sentences = list()
        for argument in arguments:
            if embeddings:
                sentences.extend(argument.sentence_embeddings)
            else:
                sentences.extend(argument.sentences)

        return sentences

    @staticmethod
    def get_args_filter_length(length=2) -> Callable[[Argument, List[Argument]], bool]:
        length_filter: Callable[[Argument, List[Argument]], bool] = lambda arg, all_arguments: len(
            arg.sentences) >= length
        return length_filter

    @staticmethod
    def get_args_filter_context_size(length=2) -> Callable[[Argument, List[Argument]], bool]:

        def _context_size_filter(argument: Argument, all_arguments: List[Argument]) -> bool:
            return len(DataHandler.get_query_context(all_arguments, argument.query)) > length

        return _context_size_filter
