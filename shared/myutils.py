import atexit
import logging
import sys
import time
from datetime import date, timedelta

import numpy as np
from nltk import word_tokenize


def cosine_similarity(u, v) -> np.array:
    """
    Computes cosine similarity for the given vectors.
    :param u: vector u
    :param v: vector v
    :return: scalar value representing the cosine similarity
    """
    # Note that for 1-D arrays np.dot yield the same results as np.inner.
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def tokenize(text: str) -> list:
    """
    Tokenizes the given text.
    :param text: string to be tokenized
    :return: list of tokens
    """
    return word_tokenize(text)


def argmax(arr, already_selected):
    """
    Finds the argmax in the given collection, considering that some of them are already selected.
    """
    max_val = float('-inf')
    max_idx = 0
    for i in range(len(arr)):
        if arr[i] > max_val and i not in already_selected:
            max_val = arr[i]
            max_idx = i

    return max_idx


def make_logging(filename: str, level=logging.DEBUG):
    # region logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log = logging.getLogger()
    log.setLevel(level)
    shandler = logging.StreamHandler(sys.stdout)
    fhandler = logging.FileHandler(f'logs/{filename}-{date.today()}.log')
    shandler.setLevel(level)
    fhandler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(name)s \t [%(levelname)s] \t %(message)s')
    shandler.setFormatter(formatter)
    fhandler.setFormatter(formatter)
    log.addHandler(shandler)
    log.addHandler(fhandler)

    log.info(f'Application startup: {filename}.')
    # endregion
    # region track execution time
    execution_start = time.time()
    atexit.register(log_execution_time, log, execution_start)
    # endregion


def log_execution_time(logger: logging.Logger, execution_start):
    end = time.time()
    exec_time = end - execution_start
    logger.info(f'Execution time: {str(timedelta(seconds=exec_time))}')
