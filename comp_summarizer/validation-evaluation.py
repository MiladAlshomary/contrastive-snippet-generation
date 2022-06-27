import logging
import os
import pickle
from typing import List

from pandas import DataFrame

from shared.DataHandler import DataHandler
from FeaturedArgument import FeaturedArgument
from evaluation_helper import evaluation
from shared.myutils import make_logging

log = logging.getLogger(__name__)
BASE_DIR = '.checkpoint'
OUTPUT = 'results/validation-evaluation.csv'


def main():
    make_logging('validation-evaluation')
    validation_prediction_files = [f'{BASE_DIR}/{fn}' for fn in os.listdir(BASE_DIR) if
                                   fn.endswith('pickle')]
    for f in validation_prediction_files:
        log.info(f'Found file {f}')

    records = list()

    for i, p in enumerate(validation_prediction_files):
        log.info(f'Evaluating file {i+1}/{len(validation_prediction_files)}')
        f, l = parse_name(p)
        predictions: List[FeaturedArgument] = pickle.load(open(p, 'rb'))
        predictions: List[FeaturedArgument] = [a for a in predictions if
                                               (len(a.excerpt_indices) == 2)]
        predictions: List[FeaturedArgument] = [a for a in predictions if
                                               len(DataHandler().get_query_context(predictions,
                                                                                   a.query)) >= 2]
        context_keys = set([a.query for a in predictions])
        log.info(
            f'Found {len(context_keys)} contexts with {len(predictions)} arguments in file {p}')
        log.debug('Starting evaluation...')
        scores = evaluation(predictions)
        log.info(f'{p} scores {scores}')
        records.append({
            'fold': f,
            'lambda': l,
            'representativeness': scores.representativeness,
            'argumentativeness': scores.argumentativeness,
            'edge_correlation': scores.edge_correlation,
            'silhouette_coefficient': scores.silhouette_coef
        })

        DataFrame.from_records(records).to_csv(OUTPUT)


def parse_name(name):
    fold = int(name[34])
    lambda_end = name.find('-args.pickle')
    lambda_start = 34 + 9
    lambda_ = float(name[lambda_start:lambda_end])
    log.debug(f'Parse filename: {name}, fold: {fold}, lambda: {lambda_}')
    return fold, lambda_


if __name__ == '__main__':
    main()
