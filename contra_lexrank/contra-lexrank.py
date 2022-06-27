import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, TextClassificationPipeline, BertForSequenceClassification

from comp_summarizer.MMDBase import MMDBase
from shared.Argument import Argument
from CentralityScorer import CentralityScorer
from ContraLexRank import ContraLexRank
from ContrastivenessScorer import ContrastivenessScorer
from ArgumentativenessScorer import ArgumentativenessScorer
from shared.DataHandler import DataHandler
from shared.EdgeCorrelation import EdgeCorrelation
from shared.SentenceArgReAllocator import SentenceArgReAllocator
from shared.SilhouetteCoefficient import SilhouetteCoefficient
from shared.TradeOffScorer import TradeOffScorer
from shared.myutils import make_logging, tokenize

log = logging.getLogger(__name__)
ARG_Q_MODEL_PATH = '../bert-finetuning/results/argQ-bert-base-uncased'
DATA_PATH = '../data/1629700068.9873986-4566-arguments-cleaned.pickle'
exp_id = 'contra-lexrank'

if torch.cuda.device_count() >= 1:
    DEVICE = torch.cuda.current_device()
    log.info(torch.cuda.current_device())
    log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    DEVICE = -1
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
model = BertForSequenceClassification.from_pretrained(ARG_Q_MODEL_PATH, local_files_only=True,
                                                      cache_dir='cache')
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', task='ArgQ')
sim_func = MMDBase(param_gamma=.0, param_lambda=.0).cosine_kernel_matrix


def main():
    make_logging(exp_id, logging.DEBUG)

    data = DataHandler()
    data.load_bin(DATA_PATH)
    X = data.get_filtered_arguments([DataHandler.get_args_filter_length(),
                                     DataHandler.get_args_filter_context_size()])
    log.info(f'Load {len(X)} arguments.')

    params = gen()
    params.extend(gen_generic())

    # region Initial base scores
    ec = EdgeCorrelation()
    corr_original = ec.edge_correlation(X)
    log.info(f'Edge correlation of arguments as given (original): {corr_original}.')
    sc = SilhouetteCoefficient()
    silh_coef = sc.silhouette_coefficient(X)
    log.info(f'Silhouette coefficient of arguments as given (original): {silh_coef}.')
    current_results = {}
    # Saving results of initial arguments
    correlations = list()
    coefficients = list()
    for context in DataHandler.get_query_context_keys(X):
        current_results[f'{context}_edge_corr'] = corr_original[context].correlation
        correlations.append(corr_original[context].correlation)
        current_results[f'{context}_silhouette_coef'] = silh_coef[context]
        coefficients.append(silh_coef[context])
    current_results['avg_edge_corr'] = np.mean(np.array(correlations))
    current_results['avg_silhouette_coef'] = np.mean(np.array(coefficients))
    # endregion

    cols = ['d_1', 'd_2', 'd_3', 'avg_edge_corr', 'avg_silhouette_coef']
    for context in data.get_query_context_keys(X):
        cols.append(f'{context}_edge_corr')
        cols.append(f'{context}_silhouette_coef')

    results_tmp = pd.DataFrame(columns=cols)
    results_tmp = results_tmp.append(current_results, ignore_index=True)

    cols = ['d_1', 'd_2', 'd_3', 'arg_id', 'argumentativeness', 'weighted_degree_centrality',
            'soc']
    arg_level_results_tmp = pd.DataFrame(columns=cols)

    global results
    results = results_tmp
    del results_tmp
    global arg_level_results
    arg_level_results = arg_level_results_tmp
    del arg_level_results_tmp

    log.info(f'Evaluation of {len(params)} parameter combinations.')
    run(X, params)


def run(X: List[Argument], parameter: List):
    counter = 0
    total = len(parameter)
    for d_1, d_2, d_3 in parameter:
        log.info(f'Starting ContraLexRank, iteration: {counter}/{total}')
        log.info(f'Running with {d_1, d_2, d_3}')
        counter += 1
        pipeline = Pipeline(steps=[
            ('argumentativeness', ArgumentativenessScorer()),
            ('contrastiveness', ContrastivenessScorer()),
            ('centrality', CentralityScorer()),
            ('clr', ContraLexRank(d_1=d_1, d_2=d_2, d_3=d_3, limit=2)),
        ])
        pipeline.predict(X)
        evaluation(X, d_1, d_2, d_3)
        arg_level_eval(X, d_1, d_2, d_3)
        results.to_csv(f'results/{exp_id}.csv', index=False)
        arg_level_results.to_csv(f'results/{exp_id}-arg-level.csv', index=False)


def evaluation(X: List[Argument], d_1, d_2, d_3):
    log.info('Starting context-level evaluation...')
    # Move sentences to closest centroid and re-compute edge correlation
    reallocator = SentenceArgReAllocator()
    reallocator.prepare_snippet_embeddings(X)
    reallocator.re_allocate(X)
    new_arguments = reallocator.convert_to_argument()

    # Compute edge correlation of new arguments
    ec = EdgeCorrelation()
    corr_realloc = ec.edge_correlation(new_arguments)
    log.info(f'Edge correlation of arguments after re-allocation: {corr_realloc}.')
    sc = SilhouetteCoefficient()
    silh_coef_realloc = sc.silhouette_coefficient(new_arguments)
    log.info(f'Silhouette coefficient of arguments after re-allocation: {silh_coef_realloc}.')

    current_results = {'d_1': d_1, 'd_2': d_2, 'd_3': d_3}
    correlations = list()
    coefficients = list()
    for context in DataHandler.get_query_context_keys(X):
        current_results[f'{context}_edge_corr'] = corr_realloc[context].correlation
        correlations.append(corr_realloc[context].correlation)
        current_results[f'{context}_silhouette_coef'] = silh_coef_realloc[context]
        coefficients.append(silh_coef_realloc[context])

    current_results['avg_edge_corr'] = np.mean(np.array(correlations))
    current_results['avg_silhouette_coef'] = np.mean(np.array(coefficients))

    global results
    results = results.append(current_results, ignore_index=True)


def arg_level_eval(arguments: List[Argument], d_1, d_2, d_3):
    log.info('Starting argument-level evaluation...')
    scorer = TradeOffScorer()
    scorer.transform(arguments)
    records = list()
    for arg in arguments:
        sim_mat = sim_func(torch.tensor(arg.sentence_embeddings))
        c0 = arg.excerpt_indices[0]
        c1 = arg.excerpt_indices[1]
        cdc = float(sum(sim_mat[c0]) + sum(sim_mat[c1]))

        tokens = list(map(tokenize, arg.excerpt))
        texts = []
        for t in tokens:
            if len(t) > 510:
                texts.append(" ".join(t[:510]))
                log.warning(f'Shortened {arg.arg_id}\'s excerpt. Cut-off: {" ".join(t[510:])}')
            else:
                texts.append(" ".join(t))
        try:
            argumentativeness = np.mean([a['score'] for a in pipeline(texts, device=DEVICE)])
        except:
            log.error(f'Could not score argumentativeness for {arg.arg_id}.')
            argumentativeness = -1
        records.append({
            'd_1': d_1,
            'd_2': d_2,
            'd_3': d_3,
            'arg_id': arg.arg_id,
            'argumentativeness': float(argumentativeness),
            'weighted_degree_centrality': cdc,
            'arg_length': len(arg.sentences),  # to normalize weighted_degree_centrality
            'soc': arg.soc_ex,
        })

    global arg_level_results
    arg_level_results = arg_level_results.append(records, ignore_index=True)


def gen():
    total = 0
    actual = 0
    parameters = list()
    for d_3 in range(1, 11, 1):
        for d_1 in range(1, 11, 1):
            total += 1
            #           1 = d_1 + d2 - d_3
            # <=> 1 - d_2 = d_1 - d_3
            # <=>   - d_2 = d_1 - d_3 - 1
            # <=>     d_2 = - d_1 + d_3 + 1
            d_2 = -d_1 + d_3 + 10
            if 0 < d_1 < 10 and 0 < d_2 < 10:
                actual += 1
                parameters.append((d_1 / 10., d_2 / 10., d_3 / 10.))

    log.debug(f'{actual}, {total}, {actual / total}')
    return parameters


def gen_generic():
    """
    Generates parameter combinations with :math:`d_3=0`.
    """
    parameters = list()
    for d_2 in range(0, 10, 1):
        d_1 = 10 - d_2
        parameters.append((d_1 / 10., d_2 / 10., 0.0))

    return parameters


if __name__ == '__main__':
    main()
