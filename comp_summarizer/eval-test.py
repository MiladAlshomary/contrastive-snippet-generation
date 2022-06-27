import logging
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline

from shared.Argument import Argument
from shared.DataHandler import DataHandler
from shared.EdgeCorrelation import EdgeCorrelation
from FeaturedArgument import FeaturedArgument
from Inference import Inference
from MMDBase import MMDBase, DEVICE
from shared.SentenceArgReAllocator import SentenceArgReAllocator
from shared.SilhouetteCoefficient import SilhouetteCoefficient
from shared.TradeOffScorer import TradeOffScorer
from Trainer import Trainer
from shared.myutils import make_logging, tokenize

TRAINER_CHECKPOINT_PATH = ''  # Set to the model you obtained from refitting
DATA_PATH = '../heuristic-data-creation/data/1632239915.4824035-3756-arguments-cleaned-test-w' \
            '-features.pickle'
ARG_Q_MODEL_PATH = '../bert-finetuning/results/argQ-bert-base-uncased'

log = logging.getLogger(__name__)
exp_id = 'supmmd'

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(ARG_Q_MODEL_PATH, local_files_only=True,
                                                      cache_dir='cache')
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', task='ArgQ')
sim_func = MMDBase(param_gamma=.0, param_lambda=.0).cosine_kernel_matrix


def main():
    make_logging('eval-test')

    parameter = load_checkpoint()

    inf = Inference(
        param_gamma=parameter['param_gamma'],
        param_lambda=parameter['param_lambda'],
        thetaA=parameter['thetaA'],
        thetaB=parameter['thetaB'],
        snippet_length=2,
    )
    log.info('Inference initialized.')

    data = get_data(normalize_surface_features=True)
    contexts = {a.query for a in data}

    cols = ['gamma', 'lambda', 'avg_edge_corr', 'avg_silhouette_coef']
    for context in contexts:
        cols.append(f'{context}_edge_corr')
        cols.append(f'{context}_silhouette_coef')

    current_results = {}
    results_tmp = pd.DataFrame(columns=cols)
    results_tmp = results_tmp.append(current_results, ignore_index=True)

    cols = ['gamma', 'lambda', 'arg_id', 'argumentativeness', 'weighted_degree_centrality', 'soc',
            'objective_value']
    arg_level_results_tmp = pd.DataFrame(columns=cols)

    global results
    results = results_tmp
    del results_tmp
    global arg_level_results
    arg_level_results = arg_level_results_tmp
    del arg_level_results_tmp

    objectives = inf.fit(data)

    if any([a.excerpt_indices is None or len(a.excerpt_indices) == 0 for a in data]):
        log.error('Excerpts were not set.')
        raise ValueError('Excerpts were not set.')
    evaluation(data, param_gamma=parameter['param_gamma'], param_lambda=parameter['param_lambda'])
    arg_level_eval(data, param_gamma=parameter['param_gamma'],
                   param_lambda=parameter['param_lambda'], objectives=objectives)
    results.to_csv(f'results/{exp_id}.csv', index=False)
    arg_level_results.to_csv(f'results/{exp_id}-arg-level.csv', index=False)


def evaluation(X: List[Argument], param_gamma, param_lambda):
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

    current_results = {'gamma': param_gamma, 'lambda': param_lambda}
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


def arg_level_eval(arguments: List[FeaturedArgument], param_gamma, param_lambda, objectives: Dict):
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
            'gamma': param_gamma,
            'lambda': param_lambda,
            'arg_id': arg.arg_id,
            'argumentativeness': float(argumentativeness),
            'weighted_degree_centrality': cdc,
            'arg_length': len(arg.sentences),  # to normalize weighted_degree_centrality
            'soc': arg.soc_ex,
            'objective_value': objectives[arg.arg_id] if arg.arg_id in objectives.keys() else None,
        })

    global arg_level_results
    arg_level_results = arg_level_results.append(records, ignore_index=True)


def load_checkpoint():
    trainer = Trainer.from_checkpoint(TRAINER_CHECKPOINT_PATH)
    return {
        'param_gamma': trainer.param_gamma,
        'param_lambda': trainer.param_lambda,
        'snippet_length': 2,
        'thetaA': trainer.model,
        'thetaB': trainer.model,
    }


def make_triplet(argument, context) -> MMDBase.triplet:
    context_wo_focal = [arg for arg in context if arg != argument]
    return MMDBase.triplet(argument, FeaturedArgument.context_argument_dummy(context_wo_focal),
                           FeaturedArgument.only_snippet_argument_dummy(argument))


def normalize(data: List[FeaturedArgument]):
    # find maximums
    max_values = data[0].surface_features[0]  # features of the first sentence of the first argument
    expected_number_of_sf = len(max_values)
    for argument in data:
        for sfs in argument.surface_features:  # iterate over sentences' surface features
            if expected_number_of_sf != len(sfs):
                raise ValueError(
                    f'Length mismatch: expected: {expected_number_of_sf}, actual: {len(sfs)}.')
            for i in range(expected_number_of_sf):
                if sfs[i] > max_values[i]:
                    max_values[i] = sfs[i]

    # normalize
    for argument in data:
        argument.surface_features = argument.surface_features / max_values


def get_data(normalize_surface_features: bool = False) -> List[FeaturedArgument]:
    with open(DATA_PATH, 'rb') as f:
        X = pickle.load(f)

    if normalize_surface_features:
        normalize(X)

    log.debug(f'Loaded {len(X)} arguments.')
    return X


if __name__ == '__main__':
    main()
