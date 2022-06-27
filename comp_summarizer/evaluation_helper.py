import logging
from collections import namedtuple
from typing import List

import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline

from shared.DataHandler import DataHandler
from shared.EdgeCorrelation import EdgeCorrelation
from FeaturedArgument import FeaturedArgument
from MMDBase import DEVICE
from shared.SentenceArgReAllocator import SentenceArgReAllocator
from shared.SilhouetteCoefficient import SilhouetteCoefficient
from shared.TradeOffScorer import TradeOffScorer
from shared.myutils import tokenize

log = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # ), cache_dir='cache')
model = BertForSequenceClassification.from_pretrained('../bert-finetuning/results/argQ-bert-base-uncased',
                                                      local_files_only=True, cache_dir='cache')
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', task='ArgQ')


Score = namedtuple('Score', ['representativeness', 'argumentativeness', 'edge_correlation', 'silhouette_coef'])

def evaluation(arguments: List[FeaturedArgument]):
    log.info('Starting context-level evaluation...')
    # Move sentences to the closest centroid and re-compute edge correlation
    reallocator = SentenceArgReAllocator()
    reallocator.prepare_snippet_embeddings(arguments)
    reallocator.re_allocate(arguments)
    new_arguments = reallocator.convert_to_argument()

    # Compute edge correlation of new arguments
    ec = EdgeCorrelation()
    corr_realloc = ec.edge_correlation(new_arguments)
    log.debug(f'Edge correlation of arguments after re-allocation: {corr_realloc}.')
    sc = SilhouetteCoefficient()
    try:
        silh_coef_realloc = sc.silhouette_coefficient(new_arguments)
        log.debug(f'Silhouette coefficient of arguments after re-allocation: {silh_coef_realloc}.')
    except Exception as e:
        log.error(f'Error computing silhouette {e}')
        silh_coef_realloc = dict()


    correlations = list()
    coefficients = list()
    for context in DataHandler.get_query_context_keys(arguments):
        correlations.append(corr_realloc[context].correlation)
        if context in silh_coef_realloc:
            coefficients.append(silh_coef_realloc[context])

    avg_edge_corr = np.nanmean(np.array(correlations))
    if len(coefficients) > 0:
        avg_silhouette_coef = np.nanmean(np.array(coefficients))
    else:
        avg_silhouette_coef = 0


    log.info('Starting argument-level evaluation...')
    scorer = TradeOffScorer()
    scorer.transform(arguments)
    argumentativeness = list()
    representativeness = list()
    for arg in arguments:
        representativeness.append(arg.soc_ex)

        tokens = list(map(tokenize, arg.excerpt))
        texts = []
        for t in tokens:
            if len(t) > 510:
                texts.append(" ".join(t[:510]))
                log.warning(f'Shortened {arg.arg_id}\'s excerpt. Cut-off: {" ".join(t[510:])}')
            else:
                texts.append(" ".join(t))
        try:
            argumentativeness.append(np.mean([a['score'] for a in pipeline(texts, device=DEVICE)]))
        except:
            log.error(f'Could not score argumentativeness for {arg.arg_id}.')

    avg_representativeness = np.nanmean(np.array(representativeness))
    if len(argumentativeness) > 0:
        avg_argumentativeness = np.nanmean(np.array(argumentativeness))
    else:
        avg_representativeness = 0

    return Score(avg_representativeness, avg_argumentativeness, avg_edge_corr, avg_silhouette_coef)