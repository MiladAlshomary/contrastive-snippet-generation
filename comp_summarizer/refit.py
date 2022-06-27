import logging
import pickle
from typing import List

import pandas as pd
import numpy as np

from FeaturedArgument import FeaturedArgument
from MMDBase import MMDBase
from Trainer import Trainer
from shared.myutils import make_logging

log = logging.getLogger(f'{__name__}-refit')
BLACKLIST_PATH = 'context_blacklist.txt'
BLACKLIST = set()
DATA_PATH = '../heuristic-data-creation/data/FeaturedArguments-w-reference-snippets-r_0.1-args-me-supmmd-train.pickle'
LAMBDA = .875
GAMMA = 2.
BETA = .16
EPOCHS = 300
CONTINUE_TRAINING = None #f'.checkpoint/mdd-trainer-lambda=0.875_gamma=2.0_beta=0.01_epochs=300.model'


def main():
    make_logging('refit-supmmd')

    data = get_data(normalize_surface_features=True)
    training_data = make_training_data(data)

    refit(LAMBDA, GAMMA, BETA, training_data)


def refit(lambda_: float, gamma: float, beta: float, data: List[MMDBase.triplet]):
    if CONTINUE_TRAINING is not None:
        log.info(f'Continue with training')
        trainer = Trainer.from_checkpoint(CONTINUE_TRAINING)
        trainer.epochs = EPOCHS - trainer.epochs
        log.info(f'Total: {EPOCHS}, epochs left: {trainer.epochs}')
    else:
        trainer = Trainer(param_lambda=lambda_, param_gamma=gamma, param_beta=beta, epochs=EPOCHS)

    model, losses = trainer.train(data, plot_losses=False, context='')

    log.info(f'Final model parameters: {model}')
    trainer.create_checkpoint(filename=f'mdd-trainer-lambda={lambda_}_gamma={gamma}_beta={beta}_epochs={EPOCHS}.model')

    pd.DataFrame.from_records([{'epoch': idx + 1, 'loss': loss} for idx, loss in enumerate(losses)]).to_csv(
        f'results/SupMMD-refit-losses-lambda={lambda_}_gamma={gamma}_beta={beta}_epochs={EPOCHS}.csv', header=True,
        index=False)


def make_training_data(data: List[FeaturedArgument]) -> List[MMDBase.triplet]:
    """
    Depending on :code:`CONTEXT_OPTIM` the function returns a list of triplets List[Trainer.triplets]
    """
    contexts_keys = set([a.query for a in data])
    keys = contexts_keys - BLACKLIST
    contexts = {c: [] for c in keys}
    for a in data:
        if a.query in contexts.keys():
            contexts[a.query].append(a)

    log.info(f'Number of contexts: {len(contexts)}')
    # keys = random.choices(list(contexts.keys()), k=10)
    log.info(f'Choose {len(contexts)} many contexts.')
    triplets = list()
    for c in keys:
        context = contexts[c]
        if len(context) > 1:
            for focal_arg in context:
                triplets.append(make_triplet(focal_arg, context))
        else:
            log.error(f'Excluded context due to too few arguments ({len(context)}): "{c}"')
    return triplets


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
                raise ValueError(f'Length mismatch: expected: {expected_number_of_sf}, actual: {len(sfs)}.')
            for i in range(expected_number_of_sf):
                if sfs[i] > max_values[i]:
                    max_values[i] = sfs[i]

    # normalize
    for argument in data:
        argument.surface_features = argument.surface_features / max_values


def get_data(normalize_surface_features: bool = False) -> List[FeaturedArgument]:
    with open(DATA_PATH, 'rb') as f:
        X = pickle.load(f)

    embs = np.load('embeddings.npa.npy', allow_pickle=True)
    for i in range(len(X)):
        X[i].sentence_embeddings = embs[i]
        X[i].query = X[i].topic
        if len(X[i].sentence_embeddings) != len(X[i].sentences):
            log.error(f'unmatching sizes: {X[i].arg_id}')

    if normalize_surface_features:
        normalize(X)

    log.debug(f'Loaded {len(X)} arguments.')
    return X


def update_blacklist():
    with open(BLACKLIST_PATH, 'r', encoding='utf-8') as f:
        BLACKLIST.update([s.rstrip() for s in f.readlines()])


if __name__ == '__main__':
    update_blacklist()
    main()
