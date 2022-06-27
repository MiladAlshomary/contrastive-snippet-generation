import json
import logging
import pickle
import re
from typing import List, Dict

import numpy as np
import torch
from lexrank import LexRank
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.Argument import Argument
from contra_lexrank.ArgumentativenessScorer import ArgumentativenessScorer
from shared.DataHandler import DataHandler
from FeaturedArgument import FeaturedArgument
from shared.myutils import make_logging, tokenize

log = logging.getLogger(__name__)

FILE_NAME = '1632239915.4824035-3756-arguments-cleaned-test'
INPUT_PATH = f'../data/{FILE_NAME}.pickle'
OUTPUT_PATH = f'data/{FILE_NAME}-w-features.pickle'


def main():
    # Run oracle.py first
    make_logging('features')

    # d = load_json_data()
    d = load_pickled_data()

    vocabulary = build_vocab(d)

    log.debug(f'Fitting vectorizer...')
    vectorizer = TfidfVectorizer(lowercase=True,
                                 preprocessor=lambda s: re.sub('[^A-Za-z,.?!]', '', s),
                                 tokenizer=tokenize,
                                 stop_words='english')
    vectorizer.fit(vocabulary)

    log.debug(f'Starting feature computation...')
    farguments: List[FeaturedArgument] = list()
    ngrams = dict()
    for argument in d:
        grams = vectorizer.transform(argument.sentences)
        ngrams[argument.arg_id] = grams

        position(argument)
        count_nouns(argument)
        count_words(argument)
        tfisf(argument, ngrams)
        lr(argument)
        farguments.append(argument)

    log.debug('Computing argumentativeness...')
    scorer = ArgumentativenessScorer(calculation=None,
                                     discourse_markers='../contra-lexrank/discourse-markers.txt',
                                     claim_lexicon='../contra-lexrank/ClaimLexicon.txt')
    scorer.transform(farguments)
    for arg in farguments:
        scores = arg.argumentativeness_scores[0]
        arg._make_surface_feature_vectors()
        arg.surface_features = torch.cat((arg.surface_features, torch.tensor(scores).unsqueeze(1)), 1)
        arg.number_of_surface_features = 6

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(farguments, f)


def load_json_data() -> List[FeaturedArgument]:
    with open('data/filtered-w-reference-snippets-r_0.1-args-me.json', 'r') as f:
        d = json.load(f)

    return [proxy_init(a) for a in d]


def proxy_init(arg_dict) -> FeaturedArgument:
    arg = FeaturedArgument(topic=arg_dict['context']['discussionTitle'],
                           arg_id=arg_dict['id'],
                           sentences=arg_dict['premises'][0]['sentences'],
                           # position=arg_dict['premises'][0]['position'],
                           # word_count=arg_dict['premises'][0]['word_counts'],
                           # noun_count=arg_dict['premises'][0]['noun_counts'],
                           # tfisf=arg_dict['premises'][0]['tfisf'],
                           # lr=arg_dict['premises'][0]['lr']
                           )
    if 'conclusion' in arg_dict:
        arg.query = arg_dict['conclusion']
    if 'reference' in arg_dict:
        arg.snippet = arg_dict['reference']
    if 'sentence_embeddings' in arg_dict:
        arg.sentence_embeddings = arg_dict['sentence_embeddings']
    return arg


def load_pickled_data() -> List[FeaturedArgument]:
    data = DataHandler()
    data.load_bin(INPUT_PATH)
    d = data.get_filtered_arguments([DataHandler.get_args_filter_length(length=3),
                                     DataHandler.get_args_filter_context_size()])

    return [proxy_init_from_argument(a) for a in d]


def proxy_init_from_argument(argument: Argument) -> FeaturedArgument:
    fa = FeaturedArgument(
        topic=argument.topic,
        query=argument.query,
        arg_id=argument.arg_id,
        sentences=argument.sentences,
        sentence_embeddings=argument.sentence_embeddings,
        snippet=list(),
        position=list(),
        word_count=list(),
        noun_count=list(),
        tfisf=list(),
        lr=list(),
    )
    return fa


def build_vocab(d: List[FeaturedArgument]):
    log.debug(f'Building vocab...')
    vocabulary = set()
    for argument in d:
        tokens = list()
        sents = argument.sentences
        for s in sents:
            tokens.extend(tokenize(s))
        vocabulary.update(tokens)

    return vocabulary


def position(argument: FeaturedArgument):
    number_of_sents = len(argument.sentences)
    values = np.zeros(number_of_sents)
    values[0] = 3
    values[1] = 2
    if number_of_sents > 2:
        values[2] = 1
    if number_of_sents > 3:
        values[-1] = 3
    if number_of_sents > 4:
        values[-2] = 2
    if number_of_sents > 5:
        values[-3] = 1
    argument.position = np.array(values)


def count_words(argument: FeaturedArgument):
    counts = list()
    for s in argument.sentences:
        counts.append(len(tokenize(s)))
    argument.word_count = np.array(counts)


def count_nouns(argument: FeaturedArgument):
    counts = list()
    for s in argument.sentences:
        tags = pos_tag(tokenize(s))
        count = sum([1 if 'NN' in t[1] else 0 for t in tags])
        counts.append(count)
    argument.noun_count = np.array(counts)


def tfisf(argument: FeaturedArgument, ngrams: Dict):
    n = len(argument.sentences)
    values = np.full(n, 0.0)
    for i in range(n):
        values[i] = np.sum(ngrams[argument.arg_id][:, 1])

    argument.tfisf = values


def btfisf(argument: FeaturedArgument, d: Dict):
    n = len(argument.sentences)
    values = np.full(n, 0.0)
    first_sent = d[argument.arg_id][:, 0]
    for i in range(n):
        if i == 0:
            values[i] = 3 * np.sum(d[argument.arg_id][:, i])
        else:
            np.sum([w * 3 if w in first_sent else w for w in d[argument.arg_id][:, i]])

    argument.btfisf = values


def lr(argument: FeaturedArgument):
    sentences = argument.sentences
    lxr = LexRank(sentences)
    scores_cont = lxr.rank_sentences(
        sentences,
        threshold=None,
        fast_power_method=False,
    )
    assert len(sentences) == len(
        scores_cont), f'Scores do not match sentences. sents = {len(sentences)}, scores = {scores_cont}'
    argument.lr = scores_cont


if __name__ == '__main__':
    main()
