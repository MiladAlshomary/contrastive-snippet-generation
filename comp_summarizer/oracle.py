import json
import logging

import numpy as np
import pandas as pd
from nltk import word_tokenize
from rouge_score import rouge_scorer
from scipy.stats import hmean
from tqdm import tqdm

from shared.myutils import make_logging, argmax

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
log = logging.getLogger(__name__)

DATA_PATH = '../data/args-me.json'


def main():
    make_logging('oracle', logging.INFO)

    with open(DATA_PATH, 'r') as f:
        d = json.load(f)
    log.debug(f'Number of arguments {len(d)}')

    opt_result = list()
    for r in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        _, ms = run_oracle(d, r)
        r = {
            'r': r,
            'mean_score': ms
        }
        log.info(r)
        opt_result.append(r)

    df = pd.DataFrame.from_records(opt_result)
    df.to_csv('results/optimize_r_results.csv', index=False)
    max_idx = argmax([a['mean_score'] for a in opt_result], [])
    log.info(f'Best value for r is {opt_result[max_idx]}.')


def run_oracle(d, r):
    scores = list()
    for argument in tqdm(d):
        conclusion = argument['conclusion']
        sentences = argument['premises'][0]['sentences']
        extract_idx, snippet = oracle_extraction(sentences, conclusion, r, 2)
        argument['reference'] = extract_idx
        scores.append(alpha(snippet, conclusion))

    with open(f'data/filtered-w-reference-snippets-r_{r}-args-me.json', 'w') as f:
        json.dump(d, f, indent=4)

    return d, np.mean(scores)


def oracle_extraction(Vt, Ht, r, L):
    """
    Extracts the most overlapping sentences (L many) according to alpha-function.

    :param Vt: argument sentences
    :param Ht: conclusion
    :param r: weight
    :param L: word budget
    :return: reference snippet
    """
    St = list()
    St_idx = list()
    while len(St_idx) < L:
        intermediate_scores = np.full(len(Vt), -1.)
        for idx, s in enumerate(Vt):
            intermediate_scores[idx] = (alpha(St + [s], Ht) + alpha(St, Ht)) / len(
                word_tokenize(s)) ** r
        s_star_idx = argmax(intermediate_scores, St_idx)
        St_idx.append(s_star_idx)
        s_star = Vt[s_star_idx]
        St.append(s_star)

    log.debug(f'Selected sentences for {St_idx}')
    return St_idx, St


def alpha(A, B):
    """
    Harmonic mean of rouge-1 and rouge-2 recall.
    :param A: summary
    :param B: reference
    :return: match score
    """
    A = " ".join(A)  # since A is a list of sentences
    score = scorer.score(B, A)
    rouge1 = score['rouge1'].recall
    rouge2 = score['rouge2'].recall
    hm = hmean([rouge1, rouge2])
    return hm


def alpha_2(A, B):
    """
    Jaccard similarity
    :param A: summary
    :param B: reference
    :return: match score
    """
    A_prep = word_tokenize(" ".join(A))
    B_prep = word_tokenize(B)
    intersection = len(list(set(A_prep).intersection(B_prep)))
    union = (len(A_prep) + len(B_prep)) - intersection
    js = float(intersection) / union
    return js


if __name__ == '__main__':
    main()
