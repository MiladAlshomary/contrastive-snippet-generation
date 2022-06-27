import logging
import os

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from shared.myutils import make_logging

os.environ['TRANSFORMERS_CACHE'] = 'cache'


def main():
    make_logging('bert-finetuning-inference', level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(torch.__version__)
    log.info(torch.cuda.is_available())
    log.info(torch.cuda.device_count())
    if torch.cuda.device_count() >= 1:
        DEVICE = torch.cuda.current_device()
        log.info(torch.cuda.current_device())
        log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        DEVICE = -1  # use cpu
    # https://huggingface.co/transformers/main_classes/pipelines.html#textclassificationpipeline

    model = BertForSequenceClassification.from_pretrained('results/argQ-bert-base-uncased',
                                                          local_files_only=True,
                                                          cache_dir='cache')

    data = pd.read_csv('test.csv')
    text_column = 'text'
    score_column = 'label'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')

    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt',
                                          task='ArgQ')
    results = pipeline(data[text_column].values.tolist(), device=DEVICE)

    log.info('Evaluating...')
    y_true = data[score_column].values.tolist()
    y_pred = [r['score'] for r in results]

    mse = mean_squared_error(
        y_true,
        y_pred
    )
    r2 = r2_score(
        y_true,
        y_pred
    )
    r = pearsonr(
        y_true,
        y_pred
    )
    p = spearmanr(
        a=y_true,
        b=y_pred
    )
    log.info(f'MSE: {mse}')
    log.info(f'R2: {r2}')
    log.info(f'Pearson r: {r}')
    log.info(f'Spearman p: {p}')
    log.debug(results)


if __name__ == '__main__':
    main()
