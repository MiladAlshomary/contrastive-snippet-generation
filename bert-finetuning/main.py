import logging
import os

import pandas as pd
import torch
from datasets import load_dataset
from torch.nn import MSELoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer

from shared.myutils import make_logging

os.environ['TRANSFORMERS_CACHE'] = 'cache'


def main():
    make_logging('bert-finetuning')
    log = logging.getLogger(__name__)

    log.info(torch.__version__)
    log.info(torch.cuda.is_available())
    log.info(torch.cuda.device_count())
    if torch.cuda.device_count() >= 1:
        NO_CUDA = False
        log.info(torch.cuda.current_device())
        log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        NO_CUDA = True
    # https://huggingface.co/transformers/training.html
    # https://huggingface.co/docs/datasets/loading_datasets.html#csv-files

    raw_data = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'},
                            cache_dir='cache')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
    encoded_data = raw_data.map(
        lambda raw: tokenizer(raw['text'], padding='max_length', truncation=True))

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1,
                                                               cache_dir='cache')
    # https://huggingface.co/transformers/_modules/transformers/training_args.html
    training_args = TrainingArguments(
        'argQ-trainer',
        no_cuda=NO_CUDA,
        num_train_epochs=5,
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        learning_rate=2e-5
    )

    # Metric https://discuss.huggingface.co/t/which-loss-function-in
    # -bertforsequenceclassification-regression/1432/3
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_data['train'],
        eval_dataset=encoded_data['train'],
        # compute_metrics=compute_metric
    )
    trainer.train()
    trainer.evaluate()

    trainer.save_model('results/argQ-bert-base-uncased')

    log.info('Finished')


def compute_metric(eval_pred):
    metric = MSELoss()
    logits, labels = eval_pred
    return metric(logits.view(-1), labels.view(-1))


def load_data():
    """
    Split entire dataset into partitions. Run separately.
    """
    data = pd.read_csv(
        '../data/IBM_Debater_(R)_arg_quality_rank_30k/arg_quality_rank_30k.csv')
    data = data[['argument', 'WA']].rename(columns={'argument': 'text', 'WA': 'label'})
    data[:24000].to_csv('train.csv', index=False, header=True)
    data[24000:].to_csv('test.csv', index=False, header=True)
    return data


if __name__ == '__main__':
    main()
    # load_data()
