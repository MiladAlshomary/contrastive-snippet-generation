import pickle

from WordEmbeddingTransformer import WordEmbeddingTransformer
from myutils import make_logging

import torch


def main():
    # run after features.py and oracle.py
    make_logging('add_embeddings-4486')
    print(torch.cuda.is_available())
    with open(
            'data/1632239915.4824035-3756-arguments-cleaned-test-w-features.pickle',
            'rb') as f:
        X = pickle.load(f)

    emb = WordEmbeddingTransformer()

    upper_bound = len(X) // 10000

    for i in range(upper_bound + 1):
        if (i + 1) * 10000 >= len(X):
            emb.transform(X[i * 10000:])
            with open(
                    'data/1632239915.4824035-3756-arguments-cleaned-test-w-features-w-arg-embedding.pickle',
                    'wb') as f:
                pickle.dump(X, f)
        else:
            emb.transform(X[i * 10000:(i + 1) * 10000])
            with open(f'data/1632239915.4824035-3756-arguments-cleaned-test-w-features-w-arg-embedding[{i * 10000}:{(i + 1) * 10000}].pickle', 'wb') as f:
                pickle.dump(X, f)

if __name__ == '__main__':
    main()
