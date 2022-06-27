# Comparative Summarization

## Surface features

To obtain feature values for, generic snippets are needed. To this end, `oracle.py` creates them 
based on the heuristic proposed by [[1]](#Bista.2020). You might want to adapt the input data 
`DATA_PATH`.

In order to compute feature values, execute `features.py`. Before running the script, the paths need
to be adapted according to the output of `oracle.py`.

## Training

The maximum mean discrepancy (MMD) implementation is divided into three classes:
* `MMDBase` provides functions and hyper-parameters for both training and inference
* `Trainer` provides functions to learn the model parameters
* `Inference` provides functions to predict snippets (needs to be initialized with model parameters
  learned by `Trainer`)

The script `training.py` loads and normalizes data in order to use `Trainer` to learn a model. If
you set the variable `CONTEXT_OPTIM` equal to `True` the is performed for each context separately,
which is incompatible with cross-validation strategy. During our experiments, we set `CONTEXT_OPTIM`
equal to `False`.

If the application or system shuts down unexpectedly during training, set `CONTINUE_FROM_CHECKPOINT`
equal to `True` and restart it. Then, the application will continue from the last saved checkpoint.

After cross-validation finishes, `training.py` automatically finds the parameter combination leading
to the smallest mean validation loss, and initiates a refit with all data.

## Predictions

The script `eval-test.py` initializes an `Inference` instance and loads the test dataset in order to
predict snippets. Subsequently, our proposed measures for representativeness, argumentativeness, and
contrastiveness are computed.

If you wish to obtain those measures for each validation fold, set `DUMP_VALIDATION_PREDICTIONS` 
equal to `True`. Then, it will perform predictions during the cross-validation and dump them to file.
Subsequently, `validation-evaluation.py` needs to be executed. It computes the three measures.

---
<span id="Bista.2020">[1] Umanga Bista, Alexander Patrick Mathews, Aditya Krishna Menon, and Lexing Xie. 2020. SupMMD: 
A Sentence Importance Model for Extractive Summarisation using Maximum Mean Discrepancy. In _Findings
of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020 
(Findings of ACL, Vol. EMNLP 2020)_, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association
for Computational Linguistics, 4108–4122. https://doi.org/10.18653/v1/2020.findings-emnlp.367 </span>