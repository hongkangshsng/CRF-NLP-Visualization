# CRF-NLP-Visualization

A technical report and Python implementation of **Linear-chain Conditional Random Fields (CRF)** for Natural Language Processing sequence labeling.

## 1. Project Goal

Sequence labeling predicts a label for every element of an observation sequence. Typical NLP applications include part-of-speech tagging, named-entity recognition and word segmentation.

This project implements a small CRF from scratch so the internal calculation is visible. It studies:

- observation sequence **X** and predicted sequence **Y**
- state and transition feature functions
- feature values and learned weights
- CRF sequence SCORE
- partition function and conditional probability
- model training and prediction
- feature-ablation experiments
- visualization and exported results

The goal is not only to obtain a prediction, but to explain **why one label sequence receives a higher score than another**.

## 2. Observation Sequence X and Label Sequence Y

Example observation:

```text
X = [自然, 語言, 處理, 很, 有趣]
```

A possible POS label sequence is:

```text
Y = [N, N, V, ADV, ADJ]
```

| Label | Meaning |
|---|---|
| N | Noun |
| V | Verb |
| ADV | Adverb |
| ADJ | Adjective |

A Linear-chain CRF considers both the observation-label relationship and dependencies between neighboring labels:

```text
自然       語言       處理       很        有趣
 |          |          |         |          |
 v          v          v         v          v
 N  ----->  N  ----->  V  -----> ADV -----> ADJ
       transition relationships
```

## 3. Conditional Probability Model

The CRF models:

```text
P(Y | X) = exp(SCORE(X,Y)) / Z(X)
```

with

```text
SCORE(X,Y)
= sum_i sum_k lambda_k * f_k(y_(i-1), y_i, X, i)
```

and partition function:

```text
Z(X) = sum over all candidate Y' of exp(SCORE(X,Y'))
```

`Z(X)` normalizes candidate scores so their probabilities sum to one.

The final prediction is:

```text
Y* = argmax_Y P(Y|X)
```

## 4. State Feature Functions

A state feature describes compatibility between an observation and a candidate label:

```text
s_k(y_i, X, i)
```

Example:

```text
Feature = WORD=處理|TAG=V
Value   = 1
```

The implementation includes features such as:

```text
WORD=<word>|TAG=<label>
LAST=<last character>|TAG=<label>
LEN=<word length>|TAG=<label>
BIAS|TAG=<label>
```

These allow the model to learn how strongly an observed word or word property supports a label.

## 5. Transition Feature Functions

A transition feature models adjacent labels:

```text
t_k(y_(i-1), y_i, X, i)
```

Examples:

```text
TRANS=N->V
TRANS=N->N
TRANS=ADV->ADJ
```

This lets CRF use sequence structure instead of treating each token as an independent classification problem.

## 6. Feature Value, Learned Weight and SCORE

A feature function produces a value, while training learns its weight.

Example:

```text
Feature       = WORD=處理|TAG=V
Feature value = 1
Weight        = +2.31

Contribution = value * weight
             = 1 * 2.31
             = +2.31
```

Positive weights increase the sequence score when a feature is active; negative weights decrease it. The complete sequence score is the sum of all active feature contributions.

The Python program prints these contributions so the calculation can be inspected directly.

## 7. From SCORE to Probability

Suppose candidate sequences receive:

| Candidate Y | SCORE |
|---|---:|
| N -> N | 1.2 |
| N -> V | 3.5 |
| V -> N | -0.3 |
| V -> V | 0.8 |

The scores are exponentiated and divided by the partition function. A larger SCORE therefore generally produces a larger `P(Y|X)`, while the exact probability also depends on all competing sequences.

## 8. Training

Weights are learned rather than manually assigned. The implementation maximizes conditional log-likelihood.

The key gradient idea is:

```text
gradient =
    empirical feature counts
    - expected feature counts under P(Y|X)
```

L2 regularization is included to discourage unnecessarily large weights. Training log-likelihood is printed across epochs.

## 9. Feature Ablation Study

The project compares four configurations:

| Experiment | Lexical | Suffix/Length | Transition |
|---|:---:|:---:|:---:|
| A_state_only | Yes | Yes | No |
| B_transition_only | No | No | Yes |
| C_full | Yes | Yes | Yes |
| D_no_suffix | Yes | No | Yes |

This allows us to investigate how different feature choices influence prediction performance.

- **State only:** observation-to-label information without label transitions.
- **Transition only:** sequence structure without lexical evidence.
- **Full CRF:** combines state and transition information.
- **No suffix:** tests the effect of additional word-form features.

## 10. Evaluation

The experiment reports:

- token accuracy
- sentence accuracy
- training conditional log-likelihood
- learned feature weights
- candidate sequence probabilities

The included dataset is deliberately small, so the results demonstrate CRF mechanics rather than benchmark production Chinese POS tagging.

## 11. Visualization and Output

Running the program generates:

```text
outputs/
├── experiment_results.csv
├── learned_weights.csv
├── prediction_details.csv
├── analysis.txt
├── 01_feature_selection_accuracy.png
├── 02_top_weights.png
├── 03_training_curves.png
└── 04_sequence_probabilities.png
```

The four figures analyze feature-selection performance, learned weights, training behavior and candidate-sequence probabilities.

### Current ablation result

| Experiment | Test token accuracy | Test sentence accuracy |
|---|---:|---:|
| A_state_only | 92.3% | 66.7% |
| B_transition_only | 69.2% | 66.7% |
| C_full | 100.0% | 100.0% |
| D_no_suffix | 84.6% | 66.7% |

In this deliberately small educational dataset, the full state + transition feature configuration obtains the highest observed test accuracy. This should **not** be interpreted as a general benchmark: the dataset is tiny and was designed to make CRF calculations inspectable. The ablation mainly demonstrates that lexical/state evidence and transition structure provide complementary information.

## 12. Project Structure

```text
CRF-NLP-Visualization/
├── README.md
├── main.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── outputs/
```

## 13. Run in VS Code

Clone:

```bash
git clone https://github.com/hongkangshsng/CRF-NLP-Visualization.git
cd CRF-NLP-Visualization
```

Create a virtual environment on Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

## 14. What to Observe in the Terminal

The calculation is displayed in a form similar to:

```text
X = [...]
Gold = [...]
Pred = [...]

Feature                    Value    Weight    Contribution
WORD=...|TAG=...             1       ...          ...
TRANS=N->V                   1       ...          ...

SCORE(X,Y*) = sum(value * weight)
log Z(X)     = ...
P(Y*|X)      = exp(SCORE - logZ)
```

This directly connects the mathematical CRF definition with its implementation.

## 15. Limitations and Future Work

For educational transparency, this version uses short sequences and enumerates candidate label sequences. The number of possible sequences grows exponentially for larger problems.

Possible extensions include:

- Viterbi decoding
- Forward-Backward dynamic programming
- larger Chinese POS or NER datasets
- precision, recall and F1
- confusion matrices
- comparison with sklearn-crfsuite
- BiLSTM-CRF or Transformer-based sequence labeling

## 16. Conclusion

CRF prediction is structured prediction rather than independent word classification. State features represent relationships between observations and labels, while transition features model dependencies between neighboring labels. Learned weights determine how strongly each active feature changes the sequence score.

The complete project flow is:

```text
Observation X
     |
     v
Feature extraction
     |
     v
Feature values * learned weights
     |
     v
SCORE(X,Y)
     |
     v
Partition function Z(X)
     |
     v
P(Y|X)
     |
     v
Best label sequence Y*
```

The feature-ablation experiments then show how different feature choices can change CRF behavior and model performance.

## License

MIT License.
