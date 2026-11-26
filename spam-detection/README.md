# Spam detection with machine learning

Classify emails as **spam** or **legitimate** from their text, with Python and scikit-learn.

## Data
- `spam-dataset.csv` (provided by the course, not included in this repository): 2,998 preprocessed emails (columns `email`, `label`; 1 = spam). Numbers and URLs are replaced by the tokens `NUMBER` and `URL`.
- **127 duplicates removed before the split**, leaving **2,871 emails** (426 spam, 14.8%). Without this step, the same email can end up in both the training and test sets, which inflates the score.

## Method
1. Stratified 80/20 train/test split (same share of spam in both sets).
2. Text turned into features with **TF-IDF** (words and word pairs, `min_df=2`, sublinear TF).
3. Three models compared with **5-fold stratified cross-validation** (F1 score), inside a pipeline so that TF-IDF is refitted on each fold (no data leakage).
4. The best model is evaluated **once** on the test set.

## Results

| Model | F1, 5-fold CV |
|---|---|
| Naive Bayes | 0.40 |
| Logistic Regression | 0.82 |
| **Linear SVC** | **0.96** |

Linear SVC on the test set (575 emails: 490 legitimate, 85 spam):

| Metric | Value |
|---|---|
| Detection rate (spam recall) | **96.5%** (82 / 85) |
| Precision (spam) | 100% |
| False positive rate | **0%** (0 / 490 legitimate emails blocked) |
| ROC-AUC | 0.997 |
| Overall accuracy | 99.5% |

Confusion matrix: TN = 490, FP = 0, FN = 3, TP = 82.

## Reading the results
- **Accuracy is not the right metric here**: 85% of the emails are legitimate, so a model that always answers "legitimate" would already reach 85%. Recall and false positive rate are what matter for a mail filter.
- The test set holds only 85 spam emails, so the detection rate has a margin of a few points.
- Some of the most "spam" terms are mailing list names (e.g. "webmake talk"): the model also learns the source of the corpus, not only the content. A next step would be to remove headers and list footers.

## Run
```bash
python spam_detection.py
```
