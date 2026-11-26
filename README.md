# Machine learning for security

Two machine learning projects applied to cybersecurity, carried out during my engineering studies in cybersecurity at ENSIBS (France).

| Project | Data | Best model | Key result |
|---|---|---|---|
| [Spam detection](spam-detection/) | 2,871 deduplicated emails (426 spam) | TF-IDF + Linear SVC | 96.5% of spam detected, 0 false positives on the test set |
| [Phishing website detection](phishing-detection/) | 11,055 websites, 11 URL features | Tuned Decision Tree | 70.8% accuracy (10-fold cross-validation) |

## What these projects show
- A clean evaluation method: duplicates removed before splitting, stratified train/test split, cross-validation, a single final evaluation on the test set.
- Security-oriented metrics: detection rate (recall), precision and **false positive rate**. In production, blocking a legitimate email has a real cost.
- An honest reading of the results, including the limits of each dataset.

## Run
```bash
pip install -r requirements.txt
cd spam-detection && python spam_detection.py
cd ../phishing-detection && python phishing_detection.py
```
Each script expects its CSV file in the same folder. The datasets were provided as part of an ENSIBS course and are not redistributed here.

## Author
Yasser Slimani, cybersecurity engineering student (ENSIBS, cyberdefense specialization).
