# Phishing website detection with machine learning

Classify websites as **phishing** or **legitimate** from features extracted from their URL and page, with Python and scikit-learn. Lab project, ENSIBS.

## Data
- `phishing-dataset.csv` (separator `;`, provided by the course, not included in this repository): **11,055 websites**, 55.7% phishing.
- 11 features: `Having_IP_Address`, `URL_Length`, `Shortining_Service`, `Having_At_Symbol`, `Double_Slash_Redirecting`, `Prefix_Suffix`, `Multi_Sub_Domain`, `Favicon`, `HTTPS_Token`, `Submitting_Email`, `Iframe`, plus the class.

## Method
1. 80/20 train/test split.
2. Three models compared: Logistic Regression, Decision Tree, Linear SVC.
3. **10-fold cross-validation** and a validation curve on the tree size.
4. Hyperparameter tuning with **GridSearchCV** (best: `max_leaf_nodes=220`, `splitter='best'`).

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 66.7% (66.2% in cross-validation) |
| **Tuned Decision Tree** | **70.8%** in 10-fold cross-validation |
## Reading the results
The ceiling comes from the data, not the model: the 11 features are almost all binary, so the 11,055 websites only produce **401 distinct feature combinations**. Many phishing and legitimate sites share the exact same profile and cannot be separated. Going further would require richer features, such as page content or domain reputation.

## Run
```bash
python phishing_detection.py
```
