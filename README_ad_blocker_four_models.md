# ML-Based Ad URL Classification — Four Pipelines + Soft Voting

This educational experiment uses **Random Forest, Logistic Regression, SVM, and XGBoost** to predict whether a URL is **ALLOW (0)** or **BLOCK (1)**. All four classifiers run inside **Scikit-learn Pipelines**, which fit feature extraction separately within each training/CV fold. A fifth candidate combines the four tuned Pipelines using **Soft Voting**.

> **Not a real ad blocker.** The code classifies strings; it does not intercept, block, or filter browser traffic. The provided CSV contains only 30 *synthetic* examples, so measured accuracy is not evidence of real-world ad-blocking performance.

## Files

| File | Purpose |
| --- | --- |
| [`Ad_Blocker_RF_LR_SVM_XGBoost_Colab.ipynb`](Ad_Blocker_RF_LR_SVM_XGBoost_Colab.ipynb) | Step-by-step Google Colab notebook; starts with CSV upload and `pd.read_csv()` |
| [`ad_blocker_four_models.py`](ad_blocker_four_models.py) | Equivalent command-line Python implementation |
| [`data/ad_urls.csv`](data/ad_urls.csv) | 30 labeled example URLs (`url`, `label`) |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`README_ad_blocker_four_models.md`](README_ad_blocker_four_models.md) | These instructions and reproducible results |

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Ad_Blocker_RF_LR_SVM_XGBoost_Colab.ipynb` using **File → Upload notebook**, or open the [notebook from GitHub](https://colab.research.google.com/github/Ruchika-choudhary8/new/blob/main/Ad_Blocker_RF_LR_SVM_XGBoost_Colab.ipynb) while signed in to your GitHub account (if required).
3. Run **Step 1** to install libraries.
4. Run **Step 2** to upload `ad_urls (1).csv` or `ad_urls.csv` (unless the CSV is already in the notebook workspace). This cell explicitly uses `pd.read_csv`.
5. Run the rest of the notebook in order and inspect training-CV scores, independent test metrics, confusion matrix, example decisions, and the downloadable `ad_blocker_four_model_results.zip`.

## Run locally

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python ad_blocker_four_models.py --csv "data/ad_urls.csv" --n-iter 4
```

Classify additional sample URLs in the demo step:

```bash
python ad_blocker_four_models.py --csv "data/ad_urls.csv" \
  --url "https://ads.example.net/banner/latest.js" \
  --url "https://news.example.com/technology"
```

The CLI outputs `outputs/ad_blocker/` by default. It saves CSV metrics, a confusion-matrix image, a JSON results summary, predictions on the held-out URLs, and the fitted CV-selected Pipeline as a `.joblib` file. Only load `.joblib` files from trusted sources.

## ML methodology

**Dataset**: 30 URLs, evenly split between class `0` and class `1`. Original filename: `ad_urls (1).csv`.

**Split**: stratified `70% train / 30% test`, using `random_state=42`. Holdout: **9 URLs**. The test set is excluded from preprocessing, hyperparameter selection, and CV.

**Feature extraction inside the Pipelines**:

1. Character-level TF-IDF with **3–5-grams**.
2. Token-level TF-IDF with **1–2-grams**.
3. A **fixed** set of common ad/tracker token indicators using `CountVectorizer` (domain knowledge, not inferred from held-out labels).

The three feature matrices are combined using `FeatureUnion`. Each of the four models receives the same feature representation.

**Optimization**: each Pipeline is tuned using `RandomizedSearchCV` (4 configurations per model, stratified **five-fold training cross-validation**). The four tuned Pipelines also form an equal-weight `VotingClassifier(voting='soft')`. The model with the highest training-CV accuracy is selected **before** evaluating the final holdout.

## Reproduced benchmark with the supplied dataset

| Candidate | Training 5-fold CV accuracy | Held-out test accuracy |
| --- | ---: | ---: |
| Random Forest | 60.0% | 55.6% (5/9) |
| Logistic Regression | 73.0% | 66.7% (6/9) |
| **SVM** | **100.0%** | **100.0% (9/9)** |
| XGBoost | 53.0% | 22.2% (2/9) |
| Soft Voting (four Pipelines) | 88.0% | 77.8% (7/9) |

**CV-selected winner: SVM**. The four-model ensemble was **not** better than SVM; more models do not automatically mean higher accuracy. The script transparently reports all candidates' test performance but does not select a new winner based on those test results. Majority-class baseline on this particular held-out split is 44.4% (4/9).

### Why these scores should not be generalized

- Only **30 synthetic URLs** and **9 test URLs** were evaluated. The SVM's perfect result may reflect simple tokens and URL-generation patterns rather than generalizable blocking ability.
- CV-based hyperparameter selection on only 21 training URLs is unstable; repeating different splits can change the outcome materially.
- The classes are perfectly balanced and do not reflect web traffic. Real-world evaluation should report false-positive allowlist mistakes and recall for trackers/ads, across **previously unseen domains and campaigns**.
- An actual browser ad blocker would require an extension/content-blocking engine or DNS/proxy integration, a safe allowlist, performance testing, and privacy safeguards.

## Training-time and inference workflow

```text
CSV (url, label)
  └── Validate, deduplicate, split (train 21 / test 9)
      └── FeatureUnion [character TF-IDF + token TF-IDF + fixed ad tokens]
          ├── Pipeline → Random Forest   ┐
          ├── Pipeline → Logistic Reg.   ├── Tune with train-only CV
          ├── Pipeline → SVM             │
          └── Pipeline → XGBoost         ┘
                  └── Four-model Soft Voting candidate
                      └── Select highest training CV score
                          └── Evaluate candidates on untouched test set
                              └── Save CV-selected Pipeline + metrics
                                  └── Predict ALLOW/BLOCK for new URLs
```

**GitHub project:** [`Ruchika-choudhary8/new`](https://github.com/Ruchika-choudhary8/new) (currently public). Keep any real browsing history, account identifiers, or browsing data out of public commits.
