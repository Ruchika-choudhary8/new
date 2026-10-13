# ML Security Labs — Google Colab, Pipelines & Ensembles

**Open the notebook in [Google Colab](https://colab.research.google.com/) using File → Upload notebook.**

**Two end-to-end educational machine-learning experiments:**

1. **ML-based ad URL classification** — predict `BLOCK` (`label=1`) or `ALLOW` (`label=0`) using lexical URL features and a Pipeline/soft-voting ensemble.
2. **Keystroke dynamics user identification** — predict `user_id` from precomputed timing features using ML Pipelines, multiple classifiers, soft voting, and stacking.

The Colab notebook walks through **reading the supplied CSV files, cleaning, train/test splitting, feature engineering, model comparison with cross-validation, ensembling, evaluation, example predictions, and exporting results**. No API keys are required to run either experiment.

> The ad classifier predicts URLs; it does **not** install a browser ad blocker. The keystroke model is a teaching example, **not** reliable biometric authentication.

## New: Keystroke four-model Pipelines (Random Forest, LR, SVM, XGBoost)

Start with the dedicated [Google Colab notebook](Keystroke_RF_LR_SVM_XGBoost_Colab.ipynb) or [standalone Python script](keystroke_four_models.py), and see [README_keystroke_four_models.md](README_keystroke_four_models.md) for a complete step-by-step explanation. **The notebook starts with CSV upload and `pd.read_csv()`**, builds four `sklearn.pipeline.Pipeline` objects, performs random hyperparameter search on 5-fold training CV, compares an equal-weight four-model soft-voting ensemble, selects by CV, then evaluates on a fixed held-out test set.

| Model | Mean CV accuracy | Held-out test accuracy |
|---|---:|---:|
| Random Forest | 21.6% | 18.4% |
| Logistic Regression | 22.7% | 17.6% |
| SVM | **23.7%** | **23.2%** |
| XGBoost | 19.5% | 19.2% |
| Soft Voting (four models) | 21.1% | 16.8% |

The training-CV-selected SVM has the same accuracy as the 23.2% majority-class baseline and is **not a reliable authenticator**. Do not use the test set to select a different model. Do not commit the uploaded keystroke CSV to GitHub.

## Project layout

```text
new/
├── ML_Security_Pipelines_Ensembles_Colab.ipynb  # Main notebook: run top to bottom
├── ad_blocker_ensemble.py                       # Standalone ad classifier
├── keystroke_ensemble.py                        # Earlier standalone timing classifier
├── Keystroke_RF_LR_SVM_XGBoost_Colab.ipynb       # New Colab: four ML Pipelines + Voting
├── keystroke_four_models.py                     # New script: RF, LR, SVM, XGBoost
├── README_keystroke_four_models.md              # Four-model-specific instructions
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── ad_urls.csv                              # 30 labeled URLs
│   └── keystroke_features.csv                   # 500 samples; in ZIP, ignored by Git
└── results/                                    # Created when notebook runs
```

## 1. Open and run in Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/) and select **File → Upload notebook**.
2. Choose `ML_Security_Pipelines_Ensembles_Colab.ipynb`.
3. Run the notebook from the first cell. At **Step 2** use **Choose Files** to upload:
   - `ad_urls.csv` **or** `ad_urls (1).csv`
   - `keystroke_features.csv` **or** `keystroke_features (1).csv`
4. The next cell calls `pd.read_csv()` on both uploaded files and displays the first few rows.
5. Continue **Runtime → Run all** (or run each cell in order). Inspect CV scores, confusion matrices, holdout accuracy, and demo predictions.
6. The `results/` directory contains `ad_cv_results.csv`, `keystroke_cv_results.csv`, and `metrics.json` after the notebook finishes. Colab runtime files are temporary; download what you want to retain.

**Important:** The dataset files inside the ZIP are not automatically mounted when opening *only* the notebook in Colab. The notebook deliberately includes an upload step. If you run locally from the extracted folder, it finds the CSVs already under `data/` and skips uploading.

## 2. Run as Python scripts (optional)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run both models from the project directory:

```bash
python ad_blocker_ensemble.py
python keystroke_ensemble.py
```

Try predicting a particular ad URL:

```bash
python ad_blocker_ensemble.py --url "https://ads.example.net/banner.js"
```

To read differently named datasets:

```bash
python ad_blocker_ensemble.py --csv "ad_urls (1).csv"
python keystroke_ensemble.py --csv "keystroke_features (1).csv"
```

## 3. How the ML Pipelines work

| Experiment | Pipeline | Compared models | Ensembling |
|---|---|---|---|
| Ad URLs | Character/word TF-IDF + URL lexical signal extraction | Logistic Regression, ComplementNB, linear SVM | Soft-voting (`LR + NB + SVM`) |
| Keystrokes | Median imputation; scaling for linear/kernel/probabilistic models | Logistic Regression, RBF SVM, GaussianNB, Random Forest, Extra Trees | Soft voting and stacking (`LR + SVM + Extra Trees`) |

Features and preprocessing are **inside cross-validated Pipelines** to avoid fitting transformers using test/validation data. A fixed **stratified train/test split** reserves an independent final test partition. **CV is run only on training data** for choosing the best ensemble.

## 4. Reproducible observations from the supplied files

The main notebook and standalone programs were run against the bundled CSV files, using the indicated random seeds:

| Experiment | Train/test split | Selected ensemble | Held-out test accuracy |
|---|---|---|---|
| Ad URLs | 21 / 9 | Soft Voting | **100.0% (9/9)** |
| Keystroke | 375 / 125 | Stacking | **22.4% (28/125)** |

**Interpret cautiously.** The ad sample contains just **30 simplified URLs** and was explored when creating the lexical signals, so `9/9` is **not** an unbiased guarantee of generalization to real websites. The keystroke model performs **below the 23.2% majority-class baseline** on the held-out split. In the notebook, Logistic Regression is the top single candidate by average training CV, but it obtains **14.4% on the holdout**—an example of why test results must not be used to select models. These keystroke features have little predictive separation between users.

For an actual keystroke-authentication study, collect independent, consented typing sessions, use a **session-separated** evaluation split, and report **FAR**, **FRR**, and **EER** along with accuracy.

## 5. Add the code to GitHub

The folder is ready for a normal Git repository. You can upload it through the GitHub website, or use these commands in a terminal **inside the extracted folder**.

**The project is already published in the private repository [`Ruchika-choudhary8/new`](https://github.com/Ruchika-choudhary8/new)**. To make future changes, clone that repository and commit your edited files. No need to create another repository.

```bash
git clone https://github.com/Ruchika-choudhary8/new.git
cd new
# Copy changes or new notebooks to this directory
git add README.md requirements.txt .gitignore *.ipynb *.py README_keystroke_four_models.md
git commit -m "Update four-model keystroke pipelines"
git push origin main
```

Sign in through your normal GitHub credential manager or SSH setup. **Never include passwords or personal access tokens in notebooks or repositories.**

To commit changes later:

```bash
git add .
git commit -m "Update ML notebook experiments"
git push
```

From Colab, you may also use **File → Save a copy in GitHub** after authorizing GitHub, or download the edited `.ipynb` and commit it locally.

### Privacy for uploaded keystroke samples

The downloadable teaching ZIP includes the original uploaded keystroke CSV so both experiments can run locally. **The `.gitignore` deliberately excludes `data/keystroke_features*.csv`** from normal Git commits, because typing-timing patterns can be behavioral biometric data. It can be committed only if you have explicit authority to share the dataset. Colab's upload step allows the public code repository to be run without publishing it.

## 6. Troubleshooting

- **`FileNotFoundError`:** Upload both CSVs in Colab Step 2, or place them in `data/` locally. Both original `(1)` filenames are supported.
- **`ModuleNotFoundError`:** Run the install cell in Colab, or `python -m pip install -r requirements.txt` locally.
- **Poor keystroke results:** This is expected with the supplied dataset; high-performing classifiers cannot invent discriminating information absent from the training features.
- **`git push` authentication error:** Configure SSH or Git Credential Manager. Do not copy a personal access token into an executable notebook cell.

## Lab safety

Use the URL predictions only in authorized experiments; add a fallback allowlist and false-positive review before implementing real blocking. Handle timing features as sensitive behavioral data and obtain consent for collection, processing, and any public release.
