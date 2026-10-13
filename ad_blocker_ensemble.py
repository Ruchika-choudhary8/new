"""ML ad URL classifier: feature Pipeline + LR/NB/SVM soft-voting ensemble.
Install: pip install pandas numpy scikit-learn
Run: python ad_blocker_ensemble.py --url 'https://ads.example.net/banner.js'
Note: Predicts decisions; it does NOT intercept browser/network requests.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC


class URLSignals(BaseEstimator, TransformerMixin):
    """Handcrafted, general ad/tracker indicators; learned weights come from ML."""

    patterns = (
        r"(?:^|[^a-z])(?:ads?|adserver|advertisements?|doubleclick|banner)(?:[^a-z]|$)",
        r"track|analytics|pixel|beacon|retarget|impression|collect",
        r"sponsor|promot|affiliate|marketing|campaign|click",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [[int(bool(re.search(pattern, str(url).lower()))) for pattern in self.patterns]
             for url in X], dtype=float
        )


def pipeline(classifier):
    features = FeatureUnion(
        [
            ("characters", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
            ("words", TfidfVectorizer(token_pattern=r"[a-zA-Z]+", ngram_range=(1, 2))),
            ("url_signals", URLSignals()),
        ],
        transformer_weights={"url_signals": 3.0},
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def main():
    parser = argparse.ArgumentParser(description="Ad URL classifier with an ML ensemble")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "ad_urls.csv"))
    parser.add_argument("--url", action="append", help="URL to classify; repeat for multiple URLs")
    args = parser.parse_args()

    data = pd.read_csv(args.csv).dropna(subset=["url", "label"])
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("CSV 'label' must contain 0=ALLOW and 1=BLOCK.")

    train_urls, test_urls, train_y, test_y = train_test_split(
        data["url"], data["label"], test_size=0.30,
        stratify=data["label"], random_state=42,
    )

    models = {
        "Logistic Regression": LogisticRegression(C=1, max_iter=1000),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Linear SVM": SVC(kernel="linear", C=1, probability=True, random_state=42),
    }
    models["Soft Voting Ensemble"] = VotingClassifier(
        estimators=[("lr", clone(models["Logistic Regression"])),
                    ("nb", clone(models["Complement Naive Bayes"])),
                    ("svm", clone(models["Linear SVM"]))],
        voting="soft",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    print("AD BLOCKING | 5-fold cross-validation on training URLs only")
    for name, classifier in models.items():
        scores = cross_val_score(pipeline(classifier), train_urls, train_y,
                                 scoring="accuracy", cv=cv)
        print(f"{name:24} CV accuracy: {scores.mean():.1%} +/- {scores.std():.1%}")

    # Evaluate on rows excluded from fitting. Dataset is too small for reliable claims.
    ensemble = pipeline(models["Soft Voting Ensemble"])
    ensemble.fit(train_urls, train_y)
    predictions = ensemble.predict(test_urls)
    print(f"\nHOLDOUT TEST ACCURACY: {accuracy_score(test_y, predictions):.1%}"
          f" ({len(test_urls)} URLs)")
    print("Confusion matrix (rows=actual, cols=predicted; 0=ALLOW, 1=BLOCK):")
    print(confusion_matrix(test_y, predictions, labels=[0, 1]))
    print(classification_report(test_y, predictions, digits=2, zero_division=0))

    # Retrain AFTER testing to provide sample predictions with all labeled examples.
    deploy_demo = clone(ensemble).fit(data["url"], data["label"])
    examples = args.url or [
        "https://ads.example.net/banner/new.js",
        "https://example.edu/student-portal",
        "https://doubleclick.net/pagead/ads?slot=1",
    ]
    print("DEMO DECISIONS (NOT actual browser blocking):")
    for url in examples:
        score = deploy_demo.predict_proba([url])[0, list(deploy_demo.classes_).index(1)]
        print(f"{'BLOCK' if score >= 0.5 else 'ALLOW':5} | score={score:.2f} | {url}")
    print("Caution: 30 labeled URLs cannot establish production ad-blocking accuracy.")


if __name__ == "__main__":
    main()