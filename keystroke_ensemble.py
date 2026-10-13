"""ML keystroke identification: Pipelines, model comparison, voting/stacking.
Install: pip install pandas numpy scikit-learn
Run: python keystroke_ensemble.py
Uses only previously collected timing features (not actual typed content).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def scaled(model):
    # Imputation and scaling are fit INSIDE each CV training fold.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def trees(model):
    return make_pipeline(SimpleImputer(strategy="median"), model)


def build_models():
    lr = scaled(LogisticRegression(C=1, max_iter=1000))
    svm = scaled(SVC(C=1, kernel="rbf", probability=True, random_state=42))
    nb = scaled(GaussianNB())
    forest = trees(RandomForestClassifier(n_estimators=180, min_samples_leaf=3,
                                          random_state=42, n_jobs=1))
    extra = trees(ExtraTreesClassifier(n_estimators=180, min_samples_leaf=3,
                                      random_state=42, n_jobs=1))

    def members():
        return [("logistic", clone(lr)), ("svm", clone(svm)),
                ("extra_trees", clone(extra))]

    return {
        "Logistic Regression": lr,
        "RBF SVM": svm,
        "Gaussian Naive Bayes": nb,
        "Random Forest": forest,
        "Extra Trees": extra,
        "Soft Voting Ensemble": VotingClassifier(members(), voting="soft"),
        "Stacking Ensemble": StackingClassifier(
            members(), final_estimator=LogisticRegression(max_iter=1000),
            cv=3, stack_method="predict_proba"),
    }


def main():
    parser = argparse.ArgumentParser(description="Keystroke timing user identification")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "keystroke_features.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    X, y = data.drop(columns=["user_id"]), data["user_id"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scores = {}
    print("KEYSTROKE ANALYSIS | 5-fold cross-validation on training samples only")
    for name, model in models.items():
        fold_scores = cross_val_score(model, X_train, y_train,
                                      cv=cv, scoring="accuracy")
        scores[name] = fold_scores.mean()
        print(f"{name:24} CV accuracy: {fold_scores.mean():.1%} +/- {fold_scores.std():.1%}")

    # Only compare ensemble methods via TRAINING CV; never select on test accuracy.
    options = ["Soft Voting Ensemble", "Stacking Ensemble"]
    best = max(options, key=lambda name: scores[name])
    final_model = clone(models[best]).fit(X_train, y_train)
    predicted = final_model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    print(f"\nBest ensemble by CV: {best}")
    print(f"HOLDOUT TEST ACCURACY: {accuracy_score(y_test, predicted):.1%}"
          f" ({len(X_test)} samples)")
    print(f"Majority-class baseline: {accuracy_score(y_test, baseline.predict(X_test)):.1%}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("Labels:", list(final_model.classes_))
    print(confusion_matrix(y_test, predicted, labels=final_model.classes_))
    print("\nPer-user metrics:")
    print(classification_report(y_test, predicted, digits=2, zero_division=0))
    one = X_test.iloc[[0]]
    print(f"Example: actual={y_test.iloc[0]} predicted={final_model.predict(one)[0]}")
    print("Caution: Near-chance accuracy is NOT adequate for user authentication.")


if __name__ == "__main__":
    main()