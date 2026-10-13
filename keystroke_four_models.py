"""Keystroke identification using four scikit-learn Pipelines and soft voting.

Train on consented feature measurements, not the actual typed text.

Usage:
    pip install -r requirements.txt
    python keystroke_four_models.py --csv 'keystroke_features (1).csv'

Model selection relies on training-set cross-validation. The untouched test
partition is used ONCE for reporting, not choosing the best model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
TARGET = 'user_id'


def read_keystroke_csv(path: str | Path):
    """Read the CSV and isolate numeric timing features and user labels."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected '{TARGET}' column; found {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET]).copy()
    features = df.drop(columns=[TARGET]).select_dtypes(include='number')
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.empty or features.isna().all().any():
        raise ValueError('Need at least one numeric feature; no feature can be entirely missing.')
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[TARGET].astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError('Need two or more users to train a classifier.')
    print(f'Dataset: {len(df)} samples | {features.shape[1]} numeric features | '
          f'{len(encoder.classes_)} users')
    print('Users:', dict(df[TARGET].value_counts()))
    print('Feature names:', features.columns.tolist())
    return features, labels, encoder


def make_models(n_classes: int):
    """All learned preprocessing is inside a Pipeline to avoid leakage."""
    return {
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(random_state=SEED, n_jobs=1)),
        ]),
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=SEED)),
        ]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBClassifier(
                objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                tree_method='hist', random_state=SEED, n_jobs=1,
            )),
        ]),
    }


SEARCH_SPACES = {
    'Random Forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 4, 8],
        'model__min_samples_leaf': [1, 3, 5],
    },
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__class_weight': [None, 'balanced'],
    },
    'SVM': {
        'model__C': [0.1, 1.0, 10.0],
        'model__gamma': ['scale', 0.01, 0.1],
        'model__class_weight': [None, 'balanced'],
    },
    'XGBoost': {
        'model__n_estimators': [100, 180, 260],
        'model__max_depth': [2, 3, 5],
        'model__learning_rate': [0.03, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
}


def train_and_compare(X_train, y_train, *, n_iter=4):
    """Tune the four Pipelines using training-set CV only; compare voting."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_pipelines = {}
    cv_rows = []
    for name, pipe in make_models(n_classes=len(np.unique(y_train))).items():
        print(f'\nTuning {name} with 5-fold CV ...', flush=True)
        search = RandomizedSearchCV(
            pipe, SEARCH_SPACES[name], n_iter=n_iter,
            scoring='accuracy', cv=folds, n_jobs=1,
            random_state=SEED, refit=True, error_score='raise',
        )
        search.fit(X_train, y_train)
        best_pipelines[name] = search.best_estimator_
        cv_rows.append({
            'model': name, 'cv_accuracy': float(search.best_score_),
            'parameters': str(search.best_params_),
        })
        print(f"{name}: CV accuracy = {search.best_score_:.1%}")
        print('  parameters:', search.best_params_)

    voters = [('rf', clone(best_pipelines['Random Forest'])),
              ('lr', clone(best_pipelines['Logistic Regression'])),
              ('svm', clone(best_pipelines['SVM'])),
              ('xgb', clone(best_pipelines['XGBoost']))]
    ensemble = VotingClassifier(estimators=voters, voting='soft', n_jobs=1)
    print('\nEvaluating four-model Soft Voting ensemble with 5-fold CV...', flush=True)
    vote_scores = cross_val_score(ensemble, X_train, y_train, cv=folds, scoring='accuracy', n_jobs=1)
    cv_rows.append({'model': 'Soft Voting (all four)',
                    'cv_accuracy': float(vote_scores.mean()),
                    'parameters': 'Four tuned Pipelines; equal soft-vote weights'})
    print('Soft Voting: CV accuracy =', f'{vote_scores.mean():.1%}')
    cv_table = pd.DataFrame(cv_rows).sort_values('cv_accuracy', ascending=False).reset_index(drop=True)
    print('\nCV COMPARISON (training data only):')
    print(cv_table[['model', 'cv_accuracy']].to_string(index=False))
    winner = cv_table.iloc[0]['model']
    print(f'\nModel chosen by training CV: {winner}')
    candidates = {**best_pipelines, 'Soft Voting (all four)': ensemble}
    return candidates, cv_table, winner


def evaluate(X_train, X_test, y_train, y_test, encoder, candidates, cv_table,
             winner, output_dir='outputs', feature_names=None):
    """Evaluate each candidate only after selection; save the CV-selected model."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f'\nMajority-class holdout baseline: {baseline_acc:.1%}')

    test_rows = []
    selected = None
    selected_pred = None
    for name, candidate in candidates.items():
        fitted = clone(candidate).fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        test_rows.append({
            'model': name, 'test_accuracy': accuracy_score(y_test, predictions),
            'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
            'macro_f1': f1_score(y_test, predictions, average='macro', zero_division=0),
        })
        if name == winner:
            selected = fitted
            selected_pred = predictions

    test_table = pd.DataFrame(test_rows).sort_values('test_accuracy', ascending=False)
    print('\nHOLDOUT PERFORMANCE (for reporting, NOT model selection):')
    print(test_table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    selected_acc = accuracy_score(y_test, selected_pred)
    print(f'\nCV-selected model: {winner} | held-out test accuracy: {selected_acc:.1%}')
    print('\nPer-user report:')
    print(classification_report(
        y_test, selected_pred, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0,
    ))
    ConfusionMatrixDisplay.from_predictions(
        encoder.inverse_transform(y_test), encoder.inverse_transform(selected_pred),
        labels=encoder.classes_, xticks_rotation=45, cmap='Blues', colorbar=False,
    )
    plt.title(f'CV-selected model: {winner}')
    plt.tight_layout()
    plt.savefig(out / 'keystroke_confusion_matrix.png', dpi=160)
    plt.close()

    cv_table.to_csv(out / 'keystroke_cv_scores.csv', index=False)
    test_table.to_csv(out / 'keystroke_holdout_scores.csv', index=False)
    model_file = out / 'keystroke_best_pipeline.joblib'
    joblib.dump({'pipeline': selected, 'label_encoder': encoder,
                 'features': list(feature_names if feature_names is not None else X_train.columns)}, model_file)
    result = {'selected_by_training_cv': winner, 'heldout_test_accuracy': float(selected_acc),
              'majority_holdout_baseline': float(baseline_acc),
              'training_rows': len(X_train), 'test_rows': len(X_test),
              'user_classes': encoder.classes_.tolist()}
    (out / 'keystroke_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nSample held-out prediction:',
          encoder.inverse_transform(selected.predict(X_test.iloc[[0]]))[0])
    print(f'Saved metrics, confusion matrix and CV-selected Pipeline to {out}/')
    print('Important: this small dataset may NOT support reliable user identification.')
    return result


def main():
    parser = argparse.ArgumentParser(description='Four-model keystroke Pipelines + voting')
    parser.add_argument('--csv', default='keystroke_features (1).csv',
                        help='Path to timing features CSV; not committed to GitHub')
    parser.add_argument('--n-iter', type=int, default=4,
                        help='Number of random hyperparameter settings per model (default 4)')
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    X, y, encoder = read_keystroke_csv(args.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED,
    )
    candidates, cv_table, winner = train_and_compare(X_train, y_train, n_iter=args.n_iter)
    evaluate(X_train, X_test, y_train, y_test, encoder, candidates, cv_table, winner,
             output_dir=args.output_dir, feature_names=X.columns)


if __name__ == '__main__':
    main()
