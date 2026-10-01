"""
Interpretable Breast Cancer Prediction — full analysis pipeline (CRISP-DM).
Rigorous evaluation: repeated stratified cross-validation with 95% CIs, plus a
leak-free held-out test (feature set and decision threshold fixed on TRAIN only).
Run:  python src/pipeline.py     (expects original csv at data/raw/data.csv)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
        StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix, accuracy_score)

DATA = "data/raw/data.csv"; SEED = 42
PARSIMONIOUS = ["radius_worst", "texture_worst", "concave points_worst",
                "concavity_worst", "smoothness_worst", "symmetry_worst"]

def load():
    df = pd.read_csv(DATA)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]

def ci(a):
    a = np.asarray(a); m = a.mean(); half = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return m, m - half, m + half

def data_quality(df):
    print("Shape:", df.shape, "| missing:", df.isna().sum().sum(),
          "| dup rows:", df.duplicated().sum(), "| dup ids:", df["id"].duplicated().sum())
    print("Class counts:\n", df["diagnosis"].value_counts().to_string())

def repeated_cv(df, y):
    feats = [c for c in df.columns if c not in ("id", "diagnosis")]
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    models = {
        "LogReg (30)": (Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]), feats),
        "Tree d3":     (DecisionTreeClassifier(max_depth=3, random_state=SEED), feats),
        "RandomForest":(RandomForestClassifier(n_estimators=300, random_state=SEED), feats),
        "GradBoost":   (GradientBoostingClassifier(random_state=SEED), feats),
        "LogReg (6)":  (Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]), PARSIMONIOUS),
    }
    print("\nRepeated 5x10-fold CV — F1 mean [95% CI]:")
    for n, (mdl, cols) in models.items():
        m, lo, hi = ci(cross_val_score(mdl, df[cols].values, y, cv=rcv, scoring="f1"))
        print(f"  {n:14s} {m:.3f} [{lo:.3f}, {hi:.3f}]")

def leak_free_test(df, y):
    """Threshold selected on TRAIN out-of-fold preds; evaluated once on held-out TEST."""
    X = df[PARSIMONIOUS].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))])
    oof = cross_val_predict(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    tstar = max(np.round(np.arange(.05, .96, .01), 2), key=lambda t: f1_score(ytr, (oof >= t).astype(int)))
    print(f"\nThreshold selected on TRAIN out-of-fold preds: t* = {tstar:.2f}")
    pipe.fit(Xtr, ytr); ptest = pipe.predict_proba(Xte)[:, 1]
    for lab, t in [("default 0.50", 0.50), (f"selected {tstar:.2f}", tstar)]:
        pr = (ptest >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
        print(f"  TEST @ {lab}: F1={f1_score(yte,pr):.3f} recall={recall_score(yte,pr):.3f} "
              f"prec={precision_score(yte,pr):.3f} missed={fn} false_alarms={fp}")
    print("\nOdds ratios (per 1 SD, fitted on the training split — consistent with report Fig 7):")
    for f, c in sorted(zip(PARSIMONIOUS, np.exp(pipe.named_steps["m"].coef_[0])), key=lambda x: -x[1]):
        print(f"  {f:22s} {c:6.2f}x")
    print("  (Note: exact OR magnitudes are sample-sensitive; direction and ranking are stable.)")

if __name__ == "__main__":
    df = load(); y = (df["diagnosis"] == "M").astype(int).values
    print("== DATA QUALITY =="); data_quality(df)
    print("\n== PRIMARY: REPEATED CROSS-VALIDATION =="); repeated_cv(df, y)
    print("\n== CONFIRMATION: LEAK-FREE HELD-OUT TEST =="); leak_free_test(df, y)
