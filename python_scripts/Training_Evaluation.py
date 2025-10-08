#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    precision_recall_fscore_support
)

# ===== Imports & setup =====
import os, time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, Binarizer

from sklearn.utils.class_weight import compute_sample_weight
from inspect import signature
dataset_name = "kkbox"


# GCS I/O
import gcsfs
fs = gcsfs.GCSFileSystem()

 import gcsfs
 fs = gcsfs.GCSFileSystem()

 gcs_path = "gs://ml-dataset-a/kkbox/kkbox_cv_folds4.pkl"
 with fs.open(gcs_path, 'rb') as f:
     folds = joblib.load(f)

 gcs_holdout_path = "gs://ml-dataset-a/kkbox/kkbox_holdout_with_labels4.csv"
 with fs.open(gcs_holdout_path, 'rb') as f:
     holdout = pd.read_csv(f)
 gcs_train_path = "gs://ml-dataset-a/kkbox/kkbox_train_with_labels4.csv"
 with fs.open(gcs_train_path, 'rb') as f:
     train = pd.read_csv(f)


#folds   = joblib.load(f"export/{dataset_name}_cv_folds4.pkl")  # list of (train_idx, val_idx)
#train   = pd.read_csv(f"export/{dataset_name}_train_with_labels4.csv")
#holdout = pd.read_csv(f"export/{dataset_name}_holdout_with_labels4.csv")

X_train_full = train.drop(columns=["Churn"])
y_train_full = train["Churn"].astype(int)
X_holdout    = holdout.drop(columns=["Churn"])
y_holdout    = holdout["Churn"].astype(int)

print("NaNs in TRAIN:\n", X_train_full.isna().sum()[X_train_full.isna().sum()>0])
print("NaNs in HOLDOUT:\n", X_holdout.isna().sum()[X_holdout.isna().sum()>0])

# Output folder
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
folder_path = f"export/{date_time}"
os.makedirs(folder_path, exist_ok=True)
print("Folder:", folder_path)

# sample_weight (reweighting)
sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train_full)

def fit_param_key(est):
    """Return the proper fit param key for sample_weight if supported."""
    if isinstance(est, Pipeline):
        last_name, last_step = est.steps[-1]
        return f"{last_name}__sample_weight" if "sample_weight" in signature(last_step.fit).parameters else None
    return "sample_weight" if "sample_weight" in signature(est.fit).parameters else None

def sliced_fit_kwargs(est, train_idx):
    key = fit_param_key(est)
    if key is None:
        return {}
    return {key: sample_weight_train[train_idx]}

# Models 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def replace_sentinel_neg1_with_nan(X):
    if isinstance(X, pd.DataFrame):
        return X.replace(-1, np.nan)
    X = np.asarray(X).copy()
    X[X == -1] = np.nan
    return X

models = [
    ("Vote_XGB_LGBM-88-12", VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ("lgbm", LGBMClassifier(random_state=42))
        ],
        voting="soft",
        weights=[0.88, 0.12]
    )),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("LightGBM", LGBMClassifier(random_state=42)),
    ("CatBoost", CatBoostClassifier(verbose=0, random_state=42)),

    ("LinearSVM", LinearSVC(dual=False, random_state=42)),  # decision_function only
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("NaiveBayes", GaussianNB_pipe),
    ("MLP", MLPClassifier(max_iter=300, random_state=42)),   # may ignore sample_weight
    ("Dummy", DummyClassifier(strategy="most_frequent")),

    ("Vote_XGB_LGBM", VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ("lgbm", LGBMClassifier(random_state=42))
        ],
        voting="soft"
    )),

    ("Vote_AllGBM", VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ("lgbm", LGBMClassifier(random_state=42)),
            ("cat", CatBoostClassifier(verbose=0, random_state=42))
        ],
        voting="soft"
    )),

    # Paper-based voting ensemble (Telecom 2021)
    ("Vote_Mix", VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("nb", GaussianNB())
        ],
        voting="soft"
    )),
 # Paper-based stacking ensemble (Bank Churn ICEMGD 2025
    ("StackGBM", StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(random_state=42)),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ("lgbm", LGBMClassifier(random_state=42)),
            ("cat", CatBoostClassifier(verbose=0, random_state=42))
        ],
        final_estimator=XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        passthrough=True
    )),

    ("BernoulliNB", BernoulliNB_pipe),
    ("MultinomialNB", MultinomialNB_pipe),
    ("Bagging", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        random_state=42
    )),
    ("SGD", SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)),
    ("SGD_NB_Ensemble", VotingClassifier(
        estimators=[
           ("SGD", SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)),
           ("nb", GaussianNB())
        ],
        voting="soft"
    )),


]

# Cross-validation 
results_cross = []
import psutil
from codecarbon import EmissionsTracker

def iter_splits(folds, X, y):
    if hasattr(folds, "split"):
        for tr, va in folds.split(X, y):
            yield tr, va
    else:
        for tr, va in folds:
            yield tr, va

for model_name, model_cv in models:
    est_proto = clone(model_cv)

    os.environ["CODECARBON_CLOUD_PROVIDER"] = "GCP"
    os.environ["CODECARBON_CLOUD_REGION"] = "us-central1"
    tracker = EmissionsTracker(
        project_name="MyBenchmark",
        output_dir=folder_path,
        output_file=f"{dataset_name}_{model_name}_cross_emissions.csv",
        save_to_file=True
    )

    try:
        tracker.start_task("Training cross")

        start_time = time.perf_counter()
        cpu_before = psutil.cpu_percent()
        ram_before = psutil.virtual_memory().used

        accs, precs, recs, f1s, aucs, loglosses = [], [], [], [], [], []

        for tr_idx, va_idx in iter_splits(folds, X_train_full, y_train_full):
            X_tr, y_tr = X_train_full.iloc[tr_idx], y_train_full.iloc[tr_idx]
            X_va, y_va = X_train_full.iloc[va_idx], y_train_full.iloc[va_idx]

            est = clone(est_proto)
            est.fit(X_tr, y_tr, **sliced_fit_kwargs(est, tr_idx))

            y_pred = est.predict(X_va)

            # Scores for AUC and LOG LOSS
            y_score = None
            y_prob = None
            if hasattr(est, "predict_proba"):
                y_prob = est.predict_proba(X_va)[:, 1]
                y_score = y_prob
            elif hasattr(est, "decision_function"):
                y_score = est.decision_function(X_va)

            accs.append(accuracy_score(y_va, y_pred))
            precs.append(precision_score(y_va, y_pred, zero_division=0))
            recs.append(recall_score(y_va, y_pred, zero_division=0))
            f1s.append(f1_score(y_va, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_va, y_score) if y_score is not None else np.nan)
            # LOG LOSS only when real probabilities are available
            loglosses.append(log_loss(y_va, y_prob, labels=[0, 1]) if y_prob is not None else np.nan)

        end_time = time.perf_counter()
        time_cv = end_time - start_time

        cpu_after = psutil.cpu_percent()
        ram_after = psutil.virtual_memory().used
        cross_training_emissions = tracker.stop_task("Training cross")

        mean_auc = float(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else np.nan
        mean_logloss = float(np.nanmean(loglosses)) if np.any(~np.isnan(loglosses)) else np.nan

        print(f"\n=== {model_name} | Cross-Validation Performance ===")
        print(f"Mean accuracy:  {np.mean(accs):.4f}")
        print(f"Mean precision: {np.mean(precs):.4f}")
        print(f"Mean recall:    {np.mean(recs):.4f}")
        print(f"Mean F1:        {np.mean(f1s):.4f}")
        if not np.isnan(mean_auc):
            print(f"Mean ROC AUC:   {mean_auc:.4f}")
        if not np.isnan(mean_logloss):
            print(f"Mean Log Loss:  {mean_logloss:.4f}")
        print(f"Total training time CV: {time_cv:.3f} sec")
        print(f"Cross emissions: {cross_training_emissions.emissions:.6f} kg CO₂")

        results_cross.append({
            "Model": model_name,
            "Mean Accuracy": np.mean(accs),
            "Mean Precision": np.mean(precs),
            "Mean Recall": np.mean(recs),
            "Mean F1-score": np.mean(f1s),
            "Mean AUC (CV)": mean_auc,
            "Mean LogLoss (CV)": mean_logloss,
            "Total Training Time CV (s)": time_cv,
            "CPU Delta (%)": (cpu_after - cpu_before),
            "RAM Delta (GB)": (ram_after - ram_before) / (1024**3),
            "Total Emissions CV (kgCO₂)": cross_training_emissions.emissions,
            "Emissions Rate (kg CO₂ / kWh)": cross_training_emissions.emissions_rate,
            "CPU energy (kWh)": cross_training_emissions.cpu_energy,
            "RAM energy (kWh)": cross_training_emissions.ram_energy,
            "Energy consumed (kWh)": cross_training_emissions.energy_consumed,
            "Cloud Provider": cross_training_emissions.cloud_provider,
            "Cloud Region": cross_training_emissions.cloud_region,
            "Country": cross_training_emissions.country_name
        })

        pd.DataFrame(results_cross).to_csv(
            f"{folder_path}/Results_10-fold_cross_{model_name}_{dataset_name}_.csv",
            index=False
        )

    finally:
        _ = tracker.stop()

pd.DataFrame(results_cross).to_csv(f"{folder_path}/Results_10-fold_cross_{dataset_name}_.csv", index=False)

# Full fit + HOLDOUT 
results_full = []

for model_name, model_full in models:
    try:
        os.environ["CODECARBON_CLOUD_PROVIDER"] = "GCP"
        os.environ["CODECARBON_CLOUD_REGION"] = "us-central1"

        tracker_full = EmissionsTracker(
            project_name="MyBenchmark",
            output_dir=folder_path,
            output_file=f"{dataset_name}_{model_name}_full_emissions.csv",
            save_to_file=True
        )

        est_full = clone(model_full)

        # Train (with weights if supported)
        tracker_full.start_task("Training full")
        t0 = time.perf_counter()
        cpu_before_full = psutil.cpu_percent()
        ram_before_full = psutil.virtual_memory().used

        # full train slice for weights
        full_idx = np.arange(len(y_train_full))
        est_full.fit(X_train_full, y_train_full, **sliced_fit_kwargs(est_full, full_idx))

        t1 = time.perf_counter()
        cpu_after_full = psutil.cpu_percent()
        ram_after_full = psutil.virtual_memory().used
        emissions_train = tracker_full.stop_task("Training full")

        train_time = t1 - t0
        print(f"\n=== {model_name} | Full Training ===")
        print(f"Training time full: {train_time:.3f} sec")
        print(f"CO₂ emitted training full: {emissions_train.emissions * 1000:.3f} g")

        # Predict holdout
        tracker_full.start_task("full model prediction holdout")
        t0p = time.perf_counter()
        cpu_before_holdout = psutil.cpu_percent()
        ram_before_holdout = psutil.virtual_memory().used

        y_pred = est_full.predict(X_holdout)
        y_score = None
        y_prob  = None
        if hasattr(est_full, "predict_proba"):
            y_prob = est_full.predict_proba(X_holdout)[:, 1]
            y_score = y_prob
        elif hasattr(est_full, "decision_function"):
            y_score = est_full.decision_function(X_holdout)

        t1p = time.perf_counter()
        cpu_after_holdout = psutil.cpu_percent()
        ram_after_holdout = psutil.virtual_memory().used
        emissions_holdout = tracker_full.stop_task("full model prediction holdout")

        pred_time = t1p - t0p
        n_rows = len(X_holdout)
        mean_latency = pred_time / n_rows if n_rows else np.nan  # s/sample
        throughput = n_rows / pred_time if pred_time > 0 else np.nan  # samples/sec

        # Metrics
        acc = accuracy_score(y_holdout, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_holdout, y_pred, average='binary')
        auc_holdout = roc_auc_score(y_holdout, y_score) if y_score is not None else np.nan
        logloss_holdout = log_loss(y_holdout, y_prob, labels=[0, 1]) if y_prob is not None else np.nan

        print("\n=== Final Evaluation on Holdout Set ===")
        print(classification_report(y_holdout, y_pred))
        if not np.isnan(auc_holdout):     print("AUC on Holdout:", f"{auc_holdout:.4f}")
        if not np.isnan(logloss_holdout): print("Log Loss on Holdout:", f"{logloss_holdout:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
        print(f"Prediction wall time: {pred_time:.4f} sec")
        print(f"Mean latency: {mean_latency:.6f} s/sample")
        print(f"Throughput: {throughput:.2f} samples/sec")

        # Persist model
        model_path = f"{folder_path}/{model_name}_full.pkl"
        joblib.dump(est_full, model_path)
        model_size_mb = os.path.getsize(model_path) / 1024**2

        results_full.append({
            "Model": model_name,
            "Accuracy hold out": acc,
            "AUC": auc_holdout,
            "Precision": prec,
            "Recall": rec,
            "F-measure": f1,
            "LogLoss Holdout": logloss_holdout,
            "Training Time (s)": train_time,
            "Prediction Time (s)": pred_time,
            "Mean Latency (s/sample)": mean_latency,
            "Throughput (samples/sec)": throughput,
            "Model Size (MB)": model_size_mb,
            "Total Emissions Prediction hold out (kgCO2)": emissions_holdout.emissions,
            "Total Emissions Training (kgCO2)": emissions_train.emissions,
            "RAM Usage During Training (GB)": (ram_after_full - ram_before_full) / (1024**3),
            "CPU Delta Training (%)": (cpu_after_full - cpu_before_full),
            "RAM Usage During Prediction (GB)": (ram_after_holdout - ram_before_holdout) / (1024**3),
            "CPU Delta Prediction (%)": (cpu_after_holdout - cpu_before_holdout),
            "Emissions Rate Prediction (kg CO2 / kWh)": emissions_holdout.emissions_rate,
            "CPU energy Prediction (kWh)": emissions_holdout.cpu_energy,
            "RAM energy Prediction (kWh)": emissions_holdout.ram_energy,
            "Energy consumed Prediction (kWh)": emissions_holdout.energy_consumed,
            "Emissions Rate Training (kg CO2 / kWh)": emissions_train.emissions_rate,
            "CPU energy Training (kWh)": emissions_train.cpu_energy,
            "RAM energy Training (kWh)": emissions_train.ram_energy,
            "Energy consumed Training (kWh)": emissions_train.energy_consumed
        })

        pd.DataFrame(results_full).to_csv(
            f"{folder_path}/Results_full_training_{model_name}_{dataset_name}_.csv",
            index=False
        )

    finally:
        _ = tracker_full.stop()

#  saves RESults
results_df_cross = pd.DataFrame(results_cross)
results_df_cross.to_csv(f"{folder_path}/Results_10-fold_cross_{dataset_name}_.csv", index=False)

results_df_full = pd.DataFrame(results_full)
print(results_df_full.to_string(index=False))
results_df_full.to_csv(f"{folder_path}/Results_full_training_{dataset_name}_.csv", index=False)





