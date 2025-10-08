#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Load Data 
df_train = pd.read_csv("train_v2.csv")      
df_members = pd.read_csv("members_v3.csv")
df_transactions = pd.read_csv("transactions_v2.csv")
df_logs = pd.read_csv("user_logs_v2.csv")

# Preprocess Dates
to_dt = lambda s: pd.to_datetime(s, format="%Y%m%d", errors="coerce")
df_transactions["transaction_date"] = to_dt(df_transactions["transaction_date"])
df_transactions["membership_expire_date"] = to_dt(df_transactions["membership_expire_date"])
df_members["registration_init_time"] = to_dt(df_members["registration_init_time"])
df_logs["date"] = to_dt(df_logs["date"])

# cutoff
CUTOFF_DATE = pd.Timestamp("2017-02-28")
df_transactions = df_transactions[df_transactions["transaction_date"] <= CUTOFF_DATE].copy()
df_logs = df_logs[df_logs["date"] <= CUTOFF_DATE].copy()

df_transactions = (
    df_transactions.sort_values("transaction_date")
                   .drop_duplicates("msno", keep="last")
)

# merge Main tables
df = df_train.rename(columns={"is_churn": "Churn"}).merge(df_members, on="msno", how="left")
df = df.merge(df_transactions, on="msno", how="left")

#feature engineering
idx_date = df["transaction_date"].fillna(CUTOFF_DATE)

df["days_since_registration"] = (idx_date - df["registration_init_time"]).dt.days
df["subscription_length"] = (df["membership_expire_date"] - df["transaction_date"]).dt.days
df["registration_month"] = df["registration_init_time"].dt.month
df["registration_dow"] = df["registration_init_time"].dt.weekday
df["transaction_month"] = df["transaction_date"].dt.month
df["transaction_dow"] = df["transaction_date"].dt.weekday

# aggregate user_logs
user_agg = df_logs.groupby("msno").agg({
    "num_25": ["sum", "mean"],
    "num_50": ["sum", "mean"],
    "num_75": ["sum", "mean"],
    "num_985": ["sum", "mean"],
    "num_100": ["sum", "mean"],
    "num_unq": ["sum", "mean"],
    "total_secs": ["sum", "mean"],
    "date": ["min", "max", "nunique"]
})
user_agg.columns = ["_".join(col).strip() for col in user_agg.columns]
user_agg = user_agg.reset_index()


user_agg["log_span_days"] = (user_agg["date_max"] - user_agg["date_min"]).dt.days
user_agg["avg_secs_per_song"] = user_agg["total_secs_sum"] / (user_agg["num_unq_sum"] + 1e-5)
user_agg["date_max"] = pd.to_datetime(user_agg["date_max"])

# add logs to main df 
df = df.merge(user_agg.drop(columns=["date_min"]), on="msno", how="left")

# days since Last Log 
df["days_since_last_log_missing"] = df["date_max"].isna().astype("int8")
df["days_since_last_log"] = (idx_date - df["date_max"]).dt.days
df = df.drop(columns=["date_max"], errors="ignore")

# handle missing Values 

for c in ["city", "registered_via", "payment_method_id"]:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("Unknown")

df[["payment_plan_days", "plan_list_price", "actual_amount_paid",
    "is_auto_renew", "is_cancel", "subscription_length"]] = df[[
    "payment_plan_days", "plan_list_price", "actual_amount_paid",
    "is_auto_renew", "is_cancel", "subscription_length"]].fillna(0)

df["transaction_month"] = df["transaction_month"].fillna(0)   # 0 = unknown
df["transaction_dow"]   = df["transaction_dow"].fillna(0)

df["days_since_registration"] = df["days_since_registration"].fillna(0)
df["registration_month"]      = df["registration_month"].fillna(0)
df["registration_dow"]        = df["registration_dow"].fillna(0)


log_cols = [col for col in df.columns if any(metric in col for metric in
    ["num_", "total_secs", "date_nunique", "log_span_days", "avg_secs_per_song"])]
df[log_cols] = df[log_cols].fillna(0)


df["days_since_last_log"] = df["days_since_last_log"].fillna(999)


df = df.drop(columns=[
    "msno", "bd", "registration_init_time", "membership_expire_date", "transaction_date"
], errors="ignore")


y = df["Churn"].astype(int)
X = df.drop(columns=["Churn"]).copy()


X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


obj_cols = X_train_full.select_dtypes(include="object").columns.tolist()
encoders = {}

for col in obj_cols:
    le = LabelEncoder()
    train_vals = X_train_full[col].astype(str).fillna("NA_SPECIAL")
    le.fit(train_vals)
    encoders[col] = le

    # Transform train
    X_train_full[col] = le.transform(train_vals)

   
    hold_vals = X_holdout[col].astype(str).fillna("NA_SPECIAL")
    is_known = np.isin(hold_vals, le.classes_)
    if not np.all(is_known):
        le.classes_ = np.concatenate([le.classes_, np.array(["__UNK__"])])
        hold_vals = np.where(is_known, hold_vals, "__UNK__")
    X_holdout[col] = le.transform(hold_vals)


from sklearn.preprocessing import MinMaxScaler


cat_int = [
    "city", "gender", "registered_via", "payment_method_id",
    "registration_month", "registration_dow",
    "transaction_month", "transaction_dow"
]
cat_int = [c for c in cat_int if c in X_train_full.columns]


bin_cols = [c for c in ["is_auto_renew", "is_cancel"] if c in X_train_full.columns]


cont_cols = [
    c for c in X_train_full.select_dtypes(include=[np.number]).columns
    if c not in cat_int + bin_cols
]

# Impute continuous with mean (fit on TRAIN only)
num_imputer = SimpleImputer(strategy="mean")
X_train_full[cont_cols] = num_imputer.fit_transform(X_train_full[cont_cols])
X_holdout[cont_cols]    = num_imputer.transform(X_holdout[cont_cols])


scaler = MinMaxScaler()
X_train_full[cont_cols] = scaler.fit_transform(X_train_full[cont_cols])
X_holdout[cont_cols]    = scaler.transform(X_holdout[cont_cols])


os.makedirs("export", exist_ok=True)
joblib.dump(encoders, "export/label_encoders4.pkl")
joblib.dump(num_imputer, "export/num_imputer4.pkl")
joblib.dump(scaler, "export/scaler4.pkl")



n_total = len(y)
n_churn = int(y.sum())
n_nonchurn = n_total - n_churn
print(f"Training class distribution: Total {n_total}")
print(f" - Churned:     {n_churn} ({n_churn / n_total:.2%})")
print(f" - Not churned: {n_nonchurn} ({n_nonchurn / n_total:.2%})")

# Quick sanity: no NaNs remain
nan_train = pd.isna(X_train_full).sum().sum()
nan_hold  = pd.isna(X_holdout).sum().sum()
print(f"NaNs remaining — train: {nan_train}, holdout: {nan_hold}")


# In[7]:


print("Scaled (cont_cols):", cont_cols)


# In[3]:


dataset_name = "kkbox"


# In[4]:


# Save with labels (for evaluation after inference)
holdout = X_holdout.copy()
holdout["Churn"] = y_holdout
holdout.to_csv("export/"+dataset_name+"_holdout_with_labels4.csv", index=False)

#save train
train = X_train_full.copy()
train["Churn"] = y_train_full  
train.to_csv(f"export/{dataset_name}_train_with_labels4.csv", index=False)


# In[5]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[6]:


import joblib
folds = list(cv.split(X_train_full, y_train_full))

# Save folds
joblib.dump(folds, "export/"+dataset_name+"_"+"cv_folds4.pkl")

