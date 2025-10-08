#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


# In[77]:


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

#Load Dataset
dataset_name = "WA_Fn-UseC_-Telco-Customer-Churn"

df = pd.read_csv(dataset_name+".csv")  # replace with your file


# In[78]:


print(df.isna().sum().sum())   # total NaNs in whole DataFrame


# In[79]:


# Preprocessing 
#df = df.dropna()
df = df.drop(columns=["customerID"])

#need for telco dataset
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# In[80]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[81]:


# Handle missing 
df["TotalCharges"].fillna(0, inplace=True)


# In[82]:


categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# In[83]:


print(df.isna().sum().sum())   #


# In[84]:


df.head()


# In[86]:


n_total = len(df['Churn'])
n_churn = df['Churn'].sum()
n_nonchurn = n_total - n_churn

print(f"Training class distribution: Total {n_total}")
print(f" - Churned:     {n_churn} ({n_churn / n_total:.2%})")
print(f" - Not churned: {n_nonchurn} ({n_nonchurn / n_total:.2%})")


# In[88]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# split holdout ===
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Hold out 20% 
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[89]:


from sklearn.preprocessing import MinMaxScaler

# minmax scaler
scaler_tc = MinMaxScaler(feature_range=(0, 1))
scaler_tc.fit(X_train_full[["TotalCharges"]])

# transform both train and holdout
X_train_full["TotalCharges"] = scaler_tc.transform(X_train_full[["TotalCharges"]])
X_holdout["TotalCharges"]    = scaler_tc.transform(X_holdout[["TotalCharges"]])


# In[90]:


from sklearn.preprocessing import MinMaxScaler

# Fit scaler on training-only 
scaler_tc = MinMaxScaler(feature_range=(0, 1))
scaler_tc.fit(X_train_full[["MonthlyCharges"]])

# Transform both train and holdout
X_train_full["MonthlyCharges"] = scaler_tc.transform(X_train_full[["MonthlyCharges"]])
X_holdout["MonthlyCharges"]    = scaler_tc.transform(X_holdout[["MonthlyCharges"]])


# In[91]:


# Save with labels (for evaluation after inference)
holdout = X_holdout.copy()
holdout["Churn"] = y_holdout
holdout.to_csv("export/"+dataset_name+"_holdout_with_labels4.csv", index=False)

#save train
train = X_train_full.copy()
train["Churn"] = y_train_full  
train.to_csv(f"export/{dataset_name}_train_with_labels4.csv", index=False)


# In[92]:


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[93]:


folds = list(cv.split(X_train_full, y_train_full))

# Save folds
joblib.dump(folds, "export/"+dataset_name+"_"+"cv_folds4.pkl")

