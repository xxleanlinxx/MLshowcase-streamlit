# train_secom.py
# Pre-compute all artifacts for the SECOM dataset showcase
# Run from the assets/ directory: python train_secom.py

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, recall_score,
                             f1_score, roc_curve, confusion_matrix)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

print("=" * 60)
print("SECOM Dataset Training Pipeline")
print("=" * 60)

# ============================================================
# 1. Load Raw Data
# ============================================================
print("\n[1/14] Loading raw data...")
data = pd.read_csv('secom/secom.data', sep=r'\s+', header=None)
labels_raw = pd.read_csv('secom/secom_labels.data', sep=r'\s+', header=None, usecols=[0])

data.columns = [f'Sensor_{i}' for i in range(data.shape[1])]
labels_raw.columns = ['label']

# -1 → Pass (0), 1 → Fail (1)
y_all = labels_raw['label'].map({-1: 0, 1: 1})
print(f"  Raw data shape: {data.shape}")
print(f"  Class distribution: Pass={int(sum(y_all==0))}, Fail={int(sum(y_all==1))}")

# ============================================================
# 2. Missing Value Analysis (for EDA)
# ============================================================
print("\n[2/14] Analyzing missing values...")
missing_pct = data.isnull().mean()
missing_count = data.isnull().sum()
missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

# ============================================================
# 3. Feature Engineering - Drop High Missing (>50%)
# ============================================================
print("\n[3/14] Dropping features with >50% missing...")
high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
data_step1 = data.drop(columns=high_missing_cols)
print(f"  Dropped {len(high_missing_cols)} high-missing features. Remaining: {data_step1.shape[1]}")

# ============================================================
# 4. Feature Engineering - Drop Zero Variance
# ============================================================
print("\n[4/14] Dropping zero-variance features...")
zero_var_cols = data_step1.columns[data_step1.nunique() <= 1].tolist()
data_step2 = data_step1.drop(columns=zero_var_cols)
print(f"  Dropped {len(zero_var_cols)} zero-variance features. Remaining: {data_step2.shape[1]}")

# ============================================================
# 5. Impute Missing with Median
# ============================================================
print("\n[5/14] Imputing remaining missing values with median...")
data_step3 = data_step2.fillna(data_step2.median())
print(f"  Remaining missing after imputation: {data_step3.isnull().sum().sum()}")

# ============================================================
# 6. Remove Highly Correlated Features (>0.95)
# ============================================================
print("\n[6/14] Removing highly correlated features (>0.95)...")
corr_matrix = data_step3.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
data_cleaned = data_step3.drop(columns=high_corr_cols)
print(f"  Dropped {len(high_corr_cols)} highly correlated features. Final: {data_cleaned.shape[1]}")

# ============================================================
# 7. PCA Analysis (for EDA visualization)
# ============================================================
print("\n[7/14] Computing PCA for visualization...")
scaler_pca = StandardScaler()
data_scaled_pca = scaler_pca.fit_transform(data_cleaned)

n_pca_components = min(50, data_cleaned.shape[1])
pca_full = PCA(n_components=n_pca_components)
pca_full.fit(data_scaled_pca)
pca_explained_variance = pca_full.explained_variance_ratio_
pca_cumulative_variance = np.cumsum(pca_explained_variance)

pca_2d = PCA(n_components=2)
pca_2d_coords = pca_2d.fit_transform(data_scaled_pca)
pca_2d_df = pd.DataFrame(pca_2d_coords, columns=['PC1', 'PC2'])
pca_2d_df['label'] = y_all.values
pca_2d_df['status'] = pca_2d_df['label'].map({0: 'Pass', 1: 'Fail'})
print(f"  PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")

# ============================================================
# 8. Prepare EDA DataFrame
# ============================================================
print("\n[8/14] Preparing EDA data...")
eda_df = data_cleaned.copy()
eda_df['label'] = y_all.values
eda_df['status'] = y_all.map({0: 'Pass', 1: 'Fail'}).values

processing_info = {
    'original_shape': data.shape,
    'n_high_missing_dropped': len(high_missing_cols),
    'high_missing_cols': high_missing_cols,
    'n_zero_var_dropped': len(zero_var_cols),
    'zero_var_cols': zero_var_cols,
    'n_high_corr_dropped': len(high_corr_cols),
    'high_corr_cols': high_corr_cols,
    'cleaned_shape': data_cleaned.shape,
    'class_pass': int(sum(y_all == 0)),
    'class_fail': int(sum(y_all == 1)),
    'imbalance_ratio': f"{sum(y_all == 0) / sum(y_all == 1):.1f}:1",
}

eda_data = {
    'df': eda_df,
    'missing_summary': missing_summary,
    'processing_info': processing_info,
    'pca_2d_df': pca_2d_df,
    'pca_explained_variance': pca_explained_variance,
    'pca_cumulative_variance': pca_cumulative_variance,
    'feature_names': data_cleaned.columns.tolist(),
}

# ============================================================
# 9. Train/Test Split + SMOTE
# ============================================================
print("\n[9/14] Splitting data and applying SMOTE...")
X = data_cleaned.copy()
y = y_all.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPORTANT: Scale BEFORE SMOTE to avoid distorting test set distribution
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns
)

# Apply SMOTE on already-scaled data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"  Train before SMOTE: Pass={int(sum(y_train==0))}, Fail={int(sum(y_train==1))}")
print(f"  Train after  SMOTE: Pass={int(sum(y_train_smote==0))}, Fail={int(sum(y_train_smote==1))}")
print(f"  Test: Pass={int(sum(y_test==0))}, Fail={int(sum(y_test==1))}")

# ============================================================
# 10. Logistic Regression Baseline
# ============================================================
print("\n[10/14] Training Logistic Regression baseline...")
lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
lr.fit(X_train_smote, y_train_smote)

lr_y_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_y_prob)
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_y_prob)

# Tune threshold using Youden's J statistic
j_scores_lr = lr_tpr - lr_fpr
optimal_idx_lr = np.argmax(j_scores_lr)
lr_optimal_threshold = lr_thresholds[optimal_idx_lr]
print(f"  Optimal threshold (Youden's J): {lr_optimal_threshold:.3f}")

lr_y_pred = (lr_y_prob >= lr_optimal_threshold).astype(int)
lr_report = classification_report(y_test, lr_y_pred, output_dict=True)
lr_recall = recall_score(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)
lr_cm = confusion_matrix(y_test, lr_y_pred)
print(f"  LR  => Recall: {lr_recall:.3f}, F1: {lr_f1:.3f}, AUC: {lr_auc:.3f}")

# ============================================================
# 11. XGBoost with GridSearchCV
# ============================================================
print("\n[11/14] Training XGBoost with GridSearchCV (scoring=recall)...")
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [1, 3],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    xgb, param_grid, cv=cv, scoring='recall', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_smote, y_train_smote)

xgb_best = grid_search.best_estimator_
xgb_y_prob = xgb_best.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_y_prob)
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_y_prob)

# Tune threshold using Youden's J statistic (maximize TPR - FPR)
j_scores = xgb_tpr - xgb_fpr
optimal_idx = np.argmax(j_scores)
xgb_optimal_threshold = xgb_thresholds[optimal_idx]
print(f"  Best Params: {grid_search.best_params_}")
print(f"  Optimal threshold (Youden's J): {xgb_optimal_threshold:.3f}")

xgb_y_pred = (xgb_y_prob >= xgb_optimal_threshold).astype(int)
xgb_report = classification_report(y_test, xgb_y_pred, output_dict=True)
xgb_recall = recall_score(y_test, xgb_y_pred)
xgb_f1 = f1_score(y_test, xgb_y_pred)
xgb_cm = confusion_matrix(y_test, xgb_y_pred)
print(f"  XGB => Recall: {xgb_recall:.3f}, F1: {xgb_f1:.3f}, AUC: {xgb_auc:.3f}")

# ============================================================
# 12. SHAP Values
# ============================================================
print("\n[12/14] Computing SHAP values for XGBoost...")
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test_scaled)
print(f"  SHAP values shape: {shap_values.shape}")

# ============================================================
# 13. Feature Importance
# ============================================================
print("\n[13/14] Computing feature importance...")
xgb_feat_imp = pd.Series(
    xgb_best.feature_importances_, index=X.columns
).sort_values(ascending=False)

shap_feat_imp = pd.Series(
    np.mean(np.abs(shap_values), axis=0), index=X.columns
).sort_values(ascending=False)

# ============================================================
# 14. Null Importance Assessment
# ============================================================
print("\n[14/14] Computing null importance (20 permutations)...")
n_null_runs = 20
null_importances = np.zeros((n_null_runs, X.shape[1]))

for i in range(n_null_runs):
    y_shuffled = y_train_smote.sample(frac=1, random_state=i).reset_index(drop=True)
    null_model = XGBClassifier(
        random_state=42, eval_metric='logloss',
        **grid_search.best_params_
    )
    null_model.fit(X_train_smote, y_shuffled)
    null_importances[i] = null_model.feature_importances_
    if (i + 1) % 5 == 0:
        print(f"  Null run {i+1}/{n_null_runs} done")

null_imp_df = pd.DataFrame(null_importances, columns=X.columns)

actual_imp = xgb_best.feature_importances_
null_95th = np.percentile(null_importances, 95, axis=0)
is_significant = actual_imp > null_95th

null_importance_result = pd.DataFrame({
    'feature': X.columns,
    'actual_importance': actual_imp,
    'null_95th_percentile': null_95th,
    'is_significant': is_significant
}).sort_values('actual_importance', ascending=False)

print(f"  Significant features: {int(sum(is_significant))}/{len(is_significant)}")

# ============================================================
# SAVE ALL ARTIFACTS
# ============================================================
print("\n" + "=" * 60)
print("Saving artifacts...")

# 1. EDA data
with open('secom_eda_data.pkl', 'wb') as f:
    pickle.dump(eda_data, f)

# 2. Model results
model_results = {
    'feature_names': X.columns.tolist(),
    'X_test': X_test_scaled,
    'y_test': y_test.reset_index(drop=True),
    # LR
    'lr_y_pred': lr_y_pred,
    'lr_y_prob': lr_y_prob,
    'lr_report': lr_report,
    'lr_recall': lr_recall,
    'lr_f1': lr_f1,
    'lr_auc': lr_auc,
    'lr_fpr': lr_fpr,
    'lr_tpr': lr_tpr,
    'lr_cm': lr_cm,
    # XGBoost
    'xgb_y_pred': xgb_y_pred,
    'xgb_y_prob': xgb_y_prob,
    'xgb_report': xgb_report,
    'xgb_recall': xgb_recall,
    'xgb_f1': xgb_f1,
    'xgb_auc': xgb_auc,
    'xgb_fpr': xgb_fpr,
    'xgb_tpr': xgb_tpr,
    'xgb_cm': xgb_cm,
    'best_params': grid_search.best_params_,
    'xgb_optimal_threshold': xgb_optimal_threshold,
    'lr_optimal_threshold': lr_optimal_threshold,
    # Feature Importance
    'xgb_feature_importance': xgb_feat_imp,
    'shap_feature_importance': shap_feat_imp,
    # Null Importance
    'null_importance_result': null_importance_result,
    'null_imp_df': null_imp_df,
    # SMOTE info
    'train_before_smote': {'Pass': int(sum(y_train==0)), 'Fail': int(sum(y_train==1))},
    'train_after_smote': {'Pass': int(sum(y_train_smote==0)), 'Fail': int(sum(y_train_smote==1))},
    'test_dist': {'Pass': int(sum(y_test==0)), 'Fail': int(sum(y_test==1))},
}
with open('secom_model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# 3. XGBoost model
with open('secom_xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_best, f)

# 4. Explainer
with open('secom_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# 5. SHAP values
np.save('secom_shap_values.npy', shap_values)

# 6. Best params
with open('secom_best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)

print("\n✅ All artifacts saved successfully!")
print("Files created in assets/:")
print("  - secom_eda_data.pkl")
print("  - secom_model_results.pkl")
print("  - secom_xgb_model.pkl")
print("  - secom_explainer.pkl")
print("  - secom_shap_values.npy")
print("  - secom_best_params.pkl")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Original features: {data.shape[1]}")
print(f"After cleaning:    {data_cleaned.shape[1]}")
print(f"Samples:           {data.shape[0]}")
print(f"Imbalance ratio:   {processing_info['imbalance_ratio']}")
print(f"\nLogistic Regression (Baseline):")
print(f"  Recall={lr_recall:.3f}  F1={lr_f1:.3f}  AUC={lr_auc:.3f}")
print(f"\nXGBoost (Best Model):")
print(f"  Recall={xgb_recall:.3f}  F1={xgb_f1:.3f}  AUC={xgb_auc:.3f}")
print(f"  Recall improvement: +{(xgb_recall - lr_recall)*100:.1f}%")
print(f"\nNull Importance: {int(sum(is_significant))}/{len(is_significant)} features are significant")
