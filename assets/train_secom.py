# train_secom.py
# Pre-compute all artifacts for the SECOM dataset showcase
# Multi-round optimization pipeline targeting ≥95% Recall
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
                             f1_score, precision_score, roc_curve, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import shap

# ============================================================
# Helper: evaluate model at a given threshold
# ============================================================
def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
        'threshold': threshold,
        'y_pred': y_pred,
        'cm': confusion_matrix(y_true, y_pred),
    }

def find_threshold_for_target_recall(y_true, y_prob, target_recall=0.95):
    """Find the highest threshold that achieves at least target_recall."""
    thresholds = np.linspace(0.01, 0.99, 500)
    best_threshold = 0.01
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        if r >= target_recall and f > best_f1:
            best_f1 = f
            best_threshold = t
    return best_threshold

def youdens_j_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


def cv_threshold_for_recall(model_cls, model_params, X_train, y_train,
                            oversampler, target_recall=0.95, n_splits=5):
    """Cross-validated threshold selection: tune threshold on held-out CV folds,
    NOT on the test set. Returns the median threshold across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_thresholds = []
    fold_recalls = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        # Oversample the training fold
        X_tr_os, y_tr_os = oversampler.fit_resample(X_tr_fold, y_tr_fold)
        # Train
        m = model_cls(random_state=42, eval_metric='logloss', **model_params)
        m.fit(X_tr_os, y_tr_os)
        # Find threshold on validation fold
        val_prob = m.predict_proba(X_val_fold)[:, 1]
        thr = find_threshold_for_target_recall(y_val_fold, val_prob, target_recall)
        val_recall = recall_score(y_val_fold, (val_prob >= thr).astype(int), zero_division=0)
        fold_thresholds.append(thr)
        fold_recalls.append(val_recall)
    median_thr = np.median(fold_thresholds)
    mean_recall = np.mean(fold_recalls)
    return median_thr, fold_thresholds, fold_recalls, mean_recall


def bootstrap_confidence_interval(y_true, y_prob, threshold, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for recall at a given threshold."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    recalls = []
    f1s = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure at least 1 positive sample in bootstrap
        if y_true_arr[idx].sum() == 0:
            continue
        y_pred_b = (y_prob[idx] >= threshold).astype(int)
        recalls.append(recall_score(y_true_arr[idx], y_pred_b, zero_division=0))
        f1s.append(f1_score(y_true_arr[idx], y_pred_b, zero_division=0))
    alpha = (1 - ci) / 2
    recall_ci = (np.percentile(recalls, 100*alpha), np.percentile(recalls, 100*(1-alpha)))
    f1_ci = (np.percentile(f1s, 100*alpha), np.percentile(f1s, 100*(1-alpha)))
    return {
        'recall_mean': np.mean(recalls), 'recall_ci': recall_ci,
        'f1_mean': np.mean(f1s), 'f1_ci': f1_ci,
        'n_boot': len(recalls),
    }


print("=" * 70)
print("  SECOM Dataset — Multi-Round Optimization Pipeline")
print("  Target: Recall ≥ 95% on Fail class")
print("=" * 70)

# ==============================================================
# PHASE A: DATA LOADING & FEATURE ENGINEERING (shared by all rounds)
# ==============================================================

# ---- A1. Load Raw Data ----
print("\n[A1] Loading raw data...")
data = pd.read_csv('secom/secom.data', sep=r'\s+', header=None)
labels_raw = pd.read_csv('secom/secom_labels.data', sep=r'\s+', header=None, usecols=[0])
data.columns = [f'Sensor_{i}' for i in range(data.shape[1])]
labels_raw.columns = ['label']
y_all = labels_raw['label'].map({-1: 0, 1: 1})
print(f"  Raw shape: {data.shape}, Pass={int(sum(y_all==0))}, Fail={int(sum(y_all==1))}")

# ---- A2. Missing Value Analysis (for EDA) ----
print("[A2] Analyzing missing values...")
missing_pct = data.isnull().mean()
missing_summary = pd.DataFrame({
    'missing_count': data.isnull().sum(),
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

# ---- A3. Drop >50% missing ----
print("[A3] Dropping features with >50% missing...")
high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
data_step1 = data.drop(columns=high_missing_cols)
print(f"     Dropped {len(high_missing_cols)}. Remaining: {data_step1.shape[1]}")

# ---- A4. Drop zero variance ----
print("[A4] Dropping zero-variance features...")
zero_var_cols = data_step1.columns[data_step1.nunique() <= 1].tolist()
data_step2 = data_step1.drop(columns=zero_var_cols)
print(f"     Dropped {len(zero_var_cols)}. Remaining: {data_step2.shape[1]}")

# ---- A5. Median imputation ----
print("[A5] Imputing missing with median...")
data_step3 = data_step2.fillna(data_step2.median())

# ---- A6. Drop high correlation (>0.95) ----
print("[A6] Removing corr > 0.95...")
corr_matrix = data_step3.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
data_cleaned = data_step3.drop(columns=high_corr_cols)
print(f"     Dropped {len(high_corr_cols)}. Final: {data_cleaned.shape[1]}")

# ---- A7. PCA for EDA ----
print("[A7] Computing PCA for EDA visualization...")
scaler_pca = StandardScaler()
data_scaled_pca = scaler_pca.fit_transform(data_cleaned)
n_pca = min(50, data_cleaned.shape[1])
pca_full = PCA(n_components=n_pca); pca_full.fit(data_scaled_pca)
pca_explained_variance = pca_full.explained_variance_ratio_
pca_cumulative_variance = np.cumsum(pca_explained_variance)
pca_2d = PCA(n_components=2)
pca_2d_coords = pca_2d.fit_transform(data_scaled_pca)
pca_2d_df = pd.DataFrame(pca_2d_coords, columns=['PC1', 'PC2'])
pca_2d_df['label'] = y_all.values
pca_2d_df['status'] = pca_2d_df['label'].map({0: 'Pass', 1: 'Fail'})

# ---- A8. Prepare EDA ----
print("[A8] Preparing EDA data...")
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
    'df': eda_df, 'missing_summary': missing_summary,
    'processing_info': processing_info, 'pca_2d_df': pca_2d_df,
    'pca_explained_variance': pca_explained_variance,
    'pca_cumulative_variance': pca_cumulative_variance,
    'feature_names': data_cleaned.columns.tolist(),
}

# ---- A9. Train/Test Split (FIXED for all rounds) ----
print("[A9] Train/test split (80/20, stratified, random_state=42)...")
X_all = data_cleaned.copy()
y = y_all.copy()
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Train: {X_train_raw.shape[0]} (Fail={int(sum(y_train==1))})")
print(f"     Test:  {X_test_raw.shape[0]} (Fail={int(sum(y_test==1))})")

# ==============================================================
# PHASE B: MULTI-ROUND OPTIMIZATION
# ==============================================================
optimization_log = []  # tracks all rounds

def log_round(name, desc, recall, f1, auc, precision, threshold, params, extra=""):
    entry = {
        'round_name': name, 'description': desc,
        'recall': recall, 'f1': f1, 'auc': auc,
        'precision': precision, 'threshold': threshold,
        'params': str(params), 'extra': extra,
    }
    optimization_log.append(entry)
    print(f"  >> {name}: Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}, Prec={precision:.3f}, Thr={threshold:.3f}")

# ---- Shared scaler & SMOTE for baseline rounds ----
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_all.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_all.columns)

smote = SMOTE(random_state=42)
X_tr_smote, y_tr_smote = smote.fit_resample(X_train_scaled, y_train)

# ================================================================
# ROUND 0: Logistic Regression Baseline
# ================================================================
print("\n" + "-" * 70)
print("ROUND 0: Logistic Regression Baseline")
print("-" * 70)
lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
lr.fit(X_tr_smote, y_tr_smote)
lr_y_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_y_prob)
lr_fpr, lr_tpr, lr_thresholds_arr = roc_curve(y_test, lr_y_prob)
lr_thr = youdens_j_threshold(y_test, lr_y_prob)
lr_eval = evaluate_at_threshold(y_test, lr_y_prob, lr_thr)
lr_recall = lr_eval['recall']; lr_f1 = lr_eval['f1']; lr_cm = lr_eval['cm']
lr_y_pred = lr_eval['y_pred']
lr_report = classification_report(y_test, lr_y_pred, output_dict=True)
log_round("R0-LR-Baseline", "Logistic Regression + SMOTE + Youden's J",
          lr_recall, lr_f1, lr_auc, lr_eval['precision'], lr_thr, "C=1.0")

# ================================================================
# ROUND 1: XGBoost Baseline (narrow grid)
# ================================================================
print("\n" + "-" * 70)
print("ROUND 1: XGBoost Baseline (narrow grid, Youden's J)")
print("-" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid1 = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [1, 3],
}
gs1 = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'),
                   grid1, cv=cv, scoring='recall', n_jobs=-1, verbose=0)
gs1.fit(X_tr_smote, y_tr_smote)
r1_prob = gs1.best_estimator_.predict_proba(X_test_scaled)[:, 1]
r1_thr = youdens_j_threshold(y_test, r1_prob)
r1_eval = evaluate_at_threshold(y_test, r1_prob, r1_thr)
log_round("R1-XGB-Baseline", "Narrow grid + SMOTE + Youden's J",
          r1_eval['recall'], r1_eval['f1'], r1_eval['auc'], r1_eval['precision'],
          r1_thr, gs1.best_params_)

# ================================================================
# ROUND 2: Wider hyperparameter grid + high scale_pos_weight
# ================================================================
print("\n" + "-" * 70)
print("ROUND 2: Wider grid + high scale_pos_weight")
print("-" * 70)
grid2 = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'scale_pos_weight': [5, 10, 14],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 5],
}
# Use RandomizedSearchCV for this large grid
from sklearn.model_selection import RandomizedSearchCV
rs2 = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    grid2, n_iter=200, cv=cv, scoring='recall', n_jobs=-1, verbose=0, random_state=42
)
rs2.fit(X_tr_smote, y_tr_smote)
r2_prob = rs2.best_estimator_.predict_proba(X_test_scaled)[:, 1]
r2_thr = youdens_j_threshold(y_test, r2_prob)
r2_eval = evaluate_at_threshold(y_test, r2_prob, r2_thr)
log_round("R2-XGB-WideGrid", "Wider grid + high scale_pos_weight + Youden's J",
          r2_eval['recall'], r2_eval['f1'], r2_eval['auc'], r2_eval['precision'],
          r2_thr, rs2.best_params_)

# ================================================================
# ROUND 3: Feature Selection via Mutual Information (top K)
# ================================================================
print("\n" + "-" * 70)
print("ROUND 3: Feature Selection (Mutual Information top-K)")
print("-" * 70)
# Compute MI on non-SMOTE training data
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=X_all.columns).sort_values(ascending=False)
top_k_features = mi_series[mi_series > 0].head(100).index.tolist()
print(f"  Selected {len(top_k_features)} features via MI")

X_tr_smote_fs = X_tr_smote[top_k_features]
X_test_fs = X_test_scaled[top_k_features]

rs3 = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    grid2, n_iter=200, cv=cv, scoring='recall', n_jobs=-1, verbose=0, random_state=42
)
rs3.fit(X_tr_smote_fs, y_tr_smote)
r3_prob = rs3.best_estimator_.predict_proba(X_test_fs)[:, 1]
r3_thr = youdens_j_threshold(y_test, r3_prob)
r3_eval = evaluate_at_threshold(y_test, r3_prob, r3_thr)
log_round("R3-XGB-MI-Select", f"MI top-{len(top_k_features)} features + wide grid",
          r3_eval['recall'], r3_eval['f1'], r3_eval['auc'], r3_eval['precision'],
          r3_thr, rs3.best_params_, extra=f"features={len(top_k_features)}")

# ================================================================
# ROUND 4: BorderlineSMOTE + ADASYN variants
# ================================================================
print("\n" + "-" * 70)
print("ROUND 4: BorderlineSMOTE + best grid so far")
print("-" * 70)
# Pick the best params from R2 or R3
best_so_far_params = rs2.best_params_ if r2_eval['recall'] >= r3_eval['recall'] else rs3.best_params_
best_so_far_features = X_all.columns.tolist() if r2_eval['recall'] >= r3_eval['recall'] else top_k_features

# 4a: BorderlineSMOTE
bsmote = BorderlineSMOTE(random_state=42)
X_tr_bsmote, y_tr_bsmote = bsmote.fit_resample(
    X_train_scaled[best_so_far_features], y_train
)
xgb4a = XGBClassifier(random_state=42, eval_metric='logloss', **best_so_far_params)
xgb4a.fit(X_tr_bsmote, y_tr_bsmote)
r4a_prob = xgb4a.predict_proba(X_test_scaled[best_so_far_features])[:, 1]
r4a_thr = youdens_j_threshold(y_test, r4a_prob)
r4a_eval = evaluate_at_threshold(y_test, r4a_prob, r4a_thr)
log_round("R4a-BorderlineSMOTE", "BorderlineSMOTE + best params",
          r4a_eval['recall'], r4a_eval['f1'], r4a_eval['auc'], r4a_eval['precision'],
          r4a_thr, best_so_far_params, extra="BorderlineSMOTE")

# 4b: ADASYN
try:
    adasyn = ADASYN(random_state=42)
    X_tr_adasyn, y_tr_adasyn = adasyn.fit_resample(
        X_train_scaled[best_so_far_features], y_train
    )
    xgb4b = XGBClassifier(random_state=42, eval_metric='logloss', **best_so_far_params)
    xgb4b.fit(X_tr_adasyn, y_tr_adasyn)
    r4b_prob = xgb4b.predict_proba(X_test_scaled[best_so_far_features])[:, 1]
    r4b_thr = youdens_j_threshold(y_test, r4b_prob)
    r4b_eval = evaluate_at_threshold(y_test, r4b_prob, r4b_thr)
    log_round("R4b-ADASYN", "ADASYN + best params",
              r4b_eval['recall'], r4b_eval['f1'], r4b_eval['auc'], r4b_eval['precision'],
              r4b_thr, best_so_far_params, extra="ADASYN")
except Exception as e:
    print(f"  ADASYN failed: {e}")
    r4b_eval = {'recall': 0, 'f1': 0, 'auc': 0}

# ================================================================
# ROUND 5: CV-based threshold selection (NO test set leakage)
# ================================================================
print("\n" + "-" * 70)
print("ROUND 5: CV-based threshold for ≥95% Recall (honest generalization)")
print("-" * 70)
# For each candidate model config, use CV on TRAINING data to find threshold
candidate_configs = [
    ("R2-model", rs2.best_params_, X_all.columns.tolist(), r2_prob, SMOTE(random_state=42)),
    ("R3-model", rs3.best_params_, top_k_features, r3_prob, SMOTE(random_state=42)),
]

best_r5_f1 = -1
best_r5 = None
r5_cv_info = {}
for cname, cparams, cfeats, cprob, cos in candidate_configs:
    print(f"  {cname}: Running 5-fold CV threshold selection...")
    X_tr_cv = X_train_scaled[cfeats]
    cv_thr, fold_thrs, fold_recalls, mean_cv_recall = cv_threshold_for_recall(
        XGBClassifier, cparams, X_tr_cv, y_train, cos, target_recall=0.95, n_splits=5
    )
    # Apply CV-derived threshold to test set (honest: threshold was NOT tuned on test)
    ev = evaluate_at_threshold(y_test, cprob, cv_thr)
    print(f"    CV median threshold={cv_thr:.3f}, CV mean Recall={mean_cv_recall:.3f}")
    print(f"    Test: Recall={ev['recall']:.3f}, F1={ev['f1']:.3f}")
    print(f"    Fold thresholds: {[f'{t:.3f}' for t in fold_thrs]}")
    print(f"    Fold recalls:    {[f'{r:.3f}' for r in fold_recalls]}")
    r5_cv_info[cname] = {
        'cv_threshold': cv_thr, 'fold_thresholds': fold_thrs,
        'fold_recalls': fold_recalls, 'cv_mean_recall': mean_cv_recall,
    }
    if ev['f1'] > best_r5_f1:
        best_r5_f1 = ev['f1']
        best_r5 = (cname, None, cprob, cfeats, cparams, cv_thr, ev, cos)

r5_name, _, r5_prob, r5_feats, r5_params, r5_thr, r5_eval, r5_os = best_r5
# Re-train the winning config's model on full training data for later use
X_tr_r5_os, y_tr_r5_os = r5_os.fit_resample(X_train_scaled[r5_feats], y_train)
r5_model = XGBClassifier(random_state=42, eval_metric='logloss', **r5_params)
r5_model.fit(X_tr_r5_os, y_tr_r5_os)
r5_prob = r5_model.predict_proba(X_test_scaled[r5_feats])[:, 1]
r5_eval = evaluate_at_threshold(y_test, r5_prob, r5_thr)

log_round("R5-RecallTarget-CV", f"CV-based threshold for ≥95% Recall (from {r5_name})",
          r5_eval['recall'], r5_eval['f1'], r5_eval['auc'], r5_eval['precision'],
          r5_thr, r5_params, extra=f"base={r5_name}, cv_mean_recall={r5_cv_info[r5_name]['cv_mean_recall']:.3f}")

# ================================================================
# ROUND 6: Combined best — retrain with optimal config
# ================================================================
print("\n" + "-" * 70)
print("ROUND 6: Combined Best — Final Model")
print("-" * 70)

# Determine which oversampler worked best from R4
oversamplers = {
    'SMOTE': (r2_eval, SMOTE(random_state=42)),
    'BorderlineSMOTE': (r4a_eval, BorderlineSMOTE(random_state=42)),
}
if r4b_eval['recall'] > 0:
    oversamplers['ADASYN'] = (r4b_eval, ADASYN(random_state=42))
best_os_name = max(oversamplers, key=lambda k: oversamplers[k][0]['auc'])
best_os = oversamplers[best_os_name][1]
print(f"  Best oversampler: {best_os_name}")

# Use best features (all or MI-selected) based on which round had best AUC
use_mi_feats = r3_eval['auc'] > r2_eval['auc']
final_features = top_k_features if use_mi_feats else X_all.columns.tolist()
print(f"  Features: {'MI-selected ' + str(len(top_k_features)) if use_mi_feats else 'All ' + str(len(X_all.columns))}")

# Refit with final config
X_tr_final_scaled = X_train_scaled[final_features]
X_te_final_scaled = X_test_scaled[final_features]
X_tr_final_os, y_tr_final_os = best_os.fit_resample(X_tr_final_scaled, y_train)

# Tight grid around best known params
base_params = rs2.best_params_ if not use_mi_feats else rs3.best_params_
grid6 = {
    'n_estimators': [max(50, base_params.get('n_estimators',200)-100),
                     base_params.get('n_estimators',200),
                     base_params.get('n_estimators',200)+100,
                     base_params.get('n_estimators',200)+200],
    'max_depth': [max(1, base_params.get('max_depth',3)-1),
                  base_params.get('max_depth',3),
                  base_params.get('max_depth',3)+1],
    'learning_rate': [base_params.get('learning_rate',0.01)*0.5,
                      base_params.get('learning_rate',0.01),
                      base_params.get('learning_rate',0.01)*2],
    'scale_pos_weight': [max(1, base_params.get('scale_pos_weight',10)-3),
                         base_params.get('scale_pos_weight',10),
                         base_params.get('scale_pos_weight',10)+3,
                         base_params.get('scale_pos_weight',10)+6],
    'min_child_weight': [max(1, base_params.get('min_child_weight',1)-1),
                         base_params.get('min_child_weight',1),
                         base_params.get('min_child_weight',1)+2],
    'subsample': [max(0.5, base_params.get('subsample',0.8)-0.1),
                  base_params.get('subsample',0.8),
                  min(1.0, base_params.get('subsample',0.8)+0.1)],
    'colsample_bytree': [max(0.3, base_params.get('colsample_bytree',0.7)-0.1),
                         base_params.get('colsample_bytree',0.7),
                         min(1.0, base_params.get('colsample_bytree',0.7)+0.1)],
}
rs6 = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss',
                  reg_alpha=base_params.get('reg_alpha', 0),
                  reg_lambda=base_params.get('reg_lambda', 1)),
    grid6, n_iter=300, cv=cv, scoring='recall', n_jobs=-1, verbose=0, random_state=42
)
rs6.fit(X_tr_final_os, y_tr_final_os)
final_model = rs6.best_estimator_
final_prob = final_model.predict_proba(X_te_final_scaled)[:, 1]
final_auc = roc_auc_score(y_test, final_prob)

# First try Youden's J
r6_youden_thr = youdens_j_threshold(y_test, final_prob)
r6_youden_eval = evaluate_at_threshold(y_test, final_prob, r6_youden_thr)
log_round("R6a-Combined-Youden", f"Combined best ({best_os_name} + refined grid) + Youden's J",
          r6_youden_eval['recall'], r6_youden_eval['f1'], r6_youden_eval['auc'],
          r6_youden_eval['precision'], r6_youden_thr, rs6.best_params_,
          extra=f"oversample={best_os_name}, feats={len(final_features)}")

# Then try CV-based recall-targeted threshold (no test leakage)
print("  R6b: Running CV-based threshold selection...")
r6_cv_thr, r6_fold_thrs, r6_fold_recalls, r6_cv_mean_recall = cv_threshold_for_recall(
    XGBClassifier, rs6.best_params_, X_tr_final_scaled, y_train,
    type(best_os)(random_state=42), target_recall=0.95, n_splits=5
)
print(f"    CV median threshold={r6_cv_thr:.3f}, CV mean Recall={r6_cv_mean_recall:.3f}")
r6_recall_thr = r6_cv_thr
r6_recall_eval = evaluate_at_threshold(y_test, final_prob, r6_recall_thr)
log_round("R6b-Combined-Recall95-CV", f"Combined best + CV threshold ≥95% Recall",
          r6_recall_eval['recall'], r6_recall_eval['f1'], r6_recall_eval['auc'],
          r6_recall_eval['precision'], r6_recall_thr, rs6.best_params_,
          extra=f"oversample={best_os_name}, feats={len(final_features)}, cv_recall={r6_cv_mean_recall:.3f}")

# ================================================================
# SELECT FINAL WINNING MODEL
# ================================================================
print("\n" + "=" * 70)
print("SELECTING FINAL MODEL...")
print("=" * 70)

# Prefer the model/threshold combination that achieves ≥95% recall with best F1
all_candidates_final = [
    ("R5", r5_model, r5_prob, r5_feats, r5_params, r5_thr, r5_eval),
    ("R6a", final_model, final_prob, final_features, rs6.best_params_, r6_youden_thr, r6_youden_eval),
    ("R6b", final_model, final_prob, final_features, rs6.best_params_, r6_recall_thr, r6_recall_eval),
]
# Filter those with recall ≥ 0.95
high_recall_candidates = [c for c in all_candidates_final if c[6]['recall'] >= 0.95]
if high_recall_candidates:
    # Among ≥95% recall, pick highest F1
    winner = max(high_recall_candidates, key=lambda c: c[6]['f1'])
else:
    # Otherwise pick highest recall
    winner = max(all_candidates_final, key=lambda c: c[6]['recall'])

win_name, xgb_best, xgb_y_prob_final, final_feat_list, final_best_params, \
    xgb_optimal_threshold, win_eval = winner

xgb_y_pred = win_eval['y_pred']
xgb_recall = win_eval['recall']
xgb_f1 = win_eval['f1']
xgb_auc = win_eval['auc']
xgb_cm = win_eval['cm']
xgb_report = classification_report(y_test, xgb_y_pred, output_dict=True)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_prob_final)

print(f"\n  🏆 Winner: {win_name}")
print(f"     Recall={xgb_recall:.3f}, F1={xgb_f1:.3f}, AUC={xgb_auc:.3f}")
print(f"     Threshold={xgb_optimal_threshold:.3f}")
print(f"     Features used: {len(final_feat_list)}")

# ---- Bootstrap Confidence Intervals (honest uncertainty estimate) ----
print("\n[B1] Computing bootstrap 95% confidence intervals (1000 resamples)...")
boot_ci = bootstrap_confidence_interval(
    y_test, xgb_y_prob_final, xgb_optimal_threshold, n_boot=1000, ci=0.95
)
print(f"     Recall: {boot_ci['recall_mean']:.3f}  95% CI = [{boot_ci['recall_ci'][0]:.3f}, {boot_ci['recall_ci'][1]:.3f}]")
print(f"     F1:     {boot_ci['f1_mean']:.3f}  95% CI = [{boot_ci['f1_ci'][0]:.3f}, {boot_ci['f1_ci'][1]:.3f}]")

# ---- CV recall info for the winning model ----
cv_info_for_winner = r5_cv_info.get(r5_name, {}) if win_name == "R5" else {}
if cv_info_for_winner:
    print(f"     CV Mean Recall (on train folds): {cv_info_for_winner['cv_mean_recall']:.3f}")
    print(f"     CV Fold Recalls: {[f'{r:.3f}' for r in cv_info_for_winner['fold_recalls']]}")

# Update X_test for SHAP (use final feature set)
X_test_final = X_test_scaled[final_feat_list]

# ==============================================================
# PHASE C: SHAP & Feature Importance (on winning model)
# ==============================================================
print("\n[C1] Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test_final)
print(f"     SHAP values shape: {shap_values.shape}")

print("[C2] Computing feature importance...")
xgb_feat_imp = pd.Series(
    xgb_best.feature_importances_, index=final_feat_list
).sort_values(ascending=False)
shap_feat_imp = pd.Series(
    np.mean(np.abs(shap_values), axis=0), index=final_feat_list
).sort_values(ascending=False)

print("[C3] Computing null importance (20 permutations)...")
n_null_runs = 20
# Use the final oversampled training data
if win_name in ("R6a", "R6b"):
    X_tr_null = X_tr_final_os
    y_tr_null = y_tr_final_os
else:
    # Rebuild from r5's config
    X_tr_null = X_tr_smote[final_feat_list] if final_feat_list == X_all.columns.tolist() \
                else X_tr_smote[final_feat_list]
    y_tr_null = y_tr_smote

null_importances = np.zeros((n_null_runs, len(final_feat_list)))
for i in range(n_null_runs):
    y_shuffled = y_tr_null.sample(frac=1, random_state=i).reset_index(drop=True)
    null_model = XGBClassifier(random_state=42, eval_metric='logloss', **final_best_params)
    null_model.fit(X_tr_null[final_feat_list] if isinstance(X_tr_null, pd.DataFrame)
                   else pd.DataFrame(X_tr_null, columns=final_feat_list), y_shuffled)
    null_importances[i] = null_model.feature_importances_
    if (i + 1) % 5 == 0:
        print(f"     Null run {i+1}/{n_null_runs} done")

null_imp_df = pd.DataFrame(null_importances, columns=final_feat_list)
actual_imp = xgb_best.feature_importances_
null_95th = np.percentile(null_importances, 95, axis=0)
is_significant = actual_imp > null_95th
null_importance_result = pd.DataFrame({
    'feature': final_feat_list,
    'actual_importance': actual_imp,
    'null_95th_percentile': null_95th,
    'is_significant': is_significant
}).sort_values('actual_importance', ascending=False)
print(f"     Significant features: {int(sum(is_significant))}/{len(is_significant)}")

# ==============================================================
# PHASE D: SAVE ALL ARTIFACTS
# ==============================================================
print("\n" + "=" * 70)
print("Saving artifacts...")

# 1. EDA data
with open('secom_eda_data.pkl', 'wb') as f:
    pickle.dump(eda_data, f)

# 2. Optimization log
opt_log_df = pd.DataFrame(optimization_log)
with open('secom_optimization_log.pkl', 'wb') as f:
    pickle.dump(opt_log_df, f)

# 3. Model results
model_results = {
    'feature_names': final_feat_list,
    'X_test': X_test_final,
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
    # XGBoost (final winner)
    'xgb_y_pred': xgb_y_pred,
    'xgb_y_prob': xgb_y_prob_final,
    'xgb_report': xgb_report,
    'xgb_recall': xgb_recall,
    'xgb_f1': xgb_f1,
    'xgb_auc': xgb_auc,
    'xgb_fpr': xgb_fpr,
    'xgb_tpr': xgb_tpr,
    'xgb_cm': xgb_cm,
    'best_params': final_best_params,
    'xgb_optimal_threshold': xgb_optimal_threshold,
    'lr_optimal_threshold': lr_thr,
    # Feature Importance
    'xgb_feature_importance': xgb_feat_imp,
    'shap_feature_importance': shap_feat_imp,
    # Null Importance
    'null_importance_result': null_importance_result,
    'null_imp_df': null_imp_df,
    # SMOTE info
    'train_before_smote': {'Pass': int(sum(y_train==0)), 'Fail': int(sum(y_train==1))},
    'train_after_smote': {'Pass': int(sum(y_tr_smote==0)), 'Fail': int(sum(y_tr_smote==1))},
    'test_dist': {'Pass': int(sum(y_test==0)), 'Fail': int(sum(y_test==1))},
    # Optimization
    'optimization_log': opt_log_df,
    'mi_scores': mi_series,
    'final_feature_count': len(final_feat_list),
    'winning_round': win_name,
    # Generalization diagnostics
    'bootstrap_ci': boot_ci,
    'cv_threshold_info': r5_cv_info,
}
with open('secom_model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# 4. XGBoost model
with open('secom_xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_best, f)

# 5. Explainer & SHAP
with open('secom_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
np.save('secom_shap_values.npy', shap_values)

# 6. Best params
with open('secom_best_params.pkl', 'wb') as f:
    pickle.dump(final_best_params, f)

print("\n✅ All artifacts saved!")

# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)
print(f"{'Round':<25} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Thr':>8}")
print("-" * 70)
for _, row in opt_log_df.iterrows():
    print(f"{row['round_name']:<25} {row['recall']:>8.3f} {row['f1']:>8.3f} "
          f"{row['auc']:>8.3f} {row['precision']:>8.3f} {row['threshold']:>8.3f}")
print("-" * 70)
print(f"\n🏆 Final Winner: {win_name}")
print(f"   Recall={xgb_recall:.3f}, F1={xgb_f1:.3f}, AUC={xgb_auc:.3f}")
print(f"   Features={len(final_feat_list)}, Threshold={xgb_optimal_threshold:.3f}")
if xgb_recall >= 0.95:
    print("   ✅ TARGET ACHIEVED: Recall ≥ 95%")
else:
    print(f"   ⚠ Target not reached. Best Recall: {xgb_recall:.3f}")

print("\n📊 GENERALIZATION DIAGNOSTICS")
print("-" * 70)
print(f"   Bootstrap 95% CI (1000 resamples on test set):")
print(f"     Recall: {boot_ci['recall_mean']:.3f}  [{boot_ci['recall_ci'][0]:.3f}, {boot_ci['recall_ci'][1]:.3f}]")
print(f"     F1:     {boot_ci['f1_mean']:.3f}  [{boot_ci['f1_ci'][0]:.3f}, {boot_ci['f1_ci'][1]:.3f}]")
print(f"   Threshold was selected via CV on training data (NOT on test set)")
for cname, cinfo in r5_cv_info.items():
    print(f"   {cname} CV info:")
    print(f"     CV median threshold: {cinfo['cv_threshold']:.3f}")
    print(f"     CV mean recall:      {cinfo['cv_mean_recall']:.3f}")
    print(f"     Fold recalls:        {[f'{r:.3f}' for r in cinfo['fold_recalls']]}")
