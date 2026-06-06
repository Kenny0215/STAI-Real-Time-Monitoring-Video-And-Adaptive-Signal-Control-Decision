"""
retrain_rf.py
Retrains the Random Forest model with optimized settings
to achieve 80-90% accuracy.

Fixes applied:
1. class_weight='balanced' — fixes class imbalance (Low=719 vs High=372)
2. GridSearchCV — finds best hyperparameters automatically
3. Feature engineering — adds interaction features
4. Saves improved model back to congestion_model.pkl

Usage:
    cd backend
    python retrain_rf.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "model", "model_output")
OUTPUT_DIR   = os.path.join(MODEL_DIR, "report")
MODEL_PKL    = os.path.join(MODEL_DIR, "congestion_model.pkl")
TRAINING_CSV = os.path.join(MODEL_DIR, "training_data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS        = ["Low", "Medium", "High"]
FEATURE_NAMES = ["vehicle_count", "density", "heavy_ratio", "avg_speed", "congestion_enc"]

print("=" * 55)
print("  SmartTraffic AI — RF Model Retraining")
print("=" * 55)

# ── Load data ──────────────────────────────────────────────────
print("\n[1/5] Loading training data...")
df = pd.read_csv(TRAINING_CSV)

# Fix congestion_enc
if df["congestion_enc"].isnull().sum() > 100:
    cong_map = {"Low": 0, "Medium": 1, "High": 2}
    df["congestion_enc"] = df["congestion"].map(cong_map).fillna(0).astype(int)

# Fix priority_level
df["priority_level"] = df["priority_level"].astype(str)\
    .str.replace(" Priority", "", regex=False).str.strip()
df["priority_level"] = df["priority_level"].replace("nan", np.nan)

# Fill NaN from 'priority' column if needed
if df["priority_level"].isnull().sum() > 0 and "priority" in df.columns:
    alt = df["priority"].astype(str)\
        .str.replace(" Priority", "", regex=False).str.strip()
    alt = alt.replace("nan", np.nan)
    df["priority_level"] = df["priority_level"].fillna(alt)

df_clean = df.dropna(subset=FEATURE_NAMES + ["priority_level"])
print(f"    Samples loaded : {len(df_clean)}")
print(f"    Distribution   :")
print(df_clean["priority_level"].value_counts().to_string(header=False))

# ── Feature engineering ────────────────────────────────────────
print("\n[2/5] Engineering features...")

df_clean = df_clean.copy()

# Add interaction features that help separate Medium vs High
df_clean["count_x_density"]    = df_clean["vehicle_count"] * df_clean["density"]
df_clean["speed_x_heavy"]      = df_clean["avg_speed"]     * df_clean["heavy_ratio"]
df_clean["count_x_heavy"]      = df_clean["vehicle_count"] * df_clean["heavy_ratio"]
df_clean["density_x_cong"]     = df_clean["density"]       * df_clean["congestion_enc"]
df_clean["speed_inv"]          = 1 / (df_clean["avg_speed"] + 1)  # inverse speed

FEATURE_NAMES_ENG = FEATURE_NAMES + [
    "count_x_density", "speed_x_heavy",
    "count_x_heavy", "density_x_cong", "speed_inv"
]

X = df_clean[FEATURE_NAMES_ENG].values
y_raw = df_clean["priority_level"].values

le = LabelEncoder()
le.fit(LABELS)
y = le.transform(y_raw)

print(f"    Features used  : {len(FEATURE_NAMES_ENG)}")
print(f"    Feature list   : {FEATURE_NAMES_ENG}")

# ── Train/test split (stratified) ─────────────────────────────
print("\n[3/5] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)}  Test: {len(X_test)}")

# ── Grid search for best RF params ────────────────────────────
print("\n[4/5] Running GridSearchCV (this may take 1-2 minutes)...")

param_grid = {
    "n_estimators":      [200, 300, 500],
    "max_depth":         [None, 15, 25],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", "log2"],
}

rf_base = RandomForestClassifier(
    class_weight="balanced",   # fixes imbalance
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_base, param_grid,
    cv=cv, scoring="f1_weighted",
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"    Best params    : {grid_search.best_params_}")
print(f"    Best CV F1     : {grid_search.best_score_*100:.1f}%")

# ── Evaluate ───────────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")

y_pred     = best_rf.predict(X_test)
acc        = accuracy_score(y_test, y_pred)
f1_w       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
y_pred_lbl = le.inverse_transform(y_pred)
y_test_lbl = le.inverse_transform(y_test)

print(f"\n    Accuracy       : {acc*100:.2f}%")
print(f"    Weighted F1    : {f1_w*100:.2f}%")
print(f"\n{classification_report(y_test_lbl, y_pred_lbl, labels=LABELS, zero_division=0)}")

# ── Save improved model ────────────────────────────────────────
joblib.dump({
    "model":           best_rf,
    "le_priority":     le,
    "le_congestion":   None,
    "feature_names":   FEATURE_NAMES_ENG,
}, MODEL_PKL)
print(f"[SAVED] Model → {MODEL_PKL}")

# ── Generate plots ─────────────────────────────────────────────
print("\n[Plotting] Generating report images...")

dark_bg  = '#0d1424'
grid_col = '#1e293b'

# 1. Confusion matrix
cm     = confusion_matrix(y_test_lbl, y_pred_lbl, labels=LABELS)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(dark_bg)
for ax in axes:
    ax.set_facecolor(dark_bg)

sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
    xticklabels=LABELS, yticklabels=LABELS,
    cmap='YlOrRd', linewidths=0.5, linecolor=grid_col,
    annot_kws={"size": 14, "weight": "bold", "color": "white"})
axes[0].set_title('Confusion Matrix (Count)', color='white', fontsize=13, pad=10)
axes[0].set_xlabel('Predicted', color='#94a3b8')
axes[0].set_ylabel('Actual',    color='#94a3b8')
axes[0].set_xticklabels(LABELS, color='white')
axes[0].set_yticklabels(LABELS, color='white', rotation=0)

sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=axes[1],
    xticklabels=LABELS, yticklabels=LABELS,
    cmap='Blues', linewidths=0.5, linecolor=grid_col,
    annot_kws={"size": 13, "weight": "bold", "color": "white"})
axes[1].set_title('Confusion Matrix (Normalised %)', color='white', fontsize=13, pad=10)
axes[1].set_xlabel('Predicted', color='#94a3b8')
axes[1].set_ylabel('Actual',    color='#94a3b8')
axes[1].set_xticklabels(LABELS, color='white')
axes[1].set_yticklabels(LABELS, color='white', rotation=0)

for ax in axes:
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor(grid_col)

plt.suptitle(f'Random Forest — Confusion Matrix  (Accuracy: {acc*100:.1f}%)',
             color='white', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("    [SAVED] confusion_matrix.png")

# 2. Classification report heatmap
from sklearn.metrics import precision_score, recall_score
report_data = {
    cls: {
        "Precision": precision_score(y_test_lbl, y_pred_lbl, labels=[cls], average='macro', zero_division=0),
        "Recall":    recall_score(y_test_lbl,    y_pred_lbl, labels=[cls], average='macro', zero_division=0),
        "F1-Score":  f1_score(y_test_lbl,        y_pred_lbl, labels=[cls], average='macro', zero_division=0),
    }
    for cls in LABELS
}
report_df = pd.DataFrame(report_data).T

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(dark_bg)
ax.set_facecolor(dark_bg)
sns.heatmap(report_df.astype(float), annot=True, fmt='.3f', ax=ax,
    cmap='RdYlGn', vmin=0, vmax=1,
    linewidths=0.5, linecolor=grid_col,
    annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Classification Report — Precision · Recall · F1',
             color='white', fontsize=13, pad=10)
ax.set_xticklabels(['Precision','Recall','F1-Score'], color='white')
ax.set_yticklabels(LABELS, color='white', rotation=0)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
fig.text(0.5, -0.05,
         f"Accuracy: {acc*100:.1f}%  |  Weighted F1: {f1_w*100:.1f}%  |  Samples: {len(X_test)}",
         ha='center', color='#10b981', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "classification_report.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("    [SAVED] classification_report.png")

# 3. Feature importance
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({
    'feature':    FEATURE_NAMES_ENG,
    'importance': importances * 100
}).sort_values('importance', ascending=True)

feat_labels = {
    'vehicle_count':   'Vehicle Count',
    'density':         'Traffic Density',
    'heavy_ratio':     'Heavy Vehicle Ratio',
    'avg_speed':       'Average Speed',
    'congestion_enc':  'Congestion Level',
    'count_x_density': 'Count × Density',
    'speed_x_heavy':   'Speed × Heavy Ratio',
    'count_x_heavy':   'Count × Heavy Ratio',
    'density_x_cong':  'Density × Congestion',
    'speed_inv':       'Inverse Speed',
}
feat_df['label'] = feat_df['feature'].map(feat_labels)
colors = ['#10b981' if v >= 15 else '#f59e0b' if v >= 8 else '#ef4444'
          for v in feat_df['importance']]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(dark_bg)
ax.set_facecolor(dark_bg)
bars = ax.barh(feat_df['label'], feat_df['importance'],
               color=colors, edgecolor=grid_col, height=0.55)
for bar, val in zip(bars, feat_df['importance']):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color='white', fontsize=10, fontweight='bold')
ax.set_xlabel('Importance (%)', color='#94a3b8')
ax.set_title('Random Forest — Feature Importance', color='white', fontsize=13, pad=10)
ax.set_xlim(0, max(feat_df['importance']) * 1.2)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("    [SAVED] feature_importance.png")

# 4. ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_prob     = best_rf.predict_proba(X_test)
colors_roc = ['#10b981','#f59e0b','#ef4444']

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(dark_bg)
ax.set_facecolor(dark_bg)
for i, (lbl, col) in enumerate(zip(LABELS, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, color=col, linewidth=2.5,
            label=f'{lbl}  (AUC = {auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1], color='#475569', linewidth=1, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate', color='#94a3b8')
ax.set_ylabel('True Positive Rate',  color='#94a3b8')
ax.set_title('ROC Curve — One-vs-Rest', color='white', fontsize=13, pad=10)
ax.tick_params(colors='white')
ax.legend(facecolor='#1e293b', labelcolor='white', edgecolor=grid_col)
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("    [SAVED] roc_curve.png")

# ── Text summary ───────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), 'w') as f:
    f.write("=" * 55 + "\n")
    f.write("  SmartTraffic AI — RF Model Report (Retrained)\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Training samples : {len(df_clean)}\n")
    f.write(f"Test samples     : {len(X_test)}\n")
    f.write(f"Features         : {FEATURE_NAMES_ENG}\n\n")
    f.write(f"Accuracy         : {acc*100:.2f}%\n")
    f.write(f"Weighted F1      : {f1_w*100:.2f}%\n\n")
    f.write(f"Best params      : {grid_search.best_params_}\n\n")
    f.write("--- Classification Report ---\n")
    f.write(classification_report(y_test_lbl, y_pred_lbl,
                                  labels=LABELS, zero_division=0))
    f.write("\n--- Feature Importances ---\n")
    for _, row in feat_df.sort_values('importance', ascending=False).iterrows():
        f.write(f"  {row['label']:<25} {row['importance']:.2f}%\n")

print("    [SAVED] model_summary.txt")

print("\n" + "=" * 55)
print(f"  Accuracy  : {acc*100:.1f}%")
print(f"  F1 Score  : {f1_w*100:.1f}%")
print(f"  Saved to  : {OUTPUT_DIR}")
print("=" * 55)