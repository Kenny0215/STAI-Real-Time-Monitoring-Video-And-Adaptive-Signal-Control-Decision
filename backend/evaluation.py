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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "model", "model_output")
OUTPUT_DIR   = os.path.join(MODEL_DIR, "report")
MODEL_PKL    = os.path.join(MODEL_DIR, "congestion_model.pkl")
TRAINING_CSV = os.path.join(MODEL_DIR, "training_data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS        = ["Low", "Medium", "High"]
# ── Removed congestion_enc — it's a bucketed copy of vehicle_count
# (classify_congestion() just thresholds vehicle_count), so including
# both gives the model two redundant views of the same signal,
# making classification artificially easy (0.81 correlation).
# Using only independent measurements forces genuine learning.
FEATURE_NAMES = ["vehicle_count", "density", "heavy_ratio", "avg_speed"]

print("=" * 55)
print("  SmartTraffic AI — RF Training")
print("=" * 55)

# ── Load + clean data ────────────────────────────────────────
print("\n[1/6] Loading training data...")
df = pd.read_csv(TRAINING_CSV)

if df["congestion_enc"].isnull().sum() > 100:
    cong_map = {"Low": 0, "Medium": 1, "High": 2}
    df["congestion_enc"] = df["congestion"].map(cong_map).fillna(0).astype(int)

df["priority_level"] = df["priority_level"].astype(str)\
    .str.replace(" Priority", "", regex=False).str.strip()
df["priority_level"] = df["priority_level"].replace("nan", np.nan)

if df["priority_level"].isnull().sum() > 0 and "priority" in df.columns:
    alt = df["priority"].astype(str)\
        .str.replace(" Priority", "", regex=False).str.strip()
    alt = alt.replace("nan", np.nan)
    df["priority_level"] = df["priority_level"].fillna(alt)

df_clean = df.dropna(subset=FEATURE_NAMES + ["priority_level"])

# ── Remove exact duplicate rows — these inflate accuracy ──────
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=FEATURE_NAMES + ["priority_level"])
after = len(df_clean)
print(f"    Removed {before - after} duplicate rows (these cause data leakage)")
print(f"    Final samples: {after}")
print(f"    Distribution:\n{df_clean['priority_level'].value_counts().to_string()}")

X = df_clean[FEATURE_NAMES].values
y_raw = df_clean["priority_level"].values

le = LabelEncoder()
le.fit(LABELS)
y = le.transform(y_raw)

# ── Add small Gaussian noise to break exact memorization ──────
# Real sensor data always has measurement noise. Adding tiny
# noise prevents the model from memorizing exact feature values.
print("\n[2/6] Adding realistic sensor noise (prevents memorization)...")
rng = np.random.RandomState(42)
X_noisy = X.copy().astype(float)
noise_scale = X_noisy.std(axis=0) * 0.03  # 3% noise — realistic sensor jitter
X_noisy += rng.normal(0, noise_scale, X_noisy.shape)
print(f"    Noise scale per feature: {dict(zip(FEATURE_NAMES, noise_scale.round(3)))}")

# ── Train / Validation / Test split (60/20/20) ─────────────────
print("\n[3/6] Splitting 60% train / 20% validation / 20% test...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_noisy, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"    Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Constrained Random Forest (anti-overfitting) ────────────────
print("\n[4/6] Training constrained Random Forest...")

best_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,                # SHALLOW trees — prevents memorizing every sample
    min_samples_split=20,       # require 20+ samples to split a node
    min_samples_leaf=10,        # require 10+ samples per leaf
    max_features="sqrt",        # only consider sqrt(n_features) per split
    class_weight="balanced",
    bootstrap=True,
    oob_score=True,             # out-of-bag score for honest evaluation
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)

print(f"    OOB Score (honest internal estimate): {best_rf.oob_score_*100:.2f}%")

# ── Cross-validation to detect overfitting gap ──────────────────
print("\n[5/6] Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"    CV scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"    CV mean: {cv_scores.mean()*100:.2f}%  (+/- {cv_scores.std()*100:.2f}%)")

train_acc = accuracy_score(y_train, best_rf.predict(X_train))
val_acc   = accuracy_score(y_val,   best_rf.predict(X_val))
test_acc  = accuracy_score(y_test,  best_rf.predict(X_test))

print(f"\n    Train accuracy : {train_acc*100:.2f}%")
print(f"    Val accuracy   : {val_acc*100:.2f}%")
print(f"    Test accuracy  : {test_acc*100:.2f}%")

overfit_gap = train_acc - test_acc
print(f"    Overfit gap    : {overfit_gap*100:.2f} percentage points")

if overfit_gap > 0.15:
    print("    [WARNING] Gap > 15% — still overfitting, model may need more constraints")
elif overfit_gap < 0.02:
    print("    [WARNING] Gap < 2% — model may be too simple, check val/test accuracy is reasonable")
else:
    print("    [GOOD] Healthy gap between train and test — model is generalizing")

# ── Final evaluation on test set ────────────────────────────────
print("\n[6/6] Final evaluation on held-out test set...")
y_pred     = best_rf.predict(X_test)
acc        = accuracy_score(y_test, y_pred)
f1_w       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
y_pred_lbl = le.inverse_transform(y_pred)
y_test_lbl = le.inverse_transform(y_test)

print(f"\n{classification_report(y_test_lbl, y_pred_lbl, labels=LABELS, zero_division=0)}")

# ── Save model ───────────────────────────────────────────────
joblib.dump({
    "model":         best_rf,
    "le_priority":   le,
    "le_congestion": None,
    "feature_names": FEATURE_NAMES,
}, MODEL_PKL)
print(f"[SAVED] Model → {MODEL_PKL}")

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
dark_bg, grid_col = '#0d1424', '#1e293b'

# 1. Train vs Val vs Test accuracy bar
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(dark_bg)
ax.set_facecolor(dark_bg)
splits  = ['Train', 'Validation', 'Test']
accs    = [train_acc*100, val_acc*100, test_acc*100]
colors  = ['#3b82f6', '#f59e0b', '#10b981']
bars = ax.bar(splits, accs, color=colors, edgecolor=grid_col, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, val+1, f'{val:.1f}%',
             ha='center', color='white', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy (%)', color='#94a3b8')
ax.set_title(f'Train/Val/Test Accuracy  (Overfit gap: {overfit_gap*100:.1f}pp)',
             color='white', fontsize=13, pad=10)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_val_test_accuracy.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("[SAVED] train_val_test_accuracy.png")

# 2. Confusion matrix
cm = confusion_matrix(y_test_lbl, y_pred_lbl, labels=LABELS)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(dark_bg)
for ax in axes: ax.set_facecolor(dark_bg)

sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], xticklabels=LABELS, yticklabels=LABELS,
    cmap='YlOrRd', linewidths=0.5, linecolor=grid_col,
    annot_kws={"size": 14, "weight": "bold", "color": "white"})
axes[0].set_title('Confusion Matrix (Count)', color='white', fontsize=13, pad=10)
axes[0].set_xlabel('Predicted', color='#94a3b8'); axes[0].set_ylabel('Actual', color='#94a3b8')

sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=axes[1], xticklabels=LABELS, yticklabels=LABELS,
    cmap='Blues', linewidths=0.5, linecolor=grid_col,
    annot_kws={"size": 13, "weight": "bold", "color": "white"})
axes[1].set_title('Confusion Matrix (Normalised %)', color='white', fontsize=13, pad=10)
axes[1].set_xlabel('Predicted', color='#94a3b8'); axes[1].set_ylabel('Actual', color='#94a3b8')

for ax in axes:
    ax.set_xticklabels(LABELS, color='white'); ax.set_yticklabels(LABELS, color='white', rotation=0)
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor(grid_col)

plt.suptitle(f'Confusion Matrix — Test Accuracy: {acc*100:.1f}%', color='white', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("[SAVED] confusion_matrix.png")

# 3. ROC curve — should show realistic curves, not perfect right angles
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_prob     = best_rf.predict_proba(X_test)
colors_roc = ['#10b981','#f59e0b','#ef4444']

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(dark_bg)
ax.set_facecolor(dark_bg)
for i, (lbl, col) in enumerate(zip(LABELS, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, linewidth=2.5, label=f'{lbl}  (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], color='#475569', linewidth=1, linestyle='--', label='Random classifier')
ax.set_xlabel('False Positive Rate', color='#94a3b8')
ax.set_ylabel('True Positive Rate',  color='#94a3b8')
ax.set_title('ROC Curve — One-vs-Rest (Test Set)', color='white', fontsize=13, pad=10)
ax.tick_params(colors='white')
ax.legend(facecolor='#1e293b', labelcolor='white', edgecolor=grid_col)
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("[SAVED] roc_curve.png")

# 4. Feature importance
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': importances*100})\
    .sort_values('importance', ascending=True)
feat_labels = {
    'vehicle_count': 'Vehicle Count', 'density': 'Traffic Density',
    'heavy_ratio': 'Heavy Vehicle Ratio', 'avg_speed': 'Average Speed',
    'congestion_enc': 'Congestion Level',
}
feat_df['label'] = feat_df['feature'].map(feat_labels)
colors = ['#10b981' if v>=25 else '#f59e0b' if v>=15 else '#ef4444' for v in feat_df['importance']]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(dark_bg); ax.set_facecolor(dark_bg)
bars = ax.barh(feat_df['label'], feat_df['importance'], color=colors, edgecolor=grid_col, height=0.55)
for bar, val in zip(bars, feat_df['importance']):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%',
             va='center', color='white', fontsize=11, fontweight='bold')
ax.set_xlabel('Importance (%)', color='#94a3b8')
ax.set_title('Feature Importance', color='white', fontsize=13, pad=10)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor(grid_col)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"),
            dpi=150, bbox_inches='tight', facecolor=dark_bg)
plt.close()
print("[SAVED] feature_importance.png")

# ── Text summary ──────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), 'w') as f:
    f.write("="*55 + "\n  RF Model Report\n" + "="*55 + "\n\n")
    f.write(f"Samples (after dedup): {len(df_clean)}\n")
    f.write(f"Train/Val/Test split : 60/20/20\n")
    f.write(f"OOB Score             : {best_rf.oob_score_*100:.2f}%\n")
    f.write(f"CV mean accuracy      : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n\n")
    f.write(f"Train accuracy        : {train_acc*100:.2f}%\n")
    f.write(f"Validation accuracy   : {val_acc*100:.2f}%\n")
    f.write(f"Test accuracy         : {test_acc*100:.2f}%\n")
    f.write(f"Overfit gap           : {overfit_gap*100:.2f} pp\n\n")
    f.write(f"Weighted F1 (test)    : {f1_w*100:.2f}%\n\n")
    f.write("--- Classification Report (Test) ---\n")
    f.write(classification_report(y_test_lbl, y_pred_lbl, labels=LABELS, zero_division=0))
    f.write("\n--- Hyperparameters ---\n")
    f.write(f"  max_depth         : 8\n")
    f.write(f"  min_samples_split : 20\n")
    f.write(f"  min_samples_leaf  : 10\n")
    f.write(f"  max_features      : sqrt\n")
    f.write(f"  n_estimators      : 150\n")

print("\n" + "="*55)
print(f"  Test Accuracy  : {acc*100:.1f}%")
print(f"  OOB Score      : {best_rf.oob_score_*100:.1f}%")
print(f"  Overfit Gap    : {overfit_gap*100:.1f}pp")
print(f"  Saved to       : {OUTPUT_DIR}")
print("="*55)