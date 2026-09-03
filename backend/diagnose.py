"""
diagnose_data.py
Checks if priority_level is a simple deterministic rule of the features
(which would explain why ANY model gets 95%+ accuracy easily).

"""
import os
import pandas as pd
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(BASE_DIR, "model", "model_output", "training_data.csv")

df = pd.read_csv(TRAINING_CSV)

# Clean labels same way as training script
df["priority_level"] = df["priority_level"].astype(str)\
    .str.replace(" Priority", "", regex=False).str.strip()
df["priority_level"] = df["priority_level"].replace("nan", np.nan)
if df["priority_level"].isnull().sum() > 0 and "priority" in df.columns:
    alt = df["priority"].astype(str).str.replace(" Priority", "", regex=False).str.strip()
    df["priority_level"] = df["priority_level"].fillna(alt.replace("nan", np.nan))

if df["congestion_enc"].isnull().sum() > 100:
    cong_map = {"Low": 0, "Medium": 1, "High": 2}
    df["congestion_enc"] = df["congestion"].map(cong_map).fillna(0).astype(int)

df_clean = df.dropna(subset=["vehicle_count","density","heavy_ratio","avg_speed","congestion_enc","priority_level"])
df_clean = df_clean.drop_duplicates()

print("=" * 60)
print("  Checking if priority_level is a deterministic rule")
print("=" * 60)

# Check 1: vehicle_count ranges per priority class
print("\n--- vehicle_count range per priority_level ---")
print(df_clean.groupby("priority_level")["vehicle_count"].agg(["min","max","mean"]))

# Check 2: Does congestion alone predict priority almost perfectly?
print("\n--- Crosstab: congestion vs priority_level ---")
print(pd.crosstab(df_clean["congestion"], df_clean["priority_level"]))

# Check 3: correlation between vehicle_count and priority (encoded)
priority_map = {"Low": 0, "Medium": 1, "High": 2}
df_clean["priority_num"] = df_clean["priority_level"].map(priority_map)

print("\n--- Correlation of each feature with priority_level ---")
for col in ["vehicle_count","density","heavy_ratio","avg_speed","congestion_enc"]:
    corr = df_clean[col].corr(df_clean["priority_num"])
    print(f"  {col:<18} correlation: {corr:.3f}")

# Check 4: Can a single threshold rule on vehicle_count alone classify well?
print("\n--- Simple rule test: vehicle_count thresholds only ---")
def simple_rule(vc):
    if vc >= 25: return "High"
    elif vc >= 15: return "Medium"
    else: return "Low"

df_clean["rule_pred"] = df_clean["vehicle_count"].apply(simple_rule)
rule_acc = (df_clean["rule_pred"] == df_clean["priority_level"]).mean()
print(f"  Simple vehicle_count-only rule accuracy: {rule_acc*100:.1f}%")

if rule_acc > 0.85:
    print("\n  [DIAGNOSIS] Your priority_level is essentially a DETERMINISTIC")
    print("  function of vehicle_count (and/or congestion). This is why ANY")
    print("  ML model gets 90%+ accuracy easily — there's no real 'learning'")
    print("  happening, just rule discovery on clean synthetic data.")
    print("\n  This happens because predict_priority() in your detection.py")
    print("  ALREADY uses hard rules (congestion == 'High' -> High priority)")
    print("  before even calling the RF model, so the RF model only sees")
    print("  the 'easy' Medium-vs-borderline cases, OR the saved CSV recorded")
    print("  the rule-based output as ground truth, not independent reality.")
else:
    print("\n  [OK] Priority is not trivially separable — model accuracy")
    print("  reflects real pattern learning.")

print("\n" + "=" * 60)