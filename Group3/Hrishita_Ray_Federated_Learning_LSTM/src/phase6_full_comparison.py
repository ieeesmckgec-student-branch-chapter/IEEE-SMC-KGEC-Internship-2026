
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, mean_squared_error
)

import tensorflow as tf

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

print("REBUILDING VALIDATION SPLIT")

drone_df = pd.read_csv("drone_clean.csv")
drone_X = drone_df.drop(columns=['label']).values.astype(np.float32)
drone_y = drone_df['label'].values.astype(np.float32)

kdd_df = pd.read_csv("Train_data.csv")
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    kdd_df[col] = le.fit_transform(kdd_df[col].astype(str))
kdd_X = kdd_df.drop(columns=['class']).values.astype(np.float32)
kdd_y = (kdd_df['class'] != 'normal').astype(np.float32).values

drone_X = StandardScaler().fit_transform(drone_X)
kdd_X   = StandardScaler().fit_transform(kdd_X)

def make_sequences(X): return X.reshape(X.shape[0], 1, X.shape[1])
drone_X = make_sequences(drone_X)
kdd_X   = make_sequences(kdd_X)

dX_tr, dX_v, dy_tr, dy_v = train_test_split(drone_X, drone_y, test_size=0.2,
                                              random_state=42, stratify=drone_y)
kX_tr, kX_v, ky_tr, ky_v = train_test_split(kdd_X,   kdd_y,   test_size=0.2,
                                              random_state=42, stratify=kdd_y)

VAL_DATA = {"Drone": (dX_v, dy_v), "NSL-KDD": (kX_v, ky_v)}


print(" LOADING MODELS")

MODEL_FILES = {
    ("Drone",   "Centralized"): "outputs/drone_lstm_baseline.keras",
    ("Drone",   "FedAvg"):      "outputs/drone_lstm_federated.keras",
    ("Drone",   "FedProx"):     "outputs/drone_lstm_fedprox.keras",
    ("Drone",   "FedNova"):     "outputs/drone_lstm_fednova.keras",
    ("NSL-KDD", "Centralized"): "outputs/kdd_lstm_baseline.keras",
    ("NSL-KDD", "FedAvg"):      "outputs/kdd_lstm_federated.keras",
    ("NSL-KDD", "FedProx"):     "outputs/kdd_lstm_fedprox.keras",
    ("NSL-KDD", "FedNova"):     "outputs/kdd_lstm_fednova.keras",
}

models = {}
for key, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"  ! Missing {path} — run the corresponding phase script first. Skipping.")
        continue
    models[key] = tf.keras.models.load_model(path)
    print(f"  Loaded {path}")
print("\nEVALUATION")

def full_eval(model, X, y):
    yp   = model.predict(X, verbose=0).flatten()
    yd   = (yp >= 0.5).astype(int)
    acc  = accuracy_score(y, yd)
    f1   = f1_score(y, yd, zero_division=0)
    prec = precision_score(y, yd, zero_division=0)
    rec  = recall_score(y, yd, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y, yp))
    cm   = confusion_matrix(y, yd)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    fpr_m = fp/(fp+tn+1e-9)
    fp_r, tp_r, _ = roc_curve(y, yp)
    roc_auc = auc(fp_r, tp_r)
    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec, fpr=fpr_m,
                rmse=rmse, auc=roc_auc, cm=cm, yprob=yp, fp_r=fp_r, tp_r=tp_r)

results = {}
for (dataset, method), model in models.items():
    X, y = VAL_DATA[dataset]
    results[(dataset, method)] = full_eval(model, X, y)

FL_METHODS = ["FedAvg", "FedProx", "FedNova"]
COLORS = {"Centralized": "#5B4CB5", "FedAvg": "#1D9E75", "FedProx": "#D95F02",
          "FedNova": "#7570B3", "Best FL": "#1D9E75"}

print("\n" + "="*70)
print("STAGE A — FEDAVG vs FEDPROX vs FEDNOVA (which FL strategy wins?)")
print("="*70)

stage_a_rows = []
best_fl = {}  
for dataset in ["Drone", "NSL-KDD"]:
    print(f"\n  ── {dataset} ──")
    scores = {}
    for method in FL_METHODS:
        key = (dataset, method)
        if key not in results:
            continue
        ev = results[key]
        scores[method] = ev['accuracy']
        stage_a_rows.append([dataset, method, ev['accuracy'], ev['f1'], ev['precision'],
                              ev['recall'], ev['fpr'], ev['rmse'], ev['auc']])
        print(f"    {method:<10} acc={ev['accuracy']:.4f}  f1={ev['f1']:.4f}  "
              f"auc={ev['auc']:.4f}  fpr={ev['fpr']:.4f}")
    if scores:
        winner = max(scores, key=scores.get)
        best_fl[dataset] = winner
        spread = max(scores.values()) - min(scores.values())
        print(f"    -> Best FL strategy on {dataset}: {winner} "
              f"(accuracy spread across FL methods: {spread*100:.2f} percentage points)")

stage_a_df = pd.DataFrame(stage_a_rows, columns=["Dataset","Method","Accuracy","F1","Precision","Recall","FPR","RMSE","AUC"])
stage_a_df.to_csv("outputs/stageA_fl_only_comparison.csv", index=False)
print("\n  Saved outputs/stageA_fl_only_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Stage A — FedAvg vs FedProx vs FedNova (federated strategies only)",
             fontsize=13, fontweight='bold')
METRICS = ["accuracy", "f1", "precision", "recall"]
for col, dataset in enumerate(["Drone", "NSL-KDD"]):
    ax = axes[col]
    x = np.arange(len(METRICS)); width = 0.25
    for i, method in enumerate(FL_METHODS):
        key = (dataset, method)
        if key not in results: continue
        vals = [results[key][m] for m in METRICS]
        label = method + ("  \u2605 best" if best_fl.get(dataset) == method else "")
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=COLORS[method])
    ax.set_title(dataset, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/stageA_fl_only_comparison.png", dpi=150, bbox_inches='tight')
print("  Saved outputs/stageA_fl_only_comparison.png")
plt.close()

print("\n" + "="*70)
print("STAGE B — BEST FEDERATED STRATEGY vs CENTRALIZED")
print("="*70)

stage_b_rows = []
for dataset in ["Drone", "NSL-KDD"]:
    cen_key = (dataset, "Centralized")
    if cen_key not in results or dataset not in best_fl:
        continue
    cen = results[cen_key]
    winner_name = best_fl[dataset]
    fl = results[(dataset, winner_name)]
    gap = cen['accuracy'] - fl['accuracy']
    stage_b_rows.append([dataset, "Centralized", cen['accuracy'], cen['f1'], cen['precision'],
                          cen['recall'], cen['fpr'], cen['rmse'], cen['auc']])
    stage_b_rows.append([dataset, f"Best FL ({winner_name})", fl['accuracy'], fl['f1'], fl['precision'],
                          fl['recall'], fl['fpr'], fl['rmse'], fl['auc']])
    print(f"\n  ── {dataset} ──")
    print(f"    Centralized       acc={cen['accuracy']:.4f}  auc={cen['auc']:.4f}")
    print(f"    Best FL ({winner_name:<7}) acc={fl['accuracy']:.4f}  auc={fl['auc']:.4f}")
    print(f"    -> Accuracy gap (Centralized - Best FL): {gap*100:.2f} percentage points")

stage_b_df = pd.DataFrame(stage_b_rows, columns=["Dataset","Method","Accuracy","F1","Precision","Recall","FPR","RMSE","AUC"])
stage_b_df.to_csv("outputs/stageB_best_fl_vs_centralized.csv", index=False)
print("\n  Saved outputs/stageB_best_fl_vs_centralized.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Stage B — Centralized vs Best Federated Strategy",
             fontsize=13, fontweight='bold')
for col, dataset in enumerate(["Drone", "NSL-KDD"]):
    ax = axes[col]
    if dataset not in best_fl:
        continue
    winner_name = best_fl[dataset]
    cen = results[(dataset, "Centralized")]
    fl  = results[(dataset, winner_name)]
    x = np.arange(len(METRICS)); width = 0.35
    ax.bar(x - width/2, [cen[m] for m in METRICS], width, label="Centralized", color=COLORS["Centralized"])
    ax.bar(x + width/2, [fl[m]  for m in METRICS], width, label=f"Best FL ({winner_name})", color=COLORS["Best FL"])
    ax.set_title(dataset, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/stageB_best_fl_vs_centralized.png", dpi=150, bbox_inches='tight')
print("  Saved outputs/stageB_best_fl_vs_centralized.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("ROC — Centralized vs Best Federated Strategy", fontsize=13, fontweight='bold')
for col, dataset in enumerate(["Drone", "NSL-KDD"]):
    ax = axes[col]
    if dataset not in best_fl:
        continue
    winner_name = best_fl[dataset]
    cen = results[(dataset, "Centralized")]
    fl  = results[(dataset, winner_name)]
    ax.plot(cen['fp_r'], cen['tp_r'], color=COLORS["Centralized"], lw=2.2, label=f"Centralized (AUC={cen['auc']:.3f})")
    ax.plot(fl['fp_r'], fl['tp_r'], color=COLORS["Best FL"], lw=2.2, ls='--', label=f"Best FL: {winner_name} (AUC={fl['auc']:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    ax.set_title(dataset, fontsize=11)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/stageB_roc.png", dpi=150, bbox_inches='tight')
print("  Saved outputs/stageB_roc.png")
plt.close()

all_rows = []
for (dataset, method), ev in results.items():
    all_rows.append([dataset, method, ev['accuracy'], ev['f1'], ev['precision'],
                      ev['recall'], ev['fpr'], ev['rmse'], ev['auc']])
full_df = pd.DataFrame(all_rows, columns=["Dataset","Method","Accuracy","F1","Precision","Recall","FPR","RMSE","AUC"])
full_df.to_csv("outputs/full_comparison_results.csv", index=False)
print("\nSaved outputs/full_comparison_results.csv (all methods, for reference)")

print("\nTWO-STAGE COMPARISON COMPLETE")
