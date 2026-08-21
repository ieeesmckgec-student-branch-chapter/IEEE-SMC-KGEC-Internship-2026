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

print("LOADING MODELS")

MODEL_FILES = {
    ("Drone", "Centralized"):       "outputs/drone_lstm_baseline.keras",
    ("Drone", "FedAvg",  "IID"):    "outputs/drone_lstm_federated.keras",
    ("Drone", "FedProx", "IID"):    "outputs/drone_lstm_fedprox.keras",
    ("Drone", "FedNova", "IID"):    "outputs/drone_lstm_fednova.keras",
    ("Drone", "FedAvg",  "NonIID"): "outputs/drone_lstm_federated_noniid.keras",
    ("Drone", "FedProx", "NonIID"): "outputs/drone_lstm_fedprox_noniid.keras",
    ("Drone", "FedNova", "NonIID"): "outputs/drone_lstm_fednova_noniid.keras",

    ("NSL-KDD", "Centralized"):       "outputs/kdd_lstm_baseline.keras",
    ("NSL-KDD", "FedAvg",  "IID"):    "outputs/kdd_lstm_federated.keras",
    ("NSL-KDD", "FedProx", "IID"):    "outputs/kdd_lstm_fedprox.keras",
    ("NSL-KDD", "FedNova", "IID"):    "outputs/kdd_lstm_fednova.keras",
    ("NSL-KDD", "FedAvg",  "NonIID"): "outputs/kdd_lstm_federated_noniid.keras",
    ("NSL-KDD", "FedProx", "NonIID"): "outputs/kdd_lstm_fedprox_noniid.keras",
    ("NSL-KDD", "FedNova", "NonIID"): "outputs/kdd_lstm_fednova_noniid.keras",
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
rows = []
for key, model in models.items():
    dataset = key[0]
    X, y = VAL_DATA[dataset]
    ev = full_eval(model, X, y)
    results[key] = ev
    label = " / ".join(key[1:])
    rows.append([dataset, label, ev['accuracy'], ev['f1'], ev['precision'],
                 ev['recall'], ev['fpr'], ev['rmse'], ev['auc']])
    print(f"  {dataset:<8} {label:<20} acc={ev['accuracy']:.4f}  f1={ev['f1']:.4f}  auc={ev['auc']:.4f}")

full_df = pd.DataFrame(rows, columns=["Dataset","Method","Accuracy","F1","Precision","Recall","FPR","RMSE","AUC"])
full_df.to_csv("outputs/iid_vs_noniid_full_results.csv", index=False)
print("\nSaved outputs/iid_vs_noniid_full_results.csv")

print("\n" + "="*70)
print("DOES THE SPREAD BETWEEN FL ALGORITHMS WIDEN UNDER NON-IID?")
print("="*70)

FL_METHODS = ["FedAvg", "FedProx", "FedNova"]
spread_summary = []
for dataset in ["Drone", "NSL-KDD"]:
    print(f"\n  ── {dataset} ──")
    for regime in ["IID", "NonIID"]:
        accs = {}
        for method in FL_METHODS:
            key = (dataset, method, regime)
            if key in results:
                accs[method] = results[key]['accuracy']
        if len(accs) < 2:
            continue
        spread = (max(accs.values()) - min(accs.values())) * 100
        best = max(accs, key=accs.get)
        print(f"    {regime:<7}: " + "  ".join(f"{m}={a:.4f}" for m, a in accs.items())
              + f"   -> spread={spread:.2f}pp, best={best}")
        spread_summary.append([dataset, regime, spread, best])

spread_df = pd.DataFrame(spread_summary, columns=["Dataset","Regime","Spread_pp","Best_Method"])
spread_df.to_csv("outputs/fl_algorithm_spread.csv", index=False)
print("\n  Saved outputs/fl_algorithm_spread.csv")
for dataset in ["Drone", "NSL-KDD"]:
    sub = spread_df[spread_df.Dataset == dataset]
    if len(sub) == 2:
        iid_spread = sub[sub.Regime == "IID"]["Spread_pp"].values
        noniid_spread = sub[sub.Regime == "NonIID"]["Spread_pp"].values
        if len(iid_spread) and len(noniid_spread):
            verdict = "WIDENED" if noniid_spread[0] > iid_spread[0] else "DID NOT widen"
            print(f"  {dataset}: spread {verdict} under non-IID "
                  f"({iid_spread[0]:.2f}pp -> {noniid_spread[0]:.2f}pp)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Accuracy per Federated Algorithm — IID vs Non-IID Clients",
             fontsize=13, fontweight='bold')
for col, dataset in enumerate(["Drone", "NSL-KDD"]):
    ax = axes[col]
    x = np.arange(len(FL_METHODS)); width = 0.35
    iid_vals    = [results.get((dataset, m, "IID"), {}).get('accuracy', 0)    for m in FL_METHODS]
    noniid_vals = [results.get((dataset, m, "NonIID"), {}).get('accuracy', 0) for m in FL_METHODS]
    ax.bar(x - width/2, iid_vals,    width, label="IID clients",     color="#1D9E75")
    ax.bar(x + width/2, noniid_vals, width, label="Non-IID clients", color="#D95F02")
    ax.set_title(dataset, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(FL_METHODS)
    ax.set_ylabel("Validation Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("outputs/iid_vs_noniid_by_algorithm.png", dpi=150, bbox_inches='tight')
print("\n  Saved outputs/iid_vs_noniid_by_algorithm.png")
plt.close()

print("\n" + "="*70)
print("BEST FEDERATED STRATEGY UNDER NON-IID vs CENTRALIZED")
print("="*70)

stage_rows = []
for dataset in ["Drone", "NSL-KDD"]:
    cen_key = (dataset, "Centralized")
    if cen_key not in results:
        continue
    cen = results[cen_key]
    accs = {m: results[(dataset, m, "NonIID")]['accuracy']
            for m in FL_METHODS if (dataset, m, "NonIID") in results}
    if not accs:
        continue
    winner = max(accs, key=accs.get)
    fl = results[(dataset, winner, "NonIID")]
    gap = cen['accuracy'] - fl['accuracy']
    stage_rows.append([dataset, "Centralized", cen['accuracy'], cen['f1'], cen['auc']])
    stage_rows.append([dataset, f"Best FL non-IID ({winner})", fl['accuracy'], fl['f1'], fl['auc']])
    print(f"\n  ── {dataset} ──")
    print(f"    Centralized                 acc={cen['accuracy']:.4f}  auc={cen['auc']:.4f}")
    print(f"    Best FL, non-IID ({winner:<7}) acc={fl['accuracy']:.4f}  auc={fl['auc']:.4f}")
    print(f"    -> Accuracy gap: {gap*100:.2f} percentage points")

stage_df = pd.DataFrame(stage_rows, columns=["Dataset","Method","Accuracy","F1","AUC"])
stage_df.to_csv("outputs/best_fl_noniid_vs_centralized.csv", index=False)
print("\n  Saved outputs/best_fl_noniid_vs_centralized.csv")

print("\nIID vs NON-IID COMPARISON COMPLETE")
