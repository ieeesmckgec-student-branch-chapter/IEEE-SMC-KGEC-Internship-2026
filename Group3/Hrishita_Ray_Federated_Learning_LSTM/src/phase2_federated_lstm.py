import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, mean_squared_error,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

print("\n" + "="*60)
print("  STEP 1 — LOAD, SCALE, SPLIT")
print("="*60)

print("\nLoading drone_clean.csv")
drone_df = pd.read_csv("drone_clean.csv")
print(f"  Shape  : {drone_df.shape}")
drone_X = drone_df.drop(columns=['label']).values.astype(np.float32)
drone_y = drone_df['label'].values.astype(np.float32)

print("\nLoading Train_data.csv")
kdd_df = pd.read_csv("Train_data.csv")
print(f"  Shape  : {kdd_df.shape}")
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

print(f"\n  Drone   — train: {dX_tr.shape}  val: {dX_v.shape}")
print(f"  NSL-KDD — train: {kX_tr.shape}  val: {kX_v.shape}")
print("Preprocessing complete")

def build_lstm(n_features, lstm_units=128, dense_units=64, dropout=0.3):
    model = Sequential([
        LSTM(lstm_units, input_shape=(1, n_features), return_sequences=False),
        LayerNormalization(),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        LayerNormalization(),
        Dropout(dropout * 0.67),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="LSTM_IDS_Federated")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def class_weights(y):
    n0, n1 = (y==0).sum(), (y==1).sum(); total = n0+n1
    return {0: total/(2*n0+1e-9), 1: total/(2*n1+1e-9)}
print("\n" + "="*60)
print("="*60)

N_CLIENTS    = 5    
N_ROUNDS     = 20   
                     
LOCAL_EPOCHS = 3    
BATCH_SIZE   = 64

print(f"  Clients      : {N_CLIENTS}")
print(f"  Rounds       : {N_ROUNDS}")
print(f"  Local epochs : {LOCAL_EPOCHS}")
print(f"  Batch size   : {BATCH_SIZE}")
print()

def partition_iid(X, y, n_clients):
    idx = np.random.permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]

def partition_non_iid(X, y, n_clients):
    idx = np.argsort(y)
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]

def fedavg(client_weights, client_sizes):
    total = sum(client_sizes)
    return [sum(w * (n/total) for w, n in zip(layer, client_sizes))
            for layer in zip(*client_weights)]

def federated_train(X_train, y_train, X_val, y_val, dataset_name, iid=True):
    split_fn = partition_iid if iid else partition_non_iid
    clients  = split_fn(X_train, y_train, N_CLIENTS)
    n_feat   = X_train.shape[2]

    print(f"\n  Partitioned into {N_CLIENTS} clients (IID={iid}):")
    for i, (cx, cy) in enumerate(clients):
        print(f"    Client {i+1}: {len(cx)} samples (attack rate: {cy.mean():.3f})")

    global_model = build_lstm(n_feat)
    local_model  = build_lstm(n_feat)           
    print(f"\n  Global model architecture:")
    global_model.summary()

    round_acc, round_loss = [], []

    print(f"\n  Starting FL training {N_ROUNDS} rounds…")
    print(f"  {'Round':>6}  {'Global Acc':>10}  {'Avg Client Loss':>15}")
    print(f"  {'-'*38}")

    for rnd in range(1, N_ROUNDS+1):
        client_weight_list, client_sizes, client_losses = [], [], []
        global_weights = global_model.get_weights()

        for cid, (cx, cy) in enumerate(clients):
            local_model.set_weights(deepcopy(global_weights))

            cw = class_weights(cy)
            local_model.fit(cx, cy, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE,
                           class_weight=cw, verbose=0)

            client_weight_list.append(deepcopy(local_model.get_weights()))
            client_sizes.append(len(cx))
            loss, _ = local_model.evaluate(cx, cy, verbose=0)
            client_losses.append(loss)

        global_model.set_weights(fedavg(client_weight_list, client_sizes))

        val_loss, val_acc = global_model.evaluate(X_val, y_val, verbose=0)
        round_acc.append(val_acc)
        round_loss.append(np.mean(client_losses))
        print(f"  {rnd:>6}  {val_acc:>10.4f}  {np.mean(client_losses):>15.4f}")

    print(f"\n  FL training complete ✓  Final global acc: {round_acc[-1]:.4f}")
    return global_model, round_acc, round_loss

print("\n" + "="*60)
print("="*60)

print("\nDrone Dataset")
drone_fl_model, drone_fl_acc, drone_fl_loss = \
    federated_train(dX_tr, dy_tr, dX_v, dy_v, "Drone", iid=True)

print("\nNSL-KDD Dataset")
kdd_fl_model, kdd_fl_acc, kdd_fl_loss = \
    federated_train(kX_tr, ky_tr, kX_v, ky_v, "NSL-KDD", iid=True)

print("\n" + "="*60)
print("="*60)

def full_eval(model, X, y, name):
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
    print(f"\n  ── {name} ──")
    print(f"  Accuracy : {acc:.4f}  |  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}  |  Recall   : {rec:.4f}")
    print(f"  FPR      : {fpr_m:.4f}  |  RMSE     : {rmse:.4f}")
    print(f"\n{classification_report(y, yd, target_names=['Normal','Attack'])}")
    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec,
                fpr=fpr_m, rmse=rmse, cm=cm, yprob=yp)

drone_fl_eval = full_eval(drone_fl_model, dX_v, dy_v, "Drone FL — Validation")
kdd_fl_eval   = full_eval(kdd_fl_model,   kX_v, ky_v, "NSL-KDD FL — Validation")

print("\n" + "="*60)
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Federated LSTM (FedAvg) — Intrusion Detection Results",
             fontsize=14, fontweight='bold')

for col, (name, acc_curve, loss_curve, Xv, yv, ev) in enumerate([
    ("Drone",   drone_fl_acc, drone_fl_loss, dX_v, dy_v, drone_fl_eval),
    ("NSL-KDD", kdd_fl_acc,   kdd_fl_loss,   kX_v, ky_v, kdd_fl_eval),
]):
    ax = axes[0][col]
    rounds = range(1, N_ROUNDS+1)
    ax.plot(rounds, acc_curve,  color='#5B4CB5', lw=2, marker='o', ms=4, label='Global accuracy')
    ax.plot(rounds, loss_curve, color='#1D9E75', lw=2, marker='s', ms=4, ls='--', label='Avg client loss')
    ax.set_title(f"{name} — FL Convergence over Rounds", fontsize=11)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(y=acc_curve[-1], color='#5B4CB5', lw=0.8, ls=':', alpha=0.6)
    ax = axes[1][col]
    cm = ev['cm']
    ax.imshow(cm, cmap='Greens')
    ax.set_title(f"{name} — Confusion Matrix (Federated)", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[r,c]>cm.max()/2 else 'black')
    ax = axes[2][col]
    fp_r, tp_r, _ = roc_curve(yv, ev['yprob'])
    roc_auc = auc(fp_r, tp_r)
    ax.plot(fp_r, tp_r, color='#1D9E75', lw=2.5, label=f'FL-LSTM (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    ax.set_title(f"{name} — ROC Curve (Federated)", fontsize=11)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/federated_lstm_results.png", dpi=150, bbox_inches='tight')
print("  Saved outputs/federated_lstm_results.png ✓")
plt.close()

print("\n" + "="*60)
print("  FEDERATED LSTM TRAINING COMPLETE")
print("="*60)

drone_fl_model.save("outputs/drone_lstm_federated.keras")
kdd_fl_model.save("outputs/kdd_lstm_federated.keras")
print("FL models saved")
