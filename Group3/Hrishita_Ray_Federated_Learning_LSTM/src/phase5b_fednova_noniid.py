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
    confusion_matrix, roc_curve, auc, mean_squared_error
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)
print("LOAD, SCALE, SPLIT")

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
print(f"  Drone   — train: {dX_tr.shape}  val: {dX_v.shape}")
print(f"  NSL-KDD — train: {kX_tr.shape}  val: {kX_v.shape}")

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
    ], name="LSTM_IDS_FedNova")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='binary_crossentropy', metrics=['accuracy'])
    return model

def class_weights(y):
    n0, n1 = (y==0).sum(), (y==1).sum(); total = n0+n1
    return {0: total/(2*n0+1e-9), 1: total/(2*n1+1e-9)}

def partition_iid(X, y, n_clients):
    idx = np.random.permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]

def partition_non_iid(X, y, n_clients):
    idx = np.argsort(y)
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]

N_CLIENTS  = 5
N_ROUNDS   = 20   
print("\nFEDNOVA TRAINING")

CLIENT_LOCAL_EPOCHS = [1, 1, 1, 1, 10]   

def fednova_local_update(model, X, y, epochs, batch_size, cw):
    """Standard local SGD/Adam training"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    n = len(X)
    sw_map = np.array([cw[0], cw[1]], dtype=np.float32)
    steps = 0
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            xb, yb = X[b], y[b]
            sample_w = sw_map[yb.astype(int)]
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                bce = tf.keras.losses.binary_crossentropy(yb.reshape(-1, 1), preds)
                bce = tf.reduce_mean(bce * sample_w)
            grads = tape.gradient(bce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            steps += 1
    return model.get_weights(), steps

ETA = 0.5 

def fednova_aggregate(global_weights, client_weights, client_sizes, client_steps, eta=ETA):
    total_n = sum(client_sizes)
    p = [n/total_n for n in client_sizes]
    tau_eff = sum(pi*ti for pi, ti in zip(p, client_steps))
    new_weights = []
    for layer_idx, gw in enumerate(global_weights):
        d_sum = np.zeros_like(gw)
        for k in range(len(client_weights)):
            wk = client_weights[k][layer_idx]
            tau_k = client_steps[k]
            d_k = (gw - wk) / tau_k
            d_sum += p[k] * d_k
        new_weights.append(gw - eta * tau_eff * d_sum)
    return new_weights

def fednova_train(X_train, y_train, X_val, y_val, name):
    clients = partition_non_iid(X_train, y_train, N_CLIENTS)  
    global_model = build_lstm(n_feat)
    local_model  = build_lstm(n_feat)                  
    round_acc, round_loss = [], []
    print(f"\n  ── {name} (FedNova, local epochs={CLIENT_LOCAL_EPOCHS}) ──")
    print(f"  {'Round':>6}  {'Global Acc':>10}  {'Avg Client Loss':>15}")
    for rnd in range(1, N_ROUNDS+1):
        cw_list, sizes, steps_list, losses = [], [], [], []
        gweights = global_model.get_weights()
        for cid, (cx, cy) in enumerate(clients):
            local_model.set_weights(deepcopy(gweights))
            cw = class_weights(cy)
            ep = CLIENT_LOCAL_EPOCHS[cid % len(CLIENT_LOCAL_EPOCHS)]
            new_w, steps = fednova_local_update(local_model, cx, cy, ep, BATCH_SIZE, cw)
            cw_list.append(deepcopy(new_w)); sizes.append(len(cx)); steps_list.append(steps)
            local_model.set_weights(new_w)
            loss, _ = local_model.evaluate(cx, cy, verbose=0)
            losses.append(loss)
        global_model.set_weights(fednova_aggregate(gweights, cw_list, sizes, steps_list))
        val_loss, val_acc = global_model.evaluate(X_val, y_val, verbose=0)
        round_acc.append(val_acc); round_loss.append(np.mean(losses))
        print(f"  {rnd:>6}  {val_acc:>10.4f}  {np.mean(losses):>15.4f}")
    return global_model, round_acc, round_loss

drone_nova_model, drone_nova_acc, drone_nova_loss = fednova_train(dX_tr, dy_tr, dX_v, dy_v, "Drone")
kdd_nova_model,   kdd_nova_acc,   kdd_nova_loss   = fednova_train(kX_tr, ky_tr, kX_v, ky_v, "NSL-KDD")

print("\nEVALUATION")

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
    fp_r, tp_r, _ = roc_curve(y, yp)
    roc_auc = auc(fp_r, tp_r)
    print(f"  {name:<24} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} fpr={fpr_m:.4f} rmse={rmse:.4f} auc={roc_auc:.4f}")
    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec, fpr=fpr_m,
                rmse=rmse, auc=roc_auc, cm=cm, yprob=yp, fp_r=fp_r, tp_r=tp_r)

drone_nova_eval = full_eval(drone_nova_model, dX_v, dy_v, "Drone FedNova")
kdd_nova_eval   = full_eval(kdd_nova_model,   kX_v, ky_v, "NSL-KDD FedNova")

drone_nova_model.save("outputs/drone_lstm_fednova_noniid.keras")
kdd_nova_model.save("outputs/kdd_lstm_fednova_noniid.keras")

res_df = pd.DataFrame([
    ["Drone",   "FedNova", drone_nova_eval['accuracy'], drone_nova_eval['f1'], drone_nova_eval['precision'], drone_nova_eval['recall'], drone_nova_eval['fpr'], drone_nova_eval['rmse'], drone_nova_eval['auc']],
    ["NSL-KDD", "FedNova", kdd_nova_eval['accuracy'],   kdd_nova_eval['f1'],   kdd_nova_eval['precision'],   kdd_nova_eval['recall'],   kdd_nova_eval['fpr'],   kdd_nova_eval['rmse'],   kdd_nova_eval['auc']],
], columns=["Dataset","Method","Accuracy","F1","Precision","Recall","FPR","RMSE","AUC"])
res_df.to_csv("outputs/fednova_results_noniid.csv", index=False)
print("\nSaved outputs/fednova_results_noniid.csv")
print(res_df.to_string(index=False))
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Federated LSTM (FedNova, NON-IID clients) Intrusion Detection Results", fontsize=14, fontweight='bold')

for col, (name, acc_curve, loss_curve, Xv, yv, ev) in enumerate([
    ("Drone",   drone_nova_acc, drone_nova_loss, dX_v, dy_v, drone_nova_eval),
    ("NSL-KDD", kdd_nova_acc,   kdd_nova_loss,   kX_v, ky_v, kdd_nova_eval),
]):
    ax = axes[0][col]
    rounds = range(1, N_ROUNDS+1)
    ax.plot(rounds, acc_curve,  color='#7570B3', lw=2, marker='^', ms=4, label='Global accuracy')
    ax.plot(rounds, loss_curve, color='#1D9E75', lw=2, marker='s', ms=4, ls='--', label='Avg client loss')
    ax.set_title(f"{name} — FedNova Convergence over Rounds", fontsize=11)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][col]
    cm = ev['cm']
    ax.imshow(cm, cmap='Purples')
    ax.set_title(f"{name} — Confusion Matrix (FedNova)", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Attack"]); ax.set_yticklabels(["Normal","Attack"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[r,c]>cm.max()/2 else 'black')

    ax = axes[2][col]
    ax.plot(ev['fp_r'], ev['tp_r'], color='#7570B3', lw=2.5, label=f"FedNova (AUC={ev['auc']:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    ax.set_title(f"{name} — ROC Curve (FedNova)", fontsize=11)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/fednova_results_noniid.png", dpi=150, bbox_inches='tight')
print("Saved outputs/fednova_results_noniid.png")
plt.close()

print("\nFEDNOVA (NON-IID) TRAINING COMPLETE")
