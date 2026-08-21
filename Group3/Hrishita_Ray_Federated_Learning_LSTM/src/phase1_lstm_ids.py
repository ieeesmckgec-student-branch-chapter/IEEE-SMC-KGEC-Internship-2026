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
    confusion_matrix, roc_curve, auc, mean_squared_error,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)
print("\n" + "="*60)
print("LOAD, SCALE, SPLIT")
print("="*60)
 
print("\nLoading drone_clean.csv")
drone_df = pd.read_csv("drone_clean.csv")
print(f"  Shape  : {drone_df.shape}")
drone_X = drone_df.drop(columns=['label']).values.astype(np.float32)
drone_y = drone_df['label'].values.astype(np.float32)
print(f"  Normal : {(drone_y==0).sum()} ({(drone_y==0).mean()*100:.1f}%)  "
      f"Attack : {(drone_y==1).sum()} ({(drone_y==1).mean()*100:.1f}%)")

print("\nLoading Train_data.csv")
kdd_df = pd.read_csv("Train_data.csv")
print(f"  Shape  : {kdd_df.shape}")

le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    kdd_df[col] = le.fit_transform(kdd_df[col].astype(str))

kdd_X = kdd_df.drop(columns=['class']).values.astype(np.float32)
kdd_y = (kdd_df['class'] != 'normal').astype(np.float32).values
print(f"  Normal : {(kdd_y==0).sum()} ({(kdd_y==0).mean()*100:.1f}%)  "
      f"Attack : {(kdd_y==1).sum()} ({(kdd_y==1).mean()*100:.1f}%)")
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
print("  Preprocessing complete")

print("\n" + "="*60)
print("="*60)

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
    ], name="LSTM_IDS")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def class_weights(y):
    n0, n1 = (y==0).sum(), (y==1).sum(); total = n0+n1
    return {0: total/(2*n0+1e-9), 1: total/(2*n1+1e-9)}

def train_lstm(X_tr, y_tr, X_v, y_v, name):
    model = build_lstm(X_tr.shape[2])
    model.summary()
    cw = class_weights(y_tr)
    print(f"\n  Class weights: normal={cw[0]:.3f}  attack={cw[1]:.3f}")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        class_weight=cw,
        epochs=80, batch_size=64,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=4, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )
    return model, history

print("\nDrone")
drone_model, drone_hist = train_lstm(dX_tr, dy_tr, dX_v, dy_v, "Drone")

print("\nNSL-KDD")
kdd_model, kdd_hist = train_lstm(kX_tr, ky_tr, kX_v, ky_v, "NSL-KDD")

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

drone_eval = full_eval(drone_model, dX_v, dy_v, "Drone — Validation")
kdd_eval   = full_eval(kdd_model,   kX_v, ky_v, "NSL-KDD — Validation")

print("\n" + "="*60)
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Centralized LSTM — Intrusion Detection Results",
             fontsize=14, fontweight='bold')

for col, (name, hist, Xv, yv, ev) in enumerate([
    ("Drone",   drone_hist, dX_v, dy_v, drone_eval),
    ("NSL-KDD", kdd_hist,   kX_v, ky_v, kdd_eval),
]):
    ax = axes[0][col]
    ax.plot(hist.history['loss'],         color='#5B4CB5', lw=2,       label='Train loss')
    ax.plot(hist.history['val_loss'],     color='#5B4CB5', lw=2, ls='--', label='Val loss')
    ax.plot(hist.history['accuracy'],     color='#1D9E75', lw=2,       label='Train acc')
    ax.plot(hist.history['val_accuracy'], color='#1D9E75', lw=2, ls='--', label='Val acc')
    ax.set_title(f"{name} — Training Curves", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][col]
    cm = ev['cm']
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"{name} — Confusion Matrix", fontsize=11)
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
    ax.plot(fp_r, tp_r, color='#5B4CB5', lw=2.5, label=f'LSTM (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    ax.set_title(f"{name} — ROC Curve", fontsize=11)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/lstm_results.png", dpi=150, bbox_inches='tight')
print("Saved outputs/lstm_results.png")
plt.close()

fig2, ax2 = plt.subplots(figsize=(9, 5))
metrics_shown = ["accuracy","f1","precision","recall"]
x = np.arange(len(metrics_shown)); w = 0.35
ax2.bar(x-w/2, [drone_eval[m] for m in metrics_shown], w, label="Drone",   color="#5B4CB5", alpha=0.85)
ax2.bar(x+w/2, [kdd_eval[m]   for m in metrics_shown], w, label="NSL-KDD", color="#1D9E75", alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(metrics_shown, fontsize=10)
ax2.set_ylim(0, 1.12)
ax2.set_title("LSTM IDS — Metric Summary (Centralized Baseline)", fontsize=12)
ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/lstm_metrics.png", dpi=150, bbox_inches='tight')
print("Saved outputs/lstm_metrics.png")
plt.close()

print("\n" + "="*60)
print("  CENTRALIZED LSTM COMPLETE")
print("  Next: run phase2_federated_lstm.py")
print("="*60)
drone_model.save("outputs/drone_lstm_baseline.keras")
kdd_model.save("outputs/kdd_lstm_baseline.keras")
print("Models saved")
