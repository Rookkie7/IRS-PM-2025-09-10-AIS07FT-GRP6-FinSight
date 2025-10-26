#!/usr/bin/env python3
import os, io, json, time, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Use all CPUs on the vnode (36 cores typical on Vanda vnodes)
tf.config.threading.set_intra_op_parallelism_threads(36)
tf.config.threading.set_inter_op_parallelism_threads(2)

# GPU config (A40 on Vanda)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU mem growth error:", e)

# Paths
JOBID = os.environ.get("PBS_JOBID", "local")
SCRATCH = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER','user')}")
OUTDIR = os.environ.get("OUTDIR", os.path.join(SCRATCH, "tf_smoke", JOBID))
os.makedirs(OUTDIR, exist_ok=True)

# Synthetic data (no downloads)
rng = np.random.default_rng(42)
X = rng.normal(size=(8000, 32)).astype("float32")
y = rng.integers(0, 4, size=(8000,)).astype("int32")

ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(8000).batch(256).prefetch(tf.data.AUTOTUNE)

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

t0 = time.time()
hist = model.fit(ds, epochs=3, verbose=2)
t1 = time.time()

# Save model + history
model.save(os.path.join(OUTDIR, "model"))
with open(os.path.join(OUTDIR, "history.json"), "w") as f:
    json.dump(hist.history, f)

# Plot
plt.figure()
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["accuracy"], label="acc")
plt.legend(); plt.title("tf_smoke")
plt.xlabel("epoch"); plt.ylabel("value")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "training.png"), dpi=140)

# System info
info = {
    "hostname": platform.node(),
    "tf_version": tf.__version__,
    "gpus": [d.name for d in gpus],
    "jobid": JOBID,
    "elapsed_s": round(t1 - t0, 2),
    "outdir": OUTDIR,
}
with open(os.path.join(OUTDIR, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)

# Also write a small text summary
sio = io.StringIO()
model.summary(print_fn=lambda s: sio.write(s + "\n"))
with open(os.path.join(OUTDIR, "model_summary.txt"), "w") as f:
    f.write(sio.getvalue())

print("DONE. Outdir:", OUTDIR)