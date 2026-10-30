import re
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

np.random.seed(0)
tf.random.set_seed(0)

PASSOS = 1500      # passos de treino por fold (suba p/ 3000 se estiver melhorando)
P = 64             # classes por batch (batch = 2*P amostras)
EMB = 128
TEMP = 0.1
PESO_TRAJ = 3.0

d = np.load("landmarks_vlibrasil_trim_signbank_slots.npz", allow_pickle=True)
y = d["y"].astype(str)
s = np.array([int(re.search(r"Articulador(\d+)", p).group(1)) for p in d["paths"]])
origem = d["origem"]
classes, yi = np.unique(y, return_inverse=True)

# --- check: frames sem mão detectada (landmarks zerados) ---
vazio = np.abs(d["X_hands"]).sum(axis=(3, 4)) == 0      # (N, 30, 2)
for o in np.unique(origem):
    print(f"origem {o}: frames sem mão (slot 0 / slot 1): "
          f"{vazio[origem == o][:, :, 0].mean():.2f} / {vazio[origem == o][:, :, 1].mean():.2f}")
print("amostras com TODOS os frames sem mão:", int(vazio.all(axis=(1, 2)).sum()))

# --- features por frame: mãos relativas ao punho, escala pela mão, + trajetória ---
h = d["X_hands"].astype(np.float32)
h = h - h[:, :, :, 0:1, :]
tam = np.linalg.norm(h[:, :, :, 9, :2], axis=-1).mean(axis=1, keepdims=True)[..., None, None]
tam[tam < 1e-6] = 1.0
h = h / tam
X = np.concatenate([
    h.reshape(len(h), 30, -1),
    PESO_TRAJ * d["X_traj"].astype(np.float32).reshape(len(h), 30, -1),
], axis=2)

acertos = {1: 0, 5: 0, 10: 0}

for teste in np.unique(s):
    te, tr = s == teste, s != teste
    idx_tr = np.where(tr)[0]

    # padroniza só com estatísticas do treino
    mu = X[tr].mean(axis=(0, 1))
    sd = X[tr].std(axis=(0, 1)) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    por_classe = {}
    for i in idx_tr:
        por_classe.setdefault(yi[i], []).append(i)
    por_classe = {c: v for c, v in por_classe.items() if len(v) >= 2}
    cls = np.array(list(por_classe))

    model = tf.keras.Sequential([
        tf.keras.layers.Input((30, X.shape[2])),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Dense(EMB),
    ])
    opt = tf.keras.optimizers.Adam(1e-3)

    for passo in range(PASSOS):
        batch = []
        for c in np.random.choice(cls, P, replace=False):
            a, b = np.random.choice(por_classe[c], 2, replace=False)
            batch += [a, b]
        xb = Xn[batch] + np.random.normal(0, 0.05, (len(batch), 30, X.shape[2])).astype(np.float32)
        lb = yi[batch]

        with tf.GradientTape() as tape:
            e = tf.math.l2_normalize(model(xb, training=True), axis=1)
            n = len(batch)
            eye = tf.eye(n)
            sim = tf.matmul(e, e, transpose_b=True) / TEMP - 1e9 * eye
            pos = tf.cast(lb[:, None] == lb[None, :], tf.float32) - eye
            logp = tf.nn.log_softmax(sim, axis=1)
            loss = -tf.reduce_sum(pos * logp) / tf.reduce_sum(pos)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if passo % 250 == 0:
            print(f"  signer {teste} passo {passo} loss {float(loss):.3f}")

    # --- avaliação: embeddings do teste vs. referências dos outros sinalizadores ---
    emb = tf.math.l2_normalize(model.predict(Xn, batch_size=512, verbose=0), axis=1).numpy()
    D = cdist(emb[te], emb[tr], "cosine")
    ytr = yi[tr]
    ok = {1: 0, 5: 0, 10: 0}
    for i, yt in enumerate(yi[te]):
        ordem = ytr[np.argsort(D[i])]
        _, first = np.unique(ordem, return_index=True)
        rank = ordem[np.sort(first)][:10]
        for k in ok:
            ok[k] += int(yt in rank[:k])
    n_te = te.sum()
    print(f"Articulador{teste}: " + " | ".join(f"Top-{k} {ok[k] / n_te:.3f}" for k in ok))
    for k in ok:
        acertos[k] += ok[k]

print("TOTAL: " + " | ".join(f"Top-{k} {acertos[k] / len(y):.3f}" for k in acertos))