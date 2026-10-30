"""Roda o train_lstm_hands_face.py com outro modelo e/ou outro aumento de dados, sem editar o arquivo.

Opcoes (antes do '--'):
  --modelo M   lstm (padrao do script) ou gru_small (GRU bidirecional pequena, como na sugestao da Maria:
               Masking -> BiGRU(32) -> Dropout 0.4 -> Dense 64 -> Dropout 0.4 -> softmax)
  --aug A      padrao (o do script: recorte de tempo, escala, rotacao, ruido) ou forte (velocidade que varia
               ao longo do sinal, perda de quadros, escala por eixo, rotacao maior, cisalhamento, deslocamento
               da trajetoria do punho, ruido)
  --seed N     SO para versoes antigas do script, que fixam a semente 42 por dentro. Na versao atual
               (que tem --seed) passe a semente depois do '--', junto dos outros argumentos.
O que vem depois do '--' vai direto para o train_lstm_hands_face.py.

Exemplo:
    python rodar_variante.py --modelo gru_small --aug forte -- --data x.npz --leave_one_signer_out \\
        --classes "@palavras100.txt" --no_face --aug_copies 10 --loso_epochs 40 --seed 1
"""
import argparse
import sys

import numpy as np
import tensorflow as tf


def build_model_gru_small(timesteps, num_features, num_classes):
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(timesteps, num_features)),
        L.Masking(mask_value=0.0),
        L.Bidirectional(L.GRU(32)),
        L.Dropout(0.4),
        L.Dense(64, activation="relu"),
        L.Dropout(0.4),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fazer_augment_forte(tem_traj):
    """seq: (T, K, 3). A trajetoria do punho, quando existe, e sempre o ultimo par de pontos."""

    def augment_forte(seq, rng):
        T, K = seq.shape[0], seq.shape[1]
        traj_slice = slice(K - 2, K) if tem_traj else None

        # 1. tempo: recorte aleatorio + velocidade que varia ao longo do sinal
        inicio, fim = rng.uniform(0.0, 0.2), rng.uniform(0.8, 1.0)
        passos = np.cumsum(rng.uniform(0.6, 1.4, T))
        passos = (passos - passos[0]) / (passos[-1] - passos[0])
        idx = np.clip(np.round((inicio + passos * (fim - inicio)) * (T - 1)).astype(int), 0, T - 1)
        seq = seq[idx].copy()
        presente = seq.any(axis=-1, keepdims=True)

        # 2. perda de quadros inteiros (ate 10%), como se o detector falhasse
        perdidos = rng.random(T) < rng.uniform(0.0, 0.1)
        seq[perdidos] = 0.0
        presente = presente & ~perdidos[:, None, None]

        # 3. geometria no plano xy: escala global, escala por eixo, rotacao, cisalhamento
        escala = rng.uniform(0.85, 1.15)
        sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
        angulo = np.deg2rad(rng.uniform(-15, 15))
        c, s = np.cos(angulo), np.sin(angulo)
        cis = rng.uniform(-0.08, 0.08)
        rot = np.array([[c, s], [-s, c]])
        cisalha = np.array([[1.0, 0.0], [cis, 1.0]])
        seq[..., :2] = seq[..., :2] @ (np.diag([sx, sy]) @ rot @ cisalha)
        seq = seq * escala

        # 4. trajetoria do punho: pequeno deslocamento em relacao aos ombros
        if traj_slice is not None:
            desloc = rng.uniform(-0.05, 0.05, size=2)
            seq[:, traj_slice, :2] += desloc * presente[:, traj_slice]

        # 5. ruido so nos pontos presentes
        seq = seq + rng.normal(0, 0.01, seq.shape) * presente
        return (seq * presente).astype(np.float32)

    return augment_forte


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    sep = sys.argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--modelo", choices=["lstm", "gru_small"], default="lstm")
    p.add_argument("--aug", choices=["padrao", "forte"], default="padrao")
    p.add_argument("--script", default="train_lstm_hands_face")
    a = p.parse_args(sys.argv[1:sep])
    resto = sys.argv[sep + 1:]

    import importlib
    t = importlib.import_module(a.script)

    if a.seed is not None:
        original = tf.keras.utils.set_random_seed

        def trocar(_ignorada):
            print(f"[rodar_variante] semente {a.seed} no lugar da fixa do script", flush=True)
            original(a.seed)

        tf.keras.utils.set_random_seed = trocar
    if a.modelo == "gru_small":
        t.build_model = build_model_gru_small
        print("[rodar_variante] modelo: BiGRU pequena no lugar da LSTM do script", flush=True)
    if a.aug == "forte":
        tem_traj = "--no_traj" not in resto
        t.augment = fazer_augment_forte(tem_traj)
        print(f"[rodar_variante] aumento de dados: forte (trajetoria {'sim' if tem_traj else 'nao'})", flush=True)

    sys.argv = [a.script + ".py"] + resto
    t.main()


if __name__ == "__main__":
    main()
