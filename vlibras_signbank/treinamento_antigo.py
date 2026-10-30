import argparse
import json
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from libras.training_utils import load_class_filter, extract_signer_ids, log_counts, save_counts_csv


# ---------------- Carga e normalizacao ----------------

def normalize_face(X_face):
    """Plano B, usado so se o .npz nao tiver X_face_norm: centra cada sequencia no centro
    do rosto e divide pelo tamanho do rosto. Frames sem rosto (tudo zero) continuam zero."""
    out = np.zeros_like(X_face)
    for i, seq in enumerate(X_face):
        valid = seq.reshape(len(seq), -1).any(axis=1)
        if not valid.any():
            continue
        pts = seq[valid]
        center = pts.mean(axis=(0, 1))
        scale = np.linalg.norm(pts[..., :2] - center[:2], axis=-1).mean() + 1e-6
        out[i][valid] = (pts - center) / scale
    return out


def load_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]
    paths = data["paths"] if "paths" in data else None
    if "X_hands" in data:
        X_hands = data["X_hands"]
    elif "X" in data:
        X_hands = data["X"]  # formato antigo (so maos)
    else:
        raise KeyError(f"{data_path} nao tem 'X_hands' nem 'X' - formato de .npz nao reconhecido.")
    N, T = X_hands.shape[:2]
    X_face = None
    if "X_face_norm" in data and "X_face" in data:
        # rosto ja normalizado pelo extrator; zera de novo os frames sem rosto detectado
        detected = data["X_face"].reshape(N, T, -1).any(axis=-1)
        X_face = data["X_face_norm"] * detected[..., None, None]
    elif "X_face" in data:
        X_face = normalize_face(data["X_face"])
    X_traj = data["X_traj"] if "X_traj" in data else None  # (N, T, 2, 2) punho relativo aos ombros
    return X_hands, X_face, X_traj, y_raw, paths


def build_points(X_hands, X_face, X_traj, use_face, use_traj):
    """Junta tudo como pontos 3D: (N, T, K, 3). K = pontos das maos + rosto (+ 2 punhos)."""
    N, T = X_hands.shape[:2]
    parts = [X_hands.reshape(N, T, -1, 3)]
    if use_face and X_face is not None:
        parts.append(X_face)
    elif use_face:
        print("Rosto nao encontrado no .npz - usando so maos.")
    if use_traj and X_traj is not None:
        traj = np.zeros(X_traj.shape[:3] + (3,), dtype=np.float32)
        traj[..., :2] = X_traj
        parts.append(traj)
        print("Usando X_traj (posicao do punho em relacao aos ombros).")
    elif use_traj:
        print("'X_traj' nao encontrado no .npz - sem a posicao do punho.")
    points = np.concatenate(parts, axis=2).astype(np.float32)
    print(f"Pontos por frame: {points.shape[2]}")
    return points


# ---------------- Augmentation e velocidade ----------------

def augment(seq, rng):
    """seq: (T, K, 3). Pontos zerados (nao detectados) continuam zerados."""
    T = len(seq)
    start, end = rng.uniform(0, 0.15), rng.uniform(0.85, 1.0)
    idx = np.linspace(start * (T - 1), end * (T - 1), T).round().astype(int)
    seq = seq[idx]
    present = seq.any(axis=-1, keepdims=True)
    seq = seq * rng.uniform(0.9, 1.1)  # escala
    angle = np.deg2rad(rng.uniform(-10, 10))  # rotacao pequena no plano xy
    c, s = np.cos(angle), np.sin(angle)
    seq[..., :2] = seq[..., :2] @ np.array([[c, s], [-s, c]])
    seq = seq + rng.normal(0, 0.01, seq.shape) * present  # ruido
    return seq.astype(np.float32)


def augment_dataset(points, y, copies, rng):
    if copies == 0:
        return points, y
    aug = [np.stack([augment(seq, rng) for seq in points]) for _ in range(copies)]
    return np.concatenate([points] + aug), np.concatenate([y] * (copies + 1))


def to_features(points, use_velocity):
    """(N, T, K, 3) -> (N, T, K*3) ou (N, T, K*6) com velocidade (diferenca entre frames)."""
    N, T = points.shape[:2]
    if use_velocity:
        present = points.any(axis=-1, keepdims=True)
        valid = present[:, 1:] & present[:, :-1]  # so onde o ponto existe nos 2 frames
        vel = np.zeros_like(points)
        vel[:, 1:] = (points[:, 1:] - points[:, :-1]) * valid
        points = np.concatenate([points, vel], axis=-1)
    return points.reshape(N, T, -1)


# ---------------- Modelo ----------------

def build_model(timesteps, num_features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, num_features)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_class_weights_dict(y_train):
    """Calcula pesos de classe para equilibrar classes com poucas amostras."""
    classes_unicas = np.unique(y_train)
    pesos = compute_class_weight(
        class_weight="balanced",
        classes=classes_unicas,
        y=y_train
    )
    return dict(zip(classes_unicas, pesos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Arquivo .npz gerado pelo extrator")
    parser.add_argument("--output", default="modelo_palavras.keras", help="Caminho para salvar o modelo treinado")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.3, help="Fracao de sinalizadores para teste")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fracao para validacao.")
    parser.add_argument("--no_face", action="store_true", help="Ignora os landmarks faciais.")
    parser.add_argument("--no_traj", action="store_true", help="Ignora X_traj.")
    parser.add_argument("--aug_copies", type=int, default=4, help="Copias aumentadas por amostra de treino.")
    parser.add_argument("--no_velocity", action="store_true", help="Nao adiciona features de velocidade.")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario.")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)", help="Regex para extrair ID do sinalizador.")
    parser.add_argument("--so_treino", default=None, help="IDs de sinalizador que ficam SEMPRE no treino.")
    parser.add_argument("--leave_one_signer_out", action="store_true", help="Validacao cruzada deixando 1 sinalizador de fora por rodada.")
    parser.add_argument("--loso_epochs", type=int, default=40, help="Epocas por rodada no LOSO.")
    parser.add_argument("--salvar_proba", default=None, help="Arquivo para salvar probabilidades por classe no LOSO.")
    parser.add_argument("--treinar_tudo", action="store_true", help="Treina com TODAS as amostras sem teste/validacao.")
    parser.add_argument("--seed", type=int, default=42, help="Semente do treino.")
    parser.add_argument("--counts_output", default=None, help="Caminho .csv para salvar contagens de amostras.")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    use_velocity = not args.no_velocity

    X_hands, X_face, X_traj, y_raw, paths = load_data(args.data)
    points = build_points(X_hands, X_face, X_traj, use_face=not args.no_face, use_traj=not args.no_traj)
    config = {
        "use_face": (not args.no_face) and X_face is not None,
        "use_traj": (not args.no_traj) and X_traj is not None,
        "use_velocity": use_velocity,
    }
    del X_hands, X_face, X_traj

    class_filter = load_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        points, y_raw = points[mask], y_raw[mask]
        if paths is not None:
            paths = paths[mask]
        print(f"Filtrando para {len(class_filter)} classes pedidas: {len(y_raw)} amostras restantes "
              f"({len(set(y_raw.tolist()))} classes de fato encontradas nos dados).")

    if paths is None:
        raise SystemExit("O .npz nao tem 'paths' - nao da pra identificar o sinalizador pra fazer o split por grupo.")

    signer_regex = re.compile(args.signer_regex)
    signer_ids = extract_signer_ids(paths, signer_regex)
    print(f"Sinalizadores encontrados ({len(set(signer_ids.tolist()))}): {sorted(set(signer_ids.tolist()))}")

    so_treino = set(s.strip() for s in args.so_treino.split(",") if s.strip()) if args.so_treino else set()
    is_extra = np.array([str(s) in so_treino for s in signer_ids.tolist()], dtype=bool)
    if so_treino:
        encontrados = sorted(set(str(s) for s in signer_ids[is_extra].tolist()))
        if not is_extra.any():
            raise SystemExit(f"--so_treino {sorted(so_treino)}: nenhuma amostra tem esses IDs de sinalizador.")
        print(f"Grupos so de treino {encontrados}: {int(is_extra.sum())} amostras.")
        if is_extra.all():
            raise SystemExit("Todas as amostras estao em --so_treino; sobra nada para avaliar.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    counts_by_split = {}

    if args.treinar_tudo:
        print(f"\n=== Treino final com TODAS as amostras ({len(y)} amostras, {num_classes} classes), "
              f"{args.loso_epochs} epocas, sem teste ===")
        log_counts("treino_total", y_raw, counts_by_split)
        p_all, y_all = augment_dataset(points, y, args.aug_copies, rng)
        X_all = to_features(p_all, use_velocity)
        
        class_weights_dict = get_class_weights_dict(y_all)
        model = build_model(timesteps=X_all.shape[1], num_features=X_all.shape[2], num_classes=num_classes)
        model.summary()
        model.fit(
            X_all, y_all, 
            epochs=args.loso_epochs, 
            batch_size=args.batch_size, 
            verbose=2,
            class_weight=class_weights_dict
        )
        _, acc_treino = model.evaluate(to_features(points, use_velocity), y, verbose=0)
        print(f"\nAcuracia no proprio treino (sem augmentation): {acc_treino:.4f}")
        model.save(args.output)
        labels_path = os.path.splitext(args.output)[0] + "_labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(list(encoder.classes_), f, ensure_ascii=False)
        config_path = os.path.splitext(args.output)[0] + "_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        if args.counts_output:
            save_counts_csv(args.counts_output, counts_by_split)
        print(f"Modelo salvo em {args.output}")
        return

    if args.leave_one_signer_out:
        if so_treino:
            avaliacao = sorted(set(str(s) for s in signer_ids[~is_extra].tolist()))
            folds = [(np.where(signer_ids != s)[0], np.where(signer_ids == s)[0]) for s in avaliacao]
        else:
            unique_signers = sorted(set(signer_ids.tolist()))
            gkf = GroupKFold(n_splits=len(unique_signers))
            folds = list(gkf.split(points, y, groups=signer_ids))
        n_splits = len(folds)
        accs, accs5, accs10 = [], [], []
        proba_salva = {}
        for fold, (train_idx, test_idx) in enumerate(folds):
            left_out = sorted(set(signer_ids[test_idx].tolist()))
            print(f"\n=== Rodada {fold + 1}/{n_splits} - sinalizador(es) de fora: {left_out} ===")
            log_counts(f"rodada{fold + 1}_treino", y_raw[train_idx], counts_by_split)
            log_counts(f"rodada{fold + 1}_teste", y_raw[test_idx], counts_by_split)
            p_train, y_train = augment_dataset(points[train_idx], y[train_idx], args.aug_copies, rng)
            X_train = to_features(p_train, use_velocity)
            X_test = to_features(points[test_idx], use_velocity)
            
            class_weights_dict = get_class_weights_dict(y_train)
            model = build_model(timesteps=X_train.shape[1], num_features=X_train.shape[2], num_classes=num_classes)
            model.fit(
                X_train, y_train, 
                epochs=args.loso_epochs, 
                batch_size=args.batch_size, 
                verbose=0,
                class_weight=class_weights_dict
            )
            test_loss, test_acc = model.evaluate(X_test, y[test_idx], verbose=0)
            proba = np.asarray(model.predict(X_test, verbose=0))
            verd = proba[np.arange(len(test_idx)), y[test_idx]]
            rank = 1 + (proba > verd[:, None]).sum(axis=1)
            top5, top10 = float(np.mean(rank <= 5)), float(np.mean(rank <= 10))
            print(f"Acuracia (sinalizador de fora {left_out}): {test_acc:.4f} | top-5: {top5:.4f} | top-10: {top10:.4f}")
            accs.append(test_acc)
            accs5.append(top5)
            accs10.append(top10)
            if args.salvar_proba:
                chave = "_".join(str(x) for x in left_out)
                proba_salva[f"proba_{chave}"] = proba.astype(np.float32)
                proba_salva[f"y_{chave}"] = y[test_idx]
                proba_salva[f"paths_{chave}"] = np.array([str(p) for p in paths[test_idx]])
        print(f"\nAcuracia media entre rodadas: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
        print(f"Top-5 medio: {np.mean(accs5):.4f} (+/- {np.std(accs5):.4f}) | Top-10 medio: {np.mean(accs10):.4f} (+/- {np.std(accs10):.4f}) "
              f"| acaso: top-1 {1 / num_classes:.4f}, top-5 {5 / num_classes:.4f}, top-10 {10 / num_classes:.4f}")
        if args.salvar_proba:
            proba_salva["classes"] = np.array(encoder.classes_)
            np.savez_compressed(args.salvar_proba, **proba_salva)
            print(f"Probabilidades salvas em {args.salvar_proba}")
        if args.counts_output:
            save_counts_csv(args.counts_output, counts_by_split)
        return

    idx_ev = np.where(~is_extra)[0]
    idx_extra = np.where(is_extra)[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    tr_rel, te_rel = next(splitter.split(points[idx_ev], y[idx_ev], groups=signer_ids[idx_ev]))
    train_idx = np.concatenate([idx_ev[tr_rel], idx_extra])
    test_idx = idx_ev[te_rel]
    print(f"Sinalizadores no teste: {sorted(set(signer_ids[test_idx].tolist()))}")

    p_train, p_test = points[train_idx], points[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    signers_train = signer_ids[train_idx]
    extra_train = is_extra[train_idx]

    log_counts("teste", y_raw[test_idx], counts_by_split)

    callbacks = []
    validation_data = None
    if args.val_split > 0:
        pos_ev = np.where(~extra_train)[0]
        pos_extra = np.where(extra_train)[0]
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
        sub_tr_rel, sub_val_rel = next(val_splitter.split(p_train[pos_ev], y_train[pos_ev], groups=signers_train[pos_ev]))
        sub_train_idx = np.concatenate([pos_ev[sub_tr_rel], pos_extra])
        sub_val_idx = pos_ev[sub_val_rel]
        p_val, y_val = p_train[sub_val_idx], y_train[sub_val_idx]
        y_raw_val = y_raw[train_idx][sub_val_idx]
        y_raw_treino = y_raw[train_idx][sub_train_idx]
        p_train, y_train = p_train[sub_train_idx], y_train[sub_train_idx]
        validation_data = (to_features(p_val, use_velocity), y_val)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        )
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        )
        log_counts("validacao", y_raw_val, counts_by_split)
        log_counts("treino", y_raw_treino, counts_by_split)
    else:
        log_counts("treino", y_raw[train_idx], counts_by_split)
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        )

    n_train_orig = len(p_train)
    p_train, y_train = augment_dataset(p_train, y_train, args.aug_copies, rng)
    X_train = to_features(p_train, use_velocity)
    X_test = to_features(p_test, use_velocity)

    print(f"\nClasses: {num_classes}")
    print(f"Treino: {n_train_orig} (com augmentation: {len(X_train)}) | "
          f"Validacao: {len(validation_data[0]) if validation_data else 0} | Teste: {len(X_test)}")
    print(f"Features por frame: {X_train.shape[2]}")

    class_weights_dict = get_class_weights_dict(y_train)

    model = build_model(timesteps=X_train.shape[1], num_features=X_train.shape[2], num_classes=num_classes)
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights_dict,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcuracia no teste: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nRelatorio de classificacao:")
    print(classification_report(
        y_test, y_pred, labels=np.arange(num_classes), target_names=encoder.classes_, zero_division=0
    ))

    model.save(args.output)
    labels_path = os.path.splitext(args.output)[0] + "_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(list(encoder.classes_), f, ensure_ascii=False)
    config_path = os.path.splitext(args.output)[0] + "_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    if args.counts_output:
        save_counts_csv(args.counts_output, counts_by_split)
    print(f"Modelo salvo em {args.output}")
    print(f"Labels salvas em {labels_path}")
    print(f"Representacao salva em {config_path}")


if __name__ == "__main__":
    main()