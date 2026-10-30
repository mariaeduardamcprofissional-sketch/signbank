"""1-NN com DTW para avaliacao final: top-1 / top-5 / top-10, um articulador de fora por rodada.

Diferencas para o diagnostico_dtw.py (feito para investigar, nao para escalar):
  - calcula so teste x treino (e nao todas contra todas), o que e ~2x mais barato;
  - --passo_tempo 2 usa so 1 de cada 2 quadros e reduz a banda na mesma proporcao: ~4x mais rapido;
  - com --espelho, imprime na MESMA execucao o resultado sem e com espelho (o espelho e aplicado
    as amostras de teste; como a distancia e uma isometria, equivale a espelhar o treino);
  - --so_treino IDS mantem grupos (ex. 99 = SignBank) sempre na galeria de treino;
  - --salvar_dist grava a menor distancia de cada amostra de teste a cada classe, para depois
    combinar com as probabilidades da LSTM (--salvar_proba no train_lstm_hands_face.py).

Uso (mesma pasta do knn_dtw_libras.py, do train_lstm_hands_face.py e da pasta libras):
    python knn_dtw_topk.py --data landmarks_vlibrasil_signbank_slots.npz \\
        --signer_regex ".*Articulador(\\d+)" --so_treino 99 --w_traj 1 --penalidade 0.2 \\
        --espelho --passo_tempo 2 --salvar_dist dist_com_sb.npz

Dica: antes de uma rodada longa, valide --passo_tempo 2 em 100 classes e compare com --passo_tempo 1.
"""
import argparse
import re
import time

import numpy as np

from knn_dtw_libras import matriz_dtw, TEM_NUMBA
from libras.training_utils import load_class_filter, extract_signer_ids
from train_lstm_hands_face import load_data, build_points


def dist_por_classe(D, y_treino, n_classes):
    """D: (n_teste, n_treino); y_treino: inteiros. Devolve (n_teste, n_classes) com a menor
    distancia a cada classe (inf se a classe nao tem amostra no treino)."""
    dist = np.full((D.shape[0], n_classes), np.inf)
    ordem = np.argsort(y_treino, kind="stable")
    ys = y_treino[ordem]
    inicios = np.r_[0, np.where(np.diff(ys) != 0)[0] + 1]
    dist[:, ys[inicios]] = np.minimum.reduceat(D[:, ordem], inicios, axis=1)
    return dist


def ranks(dist, y_teste):
    verdadeira = dist[np.arange(len(y_teste)), y_teste]
    return 1 + (dist < verdadeira[:, None]).sum(axis=1)


def resumo(r):
    return f"top1={np.mean(r <= 1):.4f}  top5={np.mean(r <= 5):.4f}  top10={np.mean(r <= 10):.4f}  rank mediano={int(np.median(r))}"


def main():
    parser = argparse.ArgumentParser(description="1-NN com DTW (avaliacao final, top-k).")
    parser.add_argument("--data", required=True)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)")
    parser.add_argument("--so_treino", default=None, help="IDs de sinalizador que so entram no treino (ex.: 99)")
    parser.add_argument("--w_hands", type=float, default=1.0)
    parser.add_argument("--w_face", type=float, default=0.0)
    parser.add_argument("--w_traj", type=float, default=1.0)
    parser.add_argument("--banda", type=int, default=10, help="Faixa de Sakoe-Chiba em quadros do video com 30 quadros")
    parser.add_argument("--penalidade", type=float, default=0.2)
    parser.add_argument("--passo_tempo", type=int, default=1, help="Usa 1 de cada N quadros (a banda e reduzida na mesma proporcao)")
    parser.add_argument("--sem_z", action="store_true")
    parser.add_argument("--espelho", action="store_true", help="Tambem calcula com distancia invariante ao espelhamento horizontal (so sem rosto)")
    parser.add_argument("--salvar_dist", default=None, help=".npz de saida com as distancias por classe de cada amostra de teste")
    args = parser.parse_args()

    if not TEM_NUMBA:
        print("Aviso: numba nao esta instalado; vai ficar muito lento (pip install numba).")
    if args.espelho and args.w_face > 0:
        raise SystemExit("--espelho so esta implementado sem rosto (use --w_face 0).")

    X_hands, X_face, X_traj, y_raw, paths = load_data(args.data)
    if paths is None:
        raise SystemExit("O .npz nao tem 'paths'.")
    usa_rosto = args.w_face > 0 and X_face is not None
    usa_traj = args.w_traj > 0 and X_traj is not None
    n_rosto = X_face.shape[2] if usa_rosto else 0
    n_traj = X_traj.shape[2] if usa_traj else 0
    n_maos = X_hands.shape[2] * X_hands.shape[3]
    points = build_points(X_hands, X_face, X_traj, use_face=usa_rosto, use_traj=usa_traj)
    del X_hands, X_face, X_traj
    grupo = np.array([0] * n_maos + [1] * n_rosto + [2] * n_traj, dtype=np.int64)

    filtro = load_class_filter(args.classes)
    if filtro is not None:
        mask = np.array([r in filtro for r in y_raw])
        points, y_raw, paths = points[mask], y_raw[mask], paths[mask]
        print(f"Filtrando para {len(filtro)} classes pedidas: {len(y_raw)} amostras restantes.")

    sig = np.array([str(s) for s in extract_signer_ids(paths, re.compile(args.signer_regex)).tolist()])
    y_str = np.array([str(v) for v in y_raw])
    classes = np.array(sorted(set(y_str.tolist())))
    indice = {c: i for i, c in enumerate(classes.tolist())}
    y = np.array([indice[v] for v in y_str], dtype=np.int64)
    n_classes = len(classes)
    sinalizadores = sorted(set(sig.tolist()))
    extras = set(x.strip() for x in args.so_treino.split(",") if x.strip()) if args.so_treino else set()
    avaliacao = [s for s in sinalizadores if s not in extras]
    if not avaliacao:
        raise SystemExit("Todos os sinalizadores estao em --so_treino.")
    print(f"Sinalizadores: {sinalizadores} | so treino: {sorted(extras)} | classes: {n_classes} | amostras: {len(y)}")

    points = np.ascontiguousarray(points, dtype=np.float32)
    if args.sem_z:
        points[..., 2] = 0.0
    if args.passo_tempo > 1:
        points = np.ascontiguousarray(points[:, ::args.passo_tempo])
    banda = max(1, int(round(args.banda / args.passo_tempo)))
    presente = np.ascontiguousarray(points.any(axis=-1))
    peso = np.array([args.w_hands, args.w_face if usa_rosto else 0.0, args.w_traj if usa_traj else 0.0], dtype=np.float64)
    print(f"Quadros por amostra: {points.shape[1]} | banda efetiva: {banda} | penalidade: {args.penalidade} | "
          f"pesos maos/rosto/traj = {peso[0]}/{peso[1]}/{peso[2]} | espelho: {args.espelho}", flush=True)

    espelhado = None
    if args.espelho:
        espelhado = np.ascontiguousarray(points.copy())
        espelhado[..., 0] *= -1.0

    r_sem, r_com, salvar = [], [], {}
    t_ini = time.time()
    pares_feitos = 0
    pares_total = sum(int((sig == s).sum()) * int((sig != s).sum()) for s in avaliacao)
    for s in avaliacao:
        te = np.where(sig == s)[0]
        tr = np.where(sig != s)[0]
        t0 = time.time()
        D = np.asarray(matriz_dtw(points[te], presente[te], points[tr], presente[tr], grupo, peso,
                                  args.penalidade, banda), dtype=np.float64)
        dist_sem = dist_por_classe(D, y[tr], n_classes)
        rs = ranks(dist_sem, y[te])
        r_sem.append(rs)
        msg = f"  sinalizador {s}: n_teste={len(te)} | sem espelho: {resumo(rs)}"
        dist_final = dist_sem
        if args.espelho:
            Dm = np.asarray(matriz_dtw(espelhado[te], presente[te], points[tr], presente[tr], grupo, peso,
                                       args.penalidade, banda), dtype=np.float64)
            dist_final = dist_por_classe(np.minimum(D, Dm), y[tr], n_classes)
            rc = ranks(dist_final, y[te])
            r_com.append(rc)
            msg += f"\n  {'':>14} {'':>13}| com espelho: {resumo(rc)}"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
        pares_feitos += len(te) * len(tr)
        if args.salvar_dist:
            salvar[f"dist_{s}"] = dist_final.astype(np.float32)
            salvar[f"y_{s}"] = y[te]
            salvar[f"paths_{s}"] = np.array([str(p) for p in paths[te]])
        restante = (time.time() - t_ini) * (pares_total - pares_feitos) / max(pares_feitos, 1)
        if pares_feitos < pares_total:
            print(f"  ... tempo estimado para as rodadas restantes: {restante / 60:.0f} min", flush=True)

    print(f"\nRESULTADO (media sobre todas as amostras de teste, {sum(len(r) for r in r_sem)} amostras; "
          f"acaso: top1={1 / n_classes:.4f} top5={5 / n_classes:.4f} top10={10 / n_classes:.4f})")
    print(f"  sem espelho: {resumo(np.concatenate(r_sem))}")
    if args.espelho:
        print(f"  com espelho: {resumo(np.concatenate(r_com))}")
    if args.salvar_dist:
        salvar["classes"] = classes
        salvar["avaliacao"] = np.array(avaliacao)
        np.savez_compressed(args.salvar_dist, **salvar)
        print(f"Distancias por classe salvas em {args.salvar_dist}")


if __name__ == "__main__":
    main()
