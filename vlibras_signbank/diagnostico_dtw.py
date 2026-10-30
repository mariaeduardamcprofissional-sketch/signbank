"""Diagnostico da separabilidade das palavras com DTW (usa as funcoes do knn_dtw_libras.py).

Calcula a matriz DTW entre TODAS as amostras e responde:
  1. Acuracia top-1 / top-5 / top-10 deixando um sinalizador de fora (media e por rodada).
  2. Acuracia para cada par (sinalizador de treino -> sinalizador de teste): mostra se algum
     articulador destoa dos outros.
  3. Distancia media entre:
       A) mesma palavra, sinalizadores diferentes
       B) palavras diferentes, sinalizadores diferentes
       C) palavras diferentes, mesmo sinalizador
     Se A nao for claramente menor que B, o DTW nao separa as palavras. Se C for bem menor que B,
     o que domina a distancia e quem esta sinalizando, nao o que esta sendo sinalizado.
  4. Taxa em que o vizinho mais proximo (qualquer classe) e do mesmo sinalizador, comparada
     com o que se esperaria ao acaso.

Uso (na mesma pasta do knn_dtw_libras.py, do train_lstm_hands_face.py e da pasta libras):
    python diagnostico_dtw.py --data landmarks_vlibrasil_completo_traj_slots.npz \\
        --signer_regex ".*Articulador(\\d+)" --classes "@palavras100.txt"
Opcoes: --sem_z (ignora a coordenada z), --w_face, --w_traj (pesos; padrao: so maos),
--espelho (distancia invariante ao espelhamento horizontal, para sinais feitos com a mao do outro
lado; so sem rosto), --so_treino IDS (grupos de sinalizador, ex. '99', que ficam sempre na galeria de
treino e nunca viram teste; use com o arquivo de landmarks_vlibrasil_signbank*.npz).

Com --espelho, a distancia entre duas amostras passa a ser min(DTW(a, b), DTW(a, espelho(b))).
Use --eixo_espelho y (ou z) como CONTROLE: inverte outro eixo, sem sentido linguistico. Se o controle
melhorar tanto quanto o espelho (x), o ganho nao vem da lateralidade.
O espelhamento inverte o sinal de x de todos os pontos e mantem os slots (no arquivo *_slots o
slot 0 ja e a mao dominante, e nao a mao esquerda/direita). Tambem imprime, para pares da mesma
palavra entre articuladores, em quantos casos a versao espelhada casa melhor, comparado com pares
de palavras diferentes: se o primeiro numero for bem maior, a lateralidade varia de verdade.
"""
import argparse
import re

import numpy as np

from knn_dtw_libras import matriz_dtw, TEM_NUMBA
from libras.training_utils import load_class_filter, extract_signer_ids
from train_lstm_hands_face import load_data, build_points


def rank_da_classe_certa(D, y_teste, y_treino, classes):
    """D: (n_teste, n_treino). Devolve o rank (1 = primeiro) da classe verdadeira de cada amostra,
    usando, para cada classe, a menor distancia entre o teste e as amostras de treino dela."""
    ranks = np.zeros(len(y_teste), dtype=int)
    for i in range(len(y_teste)):
        dist_classe = np.full(len(classes), np.inf)
        for k, c in enumerate(classes):
            m = y_treino == c
            if m.any():
                dist_classe[k] = D[i, m].min()
        verdadeira = np.where(classes == y_teste[i])[0][0]
        ranks[i] = 1 + int((dist_classe < dist_classe[verdadeira]).sum())
    return ranks


def main():
    parser = argparse.ArgumentParser(description="Diagnostico da matriz DTW.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)")
    parser.add_argument("--w_hands", type=float, default=1.0)
    parser.add_argument("--w_face", type=float, default=0.0)
    parser.add_argument("--w_traj", type=float, default=0.0)
    parser.add_argument("--banda", type=int, default=10)
    parser.add_argument("--penalidade", type=float, default=1.0)
    parser.add_argument("--sem_z", action="store_true", help="Zera a coordenada z (ruidosa no MediaPipe)")
    parser.add_argument("--eixo_espelho", choices=["x", "y", "z"], default="x",
                        help="Eixo invertido por --espelho. 'x' = espelhamento horizontal (o de verdade). 'y' (de cabeca "
                             "para baixo) e 'z' sao CONTROLES (placebo): se melhorarem tanto quanto 'x', o ganho do --espelho "
                             "vem do metodo (duas chances por par) e nao da lateralidade.")
    parser.add_argument("--so_treino", default=None,
                        help="IDs de sinalizador (separados por virgula) que so entram como treino, nunca como teste (ex.: 99 = SignBank)")
    parser.add_argument("--espelho", action="store_true",
                        help="Distancia invariante ao espelhamento horizontal (so sem rosto)")
    args = parser.parse_args()

    if not TEM_NUMBA:
        print("Aviso: numba nao esta instalado; vai ficar muito lento (pip install numba).")

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

    sig = extract_signer_ids(paths, re.compile(args.signer_regex))
    y = np.array([str(v) for v in y_raw])
    classes = np.array(sorted(set(y.tolist())))
    sinalizadores = sorted(set(sig.tolist()))
    extras = set(x.strip() for x in args.so_treino.split(",") if x.strip()) if args.so_treino else set()
    avaliacao = [s for s in sinalizadores if s not in extras]
    if not avaliacao:
        raise SystemExit("Todos os sinalizadores estao em --so_treino.")
    print(f"Sinalizadores: {sinalizadores} | so treino: {sorted(extras)} | classes: {len(classes)} | amostras: {len(y)}")

    points = np.ascontiguousarray(points, dtype=np.float32)
    presente = np.ascontiguousarray(points.any(axis=-1))
    if args.sem_z:
        points[..., 2] = 0.0
        print("Coordenada z zerada.")
    peso = np.array([args.w_hands, args.w_face if usa_rosto else 0.0, args.w_traj if usa_traj else 0.0], dtype=np.float64)
    print(f"Pesos maos/rosto/trajetoria = {peso[0]}/{peso[1]}/{peso[2]} | banda {args.banda} | penalidade {args.penalidade}")

    print("Calculando a matriz DTW completa...", flush=True)
    D = matriz_dtw(points, presente, points, presente, grupo, peso, args.penalidade, args.banda)
    D = np.asarray(D, dtype=np.float64)
    np.fill_diagonal(D, np.inf)

    def topk_todos(Dm_):
        rs = []
        for s_ in avaliacao:
            te_ = sig == s_
            rs.append(rank_da_classe_certa(Dm_[np.ix_(te_, ~te_)], y[te_], y[~te_], classes))
        return np.concatenate(rs)

    if args.espelho:
        if usa_rosto:
            raise SystemExit("--espelho so esta implementado sem rosto (use --w_face 0).")
        eixo = {"x": 0, "y": 1, "z": 2}[args.eixo_espelho]
        points_m = np.ascontiguousarray(points.copy())
        points_m[..., eixo] *= -1.0
        if args.eixo_espelho != "x":
            print(f"CONTROLE (placebo): invertendo o eixo {args.eixo_espelho}, que nao corresponde a nenhuma variacao real de sinal.")
        print("Calculando a matriz com as amostras espelhadas...", flush=True)
        Dm = np.asarray(matriz_dtw(points, presente, points_m, presente, grupo, peso,
                                   args.penalidade, args.banda), dtype=np.float64)
        np.fill_diagonal(Dm, np.inf)
        mesma = (y[:, None] == y[None, :]) & (sig[:, None] != sig[None, :])
        outra = (y[:, None] != y[None, :]) & (sig[:, None] != sig[None, :])
        melhor = Dm < D
        print(f"Versao com o eixo {args.eixo_espelho} invertido casa melhor em {melhor[mesma].mean():.3f} dos pares da MESMA palavra "
              f"(outros articuladores) e em {melhor[outra].mean():.3f} dos pares de palavras DIFERENTES.")
        r0 = topk_todos(D)
        print(f"   [SEM espelho, mesma execucao] top1={np.mean(r0 <= 1):.3f}  top5={np.mean(r0 <= 5):.3f}  "
              f"top10={np.mean(r0 <= 10):.3f}  rank mediano={int(np.median(r0))}")
        D = np.minimum(D, Dm)
        print("   (os blocos abaixo usam a distancia COM espelho)")

    # 1) top-k deixando um sinalizador de fora
    print("\n1) Acuracia deixando um sinalizador de fora (top-1 / top-5 / top-10):")
    todos = []
    for s in avaliacao:
        te = sig == s
        tr = ~te
        r = rank_da_classe_certa(D[np.ix_(te, tr)], y[te], y[tr], classes)
        todos.append(r)
        print(f"   sinalizador {s}: top1={np.mean(r <= 1):.3f}  top5={np.mean(r <= 5):.3f}  top10={np.mean(r <= 10):.3f}  "
              f"rank mediano={int(np.median(r))} de {len(classes)}")
    r = np.concatenate(todos)
    print(f"   TODOS      : top1={np.mean(r <= 1):.3f}  top5={np.mean(r <= 5):.3f}  top10={np.mean(r <= 10):.3f}  "
          f"rank mediano={int(np.median(r))} (acaso: top1={1/len(classes):.3f} top5={5/len(classes):.3f} top10={10/len(classes):.3f})")

    # 2) pares treino -> teste
    print("\n2) Top-1 para cada par (treino -> teste), com 1 exemplo por palavra no treino:")
    print("   treino \\ teste " + " ".join(f"{s:>6}" for s in avaliacao))
    for a in sinalizadores:
        linha = []
        for b in avaliacao:
            if a == b:
                linha.append("     -")
                continue
            te, tr = sig == b, sig == a
            rr = rank_da_classe_certa(D[np.ix_(te, tr)], y[te], y[tr], classes)
            linha.append(f"{np.mean(rr <= 1):6.3f}")
        print(f"   {a:>14} " + " ".join(linha))

    # 3) distancias medias A / B / C
    mesma_palavra = y[:, None] == y[None, :]
    mesmo_sinal = sig[:, None] == sig[None, :]
    finito = np.isfinite(D)
    A = D[mesma_palavra & ~mesmo_sinal & finito]
    B = D[~mesma_palavra & ~mesmo_sinal & finito]
    C = D[~mesma_palavra & mesmo_sinal & finito]
    print("\n3) Distancia DTW media:")
    print(f"   A) mesma palavra, sinalizadores diferentes   : {A.mean():.4f}")
    print(f"   B) palavras diferentes, sinalizadores difer. : {B.mean():.4f}")
    print(f"   C) palavras diferentes, mesmo sinalizador    : {C.mean():.4f}")
    print(f"   A/B = {A.mean() / B.mean():.3f}  (1.0 = a palavra nao aproxima nada; quanto menor, melhor)")
    print(f"   C/B = {C.mean() / B.mean():.3f}  (bem abaixo de 1.0 = o sinalizador pesa mais que a palavra)")

    # 4) vizinho mais proximo e do mesmo sinalizador?
    nn = np.argmin(D, axis=1)
    mesmo = (sig[nn] == sig).mean()
    esperado = np.mean([((sig == s).sum() - 1) / (len(sig) - 1) for s in sig])
    print(f"\n4) Vizinho mais proximo (qualquer classe) e do mesmo sinalizador: {mesmo:.3f} (acaso: {esperado:.3f})")


if __name__ == "__main__":
    main()
