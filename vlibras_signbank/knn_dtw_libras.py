"""Classificador por vizinho mais proximo com DTW (sem treino) para sinais isolados.

Cada sequencia de teste e comparada com todas as sequencias de treino por DTW (alinhamento
temporal elastico) e recebe a palavra da sequencia mais parecida. Com poucos exemplos por
palavra (V-Librasil: 3), costuma ser uma referencia mais justa que uma LSTM, e acrescentar uma
palavra nova ao dicionario e so guardar o video dela.

Avalia deixando um sinalizador de fora por rodada, como o train_lstm_hands_face.py com
--leave_one_signer_out, usando a mesma representacao (maos, rosto normalizado, trajetoria).

Uso (na mesma pasta do train_lstm_hands_face.py e da pasta libras):
    python knn_dtw_libras.py --data landmarks_vlibrasil_completo_traj.npz \\
        --signer_regex ".*Articulador(\\d+)" --classes "@palavras100.txt" --ablation

Cuidado com o tempo: o custo cresce com o quadrado do numero de palavras. Com 100 palavras leva
alguns segundos por rodada. Com as 1364 palavras leva horas; use --classes.
"""
import argparse
import re
import time

import numpy as np

try:
    from numba import njit, prange
    TEM_NUMBA = True
except ImportError:  # funciona sem numba, so que muito mais devagar
    TEM_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

from libras.training_utils import load_class_filter, extract_signer_ids
from train_lstm_hands_face import load_data, build_points


# ---------------- DTW ----------------

@njit(cache=True)
def _dist_quadro(Pa, Ma, Pb, Mb, grupo, peso, penalidade):
    """Distancia entre dois quadros: media, por grupo (maos, rosto, trajetoria), da distancia
    entre os pontos que existem nos dois quadros. Ponto que existe em um quadro e nao no outro
    custa 'penalidade'. Pontos ausentes nos dois nao contam."""
    n_grupos = peso.shape[0]
    soma = np.zeros(n_grupos)
    cont = np.zeros(n_grupos)
    for k in range(Pa.shape[0]):
        g = grupo[k]
        if peso[g] == 0.0:
            continue
        if Ma[k] and Mb[k]:
            dx = Pa[k, 0] - Pb[k, 0]
            dy = Pa[k, 1] - Pb[k, 1]
            dz = Pa[k, 2] - Pb[k, 2]
            soma[g] += np.sqrt(dx * dx + dy * dy + dz * dz)
            cont[g] += 1.0
        elif Ma[k] != Mb[k]:
            soma[g] += penalidade
            cont[g] += 1.0
    total = 0.0
    for g in range(n_grupos):
        if cont[g] > 0.0:
            total += peso[g] * soma[g] / cont[g]
    return total


@njit(cache=True)
def _dtw(Pa, Ma, Pb, Mb, grupo, peso, penalidade, banda):
    T = Pa.shape[0]
    U = Pb.shape[0]
    INF = 1e30
    D = np.full((T + 1, U + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, T + 1):
        lo = max(1, i - banda)
        hi = min(U, i + banda)
        for j in range(lo, hi + 1):
            c = _dist_quadro(Pa[i - 1], Ma[i - 1], Pb[j - 1], Mb[j - 1], grupo, peso, penalidade)
            m = min(D[i - 1, j], min(D[i, j - 1], D[i - 1, j - 1]))
            D[i, j] = c + m
    return D[T, U] / (T + U)


@njit(parallel=True, cache=True)
def matriz_dtw(Pteste, Mteste, Ptreino, Mtreino, grupo, peso, penalidade, banda):
    nt = Pteste.shape[0]
    nr = Ptreino.shape[0]
    saida = np.empty((nt, nr))
    for i in prange(nt):
        for j in range(nr):
            saida[i, j] = _dtw(Pteste[i], Mteste[i], Ptreino[j], Mtreino[j], grupo, peso, penalidade, banda)
    return saida


# ---------------- Avaliacao ----------------

def avaliar(points, presente, y, signer_ids, grupo, peso, penalidade, banda, nome):
    """1-NN por DTW, um sinalizador de fora por rodada. Devolve a lista de acuracias."""
    sinalizadores = sorted(set(signer_ids.tolist()))
    accs = []
    for s in sinalizadores:
        teste = signer_ids == s
        treino = ~teste
        t0 = time.time()
        d = matriz_dtw(points[teste], presente[teste], points[treino], presente[treino],
                       grupo, peso, penalidade, banda)
        prev = y[treino][np.argmin(d, axis=1)]
        acc = float((prev == y[teste]).mean())
        accs.append(acc)
        print(f"  [{nome}] Acuracia (sinalizador de fora ['{s}']): {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  [{nome}] Acuracia media entre rodadas: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})", flush=True)
    return accs


def main():
    parser = argparse.ArgumentParser(description="1-NN com DTW, um sinalizador de fora por rodada.")
    parser.add_argument("--data", required=True, help=".npz gerado pelo extrator de landmarks")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario: 'A,B,C' ou @arquivo.txt")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)",
                        help="Regex com 1 grupo para o ID do sinalizador (V-Librasil: '.*Articulador(\\d+)')")
    parser.add_argument("--no_face", action="store_true", help="Ignora o rosto")
    parser.add_argument("--no_traj", action="store_true", help="Ignora a trajetoria do punho")
    parser.add_argument("--w_hands", type=float, default=1.0, help="Peso das maos na distancia")
    parser.add_argument("--w_face", type=float, default=0.5, help="Peso do rosto na distancia")
    parser.add_argument("--w_traj", type=float, default=1.0, help="Peso da trajetoria na distancia")
    parser.add_argument("--banda", type=int, default=10, help="Faixa de Sakoe-Chiba (quadros). Limita o quanto dois quadros podem se afastar no alinhamento.")
    parser.add_argument("--penalidade", type=float, default=1.0, help="Custo de um ponto existir so em um dos quadros")
    parser.add_argument("--ablation", action="store_true",
                        help="Roda tambem so maos, maos+rosto, maos+trajetoria e tudo, para comparar as representacoes.")
    args = parser.parse_args()

    if not TEM_NUMBA:
        print("Aviso: numba nao esta instalado. Vai funcionar, mas muito mais devagar (pip install numba).")

    X_hands, X_face, X_traj, y_raw, paths = load_data(args.data)
    if paths is None:
        raise SystemExit("O .npz nao tem 'paths': nao da para identificar o sinalizador.")

    usa_rosto = (not args.no_face) and X_face is not None
    usa_traj = (not args.no_traj) and X_traj is not None
    n_rosto = X_face.shape[2] if usa_rosto else 0
    n_traj = X_traj.shape[2] if usa_traj else 0
    n_maos = X_hands.shape[2] * X_hands.shape[3]
    points = build_points(X_hands, X_face, X_traj, use_face=usa_rosto, use_traj=usa_traj)
    del X_hands, X_face, X_traj
    grupo = np.array([0] * n_maos + [1] * n_rosto + [2] * n_traj, dtype=np.int64)
    assert len(grupo) == points.shape[2], "layout dos pontos nao bate com o esperado"

    filtro = load_class_filter(args.classes)
    if filtro is not None:
        mask = np.array([r in filtro for r in y_raw])
        points, y_raw, paths = points[mask], y_raw[mask], paths[mask]
        print(f"Filtrando para {len(filtro)} classes pedidas: {len(y_raw)} amostras restantes.")

    signer_ids = extract_signer_ids(paths, re.compile(args.signer_regex))
    print(f"Sinalizadores encontrados ({len(set(signer_ids.tolist()))}): {sorted(set(signer_ids.tolist()))}")
    n_classes = len(set(y_raw.tolist()))
    print(f"Classes: {n_classes} | amostras: {len(y_raw)} | acaso: {1.0 / n_classes:.4f}")

    points = np.ascontiguousarray(points, dtype=np.float32)
    presente = np.ascontiguousarray(points.any(axis=-1))  # (N, T, K): ponto detectado ou nao

    pesos = {
        "tudo (pesos informados)": (args.w_hands, args.w_face if usa_rosto else 0.0, args.w_traj if usa_traj else 0.0),
    }
    if args.ablation:
        pesos = {
            "so maos": (1.0, 0.0, 0.0),
            "maos+rosto": (1.0, args.w_face, 0.0),
            "maos+trajetoria": (1.0, 0.0, args.w_traj),
            "tudo": (1.0, args.w_face, args.w_traj),
        }
        pesos = {k: v for k, v in pesos.items()
                 if (v[1] == 0 or usa_rosto) and (v[2] == 0 or usa_traj)}

    resultados = {}
    for nome, w in pesos.items():
        print(f"\n=== {nome} (pesos maos/rosto/trajetoria = {w[0]}/{w[1]}/{w[2]}) ===", flush=True)
        accs = avaliar(points, presente, y_raw, signer_ids, grupo,
                       np.array(w, dtype=np.float64), args.penalidade, args.banda, nome)
        resultados[nome] = accs

    if len(resultados) > 1:
        print("\nResumo (acuracia media entre rodadas):")
        for nome, accs in resultados.items():
            print(f"  {nome:<18} {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")


if __name__ == "__main__":
    main()
