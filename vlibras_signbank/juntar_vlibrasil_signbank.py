"""Junta o V-Librasil (ja unido, por exemplo com trim) com os videos do SignBank, como um articulador extra.

O SignBank entra com o nome de articulador "99" (path <nome>_Articulador99.mp4), para que
--signer_regex ".*Articulador(\\d+)" o identifique e --so_treino 99 o mantenha so no treino.
Adiciona a chave 'origem' (0 = V-Librasil, 1 = SignBank). Amostras do SignBank cuja classe nao existe
no V-Librasil sao descartadas (com aviso).

Pre-requisitos conferidos antes de gravar: mesmas chaves por amostra, mesmos pontos faciais e mesma
flag 'trimmed' nos dois arquivos (extraia o SignBank com a MESMA opcao --trim do V-Librasil).

Uso:
    python juntar_vlibrasil_signbank.py --vlibrasil landmarks_vlibrasil_trim_completo.npz \\
        --signbank landmarks_signbank_trim_traj.npz --output landmarks_vlibrasil_trim_signbank.npz
"""
import argparse
import os
from collections import Counter

import numpy as np


def carregar(caminho):
    try:
        with np.load(caminho, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except FileNotFoundError:
        raise SystemExit(f"Arquivo nao encontrado: {caminho}")


def main():
    p = argparse.ArgumentParser(description="Junta V-Librasil e SignBank (articulador 99).")
    p.add_argument("--vlibrasil", required=True)
    p.add_argument("--signbank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--grupo", default="99", help="Numero do articulador atribuido ao SignBank")
    p.add_argument("--aceitar_trimmed_diferente", action="store_true",
                   help="Permite juntar mesmo se 'trimmed' difere (os dois conjuntos ficam com representacoes diferentes!)")
    p.add_argument("--sobrescrever", action="store_true")
    a = p.parse_args()

    saida = a.output if a.output.lower().endswith(".npz") else a.output + ".npz"
    if os.path.normcase(os.path.abspath(saida)) in {os.path.normcase(os.path.abspath(a.vlibrasil)),
                                                    os.path.normcase(os.path.abspath(a.signbank))}:
        raise SystemExit("O arquivo de saida nao pode ser um dos de entrada.")
    if os.path.exists(saida) and not a.sobrescrever:
        raise SystemExit(f"{saida} ja existe. Use outro nome ou --sobrescrever.")

    v, s = carregar(a.vlibrasil), carregar(a.signbank)
    for nome, d in (("V-Librasil", v), ("SignBank", s)):
        for k in ("y", "paths"):
            if k not in d:
                raise SystemExit(f"{nome}: falta a chave '{k}'.")
    v.pop("origem", None)
    s.pop("origem", None)

    n_v, n_s = len(v["y"]), len(s["y"])
    estaticas = {k for k, x in v.items() if x.ndim == 0 or x.shape[0] != n_v}
    estaticas |= {"face_landmark_indices", "trimmed"} & set(v)
    por_amostra = [k for k in v if k not in estaticas]

    faltam = [k for k in v if k not in s]
    if faltam:
        raise SystemExit(f"O SignBank nao tem as chaves {faltam} que o V-Librasil tem. "
                         "Extraia os dois com as mesmas opcoes (por exemplo --trajectory/--trim) e aplique nos dois "
                         "a mesma correcao de slots das maos (corrigir_slots_maos.py).")
    sobram = [k for k in s if k not in v]
    if sobram:
        print(f"Aviso: chaves so do SignBank descartadas: {sobram}")

    for k in estaticas:
        if not np.array_equal(v[k], s[k]):
            if k == "trimmed" and a.aceitar_trimmed_diferente:
                print(f"AVISO: 'trimmed' difere (V-Librasil={v[k].tolist()}, SignBank={s[k].tolist()}). "
                      "Mantendo o valor do V-Librasil; o SignBank NAO tem a mesma representacao.")
                continue
            raise SystemExit(f"'{k}' difere: V-Librasil={v[k].tolist()}, SignBank={s[k].tolist()}. "
                             "Extraia o SignBank com a mesma configuracao do V-Librasil.")
    for k in por_amostra:
        if s[k].shape[1:] != v[k].shape[1:]:
            raise SystemExit(f"'{k}': formato por amostra {v[k].shape[1:]} (V-Librasil) contra {s[k].shape[1:]} (SignBank).")
        if s[k].shape[0] != n_s:
            raise SystemExit(f"SignBank: '{k}' nao tem {n_s} itens.")

    classes_v = set(str(c) for c in v["y"].tolist())
    manter = np.array([str(c) in classes_v for c in s["y"].tolist()])
    fora = sorted({str(c) for c, m in zip(s["y"].tolist(), manter) if not m})
    if fora:
        print(f"Aviso: {int((~manter).sum())} amostra(s) do SignBank descartada(s), classe inexistente no V-Librasil: {fora}")
    if not manter.any():
        raise SystemExit("Nenhuma amostra do SignBank tem classe presente no V-Librasil.")

    novos_paths = np.array([f"{os.path.splitext(os.path.basename(str(x)))[0]}_Articulador{a.grupo}.mp4"
                            for x in s["paths"][manter]])
    juntos = {}
    for k in por_amostra:
        sb = s[k][manter]
        if k == "paths":
            sb = novos_paths
        juntos[k] = np.concatenate([v[k], sb], axis=0)
    for k in estaticas:
        juntos[k] = v[k]
    juntos["origem"] = np.concatenate([np.zeros(n_v, np.int8), np.ones(int(manter.sum()), np.int8)])

    dup = [x for x, c in Counter(juntos["paths"].tolist()).items() if c > 1]
    if dup:
        raise SystemExit(f"{len(dup)} path(s) duplicado(s) (ex.: {dup[:3]}). O SignBank tem nomes repetidos ou ja foi juntado.")

    total_sb = int(manter.sum())
    classes_sb = {str(c) for c in s["y"][manter].tolist()}
    print(f"\nV-Librasil: {n_v} amostras | SignBank mantidas: {total_sb} de {n_s} | total: {len(juntos['y'])}")
    print(f"Classes do V-Librasil: {len(classes_v)} | com amostras do SignBank: {len(classes_sb)}")
    print(f"Chaves: {sorted(juntos)}")
    print(f"Use --signer_regex '.*Articulador(\\d+)' --so_treino {a.grupo} no treino.")

    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    np.savez_compressed(saida, **juntos)
    print(f"Salvo em {saida} ({os.path.getsize(saida) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
