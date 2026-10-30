"""Verifica o arquivo de landmarks antes de treinar (leva alguns segundos).

Confere: (1) o .npz abre e o zip esta integro (CRC); (2) trimmed=True (extracao com --trim);
(3) X_traj presente; (4) origem V-Librasil/SignBank; (5) sinalizadores encontrados pela regex;
(6) se os slots das maos parecem corrigidos (slot 0 = mao dominante).

Uso:
    python verificar_dados.py dados/landmarks_vlibrasil_signbank_slots.npz
Codigo de saida 0 = tudo certo para treinar; 1 = ha um problema (a mensagem diz qual).
"""
import argparse
import os
import re
import sys
import zipfile

import numpy as np


def verificar(caminho, regex=r".*Articulador(\d+)", verbose=True):
    """Devolve (ok, problemas). Imprime um resumo se verbose."""
    problemas, avisos = [], []
    say = print if verbose else (lambda *a, **k: None)

    def falha(*msgs):
        for m in msgs:
            say(f"  PROBLEMA: {m}")
        say("  => NAO treine ainda")
        return False, list(msgs)

    if not os.path.exists(caminho):
        return falha(f"arquivo nao encontrado: {caminho}")
    tam = os.path.getsize(caminho) / 1e6
    say(f"Arquivo: {caminho} ({tam:.1f} MB)")

    try:
        with zipfile.ZipFile(caminho) as z:
            ruim = z.testzip()
        if ruim is not None:
            return falha(f"zip CORROMPIDO no membro '{ruim}': recopie o arquivo")
    except zipfile.BadZipFile:
        return falha("o arquivo NAO e um .npz valido (truncado). Recopie e compare o tamanho e o hash com o original")
    say("  zip: integro")

    d = np.load(caminho, allow_pickle=True)
    chaves = set(d.keys())
    for k in ("X_hands", "X_traj", "y", "paths"):
        if k not in chaves:
            problemas.append(f"falta a chave '{k}'" + (" (extraia com --trajectory)" if k == "X_traj" else ""))
    if problemas:
        return falha(*problemas)

    N = len(d["y"])
    H = d["X_hands"]
    say(f"  amostras: {N} | classes: {len(set(d['y'].tolist()))} | X_hands {H.shape} | X_traj {d['X_traj'].shape}")
    if H.shape[0] != N or d["X_traj"].shape[0] != N or len(d["paths"]) != N:
        problemas.append("numero de amostras diferente entre os arrays")

    trimmed = bool(d["trimmed"]) if "trimmed" in chaves else None
    say(f"  trimmed: {trimmed}")
    if trimmed is not True:
        problemas.append("trimmed nao e True: este arquivo vem da extracao ANTIGA (sem recorte dos quadros parados). "
                         "Use o arquivo gerado com --trim")

    if "origem" in chaves:
        o, c = np.unique(d["origem"].astype(str), return_counts=True)
        say("  origem: " + ", ".join(f"{a}={b}" for a, b in zip(o, c)))
    else:
        avisos.append("sem a chave 'origem' (nao e o arquivo juntado V-Librasil + SignBank)")

    rx = re.compile(regex)
    ids = []
    for p in d["paths"].tolist():
        m = rx.search(str(p))
        ids.append(m.group(1) if m else None)
    sem = sum(i is None for i in ids)
    if sem:
        problemas.append(f"{sem} caminhos nao casam com a regex de sinalizador {regex!r}")
    else:
        u, c = np.unique(np.array(ids), return_counts=True)
        say("  sinalizadores: " + ", ".join(f"{a}={b}" for a, b in zip(u, c)))
        if "99" not in u and "origem" in chaves:
            avisos.append("nao ha sinalizador 99 (SignBank) nos caminhos")

    det = H[:, :, :, 1:, :].reshape(N, H.shape[1], 2, -1).any(-1)
    s0, s1 = det[..., 0].mean(), det[..., 1].mean()
    say(f"  mao detectada: slot0={s0:.2f} slot1={s1:.2f}")
    if s0 - s1 < 0.04:
        avisos.append("slot 0 e slot 1 tem deteccao parecida: os slots provavelmente NAO foram corrigidos "
                      "(corrigir_slots_maos.py). Depois da correcao o slot 0 costuma ficar ~0.10 acima")

    for a in avisos:
        say(f"  AVISO: {a}")
    for p in problemas:
        say(f"  PROBLEMA: {p}")
    say("  => " + ("PRONTO para treinar" if not problemas else "NAO treine ainda"))
    return (not problemas), problemas


def main():
    parser = argparse.ArgumentParser(description="Verifica o .npz antes de treinar.")
    parser.add_argument("data")
    parser.add_argument("--signer_regex", default=r".*Articulador(\d+)")
    args = parser.parse_args()
    ok, _ = verificar(args.data, args.signer_regex)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
