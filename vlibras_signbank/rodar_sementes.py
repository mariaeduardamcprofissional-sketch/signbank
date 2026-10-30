"""Roda o train_lstm_hands_face.py com varias sementes e resume (media, desvio, top-5, top-10).

Substitui o rodar_variante.py nesta etapa: a semente agora e uma opcao do proprio treino (--seed).
Durante a execucao mostra so as linhas importantes (rodadas, acuracias), nao as listas de contagem.

Uso:
    python rodar_sementes.py --sementes 1,2,3 --saida resultados_com_sb.json -- \\
        --data landmarks_vlibrasil_signbank_trim_slots.npz --signer_regex ".*Articulador(\\d+)" \\
        --so_treino 99 --val_split 0 --classes @palavras100_comuns.txt \\
        --leave_one_signer_out --no_face

Tudo depois do '--' e repassado ao train_lstm_hands_face.py. Nos argumentos repassados, o texto
{seed} e trocado pela semente da rodada (util em --salvar_proba proba_seed{seed}.npz).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import deque

import numpy as np

LINHAS_UTEIS = re.compile(r"(Rodada \d+/\d+|^Acuracia|Top-5 medio|Probabilidades salvas|Grupos so de treino|"
                          r"Sinalizadores encontrados|Filtrando para|Classes:|Traceback|Error|erro:)")


def rodar(semente, treino_args, script):
    args = [a.replace("{seed}", str(semente)) for a in treino_args]
    cmd = [sys.executable, "-u", script, "--seed", str(semente)] + args
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="2")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=env)
    final = deque(maxlen=30)
    saida = []
    for linha in proc.stdout:
        saida.append(linha)
        final.append(linha)
        if LINHAS_UTEIS.search(linha):
            print("   ", linha.rstrip(), flush=True)
    proc.wait()
    texto = "".join(saida)
    a = re.search(r"Acuracia media entre rodadas: ([0-9.]+)", texto)
    k = re.search(r"Top-5 medio: ([0-9.]+).*?Top-10 medio: ([0-9.]+)", texto)
    if not a:
        print("    ERRO: nao encontrei a acuracia media. Ultimas linhas:\n" + "".join(final))
        return None
    return {"top1": float(a.group(1)),
            "top5": float(k.group(1)) if k else float("nan"),
            "top10": float(k.group(2)) if k else float("nan")}


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("Uso: python rodar_sementes.py --sementes 1,2,3 --saida res.json -- <argumentos do treino>")
    i = argv.index("--")
    parser = argparse.ArgumentParser(description="Roda o treino com varias sementes.")
    parser.add_argument("--sementes", default="1,2,3")
    parser.add_argument("--saida", default=None, help=".json onde guardar os resultados por semente")
    parser.add_argument("--script", default="train_lstm_hands_face.py")
    own = parser.parse_args(argv[:i])
    treino_args = argv[i + 1:]

    sementes = [int(s) for s in own.sementes.split(",") if s.strip()]
    resultados = {}
    for s in sementes:
        print(f"=== semente {s} ===", flush=True)
        r = rodar(s, treino_args, own.script)
        if r is not None:
            resultados[str(s)] = r
            print(f"  -> semente {s}: top-1 {r['top1']:.4f} | top-5 {r['top5']:.4f} | top-10 {r['top10']:.4f}", flush=True)

    if not resultados:
        raise SystemExit("Nenhuma semente terminou com sucesso.")
    t1 = np.array([r["top1"] for r in resultados.values()])
    t5 = np.array([r["top5"] for r in resultados.values()])
    t10 = np.array([r["top10"] for r in resultados.values()])
    dp = lambda v: np.nanstd(v, ddof=1) if len(v) > 1 else float("nan")
    print(f"\nRESUMO ({len(t1)} sementes): top-1 {t1.mean():.4f} +/- {dp(t1):.4f} | "
          f"top-5 {np.nanmean(t5):.4f} +/- {dp(t5):.4f} | top-10 {np.nanmean(t10):.4f} +/- {dp(t10):.4f}")
    if own.saida:
        with open(own.saida, "w") as f:
            json.dump(resultados, f, indent=2)
        print(f"Resultados por semente salvos em {own.saida}")


if __name__ == "__main__":
    main()
