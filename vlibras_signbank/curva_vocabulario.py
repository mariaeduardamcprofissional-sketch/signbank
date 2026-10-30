"""Curva: desempenho (um articulador de fora) em funcao do tamanho do vocabulario.

Para cada tamanho N sorteia subconjuntos de N palavras (so palavras que tem video dos articuladores
1, 2 e 3) e roda o rodar_sementes.py com --leave_one_signer_out, o SignBank sempre so no treino.
Mostra top-1/5/10 medios por tamanho e o acaso. Serve para dizer ao TCC quantas palavras o sistema
consegue reconhecer com um articulador novo.

Uso (na pasta do projeto, com o .venv ativo):
    python curva_vocabulario.py --dados dados\\landmarks_vlibrasil_trim_signbank_slots.npz
    python curva_vocabulario.py --dados ... --tamanhos 11,25,50,100,250 --subconjuntos 2 --sementes 1
Saidas em resultados/curva/: listas de palavras sorteadas, json por execucao e curva_vocabulario.csv.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
REGEX = r".*Articulador(\d+)"


def classes_elegiveis(dados):
    with np.load(dados, allow_pickle=False) as d:
        y = [str(v) for v in d["y"].tolist()]
        paths = [str(v) for v in d["paths"].tolist()]
    por_classe = {}
    padrao = re.compile(REGEX)
    for c, p in zip(y, paths):
        m = padrao.search(p)
        if m:
            por_classe.setdefault(c, set()).add(m.group(1))
    return sorted(c for c, s in por_classe.items() if {"1", "2", "3"} <= s)


def rodar(cmd):
    env = dict(os.environ, PYTHONUTF8="1", TF_CPP_MIN_LOG_LEVEL="2")
    r = subprocess.run(cmd, cwd=AQUI, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit(f"Erro (codigo {r.returncode}) ao rodar: {' '.join(cmd[:4])} ...")
    return r.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dados", required=True)
    p.add_argument("--tamanhos", default="11,25,50,100,250")
    p.add_argument("--subconjuntos", type=int, default=2, help="Sorteios de palavras por tamanho")
    p.add_argument("--sementes", default="1", help="Sementes de treino, ex.: 1 ou 1,2,3")
    p.add_argument("--aug_copies", type=int, default=2)
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--saida", default=os.path.join("resultados", "curva"))
    a = p.parse_args()

    dados = a.dados if os.path.isabs(a.dados) else os.path.join(AQUI, a.dados)
    saida = a.saida if os.path.isabs(a.saida) else os.path.join(AQUI, a.saida)
    os.makedirs(saida, exist_ok=True)
    elegiveis = classes_elegiveis(dados)
    print(f"{len(elegiveis)} palavras com video dos articuladores 1, 2 e 3.", flush=True)

    linhas = []
    for n in [int(x) for x in a.tamanhos.split(",")]:
        n_eff = min(n, len(elegiveis))
        reps = 1 if n_eff == len(elegiveis) else a.subconjuntos
        for j in range(reps):
            rng = np.random.default_rng(1000 + n * 10 + j)
            escolha = sorted(rng.choice(elegiveis, size=n_eff, replace=False).tolist())
            lista = os.path.join(saida, f"palavras_N{n_eff}_s{j}.txt")
            with open(lista, "w", encoding="utf-8") as f:
                f.write("\n".join(escolha))
            cmd = [sys.executable, "rodar_sementes.py", "--sementes", a.sementes,
                   "--saida", os.path.join(saida, f"res_N{n_eff}_s{j}.json"), "--",
                   "--data", dados, "--signer_regex", REGEX, "--no_face",
                   "--aug_copies", str(a.aug_copies), "--loso_epochs", str(a.epocas),
                   "--so_treino", "99", "--val_split", "0", "--leave_one_signer_out",
                   "--classes", "@" + lista]
            print(f"\n=== {n_eff} palavras, sorteio {j} ===", flush=True)
            out = rodar(cmd)
            m = None
            for linha in out.splitlines():
                if linha.startswith("RESUMO"):
                    m = re.search(r"top-1 ([0-9.]+).*?top-5 ([0-9.]+).*?top-10 ([0-9.]+)", linha)
            if not m:
                print(out[-1500:])
                raise SystemExit("Nao achei a linha RESUMO na saida do rodar_sementes.py.")
            t1, t5, t10 = (float(x) for x in m.groups())
            print(f"    top-1 {t1:.4f} | top-5 {t5:.4f} | top-10 {t10:.4f} | acaso top-1 {1 / n_eff:.4f}", flush=True)
            linhas.append((n_eff, j, t1, t5, t10, 1 / n_eff))

    caminho = os.path.join(saida, "curva_vocabulario.csv")
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["palavras", "sorteio", "top1", "top5", "top10", "acaso_top1"])
        w.writerows(linhas)
    print("\nResumo (media dos sorteios):")
    print(f"{'palavras':>9} {'top-1':>8} {'top-5':>8} {'top-10':>8} {'acaso top-1':>12}")
    for n in sorted({l[0] for l in linhas}):
        sub = [l for l in linhas if l[0] == n]
        print(f"{n:>9} {np.mean([l[2] for l in sub]):8.3f} {np.mean([l[3] for l in sub]):8.3f} "
              f"{np.mean([l[4] for l in sub]):8.3f} {1 / n:12.3f}")
    print(f"\nSalvo em {caminho}")


if __name__ == "__main__":
    main()
