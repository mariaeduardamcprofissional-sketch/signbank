"""Compara visualmente a MESMA palavra feita pelos diferentes articuladores, a partir do .npz.

Para cada palavra e cada articulador, uma linha com tres paineis:
  1. Trajetoria do punho (X_traj): x(t) e y(t) da mao slot 0 (azul) e slot 1 (vermelho).
     Linha cheia = x, tracejada = y. Quadros sem deteccao ficam em branco.
  2. Linha de tempo de deteccao: quais quadros tem a mao slot 0 / slot 1 detectada.
  3. Formato da mao slot 0 em 3 momentos (inicio, meio e fim dos quadros detectados).

Se os tres articuladores fazem o mesmo sinal, as curvas e os formatos devem se parecer.
Se parecerem sinais diferentes, esse e um limite do dataset, nao do modelo.

Uso:
    python comparar_palavra.py --data landmarks_vlibrasil_completo_traj_slots.npz \\
        --signer_regex ".*Articulador(\\d+)" --palavras "Razão,Rico,Rosa" --saida comparar_palavras.png
Depois, no Colab:
    from IPython.display import Image; Image("comparar_palavras.png")
"""
import argparse
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONEXOES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
            (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def desenhar_mao(ax, pts, dx, cor):
    """pts: (21, 3) com o pulso (indice 0) em zero. Desenha x,y deslocado em dx. y para baixo."""
    xy = pts[:, :2] + np.array([dx, 0.0])
    for a, b in CONEXOES:
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=cor, lw=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color=cor)


def main():
    parser = argparse.ArgumentParser(description="Compara a mesma palavra entre articuladores.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)")
    parser.add_argument("--palavras", required=True, help="Lista separada por virgula, exatamente como nos rotulos")
    parser.add_argument("--saida", default="comparar_palavras.png")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    H, Tr, y = d["X_hands"], d["X_traj"], np.array([str(v) for v in d["y"]])
    rx = re.compile(args.signer_regex)
    sig = np.array([rx.search(str(p)).group(1) for p in d["paths"]])
    sigs = sorted(set(sig.tolist()))
    T = H.shape[1]
    det = H[:, :, :, 1:, :].reshape(H.shape[0], T, 2, -1).any(-1)   # (N, T, 2)

    palavras = [p.strip() for p in args.palavras.split(",") if p.strip()]
    linhas = [(p, s) for p in palavras for s in sigs]
    fig, axes = plt.subplots(len(linhas), 3, figsize=(13, 2.3 * len(linhas)),
                             gridspec_kw={"width_ratios": [2.2, 1.6, 2.2]}, squeeze=False)
    for r, (palavra, s) in enumerate(linhas):
        idx = np.where((y == palavra) & (sig == s))[0]
        ax1, ax2, ax3 = axes[r]
        if len(idx) == 0:
            ax1.set_title(f"{palavra} | articulador {s}: sem amostra", fontsize=9)
            for a in (ax1, ax2, ax3):
                a.axis("off")
            continue
        i = idx[0]
        t = np.arange(T)
        for k, cor in ((0, "tab:blue"), (1, "tab:red")):
            ok = det[i, :, k]
            for j, estilo in ((0, "-"), (1, "--")):
                v = np.where(ok, Tr[i, :, k, j], np.nan)
                ax1.plot(t, v, estilo, color=cor, lw=1.3)
        ax1.set_title(f"{palavra} | articulador {s}  (azul = slot 0, vermelho = slot 1; cheio = x, tracejado = y)", fontsize=8)
        ax1.set_xlim(0, T - 1)
        ax1.grid(alpha=0.3)

        ax2.imshow(det[i].T.astype(float), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["slot 0", "slot 1"], fontsize=7)
        ax2.set_xlabel("quadro", fontsize=7)
        ax2.set_title("mao detectada", fontsize=8)

        pres = np.where(det[i, :, 0])[0]
        if len(pres) > 0:
            for n, q in enumerate(np.unique(np.quantile(pres, [0.0, 0.5, 1.0]).round().astype(int))):
                q = pres[np.argmin(np.abs(pres - q))]
                desenhar_mao(ax3, H[i, q, 0], dx=2.4 * n, cor="tab:blue")
        ax3.set_aspect("equal")
        ax3.invert_yaxis()
        ax3.axis("off")
        ax3.set_title("formato da mao slot 0: inicio, meio, fim", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.saida, dpi=110)
    print(f"Figura salva em {args.saida}")


if __name__ == "__main__":
    main()
