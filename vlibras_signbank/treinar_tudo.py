"""Treino completo (vocabulario inteiro, V-Librasil + SignBank) com um unico comando.

Evita problemas de aspas e de continuacao de linha do PowerShell: os argumentos sao montados aqui.

  --etapa estimativa : 3 rodadas, um articulador de fora por vez (SignBank sempre no treino).
                       Mostra top-1/5/10 por rodada. E o numero de desempenho para o TCC.
  --etapa final      : treina com TODAS as amostras (sem teste) e salva o modelo.
  --etapa ambas      : estimativa e depois final (padrao).

Configuracao fixa: sem rosto (--no_face), com trajetoria e velocidade, slots corrigidos (ja no arquivo),
--aug_copies 2, 40 epocas por rodada.

Uso:
    python treinar_tudo.py --dados dados/landmarks_vlibrasil_signbank_slots.npz
    python treinar_tudo.py --dados ... --etapa estimativa --sementes 1,2,3
Saidas em resultados/: res_tudo.json, proba_tudo_seed*.npz, modelo_vlibras_signbank_final.keras (+ _labels.json, _config.json).
"""
import argparse
import os
import subprocess
import sys

from verificar_dados import verificar

AQUI = os.path.dirname(os.path.abspath(__file__))
REGEX = r".*Articulador(\d+)"


def rodar(cmd):
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    env = dict(os.environ, PYTHONUTF8="1", TF_CPP_MIN_LOG_LEVEL="2")
    r = subprocess.run(cmd, cwd=AQUI, env=env)
    if r.returncode != 0:
        raise SystemExit(f"\nO comando acima terminou com erro (codigo {r.returncode}). Veja a mensagem acima dele.")


def main():
    parser = argparse.ArgumentParser(description="Treino completo V-Librasil + SignBank.")
    parser.add_argument("--dados", default=os.path.join("dados", "landmarks_vlibrasil_signbank_slots.npz"))
    parser.add_argument("--etapa", choices=["estimativa", "final", "ambas"], default="ambas")
    parser.add_argument("--sementes", default="1", help="Sementes da estimativa, ex.: 1 ou 1,2,3")
    parser.add_argument("--aug_copies", type=int, default=2)
    parser.add_argument("--epocas", type=int, default=40, help="Epocas por rodada e no treino final")
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--pular_verificacao", action="store_true")
    args = parser.parse_args()

    dados = args.dados if os.path.isabs(args.dados) else os.path.join(AQUI, args.dados)
    if not args.pular_verificacao:
        ok, problemas = verificar(dados, REGEX)
        if not ok:
            raise SystemExit("\nCorrija o problema acima antes de treinar (ou use --pular_verificacao por sua conta e risco).")
    saida = args.saida if os.path.isabs(args.saida) else os.path.join(AQUI, args.saida)
    os.makedirs(saida, exist_ok=True)
    py = sys.executable
    comum = ["--data", dados, "--signer_regex", REGEX, "--no_face", "--aug_copies", str(args.aug_copies),
             "--loso_epochs", str(args.epocas)]

    if args.etapa in ("estimativa", "ambas"):
        rodar([py, "rodar_sementes.py", "--sementes", args.sementes, "--saida", os.path.join(saida, "res_tudo.json"), "--"]
              + comum + ["--so_treino", "99", "--val_split", "0", "--leave_one_signer_out",
                         "--salvar_proba", os.path.join(saida, "proba_tudo_seed{seed}.npz"),
                         "--counts_output", os.path.join(saida, "contagem_estimativa.csv")])
    if args.etapa in ("final", "ambas"):
        rodar([py, "train_lstm_hands_face.py"] + comum
              + ["--treinar_tudo", "--output", os.path.join(saida, "modelo_vlibras_signbank_final.keras"),
                 "--counts_output", os.path.join(saida, "contagem_final.csv")])
    print(f"\nPronto. Resultados em: {saida}")


if __name__ == "__main__":
    main()
