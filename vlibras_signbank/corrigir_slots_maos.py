"""Reorganiza os slots das maos do .npz para nao dependerem do rotulo Left/Right do MediaPipe.

Problema: o rotulo de lateralidade do MediaPipe varia de um video para outro e ate de um quadro
para outro dentro do mesmo video (no V-Librasil, ~11% dos pares de quadros consecutivos com uma
so mao trocam de slot, e so ~28% das palavras tem o mesmo slot dominante nos tres articuladores,
praticamente o acaso). Isso embaralha a entrada do modelo.

O que este script faz, amostra por amostra:
  1. Rastreia as maos quadro a quadro pela posicao do punho (X_traj): a mao que esta mais perto
     de onde a trilha estava no quadro anterior continua na mesma trilha. Isso desfaz as trocas
     no meio do video.
  2. Escolhe a trilha dominante (maior movimento total do punho; empate: mais quadros) e a
     coloca no slot 0. A outra vai para o slot 1.

Assim, slot 0 = mao dominante do sinal e slot 1 = mao de apoio, para todos os articuladores.
Todo array do .npz com formato (N, T, 2, ...) que dependa do slot (X_hands, X_traj, ...) e
reordenado do mesmo jeito. O resto (X_face, y, paths, ...) e copiado sem mudanca.

Uso:
    python corrigir_slots_maos.py --data landmarks_vlibrasil_completo_traj.npz \\
        --output landmarks_vlibrasil_completo_traj_slots.npz

Depois, rode os mesmos scripts de antes trocando so o --data pelo arquivo novo.
"""
import argparse

import numpy as np


def _dist(p, q, neutro):
    """Distancia entre posicoes; se a trilha ainda nao tem posicao (q None), custo neutro."""
    if q is None:
        return neutro
    return float(np.linalg.norm(p - q))


def reordenar_amostra(det, pos, limiar):
    """det: (T, 2) bool, mao detectada por slot. pos: (T, 2, 2) posicao do punho por slot.

    Devolve src (T, 2): src[t, novo_slot] = slot original da mao que vai para o novo slot
    (ou -1 se nao ha mao ali nesse quadro). Slot 0 novo = mao dominante."""
    T = det.shape[0]
    trilha = -np.ones((T, 2), dtype=int)  # trilha[t, k] = slot original da trilha k no quadro t
    ultimo = [None, None]

    for t in range(T):
        slots = np.where(det[t])[0]
        if len(slots) == 0:
            continue
        if len(slots) == 2:
            p0, p1 = pos[t, 0], pos[t, 1]
            if ultimo[0] is None and ultimo[1] is None:
                # primeiro quadro com duas maos: ordem estavel pela posicao horizontal
                a, b = (0, 1) if p0[0] <= p1[0] else (1, 0)
            else:
                custo_id = _dist(p0, ultimo[0], limiar) + _dist(p1, ultimo[1], limiar)
                custo_tr = _dist(p1, ultimo[0], limiar) + _dist(p0, ultimo[1], limiar)
                a, b = (0, 1) if custo_id <= custo_tr else (1, 0)
            trilha[t, 0], trilha[t, 1] = a, b
            ultimo[0], ultimo[1] = pos[t, a], pos[t, b]
        else:
            s = int(slots[0])
            p = pos[t, s]
            if ultimo[0] is None and ultimo[1] is None:
                k = 0
            elif ultimo[1] is None:
                k = 0 if _dist(p, ultimo[0], limiar) <= limiar else 1
            elif ultimo[0] is None:
                k = 1 if _dist(p, ultimo[1], limiar) <= limiar else 0
            else:
                k = 0 if _dist(p, ultimo[0], limiar) <= _dist(p, ultimo[1], limiar) else 1
            trilha[t, k] = s
            ultimo[k] = p

    # mao dominante = trilha com mais movimento do punho (desempate: mais quadros)
    mov = np.zeros(2)
    cont = np.zeros(2)
    for k in range(2):
        ok = trilha[:, k] >= 0
        cont[k] = ok.sum()
        pk = np.zeros((T, 2))
        pk[ok] = pos[np.arange(T)[ok], trilha[ok, k]]
        par = ok[1:] & ok[:-1]
        if par.any():
            mov[k] = np.linalg.norm(np.diff(pk, axis=0), axis=-1)[par].sum()
    dom = 0 if (mov[0], cont[0]) >= (mov[1], cont[1]) else 1
    return trilha[:, [dom, 1 - dom]]


def aplicar(arr, src):
    """arr: (N, T, 2, ...). src: (N, T, 2). Reordena o eixo 2 conforme src (-1 vira zero)."""
    out = np.zeros_like(arr)
    for k in range(2):
        s = src[..., k]                                   # (N, T)
        ok = s >= 0
        n_idx, t_idx = np.where(ok)
        out[n_idx, t_idx, k] = arr[n_idx, t_idx, s[n_idx, t_idx]]
    return out


def main():
    parser = argparse.ArgumentParser(description="Reorganiza os slots das maos (rastreio + mao dominante no slot 0).")
    parser.add_argument("--data", required=True, help=".npz de entrada")
    parser.add_argument("--output", required=True, help=".npz de saida")
    parser.add_argument("--limiar", type=float, default=0.5,
                        help="Distancia (unidade do X_traj) acima da qual uma mao unica e tratada como a outra mao")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    chaves = list(data.keys())
    print("Chaves no .npz:", chaves)
    if "X_hands" not in data or "X_traj" not in data:
        raise SystemExit("Preciso de 'X_hands' e 'X_traj' (posicao do punho) para rastrear as maos.")

    H = data["X_hands"]                                   # (N, T, 2, 21, 3)
    Tr = data["X_traj"]                                   # (N, T, 2, 2)
    N, T = H.shape[:2]
    det = H[:, :, :, 1:, :].reshape(N, T, 2, -1).any(-1)  # pulso e sempre 0, por isso so 1..20

    src = np.stack([reordenar_amostra(det[i], Tr[i], args.limiar) for i in range(N)])

    # estatisticas antes/depois
    novo_det = src >= 0
    mudou = (src != np.arange(2)[None, None, :]) & novo_det
    print(f"\nQuadros com mao reatribuida a outro slot: {mudou.sum() / max(novo_det.sum(), 1):.3f}")
    print(f"Ocupacao dos slots antes : slot0={det[..., 0].mean():.2f} slot1={det[..., 1].mean():.2f}")
    print(f"Ocupacao dos slots depois: slot0={novo_det[..., 0].mean():.2f} slot1={novo_det[..., 1].mean():.2f}")

    saida = {}
    reordenadas = []
    for k in chaves:
        a = data[k]
        if a.ndim >= 3 and a.shape[0] == N and a.shape[1] == T and a.shape[2] == 2 and k != "X_face" and k != "X_face_norm":
            saida[k] = aplicar(a, src)
            reordenadas.append(k)
        else:
            saida[k] = a
    print("Arrays reordenados:", reordenadas)

    np.savez_compressed(args.output, **saida)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
