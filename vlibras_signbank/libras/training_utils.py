"""Versao minima dos utilitarios de treino (so o que os scripts desta etapa usam).

Se voce ja tem a pasta libras/ do repositorio, NAO sobrescreva: use a do repositorio.
Esta versao existe para a etapa de treino rodar sem precisar clonar o repositorio.
"""
import csv


def load_class_filter(arg):
    """None -> sem filtro. 'A,B,C' -> {'A','B','C'}. '@arquivo.txt' -> classes do arquivo (uma por linha)."""
    if arg is None:
        return None
    if arg.startswith("@"):
        with open(arg[1:], encoding="utf-8-sig") as f:
            return {l.strip() for l in f if l.strip()}
    return {c.strip() for c in arg.split(",") if c.strip()}


def extract_signer_ids(paths, signer_regex):
    """Primeiro grupo da regex aplicada a cada caminho. Erro claro se algum nao casar."""
    import numpy as np
    ids = []
    for p in paths:
        m = signer_regex.search(str(p))
        if m is None:
            raise SystemExit(f"--signer_regex nao casou com o caminho: {p!r}")
        ids.append(m.group(1))
    return np.array(ids)


def log_counts(name, y, store):
    """Imprime e guarda a contagem de amostras por classe de uma divisao."""
    counts = {}
    for label in y.tolist():
        counts[label] = counts.get(label, 0) + 1
    store[name] = counts
    print(f"\nContagem de amostras - {name}:")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


def save_counts_csv(path, store):
    """CSV com uma linha por classe e uma coluna por divisao."""
    classes = sorted({c for counts in store.values() for c in counts})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classe"] + list(store.keys()))
        for c in classes:
            w.writerow([c] + [store[k].get(c, 0) for k in store])
    print(f"Contagem de amostras salva em {path}")
