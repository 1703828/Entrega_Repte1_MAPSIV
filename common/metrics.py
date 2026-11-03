"""Mètriques de lectura. Sense pandas ni numpy: s'han de poder fer servir a l'entorn de Codabench.

  plate_accuracy / char_accuracy: les OFICIALS, idèntiques a evaluation/run_local.py (text normalitzat a majúscules
      i només lletres i números; caràcter a caràcter per POSICIÓ, sobre la llargada de la veritat).
  cer: Character Error Rate (distància d'edició / llargada de la veritat). Només diagnòstic: NO és la mètrica del repte.
"""
import re


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def plate_accuracy(pairs):
    """pairs: [(veritat, predicció)]."""
    return sum(norm(p) == norm(t) for t, p in pairs) / len(pairs)


def char_accuracy(pairs):
    ok = sum(sum(a == b for a, b in zip(norm(p), norm(t))) for t, p in pairs)
    return ok / sum(len(norm(t)) for t, _ in pairs)


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, start=1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def cer(pairs):
    return sum(edit_distance(norm(p), norm(t)) for t, p in pairs) / sum(len(norm(t)) for t, _ in pairs)
