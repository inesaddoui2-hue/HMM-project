"""
Comparaison quantitative de l'intensite de modification par fold-change,
avec correction bayesienne. Section 2.2 du papier.

        E(p1 | x1)       (alpha + x1) (alpha + beta + n2)
     ---------------  =  --------------------------------
        E(p2 | x2)       (alpha + x2) (alpha + beta + n1)

alpha = 1, beta = m (le nombre de bins).
"""

import numpy as np
import pandas as pd

from preprocessing import comptages, BIN_SIZE

ALPHA = 1.0        # prior Beta, section 2.2
TAU = 3.0          # seuil de fold-change, section 3.1


def fold_change_naif(x1, x2):
    """Rapport brut des comptages, sans correction. Mentionne en introduction.

    Renvoie inf quand x2 vaut 0 et nan quand les deux valent 0 : c'est le
    defaut qu'on veut montrer, on le laisse apparaitre.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return x1 / x2


def classer(ratio, tau=TAU):
    """Etat de chaque bin : 0 non differentiel, 1 enrichi L1, 2 enrichi L2.

    Les nan restent a 0, toute comparaison avec nan etant fausse.
    """
    etat = np.zeros(len(ratio), dtype=np.int8)
    etat[ratio > tau] = 1
    etat[ratio < 1 / tau] = 2
    return etat


def facteur_profondeur(x1, x2, alpha=ALPHA, beta=None):
    """(alpha + beta + n2) / (alpha + beta + n1), la correction de profondeur."""
    if beta is None:
        beta = len(x1)
    return (alpha + beta + x2.sum()) / (alpha + beta + x1.sum())


def fold_change(x1, x2, alpha=ALPHA, beta=None):
    """Equation 3 : rapport des intensites attendues a posteriori."""
    if beta is None:
        beta = len(x1)
    return (alpha + x1) / (alpha + x2) * facteur_profondeur(x1, x2, alpha, beta)


def log_fold_change(x1, x2, alpha=ALPHA, beta=None):
    """Le meme rapport en log2.

    Le papier ecrit un log sans preciser la base ; log2 se lit mieux sur un
    graphe, un doublement valant exactement 1.
    """
    return np.log2(fold_change(x1, x2, alpha, beta))


if __name__ == "__main__":
    x1 = comptages("ES").astype(float)
    x2 = comptages("NP").astype(float)

    table = pd.DataFrame({
        "debut": np.arange(len(x1)) * BIN_SIZE,
        "ES": x1.astype(int),
        "NP": x2.astype(int),
        "log2": log_fold_change(x1, x2),
        "etat": classer(fold_change(x1, x2)),
    })
    table.index.name = "bin"
    print(table)
