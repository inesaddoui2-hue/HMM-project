import numpy as np
import pandas as pd
from preprocessing import comptages, CHR19_LENGTH, BIN_SIZE

ETA = 0.7
M = CHR19_LENGTH // BIN_SIZE + 1      # 61 322 bins
SEUIL_F = 2 / (M * ETA)               # 4,659e-05

"""Score F : enrichissement combiné, normalisé par la profondeur."""
def f_score(x1, x2):
    return x1 / x1.sum() + x2 / x2.sum()

"""Masque : ce bin porte-t-il assez de signal pour être analysé ?"""
def bins_retenus(x1, x2):
    return f_score(x1, x2) > SEUIL_F


MAX_GAP = 1        # bins non retenus tolérés à l'intérieur d'une région


def merge_regions(masque, max_gap=MAX_GAP):
    """Fusionne les bins retenus en régions continues (début, fin), fin exclue."""
    idx = np.arange(len(masque))[masque]                    # numéros des bins retenus
    if idx.size == 0:
        return []

    ecarts = np.diff(idx)                                   # distance entre bins retenus voisins
    coupures = np.arange(len(ecarts))[ecarts > max_gap + 1]

    debuts = np.concatenate(([idx[0]], idx[coupures + 1]))
    fins   = np.concatenate((idx[coupures], [idx[-1]])) + 1
    return list(zip(debuts, fins))




if __name__ == "__main__":
    x1 = comptages("ES").astype(float)
    x2 = comptages("NP").astype(float)
 
    masque = bins_retenus(x1, x2)
    regions = merge_regions(masque)
 
    table = pd.DataFrame(regions, columns=["debut_bin", "fin_bin"])
    table["n_bins"] = table["fin_bin"] - table["debut_bin"]
    table["debut_pb"] = table["debut_bin"] * BIN_SIZE
    table["fin_pb"] = table["fin_bin"] * BIN_SIZE
    print(table)

   