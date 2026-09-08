"""
[Fichier genere par Claude — reference, a recopier, pas a lancer]

hmm.py — section 2.3 du papier ChIPDiff (Xu et al. 2008).

Assemble les trois pieces :

    emission.py     l'equation 4, specifique a ChIPDiff        <- ecrit par nous
    hmmlearn        forward-backward et Baum-Welch             <- bibliotheque
    ce fichier      le branchement, le seuil rho, la fusion    <- ecrit par nous

L'ADAPTATEUR
------------
hmmlearn calcule normalement les emissions lui-meme, a partir d'observations
et d'une loi parametrique (gaussienne, multinomiale...). Aucune de ces lois ne
correspond a l'equation 4.

L'astuce : on lui passe directement log(E) comme "observations", et on
redefinit _compute_log_likelihood pour qu'elle rende son entree telle quelle.
hmmlearn ne sert plus que de moteur : forward-backward et Baum-Welch.

Deux reglages sont indispensables :

    params="t"        ne reestimer QUE les transitions. Sans lui, hmmlearn
                      apprendrait aussi les emissions et l'equation 4 serait
                      ecrasee des la premiere iteration.
    init_params=""    ne rien initialiser tout seul : startprob_ et transmat_
                      sont poses a la main.

Le parametre `lengths` de fit() dit ou s'arrete une region et ou commence la
suivante. Sans lui, hmmlearn croirait que les 4 634 regions n'en forment qu'une.

REFERENCES
----------
    hmmlearn, licence BSD-3, github.com/hmmlearn/hmmlearn
    Rabiner, L. (1989) Proc. IEEE, 77, 257-286, pour les algorithmes eux-memes.
"""

import numpy as np
import pandas as pd
from hmmlearn.base import BaseHMM

from emission import emission
from preprocessing import comptages, BIN_SIZE
from scoreF import bins_retenus, merge_regions

RHO = 0.95              # seuil de confiance du decodage, section 3.1
N_ITER = 50             # iterations maximales de Baum-Welch
TOL = 1e-6              # arret quand la log-vraisemblance stagne

ETAT_NEUTRE, ETAT_L1, ETAT_L2 = 0, 1, 2


class HMMEmissionFixe(BaseHMM):
    """HMM dont les emissions sont fournies, pas apprises.

    X n'est pas une observation brute : c'est deja log(E). La methode rend
    donc son entree telle quelle.
    """

    def _compute_log_likelihood(self, X):
        return X

    def _init(self, X, lengths=None):
        pass

    def _check(self):
        pass


def entrainer(emissions, n_iter=N_ITER, tol=TOL):
    """Apprend la matrice de transition sur toutes les regions.

    Parametres
    ----------
    emissions : liste de tableaux (k, 3), un par region

    Retour
    ------
    le modele entraine ; sa matrice est dans modele.transmat_
    """
    modele = HMMEmissionFixe(n_components=3, params="t", init_params="",
                             n_iter=n_iter, tol=tol)
    modele.startprob_ = np.array([1.0, 0.0, 0.0])      # l'etat initial est a0
    modele.transmat_ = np.full((3, 3), 1.0 / 3.0)      # depart uniforme

    X = np.log(np.vstack(emissions))
    lengths = [len(E) for E in emissions]
    modele.fit(X, lengths)
    return modele


def decoder(modele, E):
    """Probabilites a posteriori de chaque etat, pour une region. Forme (k, 3)."""
    return modele.predict_proba(np.log(E))


def etats_seuilles(gamma, rho=RHO):
    """Etat retenu par bin : 0, 1 ou 2.

    Un bin n'est declare differentiel que si sa probabilite depasse rho.
    """
    etats = np.zeros(len(gamma), dtype=np.int8)
    etats[gamma[:, ETAT_L1] > rho] = ETAT_L1
    etats[gamma[:, ETAT_L2] > rho] = ETAT_L2
    return etats


def fusionner_dhms(etats, decalage=0):
    """Regroupe les bins differentiels consecutifs de MEME sens.

    "Consecutive DHMSs with no gap between them were merged into DHMS
    regions" (section 2.3). Contrairement a la fusion du score F, aucun trou
    n'est tolere ici.

    Retour
    ------
    liste de (debut, fin, etat), en indices de bins du chromosome, fin exclue.
    """
    resultat = []
    debut = None

    for i, e in enumerate(etats):
        if e != 0 and (debut is None or etats[debut] != e):
            if debut is not None:
                resultat.append((debut + decalage, i + decalage, int(etats[debut])))
            debut = i
        elif e == 0 and debut is not None:
            resultat.append((debut + decalage, i + decalage, int(etats[debut])))
            debut = None

    if debut is not None:
        resultat.append((debut + decalage, len(etats) + decalage, int(etats[debut])))
    return resultat


def chipdiff(x1, x2, rho=RHO):
    """La chaine complete : comptages -> regions DHMS.

    Retour
    ------
    dhms   : liste de (debut_bin, fin_bin, etat)
    modele : le HMM entraine, pour inspecter transmat_
    """
    n1, n2, m = x1.sum(), x2.sum(), len(x1)

    regions = merge_regions(bins_retenus(x1, x2))
    emissions = [emission(x1[d:f], x2[d:f], n1, n2, m) for d, f in regions]

    modele = entrainer(emissions)

    dhms = []
    for (d, _), E in zip(regions, emissions):
        etats = etats_seuilles(decoder(modele, E), rho)
        dhms.extend(fusionner_dhms(etats, decalage=d))

    return dhms, modele


if __name__ == "__main__":
    x1 = comptages("ES").astype(float)
    x2 = comptages("NP").astype(float)

    dhms, modele = chipdiff(x1, x2)

    print("matrice de transition apprise par Baum-Welch")
    print(pd.DataFrame(modele.transmat_.round(4),
                       index=["depuis a0", "depuis a1", "depuis a2"],
                       columns=["vers a0", "vers a1", "vers a2"]), "\n")

    table = pd.DataFrame(dhms, columns=["debut_bin", "fin_bin", "etat"])
    table["n_bins"] = table["fin_bin"] - table["debut_bin"]
    table["sens"] = table["etat"].map({1: "ES", 2: "NP"})
    table["debut_pb"] = table["debut_bin"] * BIN_SIZE
    table["fin_pb"] = table["fin_bin"] * BIN_SIZE
    print(table[["debut_bin", "fin_bin", "n_bins", "sens", "debut_pb", "fin_pb"]])

    n_es = int((table["etat"] == 1).sum())
    n_np = int((table["etat"] == 2).sum())
    total = len(table)
    print(f"\nregions DHMS : {total:,}")
    if total:
        print(f"  enrichies ES : {n_es:>5,}  ({100*n_es/total:.1f} %)")
        print(f"  enrichies NP : {n_np:>5,}  ({100*n_np/total:.1f} %)")
        print(f"  bins couverts : {int(table['n_bins'].sum()):,}")
        print(f"\npour comparaison, le papier sur le genome entier :")
