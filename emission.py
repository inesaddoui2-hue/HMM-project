# ---------------------------------------------------------------------------
# [Fichier genere par Claude — reference, a recopier, pas a lancer]
#
# emission.py — equation 4 du papier ChIPDiff (Xu et al. 2008, section 2.3).
#
# C'est la seule partie du HMM qui soit specifique a ChIPDiff. Forward-backward
# et Baum-Welch sont des algorithmes generiques qu'on prend dans hmmlearn ;
# l'equation 4, elle, n'existe dans aucune bibliotheque.
#
# CE QU'ELLE CALCULE
# ------------------
# Pour un bin, elle rend trois facteurs : a quel point ce bin, tout seul, plaide
# pour chacun des trois etats.
#
#     e = 1     ce bin n'a pas d'avis
#     e > 1     il pousse vers cet etat
#     e < 1     il l'ecarte
#
# Ces facteurs sont bornes : au plus 2 pour a0 et 4 pour a1 et a2, parce qu'une
# probabilite ne depasse pas 1 et qu'on divise par le prior [1/2, 1/4, 1/4].
#
# LE RAISONNEMENT, EN QUATRE TEMPS
# --------------------------------
# 1. Le comptage x n'est pas l'intensite p, c'est un tirage gouverne par elle
#    (equation 1, binomiale). Avec un prior Beta (equation 2), la conjugaison
#    donne un posterior Beta(alpha + x, beta + n - x) : la liste des valeurs
#    plausibles de p, avec leur credibilite.
#
# 2. On veut la loi du RAPPORT p1/p2, et plus precisement la masse qu'elle place
#    au-dessus d'un seuil. Comme les profondeurs sont enormes devant les
#    comptages, les Beta se comportent comme des Gamma, et le rapport de deux
#    Gamma se ramene a une Beta : la reponse s'obtient par la fonction beta
#    incomplete regularisee, betainc.
#
# 3. Les seuils tau et 1/tau decoupent l'axe des rapports en trois plages, une
#    par etat. Deux appels suffisent, la troisieme masse s'obtient par
#    complement.
#
# 4. On divise par le prior. Sans donnees, deux intensites de meme loi ont une
#    chance sur deux de rester dans un facteur tau : cette preference pour a0
#    vient de la geometrie du seuil, pas de la biologie. La diviser fait qu'un
#    bin sans aucun fragment emet exactement [1, 1, 1], donc ne vote pas.
#
# VALIDATION
# ----------
# La forme close du point 2 est verifiee par tirage aleatoire dans les lois Beta
# (voir test_emission.py) : les deux coincident a 1e-3 pres, ce qui est le bruit
# du tirage.
# ---------------------------------------------------------------------------

import numpy as np
from scipy.special import betainc

ALPHA = 1.0        # prior Beta, section 2.2
TAU = 3.0          # seuil de fold-change definissant les etats, section 3.1

ETATS = ("a0", "a1", "a2")   # non differentiel, enrichi L1, enrichi L2

PLANCHER = 1e-300           # evite log(0) = -inf en aval


# Parametres de la loi Beta du bin, apres avoir vu x fragments sur n.
# Conjugaison binomiale-Beta : Beta(alpha + x, beta + n - x), avec beta = m.
def posterior_beta(x, n, m, alpha=ALPHA):
    return alpha + x, m + n - x


# P(p1/p2 > t), au vu des comptages x1 et x2.
# x1 et x2 peuvent etre des tableaux : la sortie a alors la meme forme.
def proba_ratio_sup(t, x1, x2, n1, n2, m, alpha=ALPHA):
    a1, _ = posterior_beta(np.asarray(x1, dtype=float), n1, m, alpha)
    a2, _ = posterior_beta(np.asarray(x2, dtype=float), n2, m, alpha)

    r1 = alpha + m + n1        # "taille" de la librairie 1
    r2 = alpha + m + n2

    c = t * r1 / r2
    return 1.0 - betainc(a1, a2, c / (1.0 + c))


# Les trois probabilites P(a0), P(a1), P(a2). Somment a 1.
# Sortie de forme (k, 3) pour k bins.
def probas_trois_etats(x1, x2, n1, n2, m, tau=TAU, alpha=ALPHA):
    p_a1 = proba_ratio_sup(tau, x1, x2, n1, n2, m, alpha)
    p_a2 = 1.0 - proba_ratio_sup(1.0 / tau, x1, x2, n1, n2, m, alpha)
    p_a0 = 1.0 - p_a1 - p_a2
    return np.stack([p_a0, p_a1, p_a2], axis=-1)


# Les trois probabilites SANS aucune donnee.
# Meme calcul, mais les deux intensites ont la meme loi : le resultat ne depend
# que de alpha et tau. Pour alpha = 1 cela vaut [1/2, 1/4, 1/4].
def prior_trois_etats(tau=TAU, alpha=ALPHA):
    q_a1 = 1.0 - betainc(alpha, alpha, tau / (1.0 + tau))
    q_a2 = betainc(alpha, alpha, (1.0 / tau) / (1.0 + 1.0 / tau))
    return np.array([1.0 - q_a1 - q_a2, q_a1, q_a2])


# Les facteurs d'emission, prets pour le HMM.
#
# Parametres
#   x1, x2 : tableaux de k comptages, les bins d'une region
#   n1, n2 : profondeurs des deux librairies
#   m      : nombre de bins du chromosome
#
# Retour
#   tableau (k, 3), colonnes dans l'ordre a0, a1, a2.
#
# Les valeurs sont planchees a PLANCHER : une emission exactement nulle
# donnerait log(0) = -inf, que hmmlearn refuse. Le plancher est assez bas pour
# ne rien changer au resultat.
def emission(x1, x2, n1, n2, m, tau=TAU, alpha=ALPHA):
    P = probas_trois_etats(x1, x2, n1, n2, m, tau, alpha)
    return np.clip(P / prior_trois_etats(tau, alpha), PLANCHER, None)


# Une matrice d'emission par region.
#
# Parametres
#   x1, x2  : les vecteurs de comptages du chromosome entier
#   regions : liste de couples (debut, fin), fin exclue
#
# Retour
#   liste de tableaux (k, 3), un par region.
def emissions_par_region(x1, x2, regions, tau=TAU, alpha=ALPHA):
    n1, n2 = float(x1.sum()), float(x2.sum())
    m = len(x1)
    return [emission(x1[d:f], x2[d:f], n1, n2, m, tau, alpha) for d, f in regions]


if __name__ == "__main__":
    import pandas as pd

    from preprocessing import comptages, BIN_SIZE
    from scoreF import bins_retenus, merge_regions

    x1 = comptages("ES").astype(float)
    x2 = comptages("NP").astype(float)
    n1, n2, m = x1.sum(), x2.sum(), len(x1)

    regions = merge_regions(bins_retenus(x1, x2))
    matrices = emissions_par_region(x1, x2, regions)

    print(f"{len(regions):,} regions   ->   {len(matrices):,} matrices d'emission")
    print("chacune de forme (nombre de bins de la region, 3)\n")
    for (d, f), E in list(zip(regions, matrices))[:5]:
        print(f"  region bins {d:>6} -> {f:>6}   E.shape = {E.shape}")

    # --- une region en detail ---
    d, f = max(regions, key=lambda r: r[1] - r[0])
    E = emission(x1[d:f], x2[d:f], n1, n2, m)
    table = pd.DataFrame(E, columns=["e_a0", "e_a1", "e_a2"])
    table.insert(0, "NP", x2[d:f].astype(int))
    table.insert(0, "ES", x1[d:f].astype(int))
    table.index = range(d, f)
    table.index.name = "bin"

    print(f"\nla plus grande region : bins {d} a {f}, soit {f-d} bins")
    print(f"chr19:{d*BIN_SIZE:,}-{f*BIN_SIZE:,}\n")
    print(table.round(2))

    # --- verifications ---
    tous = np.vstack(matrices)
    print(f"\n--- verifications sur les {len(tous):,} bins analyses ---")
    print(f"prior                        {np.round(prior_trois_etats(), 4)}")
    print(f"bin vide -> [1, 1, 1]        {np.allclose(emission(0, 0, n1, n2, m), 1.0, atol=1e-9)}")
    print(f"e(a0) <= 2                   {bool((tous[:, 0] <= 2 + 1e-9).all())}")
    print(f"e(a1), e(a2) <= 4            {bool((tous[:, 1:] <= 4 + 1e-9).all())}")
    print(f"aucun nan                    {not bool(np.isnan(tous).any())}")