# main.py — reproduit les chiffres de la section Resultats du rapport.
# affiche leurs sorties dans l'ordre du rapport.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import comptages, BIN_SIZE
from scoreF import bins_retenus, merge_regions
from FCbay import fold_change_naif, fold_change, log_fold_change, classer, TAU
from chipdiff import chipdiff, fusionner_dhms, RHO


# ===========================================================================
# 3.1  Comparaison par fold-change
# ===========================================================================

x1 = comptages("ES").astype(float)
x2 = comptages("NP").astype(float)
n1, n2, m = x1.sum(), x2.sum(), len(x1)

masque = bins_retenus(x1, x2)       # un True/False par bin du chromosome
regions = merge_regions(masque)     # fusionnes quand separes d'au plus 1 kb
n_retenus = int(masque.sum())       # nombre de True


# Masque des bins couverts par les regions candidates. Il y en a plus que de
# bins retenus : la fusion avale les trous d'un bin. C'est ce que voit le HMM.
dans_regions = np.zeros(m, dtype=bool).   # tout a False, longueur du chromosome
for d, f in regions:
    dans_regions[d:f] = True            # on allume la tranche de chaque region

print("=" * 62)
print("3.1  Comparaison par fold-change")
print("=" * 62)
print(f"profondeurs                 ES {int(n1):>8,}   NP {int(n2):>8,}")
print(f"bins du chromosome 19       {m:>8,}   ({BIN_SIZE} pb chacun)")
print(f"bins retenus par le score F {n_retenus:>8,}")
print(f"regions candidates          {len(regions):>8,}")
print(f"bins couverts               {int(dans_regions.sum()):>8,}")

# Les deux rapports sont calcules sur le chromosome entier, parce que
# fold_change a besoin des profondeurs globales et du nombre total de bins.
# On annule ensuite tout ce qui est hors regions candidates, pour que les
# trois methodes jugent exactement les memes bins.
etats_naif = classer(fold_change_naif(x1, x2)) * dans_regions
etats_bayes = classer(fold_change(x1, x2)) * dans_regions

print(f"\nbins declares differentiels")
print(f"  rapport brut              {int((etats_naif != 0).sum()):>8,}")
print(f"  version bayesienne        {int((etats_bayes != 0).sum()):>8,}")


# ===========================================================================
# 3.2  Identification des DHMS par HMM
# ===========================================================================

print("\n" + "=" * 62)
print("3.2  Identification des DHMS par HMM")
print("=" * 62)

dhms, modele = chipdiff(x1, x2)                 # entrainement, decodage, fusion

# Rien n'a ete impose a cette matrice : sa diagonale mesure la persistance des
# domaines, et les zeros entre a1 et a2 disent qu'on ne passe jamais d'un
# domaine perdu a un domaine gagne sans traverser une zone sans difference.
print("\nmatrice de transition apprise par Baum-Welch")
print(pd.DataFrame(modele.transmat_.round(4),
                   index=["depuis a0", "depuis a1", "depuis a2"],
                   columns=["vers a0", "vers a1", "vers a2"]))

#dhms: les regions est une liste de tripelts [(d,f,etat), ...] debut bin jusqua fin bin 
n_bins_hmm = sum(f - d for d, f, _ in dhms) # on compte le nombre total de bin DHMS
n_es = sum(1 for _, _, e in dhms if e == 1) # on compte total etat 1="ES"

print(f"\nseuil de confiance rho      {RHO}")
print(f"bins retenus                {n_bins_hmm:>8,}")
print(f"regions DHMS                {len(dhms):>8,}")
print(f"   regions enrichies ES              {n_es:>8,}  ({100*n_es/len(dhms):.1f} %)")
print(f"  regions enrichies NP              {len(dhms)-n_es:>8,}  ({100*(len(dhms)-n_es)/len(dhms):.1f} %)")


# ===========================================================================
# 3.3  Comparaison des trois methodes
# ===========================================================================

print("\n" + "=" * 62)
print("3.3  Comparaison des trois methodes")
print("=" * 62)

# On fusionne les bins des deux fold-changes region par region, exactement
# comme le fait chipdiff.py pour le HMM. Sans ce decoupage, deux regions
# candidates voisines seraient recollees et la comparaison serait faussee.
dhms_naif, dhms_bayes = [], []
for d, f in regions:
    dhms_naif += fusionner_dhms(etats_naif[d:f], decalage=d)
    dhms_bayes += fusionner_dhms(etats_bayes[d:f], decalage=d)

lignes = []
for nom, liste in [("rapport brut", dhms_naif), ("bayesien", dhms_bayes), ("HMM", dhms)]:
    n_bins = sum(f - d for d, f, _ in liste)
    n_es_m = sum(1 for _, _, e in liste if e == 1)
    lignes.append({
        "methode": nom,
        "bins": n_bins,
        "regions": len(liste),
        "bins/region": round(n_bins / len(liste), 2),
        "% ES": round(100 * n_es_m / len(liste), 1),
    })

print()
print(pd.DataFrame(lignes).to_string(index=False))

# Le recouvrement dit ce que le HMM apporte reellement. S'il retrouve surtout
# des bins que le fold-change avait deja, son gain est un gain de specificite :
# il retire du bruit, il ne decouvre pas de nouveaux sites.
bins_hmm = np.zeros(m, dtype=bool)
for d, f, _ in dhms:
    bins_hmm[d:f] = True

commun = int((bins_hmm & (etats_bayes != 0)).sum())
ecartes = int(((etats_bayes != 0) & ~bins_hmm).sum())

print(f"\nrecouvrement HMM / bayesien")
print(f"  bins du HMM deja vus par le fold-change   {commun:,} / {n_bins_hmm:,}"
      f"  ({100*commun/n_bins_hmm:.0f} %)")
print(f"  bins du fold-change ecartes par le HMM    {ecartes:,}")


# ===========================================================================
# Figure 1 : RI-plot, reproduction de la Figure 1b de l'article
# ===========================================================================

# Aux faibles couvertures le log-ratio part dans tous les sens et franchit le
# seuil par hasard ; aux fortes il reste serre autour de zero. C'est cet effet
# d'entonnoir que le HMM corrige en exigeant que les voisins confirment.
idx = np.flatnonzero(dans_regions)

plt.figure(figsize=(6, 4.5))
plt.scatter((x1 + x2)[idx], log_fold_change(x1, x2)[idx], s=2, alpha=0.3)
plt.axhline(np.log2(TAU), ls="--")
plt.axhline(-np.log2(TAU), ls="--")
plt.xscale("log")
plt.xlabel("couverture du bin (ES + NP)")
plt.ylabel("log2 fold-change ES / NP")
plt.tight_layout()
plt.savefig("figure1.png", dpi=300)

print(f"\nfigure1.png enregistree  ({len(idx):,} bins)")