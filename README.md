# HMM-project

Réimplémentation de ChIPDiff (Xu et al., Bioinformatics 2008) : identification
des DHMS (Differential Histone Modification Sites) entre cellules souches
embryonnaires (ES) et progéniteurs neuraux (NP), sur le chromosome 19 de la souris.

## Données

Chromosome 19, marque H3K27me3, deux types cellulaires :

- **ES** — cellules souches embryonnaires de souris, état pluripotent
- **NP** — progéniteurs neuraux, dérivés des ES, déjà engagés vers le neural

H3K27me3 est une marque répressive déposée par Polycomb. Comparer ES et NP
revient à demander quels gènes se font réprimer, ou dé-réprimer, quand la
cellule quitte la pluripotence pour s'engager dans une voie.

Source : GEO GSE12241 (Mikkelsen et al., 2007). Fichiers attendus ici :

```
data/chr19-data/ES.chr19.tsv
data/chr19-data/NP.chr19.tsv
```

## Pipeline

```
  fichiers .tsv
        |
  readfile.py           lit les fichiers
        |
  preprocessing.py      découpe chr19 en bins, compte les fragments
        |
        |               x1, x2 : un comptage par bin
        |
  scoreF.py             garde les bins assez couverts, les fusionne en régions
        |
  emission.py           équation 4 : score de chaque bin pour les 3 états
        |
  chipdiff.py           HMM (Baum-Welch + décodage), seuil rho, fusion
        |
  régions DHMS
```

`FCbay.py` est à part : il compare le fold-change brut au fold-change corrigé
(équation 3) pour montrer pourquoi la correction bayésienne est nécessaire.
Il n'alimente pas la suite du pipeline.

## Les scripts

**readfile.py** : lit les `.tsv` bruts et renvoie les positions des fragments.

**preprocessing.py** : découpe le chromosome en bins de taille `BIN_SIZE` et
compte les fragments par bin. Expose `comptages(nom)`, qui renvoie le vecteur
de comptages d'une condition.

**scoreF.py** : écarte les bins trop peu couverts (`bins_retenus`) puis regroupe
les bins voisins en régions candidates (`merge_regions`). Le HMM ne tournera que
sur ces régions, pas sur le chromosome entier.

**FCbay.py** : fold-change entre les deux conditions, brut et corrigé par un
prior Beta plus une correction de profondeur de librairie.

**emission.py** : équation 4 du papier, la seule partie vraiment spécifique à
ChIPDiff. Pour chaque bin, trois facteurs disant à quel point ce bin plaide pour
chacun des états. Sortie : une matrice `(nombre de bins, 3)` par région.

**chipdiff.py** : branche ces matrices sur `hmmlearn`, apprend la matrice de
transition par Baum-Welch, décode avec forward-backward, ne retient que les bins
dont la probabilité dépasse `RHO`, et fusionne les bins consécutifs de même sens.
Sortie : liste de `(début_bin, fin_bin, état)`.

Les trois états : `a0` non différentiel, `a1` enrichi ES, `a2` enrichi NP.

## Exécution
Pour reproduire tous les résultats du rapport :

```
python resultat.py
```

Chaque module peut aussi se lancer seul, ce qui permet de vérifier une étape
sans relancer le reste :

```
python readfile.py
python preprocessing.py
python scoreF.py
python emission.py
python chipdiff.py
python FCbay.py
```
Chaque script a un bloc `if __name__ == "__main__"` qui affiche ses sorties
intermédiaires, ce qui permet de vérifier une étape sans lancer tout le reste.
`python FCbay.py` se lance quand on veut, indépendamment.

## Dépendances

```
numpy pandas scipy hmmlearn matplotlib
```

## Référence

Xu H., Wei C.-L., Lin F., Sung W.-K. (2008) *An HMM approach to genome-wide
identification of differential histone modification sites from ChIP-seq data.*
Bioinformatics 24(20):2344-2349.
